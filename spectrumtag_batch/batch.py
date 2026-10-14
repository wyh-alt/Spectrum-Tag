"""批处理引擎 —— 对一批音频/视频文件施加同一套水印设置。

设计要点：

* **图案按需生成、按需缓存**：固定文字/图片模式下所有文件共用同一张图案，
  只在第一个文件时渲染一次；"按文件名生成文字"模式下每个文件各自渲染。
* **单文件失败不中断整批**：每个文件包在自己的 try 里，异常记进结果继续跑。
* **进度是两级**：文件内进度（渲染）与整体进度（含文件计数），供 UI 显示。
* **取消是协作式的**：回调返回 False 即停止，已写完的文件保留，当前文件回滚。
* **输出保留输入的层级**：文件夹拖进来就在输出目录里建同名文件夹，里面的
  子目录照搬；单独拖入的文件平铺在输出目录根部。这层信息由
  :class:`SourceFile` 一路带到写入处。
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .core.params import OutputSpec, PatternSource, RenderJob
from .core.pattern import PatternError, build_pattern
from .core.render import EmptyRender, plan_render, render_audio
from .media.audio_io import (
    AUDIO_EXTENSIONS,
    MediaError,
    output_subtype_for,
    unique_output_path,
    write_audio,
)
from .media.decode import decode_media
from .media.video_io import (
    VIDEO_EXTENSIONS,
    is_video_file,
    mux_video,
)

# 接受的文件类型 = 音频 ∪ 视频
SUPPORTED_EXTENSIONS = tuple(sorted(set(AUDIO_EXTENSIONS) | set(VIDEO_EXTENSIONS)))


class ItemStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class TextMode(Enum):
    """批量处理时水印文字怎么来。"""

    FIXED = "fixed"                  # 整批共用一段固定文字
    FROM_FILENAME = "from_filename"  # 用每个文件的文件名（不含扩展名）


@dataclass(frozen=True)
class SourceFile:
    """一个待处理文件，连同它在输出目录里的落点。

    ``relative_dir`` 是相对输出目录的子目录：拖入文件夹时它是该文件在文件夹
    内的位置（首段就是被拖入文件夹的名字），单独拖入的文件为空串、直接落在
    输出目录根部。
    """

    path: str
    relative_dir: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def display_name(self) -> str:
        """带层级的显示名 —— 不同文件夹里的同名文件光看主名分不出来。"""
        if not self.relative_dir:
            return self.name
        return os.path.join(self.relative_dir, self.name)


@dataclass
class BatchItem:
    """单个文件的处理结果。"""

    source_path: str
    status: ItemStatus = ItemStatus.PENDING
    output_path: str = ""
    duration_sec: float = 0.0
    elapsed_sec: float = 0.0
    message: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.source_path)


@dataclass
class BatchProgress:
    """一次进度汇报。"""

    file_index: int          # 从 1 开始的序号
    file_count: int
    file_path: str
    file_progress: float     # 当前文件内 [0, 1]
    stage: str               # 当前阶段的中文描述

    @property
    def overall(self) -> float:
        """整体进度 [0, 1]。"""
        if self.file_count <= 0:
            return 0.0
        done = (self.file_index - 1) + max(0.0, min(1.0, self.file_progress))
        return done / self.file_count

    @property
    def overall_percent(self) -> int:
        return int(round(self.overall * 100))


@dataclass
class BatchResult:
    """整批的结果汇总。"""

    items: list[BatchItem] = field(default_factory=list)
    cancelled: bool = False

    @property
    def succeeded(self) -> int:
        return sum(1 for it in self.items if it.status is ItemStatus.DONE)

    @property
    def failed(self) -> list[BatchItem]:
        return [it for it in self.items if it.status is ItemStatus.FAILED]

    @property
    def skipped(self) -> list[BatchItem]:
        return [it for it in self.items if it.status is ItemStatus.SKIPPED]

    @property
    def total_seconds(self) -> float:
        return sum(it.elapsed_sec for it in self.items)


ProgressFn = Callable[[BatchProgress], bool]
"""进度回调，返回 False 表示请求取消。"""


def _subdir(*parts: str) -> str:
    """拼出相对输出目录的子目录；空的层级直接跳过。

    ``os.path.join("album", "")`` 会留下一个多余的尾分隔符，而驱动器根这类
    没有名字的输入根本不产生首段 —— 两种都在这儿抹平。
    """
    kept = [part for part in parts if part]
    return os.path.join(*kept) if kept else ""


def collect_sources(paths: Iterable[str], recursive: bool = True) -> list[SourceFile]:
    """把混合的「文件 / 文件夹」输入展开成待处理清单，并记下各自的落点。

    落点规则：

    * **文件夹** —— 以该文件夹的名字为顶层在输出目录里重建一套同样的层级，
      子目录照搬；
    * **单个文件** —— 平铺在输出目录根部，不为它单独建文件夹。

    两者混着拖时各按各的规矩走。顺序保持稳定：先出现的先处理，文件夹内的
    文件按名称排序。同一个文件被重复传入（或既在文件夹里又被单独拖入）只算
    一次，以先出现的那个身份为准。
    """
    found: list[SourceFile] = []
    seen: set[str] = set()

    def _add(path: str, relative_dir: str) -> None:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            return
        seen.add(key)
        found.append(SourceFile(path=path, relative_dir=relative_dir))

    for raw in paths:
        if not raw:
            continue
        if os.path.isdir(raw):
            # 只有被拖入的那个文件夹本身作为顶层 —— 它的上级路径不进输出
            root_name = os.path.basename(os.path.normpath(raw))
            if recursive:
                for dirpath, dirs, files in os.walk(raw):
                    dirs.sort()
                    inner = os.path.relpath(dirpath, raw)
                    inner = "" if inner == os.curdir else inner
                    for name in sorted(files):
                        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                            _add(os.path.join(dirpath, name), _subdir(root_name, inner))
            else:
                for name in sorted(os.listdir(raw)):
                    full = os.path.join(raw, name)
                    if os.path.isfile(full) and (
                        os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
                    ):
                        _add(full, root_name)
        elif os.path.isfile(raw):
            # 单个文件同样要过扩展名 —— 拖进来一张图片或一份文档时，
            # 那是想拿去做水印图案或压根不相干，不该混进待处理清单
            if os.path.splitext(raw)[1].lower() in SUPPORTED_EXTENSIONS:
                _add(raw, "")

    return found


def collect_files(paths: Iterable[str], recursive: bool = True) -> list[str]:
    """同 :func:`collect_sources`，但只取路径 —— 不关心输出层级的场合用它。"""
    return [item.path for item in collect_sources(paths, recursive)]


def _job_for_file(base_job: RenderJob, path: str, text_mode: TextMode) -> RenderJob:
    """按文件派生 job —— 只有"按文件名生成文字"模式会改变图案。"""
    if text_mode is not TextMode.FROM_FILENAME:
        return base_job
    if base_job.pattern.source is not PatternSource.TEXT:
        return base_job
    stem = Path(path).stem
    return base_job.with_pattern(text=stem or base_job.pattern.text)


def _target_dir(output_dir: str, source: SourceFile) -> str:
    """该文件最终写进哪个目录 —— 拖入文件夹来的文件落回对应的子目录。"""
    if not source.relative_dir:
        return output_dir
    return os.path.join(output_dir, source.relative_dir)


def _as_source(item: "SourceFile | str") -> SourceFile:
    """容忍直接给路径字符串的调用方 —— 那等同于"单独拖入的文件"。"""
    return item if isinstance(item, SourceFile) else SourceFile(path=str(item))


def _write_video_result(
    source_path: str,
    samples: np.ndarray,
    sample_rate: int,
    target_dir: str,
    output: OutputSpec,
) -> str:
    """把处理后的音轨合回原视频，画面原样保留。"""
    container = output.video_format.resolve(source_path)
    out_path = unique_output_path(
        target_dir, Path(source_path).stem, f".{container}", source_path
    )
    # 先落一个无损的中间 WAV，再让 ffmpeg 封进容器
    with tempfile.TemporaryDirectory(prefix="stbatch_mux_") as tmp:
        audio_path = os.path.join(tmp, "tagged.wav")
        write_audio(audio_path, samples, sample_rate, "wav", "PCM_24")
        mux_video(source_path, audio_path, out_path, container)
    return out_path


def _write_audio_result(
    source_path: str,
    samples: np.ndarray,
    sample_rate: int,
    source_subtype: str,
    target_dir: str,
    output: OutputSpec,
) -> str:
    """只导出音频，格式按设置或源文件定。"""
    fmt = output.audio_format.resolve(source_path)
    out_path = unique_output_path(
        target_dir, Path(source_path).stem, f".{fmt}", source_path
    )
    write_audio(out_path, samples, sample_rate, fmt, output_subtype_for(source_subtype))
    return out_path


def run_batch(
    files: Sequence["SourceFile | str"],
    job: RenderJob,
    output_dir: str,
    output: OutputSpec = OutputSpec(),
    text_mode: TextMode = TextMode.FIXED,
    progress: Optional[ProgressFn] = None,
    pattern_cache: Optional[dict] = None,
) -> BatchResult:
    """顺序处理一批文件。

    Parameters
    ----------
    files
        源文件列表（应已由 :func:`collect_sources` 展开）。用 :class:`SourceFile`
        可以带上各自的输出子目录；直接给路径字符串则一律写在输出目录根部。
    job
        水印设置。``text_mode`` 为 FROM_FILENAME 时文字会被逐文件替换。
    output_dir
        输出目录，不存在时创建。带层级的文件会在它下面重建同名目录。
    output
        输出形态：只出音频，还是把音轨合回视频；以及各自的容器格式。
    progress
        进度回调；返回 False 请求取消。
    pattern_cache
        可复用的图案缓存（键为 :meth:`PatternSpec.cache_key`）。跨批次复用时传入
        同一个 dict 可以省掉重复的二值化。

    Returns
    -------
    BatchResult
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    steps = [_as_source(item) for item in files]
    file_count = len(steps)
    result = BatchResult()

    # 固定模式下所有文件共用一个 job 与图案，先算一次
    cache = pattern_cache if pattern_cache is not None else {}
    fixed_pattern: Optional[np.ndarray] = None
    if text_mode is TextMode.FIXED:
        key = job.pattern.cache_key()
        if key in cache:
            fixed_pattern = cache[key]
        else:
            fixed_pattern = build_pattern(job.pattern)
            cache[key] = fixed_pattern

    def emit(index: int, path: str, file_progress: float, stage: str) -> bool:
        """上报进度；返回 False 表示用户请求取消。"""
        if progress is None:
            return True
        return progress(
            BatchProgress(
                file_index=index,
                file_count=file_count,
                file_path=path,
                file_progress=file_progress,
                stage=stage,
            )
        )

    for index, source in enumerate(steps, start=1):
        path = source.path
        item = BatchItem(source_path=path, status=ItemStatus.RUNNING)
        result.items.append(item)
        started = time.monotonic()

        try:
            if not emit(index, path, 0.0, "读取"):
                result.cancelled = True
                item.status = ItemStatus.SKIPPED
                item.message = "已取消"
                break

            audio = decode_media(path)
            item.duration_sec = audio.duration_sec

            # 图案：固定模式复用，逐文件名模式现算
            if text_mode is TextMode.FIXED:
                pattern = fixed_pattern
                file_job = job
            else:
                file_job = _job_for_file(job, path, text_mode)
                pattern = build_pattern(file_job.pattern)

            plan = plan_render(file_job, audio.sample_rate, audio.duration_sec, pattern)

            # 渲染占进度条的 15% ~ 90%
            processed = render_audio(
                audio.samples,
                audio.sample_rate,
                file_job,
                binary_pattern=pattern,
                plan=plan,
                progress=lambda v: emit(index, path, 0.15 + v * 0.75, "渲染"),
            )

            if not emit(index, path, 0.92, "写入"):
                result.cancelled = True
                item.status = ItemStatus.SKIPPED
                item.message = "已取消"
                break

            # 到这一步才建目录：前面的失败或取消不会留下一串空文件夹
            target_dir = _target_dir(output_dir, source)
            if target_dir != output_dir:
                os.makedirs(target_dir, exist_ok=True)

            if is_video_file(path) and output.mux_video:
                out_path = _write_video_result(
                    path, processed, audio.sample_rate, target_dir, output
                )
                suffix_note = f"合成 {Path(out_path).suffix.lstrip('.').upper()}"
            else:
                out_path = _write_audio_result(
                    path, processed, audio.sample_rate, audio.subtype,
                    target_dir, output,
                )
                suffix_note = Path(out_path).suffix.lstrip(".").upper()

            item.output_path = out_path
            item.status = ItemStatus.DONE
            item.message = (
                f"{audio.channels} 声道 / {audio.sample_rate / 1000:.1f} kHz / "
                f"{len(plan.intervals)} 处印章 / {suffix_note}"
            )
            emit(index, path, 1.0, "完成")

        except (MediaError, PatternError, EmptyRender) as exc:
            item.status = ItemStatus.FAILED
            item.message = str(exc)
        except Exception as exc:    # noqa: BLE001 - 单个失败不能拖垮整批
            item.status = ItemStatus.FAILED
            item.message = f"{type(exc).__name__}: {exc}"
        finally:
            item.elapsed_sec = time.monotonic() - started

    return result
