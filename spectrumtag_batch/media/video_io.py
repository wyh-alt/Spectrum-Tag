"""视频支持 —— 用 ffmpeg 把视频文件的音轨抽成 WAV，再交给音频链路处理。

视频画面不做保留（按需求：输出纯音频 WAV）。

关于查找 ffmpeg：包管理器（Chocolatey / Scoop）装出来的 ``ffmpeg.exe`` 常常只是
一个转发用的 shim，直接复制到别处就失效。所以这里的策略是：

1. 显式环境变量 ``SPECTRUMTAG_FFMPEG`` 优先；
2. 其次是随程序分发的 ``vendor/ffmpeg`` 目录；
3. 再扫 PATH，但**跳过包管理器目录**里的候选；
4. 最后试几个常见安装位置。

每个候选都会真的跑一次 ``-version`` 验证，不通过就换下一个。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional

from .audio_io import MediaError

# 交给 ffmpeg 去识别的视频扩展名
VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".ts", ".m2ts", ".3gp", ".ogv", ".rmvb",
)

# 环境变量覆盖
_ENV_VAR = "SPECTRUMTAG_FFMPEG"

# 包管理器造的 shim 常见所在，扫 PATH 时跳过
_SHIM_DIR_MARKERS = ("chocolatey", "scoop", "winget", "msys64")

# 兜底探测路径
_FALLBACK_PATHS = (
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
)

# 提取超时（秒）——给足余量，长的视频文件音轨抽离也可能要一会儿
_EXTRACT_TIMEOUT = 900

_resolved_ffmpeg: Optional[str] = None
_searched = False


def _verify(candidate: str) -> bool:
    """真的执行一次 -version，确认这个路径能独立工作。"""
    if not candidate or not os.path.isfile(candidate):
        return False
    try:
        result = subprocess.run(
            [candidate, "-version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and b"ffmpeg" in result.stdout.lower()


def _bundled_candidate() -> Optional[str]:
    """随程序分发的 ffmpeg。

    打包成 exe 后，PyInstaller 会把它解压到 ``sys._MEIPASS`` 下的 ``ffmpeg/``；
    开发态则放在项目的 ``vendor/ffmpeg/`` 里。两处都找一遍。
    """
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    roots: list[str] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(meipass)
    here = os.path.dirname(os.path.abspath(__file__))
    roots.append(os.path.dirname(here))                # spectrumtag_batch/

    for root in roots:
        for parts in (("ffmpeg",), ("vendor", "ffmpeg", "bin"), ("vendor", "ffmpeg")):
            path = os.path.join(root, *parts, name)
            if os.path.isfile(path):
                return path
    return None


def find_ffmpeg(refresh: bool = False) -> Optional[str]:
    """定位可用的 ffmpeg，找不到返回 None。结果会缓存。"""
    global _resolved_ffmpeg, _searched
    if _searched and not refresh:
        return _resolved_ffmpeg

    _searched = True
    _resolved_ffmpeg = None

    env_path = os.environ.get(_ENV_VAR)
    if env_path and _verify(env_path):
        _resolved_ffmpeg = env_path
        return _resolved_ffmpeg

    bundled = _bundled_candidate()
    if bundled and _verify(bundled):
        _resolved_ffmpeg = bundled
        return _resolved_ffmpeg

    on_path = shutil.which("ffmpeg")
    if on_path:
        lowered = on_path.lower()
        is_shim = any(marker in lowered for marker in _SHIM_DIR_MARKERS)
        if not is_shim and _verify(on_path):
            _resolved_ffmpeg = on_path
            return _resolved_ffmpeg

    for candidate in _FALLBACK_PATHS:
        if _verify(candidate):
            _resolved_ffmpeg = candidate
            return _resolved_ffmpeg

    return None


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def _run_extract(ffmpeg: str, video_path: str, out_path: str) -> None:
    cmd = [
        ffmpeg,
        "-v", "error",          # 只保留错误输出
        "-y",                   # 覆盖已有文件
        "-i", video_path,
        "-vn",                  # 丢掉画面
        "-map", "0:a:0",        # 只取第一条音轨
        "-c:a", "pcm_s24le",    # 24-bit PCM，避免解码后再量化
        out_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=_EXTRACT_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            f"提取音轨超时（{_EXTRACT_TIMEOUT}s）：{os.path.basename(video_path)}"
        ) from exc
    except OSError as exc:
        raise MediaError(f"无法启动 ffmpeg：{exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        detail = detail.splitlines()[-1] if detail else f"退出码 {result.returncode}"
        raise MediaError(f"提取音轨失败 {os.path.basename(video_path)}：{detail}")
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise MediaError(f"视频没有可用音轨：{os.path.basename(video_path)}")


# 各容器能装下的音频编码。交给 ffmpeg 自己猜容易选出容器不认的组合，
# 干脆列清楚；表里没有的一律用 aac（兼容性最好）。
_MUX_AUDIO_CODEC = {
    "mp4": "aac",
    "mov": "aac",
    "mkv": "aac",
    "avi": "mp3",
    "webm": "libopus",
}


def mux_video(
    source_video: str,
    audio_path: str,
    output_path: str,
    container: str,
) -> None:
    """把处理后的音轨合回视频，画面直接复制、不重编码。

    画面不重编码既快又无损 —— 我们只动过音频，没有理由去碰画面。
    代价是容器得装得下原来的视频编码；装不下时 ffmpeg 会报错，
    这里如实转达，而不是默默重编码（那样可能耗时几十分钟）。
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError("合成视频需要 ffmpeg，但没找到它")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    codec = _MUX_AUDIO_CODEC.get(container.lower(), "aac")
    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-i", source_video,
        "-i", audio_path,
        "-map", "0:v:0",        # 原来的画面
        "-map", "1:a:0",        # 处理过的音轨
        "-c:v", "copy",
        "-c:a", codec,
        "-shortest",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=_EXTRACT_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            f"合成视频超时（{_EXTRACT_TIMEOUT}s）：{os.path.basename(source_video)}"
        ) from exc
    except OSError as exc:
        raise MediaError(f"无法启动 ffmpeg：{exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        detail = detail.splitlines()[-1] if detail else f"退出码 {result.returncode}"
        raise MediaError(
            f"合成视频失败 {os.path.basename(output_path)}：{detail}"
        )


@contextmanager
def extract_audio_track(video_path: str) -> Iterator[str]:
    """把视频音轨抽成临时 WAV，退出上下文时自动清理。

    Yields
    ------
    str
        临时 WAV 的路径。

    Raises
    ------
    MediaError
        找不到 ffmpeg，或该视频没有可提取的音轨。
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError(
            "需要 ffmpeg 才能处理视频。请安装 ffmpeg 并加入 PATH，"
            f"或设置环境变量 {_ENV_VAR} 指向 ffmpeg 可执行文件。"
        )

    with tempfile.TemporaryDirectory(prefix="stbatch_") as tmpdir:
        out_path = os.path.join(tmpdir, "audio.wav")
        _run_extract(ffmpeg, video_path, out_path)
        yield out_path
