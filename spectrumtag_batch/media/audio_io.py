"""音频读写 —— 全部走 libsndfile（soundfile），不依赖任何外部工具。

libsndfile 1.2 起原生支持 MP3 解码，因此 WAV / MP3 / FLAC / AIFF / OGG
都在这里搞定，只有视频需要外部 ffmpeg（见 :mod:`.video_io`）。

约定：内部一律用 ``(channels, samples)`` 的 float32 数组，与 DSP 层一致；
文件里的原始位深会被记录下来，导出时尽量原样保留（对应 SpectrumTag
"保持原音频的采样率、位深与声道数"的行为）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

# 可解码的音频扩展名（与 SpectrumTag 的白名单一致）
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg")

# 位深"穿透"保存的 PCM 子类型；其余（有损格式）统一落到 _FALLBACK_SUBTYPE
_PASSTHROUGH_SUBTYPES = ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE")
_FALLBACK_SUBTYPE = "PCM_24"

# libsndfile 直接能写的格式 → 需要显式指定的容器名（None = 它能从扩展名认出来）。
# MP3 编码从 libsndfile 1.2 起支持，所以不必为它去借 ffmpeg。
_SF_FORMATS: dict[str, Optional[str]] = {
    "wav": None,
    "flac": None,
    "aiff": None,
    "aif": "AIFF",      # 这个扩展名它认不出来
    "ogg": None,
    "mp3": None,
}

# 只有这些格式谈得上"保留源位深"，其余（有损）交给编码器自己定
_SF_PCM_CAPABLE = ("wav", "flac", "aif", "aiff")

# 16-bit 有符号整数满量程。libsndfile 读入时按 `int / 32768` 归一化，
# 写出必须用同一个 32768 才能保证 round-trip 不丢 1 个 LSB。
_PCM16_SCALE = 32768.0
_PCM16_MIN = -32768
_PCM16_MAX = 32767


class MediaError(RuntimeError):
    """媒体读写失败（文件损坏、格式不支持、缺 ffmpeg 等）。"""


@dataclass(frozen=True)
class AudioProbe:
    """不读音频数据就能拿到的元信息，用于批量任务的前置校验与规划。"""

    sample_rate: int
    channels: int
    frames: int
    subtype: str
    format: str

    @property
    def duration_sec(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0


@dataclass(frozen=True)
class AudioData:
    """已解码到内存的音频。"""

    samples: np.ndarray        # (channels, samples) float32
    sample_rate: int
    subtype: str               # 源文件的原始子类型，用于导出时保留位深
    source_path: str = ""

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_sec(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0


def output_subtype_for(probe_subtype: str, fallback: str = _FALLBACK_SUBTYPE) -> str:
    """决定导出 WAV 时用的子类型。

    源是 PCM/FLOAT 就原样保留位深；有损格式（MP3/OGG）与 8-bit 之类没有对应的
    高质量 WAV 子类型，退回到 fallback。
    """
    if probe_subtype in _PASSTHROUGH_SUBTYPES:
        return probe_subtype
    return fallback


def probe_audio(path: str) -> AudioProbe:
    """读取音频元信息，不解码样本。"""
    try:
        info = sf.info(path)
    except Exception as exc:    # noqa: BLE001 - 统一包装成 MediaError
        raise MediaError(f"无法读取音频 {os.path.basename(path)}：{exc}") from exc

    if info.frames <= 0:
        raise MediaError(f"音频为空：{os.path.basename(path)}")

    return AudioProbe(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        subtype=str(info.subtype),
        format=str(info.format),
    )


def read_audio(path: str) -> AudioData:
    """把整个音频文件解码到内存，返回 ``(channels, samples)`` 的 float32。"""
    probe = probe_audio(path)
    try:
        data, sample_rate = sf.read(
            path, dtype="float32", always_2d=True, frames=probe.frames
        )
    except Exception as exc:    # noqa: BLE001
        raise MediaError(f"解码失败 {os.path.basename(path)}：{exc}") from exc

    # soundfile 返回 (samples, channels)，DSP 层要 (channels, samples)
    samples = np.ascontiguousarray(data.T)
    return AudioData(
        samples=samples,
        sample_rate=int(sample_rate),
        subtype=probe.subtype,
        source_path=path,
    )


def _float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """float [-1,1] → int16，往返无损。

    用 32768 缩放（与 libsndfile 的解码方向一致），再把 +1.0 造成的 32768 夹回
    32767，既能精确还原原本就是 16-bit 的素材，又不会溢出环绕成噪声。
    """
    scaled = np.round(np.clip(samples, -1.0, 1.0) * _PCM16_SCALE)
    return np.clip(scaled, _PCM16_MIN, _PCM16_MAX).astype(np.int16)


def write_audio(
    path: str,
    samples: np.ndarray,
    sample_rate: int,
    fmt: str = "wav",
    subtype: str = _FALLBACK_SUBTYPE,
) -> None:
    """按指定格式写出音频。

    ``samples`` 为 ``(channels, samples)``。整型子类型会先限幅再量化，防止
    处理过程中略微超出 [-1,1] 的样本在写盘时发生环绕失真。

    wav / flac / aiff / aif / ogg / mp3 都由 libsndfile 直接写（1.2 起它连
    MP3 编码都支持），只有 m4a 得借道 ffmpeg。
    """
    fmt = fmt.lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if fmt in _SF_FORMATS:
        _write_with_soundfile(path, samples, sample_rate, fmt, subtype)
        return
    _write_with_ffmpeg(path, samples, sample_rate, fmt)


def _write_with_soundfile(
    path: str,
    samples: np.ndarray,
    sample_rate: int,
    fmt: str,
    subtype: str,
) -> None:
    data = np.ascontiguousarray(samples.T)     # 回到 (samples, channels)
    kwargs: dict = {}
    # .aif 这个扩展名 libsndfile 认不出来，得显式告诉它容器是什么
    explicit = _SF_FORMATS[fmt]
    if explicit:
        kwargs["format"] = explicit
    if fmt in _SF_PCM_CAPABLE:
        kwargs["subtype"] = subtype

    try:
        if kwargs.get("subtype") == "PCM_16":
            # 16-bit 走自己的量化，保证与解码方向对称、往返不丢 LSB
            sf.write(path, _float_to_pcm16(data), sample_rate, **kwargs)
        else:
            sf.write(path, data, sample_rate, **kwargs)
    except Exception as exc:    # noqa: BLE001
        raise MediaError(f"写入失败 {os.path.basename(path)}：{exc}") from exc


def _write_with_ffmpeg(
    path: str,
    samples: np.ndarray,
    sample_rate: int,
    fmt: str,
) -> None:
    """libsndfile 写不了的格式（目前只有 m4a）：先落临时 WAV 再让 ffmpeg 转。"""
    import subprocess
    import tempfile

    from .video_io import find_ffmpeg      # 延迟导入，避开两个模块互相引用

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError(f"导出 .{fmt} 需要 ffmpeg，但没找到它")

    with tempfile.TemporaryDirectory(prefix="stbatch_enc_") as tmp:
        wav_path = os.path.join(tmp, "src.wav")
        sf.write(
            wav_path,
            np.ascontiguousarray(samples.T),
            sample_rate,
            subtype="PCM_24",
        )
        try:
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", wav_path, path],
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MediaError(f"调用 ffmpeg 失败：{exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        detail = detail.splitlines()[-1] if detail else f"退出码 {result.returncode}"
        raise MediaError(f"写入失败 {os.path.basename(path)}：{detail}")


def write_wav(
    path: str,
    samples: np.ndarray,
    sample_rate: int,
    subtype: str = _FALLBACK_SUBTYPE,
) -> None:
    """写 WAV —— 保留这个名字，内部就是 ``write_audio(..., "wav", ...)``。"""
    write_audio(path, samples, sample_rate, "wav", subtype)


def unique_output_path(
    directory: str,
    stem: str,
    extension: str = ".wav",
    source_path: str = "",
) -> str:
    """生成输出路径：``<主文件名><扩展名>``，重名时追加 _1 / _2 …

    ``source_path`` 用于避免踩到源文件本身：输出目录就是源目录、扩展名又恰好
    相同时（比如 wav 转 wav），会正好压在原件上。那种情况下退回加 ``_tagged``
    后缀 —— 宁可名字和源文件不一致，也不能把原始素材覆盖掉。
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    base = stem
    candidate = os.path.join(directory, f"{base}{extension}")

    if source_path and _same_file(candidate, source_path):
        base = f"{stem}_tagged"
        candidate = os.path.join(directory, f"{base}{extension}")

    if not os.path.exists(candidate):
        return candidate

    index = 1
    while True:
        candidate = os.path.join(directory, f"{base}_{index}{extension}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _same_file(a: str, b: str) -> bool:
    """两个路径是否指向同一个文件（只在两边都存在时才有意义）。"""
    try:
        return os.path.samefile(a, b)
    except OSError:
        # 目标还不存在时 samefile 会抛错，退回按规范化路径比较
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
