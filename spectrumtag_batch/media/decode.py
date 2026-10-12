"""统一的解码入口 —— 把任意受支持的媒体文件读成音频。

libsndfile 覆盖了 wav / mp3 / flac / aiff / ogg，但 m4a、AAC、WMA 这类容器
它解不了，视频更得先把音轨抽出来。这些一律借 ffmpeg 转成临时 WAV 再交给
libsndfile —— 调用方不必关心手上这个是哪一种。

批处理与界面预览都走这里，免得两处的判断各写一份、哪天添了新格式只改一边。
"""

from __future__ import annotations

import os

from .audio_io import (
    FFMPEG_AUDIO_EXTENSIONS,
    AudioData,
    read_audio,
)
from .video_io import extract_audio_track, is_video_file


def needs_ffmpeg(path: str) -> bool:
    """这个文件要不要先过一道 ffmpeg。"""
    if is_video_file(path):
        return True
    return os.path.splitext(path)[1].lower() in FFMPEG_AUDIO_EXTENSIONS


def decode_media(path: str) -> AudioData:
    """把媒体文件解码到内存。

    Raises
    ------
    MediaError
        文件损坏、格式不受支持，或者需要的 ffmpeg 不在。
    """
    if not needs_ffmpeg(path):
        return read_audio(path)

    with extract_audio_track(path) as wav_path:
        return read_audio(wav_path)
