"""媒体读写 —— 音频解码、WAV 编码，以及视频音轨提取。"""

from .audio_io import (
    AudioData,
    AudioProbe,
    MediaError,
    output_subtype_for,
    probe_audio,
    read_audio,
    write_wav,
)
from .video_io import (
    VIDEO_EXTENSIONS,
    extract_audio_track,
    find_ffmpeg,
    is_video_file,
)

__all__ = [
    "AudioData",
    "AudioProbe",
    "MediaError",
    "probe_audio",
    "read_audio",
    "write_wav",
    "output_subtype_for",
    "VIDEO_EXTENSIONS",
    "extract_audio_track",
    "find_ffmpeg",
    "is_video_file",
]
