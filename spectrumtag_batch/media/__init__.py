"""媒体读写 —— 音频解码、WAV 编码，以及视频音轨提取。"""

from .audio_io import (
    AUDIO_EXTENSIONS,
    AudioData,
    AudioProbe,
    MediaError,
    output_subtype_for,
    probe_audio,
    read_audio,
    write_wav,
)
from .decode import decode_media, needs_ffmpeg
from .video_io import (
    VIDEO_EXTENSIONS,
    extract_audio_track,
    find_ffmpeg,
    is_video_file,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "AudioData",
    "AudioProbe",
    "MediaError",
    "probe_audio",
    "read_audio",
    "write_wav",
    "output_subtype_for",
    "VIDEO_EXTENSIONS",
    "decode_media",
    "needs_ffmpeg",
    "extract_audio_track",
    "find_ffmpeg",
    "is_video_file",
]
