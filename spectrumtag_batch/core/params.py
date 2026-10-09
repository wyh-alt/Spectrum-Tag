"""参数数据类 —— UI 与 DSP 核心之间的契约。

这些结构刻意不依赖任何 UI 框架，便于在后台线程 / 子进程中直接传递。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

# 显示频率范围（与 SpectrumTag 的 Standalone FreqMap 一致）
MIN_DISPLAY_HZ = 20.0
MAX_DISPLAY_HZ = 22000.0

# 默认水印落点：低频 9 kHz、高频 10.5 kHz。
# 归一化值本身与采样率无关，这里的基准 Nyquist 只是把"9 kHz"折算成一个
# 默认数字（44.1 kHz 素材的 Nyquist），界面上显示的仍是实实在在的 Hz。
#
# 跨度给到 1.5 kHz：44.1 kHz 素材上相当于 139 个频率采样点。再窄的话，
# 一行文字挤进几十个点里就只剩一团糊影。
DEFAULT_LOW_FREQ_HZ = 9_000.0
DEFAULT_HIGH_FREQ_HZ = 10_500.0
DEFAULT_REF_NYQUIST_HZ = 22_050.0

# FFT 档位
FFT_SIZE_CHOICES = (1024, 2048, 4096, 8192)
DEFAULT_FFT_SIZE = 4096

# 掩码列数上限（与 SpectrumTag 的 computeMaskCols 一致）
MIN_MASK_COLS = 4
MAX_MASK_COLS = 4096


class PatternSource(Enum):
    """水印图案的来源。"""

    IMAGE = "image"
    TEXT = "text"


class EngraveMode(Enum):
    """水印怎么印上去 —— 前两种印法性质不同，第三种是图案描边专用的。

    ``CUT``（衰减）是**乘法增益**：``out = in × gain``，把图案区域的能量削弱。
    素材里本来就没有内容的频段，乘任何系数仍然是空的。

    ``DRAW``（注入）是**加性合成**：``out = in + synth``，在图案区域造出内容。
    因此即使超高频原本一片空白，也能"画"出图案 —— 这正是衰减做不到的。

    ``BOOST``（凸起）同样走乘法，只是增益大于 1。它不单独作为印法暴露给用户，
    只给图案描边用：外圈那一道被抬起来的边，与图案本体的凹陷配成峰谷对。
    """

    CUT = "cut"
    DRAW = "draw"
    BOOST = "boost"


class AudioFormat(Enum):
    """导出音频的容器格式。``SOURCE`` 表示跟着源文件走。"""

    SOURCE = "source"
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    AIF = "aif"
    AIFF = "aiff"
    OGG = "ogg"

    @property
    def suffix(self) -> str:
        return self.value

    def resolve(self, source_path: str) -> str:
        """定下真正要用的格式：要求"与源一致"时就取源文件的扩展名。

        视频源的"与源一致"没有意义（它没有音频容器），退到 wav。
        """
        if self is not AudioFormat.SOURCE:
            return self.value
        ext = os.path.splitext(source_path)[1].lstrip(".").lower()
        if ext in {f.value for f in AudioFormat} and ext != "source":
            return "aif" if ext == "aiff" else ext
        return AudioFormat.WAV.value


class VideoFormat(Enum):
    """导出视频的容器格式。``SOURCE`` 表示跟着源文件走。"""

    SOURCE = "source"
    MP4 = "mp4"
    MKV = "mkv"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"

    @property
    def suffix(self) -> str:
        return self.value

    def resolve(self, source_path: str) -> str:
        """要求"与源一致"时取源容器的扩展名；取不到就落到 mp4。"""
        if self is not VideoFormat.SOURCE:
            return self.value
        ext = os.path.splitext(source_path)[1].lstrip(".").lower()
        if ext in {f.value for f in VideoFormat} and ext != "source":
            return ext
        return VideoFormat.MP4.value


@dataclass(frozen=True)
class OutputSpec:
    """导出形态。

    ``mux_video`` 只对视频素材起作用：打开时把处理后的音轨合回视频（画面原样
    复制、不重编码），关掉就只导出音频。音频素材不受它影响。
    """

    mux_video: bool = False
    audio_format: AudioFormat = AudioFormat.SOURCE
    video_format: VideoFormat = VideoFormat.SOURCE


class PositionMode(Enum):
    """水印时间位置的解释方式 —— 决定批量处理不同时长文件时的行为。"""

    RELATIVE = "relative"   # 按比例定位（0~1 相对总时长），适合长短不一的文件
    ABSOLUTE = "absolute"   # 按绝对秒数定位，适合对齐固定的时间点


@dataclass(frozen=True)
class PatternSpec:
    """水印图案本身：来自图片还是文字，以及如何二值化。"""

    source: PatternSource = PatternSource.TEXT

    # --- 图片来源 ---
    image_path: Optional[str] = None

    # --- 文字来源 ---
    text: str = "WATERMARK"
    font_path: Optional[str] = None      # None = 按平台自动挑选中文字体
    # 字形粗细，0 = 最细、1 = 最粗。
    # 不暴露字号：图案最终总会被拉伸铺满掩码网格，字号只影响渲染精度，
    # 对成品没有可见影响，因此内部固定用一个足够大的渲染尺寸。
    weight: float = 0.25

    # --- 二值化 ---
    # 与 SpectrumTag 的 preprocessImage 一致：
    #   * 透明像素占比 > 5% 时走 alpha 通道（不透明 = 图案本体）
    #   * 否则走亮度（暗像素 = 图案本体，即黑字白底）
    # 这里的 threshold 为 None 表示沿用"平均亮度"自动阈值。
    binary_threshold: Optional[float] = None
    invert_pattern: bool = False         # 图案内外互换（白字黑底场景可手动纠正）

    # --- 描边 ---
    # 沿图案外沿描一圈"反向"处理，与图案本体配成峰谷对：衰减时外圈凸起、
    # 注入时外圈凹陷（见 render.plan_render）。想抹掉图案就得同时处理外圈，
    # 而外圈一动图案又露出来 —— 两头的代价互相牵制。
    outline: bool = True

    def cache_key(self) -> tuple:
        """用于缓存已渲染二值图的键。"""
        return (
            self.source.value,
            self.image_path,
            self.text,
            self.font_path,
            self.weight,
            self.binary_threshold,
            self.invert_pattern,
            self.outline,
        )


# 注入模式：强度 0~1 映射到的注入电平（dBFS）。0 = 满量程正弦。
DRAW_LEVEL_MIN_DBFS = -80.0
DRAW_LEVEL_MAX_DBFS = -25.0

# 两种印法各自的默认强度，取值考虑不一样：
#
#   * 衰减要更狠一点才听得出差别 —— 图案只占频段的一部分，摊到整段往往只剩
#     几个 dB。0.95 对应图案区增益 0.05（约 -26 dB），轮廓在频谱上是实心的；
#   * 注入不宜过强 —— 它是凭空造内容，铺得宽一点就容易削波，0.6 对应每个
#     频点约 -47 dBFS。
DEFAULT_CUT_STRENGTH = 0.95
DEFAULT_DRAW_STRENGTH = 0.6


def draw_dbfs_for(strength: float) -> float:
    """注入强度（0~1）映射到的目标电平（dBFS，满量程正弦 = 0）。"""
    return DRAW_LEVEL_MIN_DBFS + (
        DRAW_LEVEL_MAX_DBFS - DRAW_LEVEL_MIN_DBFS
    ) * strength


# 图案描边的凸起增益随强度线性增长：强度 1 时约 +9 dB（2.8 倍），
# 强度 0.95（衰减的默认值）时约 +8.7 dB —— 都落在"听得出来但不刺耳"的区间里。
#
# 斜率定得比凹陷温和（凹陷 0.95 是 -26 dB）：凸起是**抬升已有内容**，
# 幅度上比削弱更容易撞到削波，不该和凹陷等量。
OUTLINE_BOOST_SLOPE = 1.8


def outline_boost_for(strength: float) -> float:
    """图案描边的凸起增益（线性倍数）。

    与衰减的 ``1 - strength`` 方向相反、量级也略小 —— 凸起是**抬升已有内容**，
    幅度上比削弱更容易撞到削波，所以不宜和凹陷等量。
    """
    return 1.0 + OUTLINE_BOOST_SLOPE * strength


def draw_amplitude_for(strength: float, fft_size: int) -> float:
    """注入模式下写到频谱 bin 上的线性幅度。

    频谱图上衡量一个 bin 的电平用的是 ``20·log10(2·|X[k]|/N)``（即"该频率正弦的
    幅度"），所以要让某个 bin 显示成指定 dBFS，写进频谱的复数模必须是
    ``A · N / 2`` —— 直接写 ``A`` 会小掉 N/2 倍（4096 点时是 66 dB）。
    """
    linear = float(10.0 ** (draw_dbfs_for(strength) / 20.0))
    return linear * fft_size / 2.0


@dataclass(frozen=True)
class DspSpec:
    """STFT / OLA 与掩码处理参数。"""

    fft_size: int = DEFAULT_FFT_SIZE

    mode: EngraveMode = EngraveMode.CUT

    # 0.0 = 无效果；1.0 = 效果拉满。
    # 衰减模式下 1.0 表示完全抹掉图案区域，注入模式下 1.0 表示注入最强。
    strength: float = DEFAULT_CUT_STRENGTH

    # 掩码反相：图案区域与背景区域的处理方向互换
    invert: bool = False

    def __post_init__(self) -> None:
        if self.fft_size not in FFT_SIZE_CHOICES:
            raise ValueError(f"fft_size 必须是 {FFT_SIZE_CHOICES} 之一，收到 {self.fft_size}")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength 需在 [0.0, 1.0]，收到 {self.strength}")

    @property
    def cut_gain(self) -> float:
        """衰减模式下图案区域被乘上的增益：强度 1 → 完全抹除。"""
        return 1.0 - self.strength

    @property
    def draw_dbfs(self) -> float:
        """注入模式下每个 bin 的目标电平（dBFS，满量程正弦 = 0）。"""
        return draw_dbfs_for(self.strength)

    @property
    def draw_amplitude(self) -> float:
        """注入模式下写到频谱 bin 上的线性幅度。"""
        return draw_amplitude_for(self.strength, self.fft_size)


@dataclass(frozen=True)
class PlacementSpec:
    """水印在频谱图上的落点与尺寸。

    频率范围用归一化值表达（0 = DC，1 = Nyquist），时间范围由 PositionMode 决定
    其单位是"秒"还是"占总时长的比例"。
    """

    freq_low_norm: float = DEFAULT_LOW_FREQ_HZ / DEFAULT_REF_NYQUIST_HZ
    freq_high_norm: float = DEFAULT_HIGH_FREQ_HZ / DEFAULT_REF_NYQUIST_HZ

    start: float = 0.0         # 起点（秒 或 0~1 比例）
    duration: float = 0.30     # 单次印刷的持续长度（秒 或 0~1 比例）

    position_mode: PositionMode = PositionMode.RELATIVE

    def __post_init__(self) -> None:
        if not (0.0 <= self.freq_low_norm <= 1.0):
            raise ValueError("freq_low_norm 需在 [0, 1]")
        if not (0.0 <= self.freq_high_norm <= 1.0):
            raise ValueError("freq_high_norm 需在 [0, 1]")
        if self.freq_low_norm >= self.freq_high_norm:
            raise ValueError("freq_low_norm 必须小于 freq_high_norm")

    @property
    def freq_range_valid(self) -> bool:
        return 0.0 <= self.freq_low_norm < self.freq_high_norm <= 1.0


@dataclass(frozen=True)
class LoopSpec:
    """循环印刷：勾选后每隔 interval_sec 在同样的频段位置重复印刷一次。"""

    enabled: bool = True
    # 上一次印完到下一次开始之间的距离（不是"从起点算的固定周期"，
    # 所以印章本身的长度不会挤占间隔）
    interval_sec: float = 5.0
    # 起点偏移：第一次印刷在 start 之后的第几个周期（0 = 从 start 立即开始）
    max_repeats: int = 0          # 0 = 印满整段音频；>0 = 最多重复这么多次

    def __post_init__(self) -> None:
        if self.enabled and self.interval_sec <= 0.0:
            raise ValueError("循环间隔必须大于 0")


@dataclass(frozen=True)
class RenderJob:
    """一次完整的渲染任务：图案 + DSP + 位置 + 循环。"""

    pattern: PatternSpec = field(default_factory=PatternSpec)
    dsp: DspSpec = field(default_factory=DspSpec)
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    loop: LoopSpec = field(default_factory=LoopSpec)

    def with_pattern(self, **kw) -> "RenderJob":
        return replace(self, pattern=replace(self.pattern, **kw))

    def with_dsp(self, **kw) -> "RenderJob":
        return replace(self, dsp=replace(self.dsp, **kw))

    def with_placement(self, **kw) -> "RenderJob":
        return replace(self, placement=replace(self.placement, **kw))

    def with_loop(self, **kw) -> "RenderJob":
        return replace(self, loop=replace(self.loop, **kw))


@dataclass(frozen=True)
class Interval:
    """一次印刷在时间轴上的区间（秒）。"""

    start_sec: float
    end_sec: float

    # 图案列映射用的跨度。正常就等于 ``end - start``；当区间被音频末尾截断时，
    # 这里仍保留**原始时长** —— 末端该是"切掉"图案，而不是把图案压扁塞进
    # 剩下的时间里（那会让最后一个印章明显变小，跟前面几个对不上）。
    span_sec: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def mapping_span(self) -> float:
        """图案列映射用的跨度。"""
        return self.span_sec if self.span_sec is not None else self.duration

    def clamp_to(self, total_sec: float) -> Optional["Interval"]:
        """裁剪到 [0, total_sec]；完全落在范围外时返回 None。"""
        start = max(0.0, min(self.start_sec, total_sec))
        end = max(0.0, min(self.end_sec, total_sec))
        if end - start <= 1e-9:
            return None
        # 带上原始的映射跨度：裁剪只改变"印到哪儿为止"，不改变图案本身的尺寸
        return Interval(start, end, self.mapping_span)


def resolve_intervals(
    placement: PlacementSpec,
    loop: LoopSpec,
    total_sec: float,
) -> list[Interval]:
    """把位置 + 循环设置展开成实际要印刷的时间区间列表。

    单次模式返回一个区间；循环模式按 interval_sec 生成一串区间，
    每个区间的长度始终等于 placement.duration。
    """
    if total_sec <= 0.0:
        return []

    if placement.position_mode is PositionMode.RELATIVE:
        start_sec = placement.start * total_sec
        duration_sec = placement.duration * total_sec
    else:
        start_sec = placement.start
        duration_sec = placement.duration

    if duration_sec <= 0.0:
        return []

    if not loop.enabled:
        single = Interval(start_sec, start_sec + duration_sec).clamp_to(total_sec)
        return [single] if single else []

    # 循环：每次前进「印章时长 + 间隔」——间隔量的是上一次印完到下一次开始，
    # 因此印章越长、两次之间的空隙仍然等于设定的间隔
    intervals: list[Interval] = []
    gap = max(loop.interval_sec, 1e-6)
    cursor = start_sec
    guard = 0
    max_count = loop.max_repeats if loop.max_repeats > 0 else 10_000
    while cursor < total_sec and guard < max_count:
        # 显式带上原始时长：末尾那一次若放不下，会被 clamp 截短，但图案尺寸不变
        piece = Interval(cursor, cursor + duration_sec, duration_sec).clamp_to(total_sec)
        if piece:
            intervals.append(piece)
        cursor += duration_sec + gap
        guard += 1
    return intervals
