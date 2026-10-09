"""STFT / OLA 频谱水印核心 —— SpectrumTag 离线渲染算法的 Python 移植。

与 `SpectrumTag/Standalone/OfflineRenderer.cpp` 数学一致，但做了两点改造：

1. **批量 STFT**：原实现是 sample-major 的流式环形缓冲，这里改成按帧矩阵化
   （`rfft` / `irfft` 一次算一整块），结果完全相同但快一到两个数量级。
   推导见下方「帧几何」注释。
2. **多区间**：原实现只有一个 `[startSec, endSec]` 印章区间，这里支持任意多个
   区间 —— 循环印刷只是同一套逻辑下多给几个区间。

帧几何（与 C++ 逐样本循环等价，已按 `latencySamples = N - 1` 推导验证）：

    * 帧 k（k = 1, 2, 3 …）读取输入样本 ``[k*hop - N, k*hop)``，
      并把它的 N 个输出样本贡献到**同样的**样本区间 ``[k*hop - N, k*hop)``。
    * 因此每个输出样本恰好被 ``N/hop = 4`` 个帧覆盖，WOLA 归一化即
      ``out = Σ(frame_i * w) / Σ(w²)``。
    * 帧 k 的增益由它覆盖区间的**起点**决定，即输出时间
      ``t_k = (k*hop - N) / sampleRate`` —— 与原实现里
      ``m = n - latencySamples`` 的选择一致。

因为帧的输出区间用全局样本坐标表示后恰好是 ``[k*hop, k*hop + N)``（含前补的 N 个
零样本），重叠相加时同余类内的帧互不重叠，可以整段 reshape 累加，无需逐帧循环。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .params import (
    DEFAULT_CUT_STRENGTH,
    EngraveMode,
    Interval,
    draw_amplitude_for,
    outline_boost_for,
)

# WOLA 归一化时判定"窗能量足够"的阈值（对应 C++ 的 kNormEps）
_NORM_EPS = 1.0e-8

# 幅频增益的每帧时间平滑时间常数（秒），对应 C++ 的 0.010f
_GAIN_SMOOTH_SECONDS = 0.010

# 印章区间端点的淡入淡出长度（秒），对应 C++ OfflineRenderer 的 20ms
_EDGE_FADE_SECONDS = 0.020

# 每块处理的帧数。块越小内存越省，越大吞吐越高。
_DEFAULT_BLOCK_FRAMES = 512

ProgressFn = Callable[[float], bool]
"""进度回调：收到 [0, 1] 的进度，返回 False 表示请求取消。"""


def hann_window(n: int) -> np.ndarray:
    """对称 Hann 窗 —— 分母为 N-1，与 JUCE 端逐字一致。

    SpectrumTag 刻意不用周期窗（分母 N），改动这里会改变印章的边缘形状。
    """
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float32)
    idx = np.arange(n, dtype=np.float64)
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * idx / (n - 1)))).astype(np.float32)


def _juce_round(values: np.ndarray) -> np.ndarray:
    """复刻 JUCE 的 roundToInt：四舍五入（.5 向上），而非 numpy 的银行家舍入。"""
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


@dataclass(frozen=True)
class EngraveBand:
    """一枚印章铺在频谱上的一个频段 —— 掩码 + 频率落点 + 印法。

    只有一层时就是常规水印；图案描边再加一层：掩码换成图案外沿的那一圈，
    频率落点照抄，印法取本体的反面（衰减配凸起、注入配衰减）。两层共用同一批
    时间区间，因此在时间轴上天然同步 —— "峰谷成对"不需要任何对齐逻辑。
    """

    # 掩码不参与比较与哈希：numpy 数组的 == 返回的是逐元素结果，
    # 落进 dataclass 自动生成的 __eq__ / __hash__ 里只会炸掉
    mask: np.ndarray = field(compare=False)
    freq_low_norm: float = 0.0
    freq_high_norm: float = 1.0
    mode: EngraveMode = EngraveMode.CUT
    strength: float = DEFAULT_CUT_STRENGTH
    invert: bool = False

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.size == 0:
            raise ValueError("掩码必须是二维且非空")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength 需在 [0.0, 1.0]，收到 {self.strength}")
        if not (0.0 <= self.freq_low_norm < self.freq_high_norm <= 1.0):
            raise ValueError(
                f"频率范围需满足 0 <= low < high <= 1，"
                f"收到 {self.freq_low_norm} ~ {self.freq_high_norm}"
            )

    @property
    def fft_size(self) -> int:
        """由掩码行数反推 —— 行数恒为 ``fft_size // 2 + 1``。"""
        return (self.mask.shape[0] - 1) * 2

    @property
    def is_draw(self) -> bool:
        return self.mode is EngraveMode.DRAW

    @property
    def gain(self) -> float:
        """乘法模式下图案区域的增益：衰减 < 1、描边 > 1。

        注入不走这条路（它的效果量是加性幅度），这时返回值没有意义。
        """
        if self.mode is EngraveMode.BOOST:
            return outline_boost_for(self.strength)
        return 1.0 - self.strength

    def levels(self) -> tuple[float, float]:
        """图案内 / 图案外各自的效果量。

        第二个值同时充当"无效果"的基准：乘法是 1.0（增益不变），注入是 0.0
        （不注入）。``invert`` 通过交换两者表达。
        """
        if self.is_draw:
            amplitude = draw_amplitude_for(self.strength, self.fft_size)
            return (0.0, amplitude) if self.invert else (amplitude, 0.0)
        gain = self.gain
        return (1.0, gain) if self.invert else (gain, 1.0)


def build_binary_mask(
    binary: np.ndarray,
    num_freq_bins: int,
    num_cols: int,
) -> np.ndarray:
    """把二值图案（高×宽，True = 图案本体）栅格化成 DSP 用的掩码。

    对应 ``SharedUI.cpp::ImageBoxComponent::generateMask``：

    * 行 = 频率 bin，**row 0 = 最低频**，所以图案在纵向上翻转
      （图片 y=0 是顶部 = 高频）。
    * 列 = 时间，按 ``col / (num_cols - 1)`` 线性铺满整列。
    * 整张图被拉伸铺满 ``num_freq_bins × num_cols``。
    """
    if num_freq_bins <= 0 or num_cols <= 0:
        raise ValueError("num_freq_bins 与 num_cols 必须为正")

    h, w = binary.shape
    if h == 0 or w == 0:
        raise ValueError("图案尺寸为空")

    rows = np.arange(num_freq_bins, dtype=np.float64)
    if num_freq_bins > 1:
        ny = (num_freq_bins - 1 - rows) / (num_freq_bins - 1)
    else:
        ny = np.zeros(1, dtype=np.float64)
    img_y = np.clip(_juce_round(ny * (h - 1)), 0, h - 1).astype(np.intp)

    cols = np.arange(num_cols, dtype=np.float64)
    if num_cols > 1:
        nx = cols / (num_cols - 1)
    else:
        nx = np.zeros(1, dtype=np.float64)
    img_x = np.clip(_juce_round(nx * (w - 1)), 0, w - 1).astype(np.intp)

    return np.ascontiguousarray(binary[np.ix_(img_y, img_x)], dtype=np.float32)


def compute_column_field(
    mask: np.ndarray,
    sample_rate: float,
    fft_size: int,
    freq_low_norm: float,
    freq_high_norm: float,
    inside_value: float,
    outside_value: float,
) -> np.ndarray:
    """预先算出**所有列**的逐 bin 效果量，形状 ``(num_bins, num_cols)``。

    衰减与注入共用这一套几何：图案内取 ``inside_value``、图案外取 ``outside_value``，
    再套上**保护带**、**边缘斜坡**与频域 3-tap 平滑。``outside_value`` 同时充当
    "无效果"基准值 —— 衰减是 1.0（增益不变），注入是 0.0（不注入）。

    保护带对应 ``OfflineRenderer.cpp::computeBinGainsForCol``：图案实际占据的频率
    范围之外强制回到基准，范围内侧再给一条陡斜坡，抑制"误挖孔"造成的竖直细线。

    ``invert`` 的语义通过交换 ``inside_value`` / ``outside_value`` 表达。
    """
    num_bins, num_cols = mask.shape
    max_hz = sample_rate * 0.5
    f_low = freq_low_norm * max_hz
    f_high = freq_high_norm * max_hz
    denom = max(1.0e-6, f_high - f_low)
    bin_hz = sample_rate / max(1, fft_size)

    # 每个 bin 的中心频率（与 C++ 的 hz = k / (N/2) * maxHz 一致）
    hz = (np.arange(num_bins, dtype=np.float64) / (fft_size / 2.0)) * max_hz
    in_range = (hz >= f_low) & (hz <= f_high)

    # bin → mask 行的映射（行 0 = 最低频）
    norm = (hz - f_low) / denom
    if num_bins > 1:
        row_idx = np.clip(
            np.floor(norm * (num_bins - 1)).astype(np.intp), 0, num_bins - 1
        )
    else:
        row_idx = np.zeros(1, dtype=np.intp)

    # --- 每列实际被图案占据的行范围（用于保护带）---
    active = mask > 0.5
    has_active = active.any(axis=0)
    active_min = np.argmax(active, axis=0).astype(np.float64)
    active_max = (num_bins - 1 - np.argmax(active[::-1], axis=0)).astype(np.float64)

    row_to_norm = 1.0 / (num_bins - 1) if num_bins > 1 else 0.0
    protect_pad = max(2.0 * bin_hz, 0.005 * denom)
    edge_slope = max(1.5 * bin_hz, 0.002 * denom)
    protect_low = (f_low + active_min * row_to_norm * denom) - protect_pad
    protect_high = (f_low + active_max * row_to_norm * denom) + protect_pad

    # --- 图案内外的取值（JUCE jmap 语义：outside + m*(inside-outside)）---
    m = mask[row_idx, :]                       # (num_bins, num_cols)
    value = outside_value + m * (inside_value - outside_value)

    # --- 保护带 + 边缘斜坡（基准值即无效果值）---
    hz_col = hz[:, None]
    lo_edge = np.clip((hz_col - protect_low[None, :]) / edge_slope, 0.0, 1.0)
    hi_edge = np.clip((protect_high[None, :] - hz_col) / edge_slope, 0.0, 1.0)
    edge_keep = np.minimum(lo_edge, hi_edge)
    outside_band = (hz_col < protect_low[None, :]) | (hz_col > protect_high[None, :])

    shaped = outside_value + (value - outside_value) * edge_keep
    field = np.where(
        has_active[None, :],
        np.where(outside_band, outside_value, shaped),
        value,
    )

    # --- 频段外一律回到基准 ---
    field = np.where(in_range[:, None], field, outside_value)

    # --- 频域 3-tap 平滑（仅范围内）---
    if num_bins >= 3:
        smoothed = field.copy()
        smoothed[1:-1, :] = (
            0.2 * field[:-2, :] + 0.6 * field[1:-1, :] + 0.2 * field[2:, :]
        )
        field = np.where(in_range[:, None], smoothed, field)

    return np.ascontiguousarray(field, dtype=np.float32)


def _assign_columns(
    times: np.ndarray,
    intervals: Sequence[Interval],
    num_cols: int,
) -> np.ndarray:
    """给每个帧分配掩码列号；不在任何区间内的帧返回 -1（表示旁通）。"""
    col_of_frame = np.full(times.shape, -1, dtype=np.intp)
    for interval in intervals:
        # 用 mapping_span 而不是实际跨度：末尾那次印章被音频截短时，
        # 列号只走到图案的一部分，于是图案是"被切掉"而非"被压扁"
        span = interval.mapping_span
        if span <= 0.0:
            continue
        inside = (times >= interval.start_sec) & (times < interval.end_sec)
        if not inside.any():
            continue
        rel = (times[inside] - interval.start_sec) / span
        col_of_frame[inside] = np.clip(
            np.floor(rel * num_cols).astype(np.intp), 0, num_cols - 1
        )
    return col_of_frame


def _build_mix_curve(
    num_samples: int,
    sample_rate: float,
    intervals: Sequence[Interval],
) -> np.ndarray:
    """构造 dry/wet 混合曲线：区间内为 1（全湿），端点做 20ms 线性淡入淡出。

    斜坡值与 C++ 的 ``(m - startSample)/(fadeInEnd - startSample)`` 及
    ``(endSample - 1 - m)/(endSample - fadeOutStart)`` 逐点对应。多区间重叠取较大值。
    """
    mix = np.zeros(num_samples, dtype=np.float32)
    fade = max(64, int(round(_EDGE_FADE_SECONDS * sample_rate)))

    for interval in intervals:
        start = int(round(interval.start_sec * sample_rate))
        end = int(round(interval.end_sec * sample_rate))
        start = max(0, min(start, num_samples))
        end = max(start, min(end, num_samples))
        if end <= start:
            continue

        fade_len = min(fade, max(1, (end - start) // 2))
        seg = np.ones(end - start, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
        seg[:fade_len] = ramp
        seg[-fade_len:] = ramp[::-1]

        np.maximum(mix[start:end], seg, out=mix[start:end])

    return mix


def render_watermark(
    audio: np.ndarray,
    sample_rate: float,
    bands: Sequence[EngraveBand],
    intervals: Sequence[Interval],
    progress: Optional[ProgressFn] = None,
    block_frames: int = _DEFAULT_BLOCK_FRAMES,
) -> np.ndarray:
    """把一枚或多枚印章印到音频上，返回与输入同形状的音频。

    Parameters
    ----------
    audio
        形状 ``(channels, samples)`` 或 ``(samples,)`` 的浮点音频，取值范围约 [-1, 1]。
    sample_rate
        采样率。输出保持原采样率，不做重采样。
    bands
        印章频段落列表，每段自带掩码、频率范围与印法（见 :class:`EngraveBand`）。
        各段的掩码形状必须一致 —— 它们本来就是同一张图案。

        衰减是乘法增益（``out = in × gain``），只能削弱已有内容；注入是加性注入
        （在原幅度不足的 bin 上替换为合成信号），因此空白频段也能"画"出图案。
    intervals
        要施加印章的时间区间列表（秒）。**所有层共用这一批区间**，所以图案本体
        与描边严格同步，循环印刷也自动一起走。空列表直接返回输入副本。

    Returns
    -------
    numpy.ndarray
        与输入同形状的 float32 结果。
    """
    def _report(value: float) -> bool:
        return True if progress is None else progress(min(max(value, 0.0), 1.0))

    if not bands:
        raise ValueError("至少要有一个印章频段")

    single_channel = audio.ndim == 1
    work = audio[np.newaxis, :] if single_channel else np.asarray(audio)
    work = np.ascontiguousarray(work, dtype=np.float32)
    num_ch, num_samples = work.shape

    def _passthrough() -> np.ndarray:
        return work[0].copy() if single_channel else work.copy()

    if num_samples == 0:
        return _passthrough()

    n = bands[0].fft_size
    hop = n // 4
    num_bins = n // 2 + 1
    group = n // hop                       # 恒为 4
    shape = (num_bins, bands[0].mask.shape[1])
    for band in bands:
        if band.mask.shape != shape:
            raise ValueError(
                f"所有频段的掩码必须同为 {shape}，收到 {band.mask.shape}"
            )

    if not intervals:
        _report(1.0)
        return _passthrough()

    window = hann_window(n)

    # --- 帧编号范围：帧 k 覆盖输出样本 [k*hop - N, k*hop) ---
    k_start = 1
    k_end = (num_samples + n) // hop + 2
    total_frames = k_end - k_start
    all_k = np.arange(k_start, k_end, dtype=np.int64)

    # 帧 k 的输出时间（用其覆盖区间的起点，与 C++ 一致）
    times = (all_k.astype(np.float64) * hop - n) / sample_rate

    num_cols = shape[1]
    cols_of_frame = _assign_columns(times, intervals, num_cols)
    touched = cols_of_frame >= 0
    if not touched.any():
        _report(1.0)
        return _passthrough()

    # --- 每个频段各自的逐 bin 效果量（一次性算好，之后按帧查表）---
    # 每段一份：(效果量表, 无效果基准值, 逐帧平滑状态, 是否走注入)
    layers: list[tuple[np.ndarray, float, np.ndarray, bool]] = []
    for band in bands:
        inside, outside = band.levels()
        field = compute_column_field(
            band.mask, sample_rate, n,
            band.freq_low_norm, band.freq_high_norm,
            inside, outside,
        )                                   # (num_bins, num_cols)
        layers.append((
            field,
            float(outside),
            np.full(num_bins, outside, dtype=np.float32),
            band.is_draw,
        ))

    # 一阶 IIR 平滑的时间常数（与 C++ 的 smoothAlpha 一致）
    alpha = 1.0 - math.exp(-(hop / float(sample_rate)) / _GAIN_SMOOTH_SECONDS)

    # 注入用的合成相位：每个 bin 有固定初相，跨帧按自身频率线性推进。
    # 这样注入的信号在帧与帧之间是连续的（听起来像稳态音而非咔哒声），
    # 不同 bin 的初相互不相同，避免所有频率同相叠成一个脉冲。
    bin_index = np.arange(num_bins, dtype=np.float64)
    phase_step = 2.0 * np.pi * bin_index * hop / float(n)
    phase_origin = np.random.default_rng(20240501).uniform(
        0.0, 2.0 * np.pi, num_bins
    )

    # --- 输入补零：帧 k 读 x_pad[:, k*hop : k*hop + N] ---
    x_pad = np.zeros((num_ch, k_end * hop + n), dtype=np.float32)
    x_pad[:, n : n + num_samples] = work

    # --- 全局 OLA 累加器：索引 j 对应输出样本 j - N ---
    # 最大帧 k_end-1 贡献到 [k*hop, k*hop + N)，所以数组至少要这么长。
    ola_len = (k_end - 1) * hop + n
    ola = np.zeros((num_ch, ola_len), dtype=np.float32)
    ola_norm = np.zeros(ola_len, dtype=np.float32)
    window_sq = (window * window).astype(np.float32)

    next_report = 0.0

    for block_begin in range(0, total_frames, block_frames):
        block_end = min(block_begin + block_frames, total_frames)
        ks = all_k[block_begin:block_end]
        block_n = ks.size

        # 取帧：(num_ch, block_n, N)
        offsets = (ks * hop)[:, None] + np.arange(n, dtype=np.int64)[None, :]
        frames = x_pad[:, offsets]

        spectrum = np.fft.rfft(frames * window[None, None, :], axis=-1)

        # 只有落在印章区间内的帧才需要施加效果
        block_cols = cols_of_frame[block_begin:block_end]
        hit = block_cols >= 0
        if hit.any():
            hit_cols = block_cols[hit]
            # 时频坐标对所有频段都一样，逐段套上自己的效果量即可
            for field, neutral, smooth_state, is_draw in layers:
                targets = np.full((block_n, num_bins), neutral, dtype=np.float32)
                targets[hit] = field[:, hit_cols].T
                # 逐帧一阶平滑（递推是串行的，但每步只做一次 num_bins 向量运算）
                for i in range(block_n):
                    smooth_state += (targets[i] - smooth_state) * alpha
                    targets[i] = smooth_state

                if not is_draw:
                    spectrum *= targets[None, :, :]
                else:
                    # 注入：只在原幅度还不到目标值的 bin 上用合成信号补齐。
                    # 已有内容的地方保持原样，避免与素材本身产生干涉。
                    theta = phase_origin[None, :] + phase_step[None, :] * ks[:, None]
                    synth = (targets * np.exp(1j * theta)).astype(spectrum.dtype)
                    replace = np.abs(spectrum) < targets[None, :, :]
                    np.copyto(spectrum, synth[None, :, :], where=replace)

        restored = np.fft.irfft(spectrum, n=n, axis=-1).astype(np.float32)
        weighted = restored * window[None, None, :]

        # 重叠相加：同余类内的帧互不重叠，可整段 reshape 累加
        base = int(ks[0] * hop)
        for g in range(group):
            sub = weighted[:, g::group, :]              # (num_ch, T, N)
            if sub.shape[1] == 0:
                continue
            start = base + g * hop
            span = sub.shape[1] * n
            ola[:, start : start + span] += sub.reshape(num_ch, span)
            ola_norm[start : start + span] += np.tile(window_sq, sub.shape[1])

        done = block_end / total_frames
        if done >= next_report:
            next_report = done + 0.01
            if not _report(done):
                return _passthrough()

    # --- WOLA 归一化 ---
    region = ola_norm[n : n + num_samples]
    valid = region > _NORM_EPS
    wet = np.zeros((num_ch, num_samples), dtype=np.float32)
    np.divide(
        ola[:, n : n + num_samples],
        np.where(valid, region, 1.0)[None, :],
        out=wet,
        where=valid[None, :],
    )

    # --- 与干信号混合 ---
    mix = _build_mix_curve(num_samples, sample_rate, intervals)
    output = work * (1.0 - mix[None, :]) + wet * mix[None, :]

    _report(1.0)
    return output[0] if single_channel else output
