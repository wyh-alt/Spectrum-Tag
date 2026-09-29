"""渲染编排 —— 把「图案 + 位置 + 循环 + DSP 参数」变成实际的音频处理。

这一层负责三件 DSP 层不该关心的事：

1. 决定掩码的列数（时间方向的采样密度）；
2. 把位置/循环设置展开成具体的时间区间；
3. 兜住"区间落在音频之外"这类空操作。

掩码列数的选择依据：一列对应一个 STFT 帧时时间分辨率最高，再多的列也用不上
（同一列会被相邻帧重复取用）。所以取 ``时长 / 帧跳距``，并夹在 [4, 4096] 内 ——
上界与 SpectrumTag 的 ``computeMaskCols`` 保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .dsp import build_binary_mask, render_watermark
from .params import (
    MAX_MASK_COLS,
    MIN_MASK_COLS,
    Interval,
    RenderJob,
    resolve_intervals,
)
from .pattern import build_pattern

class EmptyRender(RuntimeError):
    """没有任何区间落在音频范围内（例如起点超出音频时长）。"""


@dataclass(frozen=True)
class RenderPlan:
    """一次渲染的完整计划，UI 预览与批处理共用。"""

    intervals: tuple[Interval, ...]
    mask: np.ndarray
    freq_low_norm: float
    freq_high_norm: float

    @property
    def total_stamp_seconds(self) -> float:
        return sum(iv.duration for iv in self.intervals)


def choose_num_cols(
    intervals: Sequence[Interval],
    sample_rate: float,
    fft_size: int,
) -> int:
    """按最短区间决定掩码列数，保证每个区间的图案都不丢细节。"""
    if not intervals:
        return MIN_MASK_COLS
    hop = max(1, fft_size // 4)
    shortest = min(iv.duration for iv in intervals)
    frames = shortest * sample_rate / hop
    return int(np.clip(round(frames), MIN_MASK_COLS, MAX_MASK_COLS))


def plan_render(
    job: RenderJob,
    sample_rate: float,
    total_sec: float,
    binary_pattern: Optional[np.ndarray] = None,
) -> RenderPlan:
    """算好区间与掩码，但不处理音频 —— 供 UI 预览复用。

    Parameters
    ----------
    binary_pattern
        已生成的布尔图案。为 None 时按 ``job.pattern`` 现场生成（批处理走这条路，
        图案只在每个文件开头生成一次）。
    """
    intervals = resolve_intervals(job.placement, job.loop, total_sec)
    if not intervals:
        raise EmptyRender("没有印章落在音频范围内，请检查起点与时长")

    num_cols = choose_num_cols(intervals, sample_rate, job.dsp.fft_size)
    num_bins = job.dsp.fft_size // 2 + 1
    pattern = build_pattern(job.pattern) if binary_pattern is None else binary_pattern
    mask = build_binary_mask(pattern, num_bins, num_cols)

    return RenderPlan(
        intervals=tuple(intervals),
        mask=mask,
        freq_low_norm=job.placement.freq_low_norm,
        freq_high_norm=job.placement.freq_high_norm,
    )


def render_audio(
    audio: np.ndarray,
    sample_rate: float,
    job: RenderJob,
    binary_pattern: Optional[np.ndarray] = None,
    plan: Optional[RenderPlan] = None,
    progress: Optional[Callable[[float], bool]] = None,
) -> np.ndarray:
    """按 job 处理整段音频，返回同形状的结果。

    ``progress`` 收到 [0, 1] 的处理进度，返回 False 表示请求取消
    （取消时返回干信号副本，不做部分写入）。

    Raises
    ------
    EmptyRender
        没有任何区间落在音频内。
    """
    if plan is None:
        total_sec = audio.shape[-1] / float(sample_rate)
        plan = plan_render(job, sample_rate, total_sec, binary_pattern)

    return render_watermark(
        audio,
        sample_rate,
        plan.mask,
        job.dsp,
        plan.intervals,
        plan.freq_low_norm,
        plan.freq_high_norm,
        progress=progress,
    )
