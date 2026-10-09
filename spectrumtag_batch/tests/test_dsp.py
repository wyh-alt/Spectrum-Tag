"""DSP 核心的自检 —— 验证 Python 移植与 SpectrumTag 的 STFT/OLA 数学等价。

直接运行：``python -m spectrumtag_batch.tests.test_dsp``

检查项：

1. **恒等性**：``amplitude_ratio = 1.0`` 时所有 bin 增益都是 1，
   整条链路应当是无损重建。这一步同时检验帧几何、WOLA 归一化、重叠相加分组。
2. **脉冲对齐**：脉冲位置不得移动 —— 检验 N-1 的群延迟补偿是否正确。
3. **掩码生效**：``ratio = 0`` 应当把目标频段的能量显著压下去。
4. **循环印刷**：多个区间都要留下印章，且区间之外保持原样。
5. **注入模式**：静音素材上也能"画"出图案。
6. **图案描边**：本体与外圈朝相反方向处理，配成峰谷对。
"""

from __future__ import annotations

import sys

import numpy as np

from ..core.dsp import EngraveBand, build_binary_mask, render_watermark
from ..core.params import (
    DEFAULT_CUT_STRENGTH,
    DspSpec,
    EngraveMode,
    Interval,
    LoopSpec,
    PlacementSpec,
    PositionMode,
    outline_boost_for,
    resolve_intervals,
)

SR = 44100
FFT_SIZE = 4096
NUM_BINS = FFT_SIZE // 2 + 1


def _band_energy(x: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    sel = (freqs >= lo_hz) & (freqs <= hi_hz)
    return float(np.sum(np.abs(spec[sel]) ** 2))


def _full_mask(num_cols: int = 64) -> np.ndarray:
    """整块都是图案本体的掩码。"""
    return np.ones((NUM_BINS, num_cols), dtype=np.float32)


def _band(
    dsp: DspSpec,
    low_norm: float,
    high_norm: float,
    mode: EngraveMode | None = None,
) -> EngraveBand:
    """按 DspSpec 造一个频段；``mode`` 可以覆盖掉它自带的印法。"""
    return EngraveBand(
        _full_mask(), low_norm, high_norm,
        mode or dsp.mode, dsp.strength, dsp.invert,
    )


def _render(
    audio: np.ndarray,
    dsp: DspSpec,
    intervals: list[Interval],
    low_norm: float = 0.10,
    high_norm: float = 0.40,
    extra: list[EngraveBand] | None = None,
) -> np.ndarray:
    bands = [_band(dsp, low_norm, high_norm)] + list(extra or [])
    return render_watermark(audio, SR, bands, intervals)


def check_identity() -> bool:
    """强度 0 → 增益恒为 1 → 应无损重建。"""
    num = SR  # 1 秒
    rng = np.random.default_rng(1234)
    audio = (rng.standard_normal(num) * 0.1).astype(np.float32)

    dsp = DspSpec(fft_size=FFT_SIZE, mode=EngraveMode.CUT, strength=0.0)
    out = _render(audio, dsp, [Interval(0.0, 1.0)], 0.1, 0.4)

    err = float(np.max(np.abs(out - audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    ok = err < 1e-4 and out.shape == audio.shape
    print(f"[1] 恒等性 (ratio=1.0)      最大误差 = {err:.3e}  "
          f"(信号 RMS = {rms:.3e})  ->  {'通过' if ok else '失败'}")
    return ok


def check_impulse_alignment() -> bool:
    """脉冲位置必须原地不动 —— 群延迟已被 N-1 补偿。"""
    num = SR
    audio = np.zeros(num, dtype=np.float32)
    positions = [500, 11000, 23999, 40000]
    for p in positions:
        audio[p] = 1.0

    dsp = DspSpec(fft_size=FFT_SIZE, mode=EngraveMode.CUT, strength=0.0)
    out = _render(audio, dsp, [Interval(0.0, 1.0)], 0.05, 0.95)

    # 峰值位置应与输入一致（脉冲经过加窗重建后会略微扩散，取邻域最大值）
    shifted = 0
    for p in positions:
        lo, hi = max(0, p - 8), min(num, p + 9)
        peak = lo + int(np.argmax(np.abs(out[lo:hi])))
        if abs(peak - p) > 1:
            shifted += 1

    err = float(np.max(np.abs(out - audio)))
    ok = shifted == 0 and err < 1e-3
    print(f"[2] 脉冲对齐                偏移数量 = {shifted}/4   最大误差 = {err:.3e}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_mask_effect() -> bool:
    """ratio = 0 应把掩码覆盖频段的能量压低。"""
    num = SR
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(num) * 0.1).astype(np.float32)

    low_norm, high_norm = 0.10, 0.20
    lo_hz = low_norm * SR / 2
    hi_hz = high_norm * SR / 2

    dsp = DspSpec(fft_size=FFT_SIZE, mode=EngraveMode.CUT, strength=1.0)
    out = _render(audio, dsp, [Interval(0.0, 1.0)], low_norm, high_norm)

    # 只测区间中部：两端各有 20ms 的淡入淡出，那里按设计本来就保留干信号
    a, b = int(0.10 * SR), int(0.90 * SR)
    before = _band_energy(audio[a:b], lo_hz, hi_hz)
    after = _band_energy(out[a:b], lo_hz, hi_hz)
    # 带外能量应基本不变
    out_before = _band_energy(audio[a:b], hi_hz * 1.5, SR / 2 * 0.9)
    out_after = _band_energy(out[a:b], hi_hz * 1.5, SR / 2 * 0.9)

    ratio_db = 10 * np.log10(max(after, 1e-30) / max(before, 1e-30))
    keep_db = 10 * np.log10(max(out_after, 1e-30) / max(out_before, 1e-30))

    # 整段（含淡入淡出）用于交叉验证：4% 的时长保留干信号 → 约 -14 dB
    whole_db = 10 * np.log10(
        max(_band_energy(out, lo_hz, hi_hz), 1e-30) / max(_band_energy(audio, lo_hz, hi_hz), 1e-30)
    )

    ok = ratio_db < -25.0 and abs(keep_db) < 1.0
    print(f"[3] 掩码生效 (ratio=0)      中段带内 {ratio_db:+.1f} dB  "
          f"整段带内 {whole_db:+.1f} dB  带外 {keep_db:+.2f} dB"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_loop_intervals() -> bool:
    """循环印刷：每个区间都应留下印章，区间外保持原样。"""
    num = SR * 3
    rng = np.random.default_rng(99)
    audio = (rng.standard_normal(num) * 0.1).astype(np.float32)

    # 区间要明显长于一个 STFT 帧（N/sr ≈ 93ms），否则整个印章都落在
    # 「只有部分帧被改增益」的边界稀释区里，效果会弱很多。
    placement = PlacementSpec(
        freq_low_norm=0.10, freq_high_norm=0.20,
        start=0.20, duration=0.50,
        position_mode=PositionMode.ABSOLUTE,
    )
    loop = LoopSpec(enabled=True, interval_sec=1.0)
    intervals = resolve_intervals(placement, loop, num / SR)

    # 间隔量的是「上一次印完 → 下一次开始」，所以步进 = 时长 0.5 + 间隔 1.0 = 1.5
    #   0.2~0.7 , 1.7~2.2 , 下一次 3.2 已越过 3 秒的音频末尾
    expected = [0.2, 1.7]
    got = [round(iv.start_sec, 3) for iv in intervals]
    intervals_ok = len(intervals) == len(expected) and all(
        abs(a - b) < 1e-6 for a, b in zip(got, expected)
    )

    dsp = DspSpec(fft_size=FFT_SIZE, mode=EngraveMode.CUT, strength=1.0)
    out = _render(
        audio, dsp, intervals,
        placement.freq_low_norm, placement.freq_high_norm,
    )

    lo_hz, hi_hz = 0.10 * SR / 2, 0.20 * SR / 2
    # 各印章区间的能量都应被压低
    stamps_ok = True
    per_stamp: list[str] = []
    for start in expected:
        # 取印章中段，避开两端各一个帧长（≈93ms）的稀释区
        a = int((start + 0.20) * SR)
        b = int((start + 0.40) * SR)
        before = _band_energy(audio[a:b], lo_hz, hi_hz)
        after = _band_energy(out[a:b], lo_hz, hi_hz)
        db = 10 * np.log10(max(after, 1e-30) / max(before, 1e-30))
        per_stamp.append(f"{db:+.1f}")
        if db > -10.0:
            stamps_ok = False

    # 区间之外应原样保留（区间是 [0.2,0.7]、[1.2,1.7]、[2.2,2.7]，
    # 所以取两段印章之间的空隙，并避开端点 20ms 淡出）
    a, b = int(0.75 * SR), int(1.15 * SR)
    untouched = float(np.max(np.abs(out[a:b] - audio[a:b])))

    ok = intervals_ok and stamps_ok and untouched < 1e-5
    print(f"[4] 循环印刷                区间 = {got}  各章衰减 = {per_stamp} dB  "
          f"区间外偏差 = {untouched:.2e}  ->  {'通过' if ok else '失败'}")
    return ok


def check_draw_mode() -> bool:
    """注入模式：在**完全静音**的素材上也要能"画"出图案。

    用衰减模式做对照 —— 静音乘任何增益仍然是静音，所以它一个样本都改不动。
    这个对比正是两种模式的本质区别。
    """
    num = SR
    silent = np.zeros(num, dtype=np.float32)
    low_norm, high_norm = 0.10, 0.20
    lo_hz, hi_hz = low_norm * SR / 2, high_norm * SR / 2
    stamp = [Interval(0.30, 0.70)]

    def render(dsp):
        return _render(silent, dsp, stamp, low_norm, high_norm)

    cut_out = render(DspSpec(fft_size=FFT_SIZE, mode=EngraveMode.CUT, strength=1.0))
    cut_peak = float(np.max(np.abs(cut_out)))

    draw_out = render(DspSpec(fft_size=FFT_SIZE, mode=EngraveMode.DRAW, strength=1.0))

    # 印章区间内应当出现能量
    a, b = int(0.40 * SR), int(0.60 * SR)
    seg = draw_out[a:b]
    peak = float(np.max(np.abs(seg)))

    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / (len(seg) / 2.0)
    freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
    in_band = float(spec[(freqs >= lo_hz) & (freqs <= hi_hz)].max())
    out_band = float(spec[freqs > hi_hz * 1.5].max())
    in_db = 20 * np.log10(max(in_band, 1e-12))
    out_db = 20 * np.log10(max(out_band, 1e-12))

    # 印章区间之外必须仍是静音
    before = float(np.max(np.abs(draw_out[: int(0.20 * SR)])))
    after = float(np.max(np.abs(draw_out[int(0.80 * SR):])))

    # 目标电平 -25 dBFS，重建后允许有若干 dB 偏差
    level_ok = -45.0 < in_db < -10.0
    # 注入必须落在指定频段里
    band_ok = in_db - out_db > 20.0

    ok = (
        cut_peak == 0.0            # 衰减对静音无能为力
        and peak > 1e-3            # 注入真的造出了信号
        and level_ok
        and band_ok
        and before < 1e-6
        and after < 1e-6
    )
    print(f"[6] 注入模式（静音上作画）  衰减产出 = {cut_peak:.1e}  注入峰值 = {peak:.4f}  "
          f"带内 {in_db:.1f} dB / 带外 {out_db:.1f} dB  章外 {max(before, after):.1e}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_truncated_stamp_not_squashed() -> bool:
    """末尾放不下的印章该被**切掉**，而不是压扁到剩余时间里。

    压扁会让最后一个印章明显比前面几个小，看起来像出了错。列号必须按原始
    时长推进，走到哪儿算哪儿。
    """
    from ..core.dsp import _assign_columns

    # 音频 4 秒、印章 1.5 秒、间隔 1.5 秒：
    #   0.0~1.5 完整；3.0~4.5 被末尾截成 3.0~4.0
    placement = PlacementSpec(
        start=0.0, duration=1.5, position_mode=PositionMode.ABSOLUTE
    )
    loop = LoopSpec(enabled=True, interval_sec=1.5)
    intervals = resolve_intervals(placement, loop, 4.0)

    if len(intervals) != 2:
        print(f"[7] 末尾截断                区间数应为 2，实得 {len(intervals)}  ->  失败")
        return False

    last = intervals[-1]
    truncated = last.duration < last.mapping_span - 1e-9

    num_cols = 100
    tail_cols = _assign_columns(
        np.array([last.start_sec, last.end_sec - 1e-9]), [last], num_cols
    )
    full_col = _assign_columns(
        np.array([intervals[0].end_sec - 1e-9]), [intervals[0]], num_cols
    )[0]

    ok = (
        truncated                          # 确实被截短了
        and tail_cols[0] == 0              # 起点还是从图案头开始
        and tail_cols[-1] < num_cols - 1   # 没走到图案尾 —— 是切掉不是压扁
        and full_col == num_cols - 1       # 对照：完整的那次应当铺满
    )
    print(f"[7] 末尾截断而非压扁          实际 {last.duration:.2f}s / 映射 "
          f"{last.mapping_span:.2f}s  末尾列号 {tail_cols[-1]}"
          f"（完整印章 {full_col}）  ->  {'通过' if ok else '失败'}")
    return ok


def check_outline_peak_valley() -> bool:
    """图案描边：本体削弱时外圈被抬起来，本体注入时外圈被压下去 —— 峰谷成对。

    掩码手工摆成两片：下半是"本体"、紧挨着上面一条是"外圈"。真实路径里
    这两片由 ``pattern.build_outline`` 从图案算出来（形状检查在 test_pattern），
    这里只验 DSP 层对这两片确实朝相反方向处理。
    """
    num = SR * 2
    rng = np.random.default_rng(21)
    audio = (rng.standard_normal(num) * 0.12).astype(np.float32)
    stamp = [Interval(0.30, 1.70)]

    low_norm, high_norm = 0.10, 0.20
    band_lo = low_norm * SR / 2
    span_hz = (high_norm - low_norm) * SR / 2
    # 掩码行号大 = 频率高：本体占上半段，外圈紧贴在它下方
    ring_rows = 200
    body = np.zeros((NUM_BINS, 64), dtype=np.float32)
    body[NUM_BINS // 2 :, :] = 1.0
    ring = np.zeros_like(body)
    ring[NUM_BINS // 2 - ring_rows : NUM_BINS // 2, :] = 1.0

    hz_of_row = lambda row: band_lo + span_hz * row / (NUM_BINS - 1)  # noqa: E731
    body_hz = (hz_of_row(NUM_BINS // 2 + 20), hz_of_row(NUM_BINS - 20))
    ring_hz = (hz_of_row(NUM_BINS // 2 - ring_rows + 20), hz_of_row(NUM_BINS // 2 - 20))

    def render(mode: EngraveMode, strength: float) -> np.ndarray:
        # 外圈与本体反着来：衰减配凸起、注入配衰减。这条规则住在
        # render.plan_render 里（DSP 层不知道"描边"是什么），test_pattern
        # 会另行确认它确实是这么组装的。
        ring_mode = (
            EngraveMode.CUT if mode is EngraveMode.DRAW else EngraveMode.BOOST
        )
        bands = [
            EngraveBand(body, low_norm, high_norm, mode, strength),
            EngraveBand(ring, low_norm, high_norm, ring_mode, strength),
        ]
        return render_watermark(audio, SR, bands, stamp)

    a, b = int(0.60 * SR), int(1.40 * SR)

    def delta_db(out: np.ndarray, lo: float, hi: float) -> float:
        before = _band_energy(audio[a:b], lo, hi)
        after = _band_energy(out[a:b], lo, hi)
        return 10 * np.log10(max(after, 1e-30) / max(before, 1e-30))

    # 衰减 + 描边：本体塌下去、外圈鼓起来
    cut_out = render(EngraveMode.CUT, 0.8)
    cut_body = delta_db(cut_out, *body_hz)
    cut_ring = delta_db(cut_out, *ring_hz)

    # 注入 + 描边：本体亮起来、外圈压下去
    draw_out = render(EngraveMode.DRAW, 0.8)
    draw_body = delta_db(draw_out, *body_hz)
    draw_ring = delta_db(draw_out, *ring_hz)

    ok = (
        cut_body < -8.0        # 本体被削弱
        and cut_ring > 3.0     # 外圈被抬起来（+7.7 dB 的目标值，留足余量）
        and draw_body > 3.0    # 本体被注入出内容
        and draw_ring < -3.0   # 外圈被压下去
    )
    print(f"[8] 图案描边（峰谷成对）    衰减：本体 {cut_body:+.1f} dB / 外圈 {cut_ring:+.1f} dB   "
          f"注入：本体 {draw_body:+.1f} dB / 外圈 {draw_ring:+.1f} dB"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_outline_boost_range() -> bool:
    """描边的量要与凹陷配套：凸起落在 +6~9 dB，凹陷随强度线性加深。

    数值本身是取舍（听得出来但不刺耳、不撞削波），但它与凹陷是一对 ——
    这条守着两者不会在后续调参里跑偏到一边去。
    """
    default_db = 20 * np.log10(outline_boost_for(DEFAULT_CUT_STRENGTH))
    full_db = 20 * np.log10(outline_boost_for(1.0))
    cut_default_db = 20 * np.log10(1.0 - DEFAULT_CUT_STRENGTH)

    ok = (
        6.0 < default_db < 9.0                 # 默认强度下的凸起在设计区间里
        and 8.0 < full_db < 9.5                # 拉满也不过 +9.5 dB，别撞削波
        and abs(cut_default_db + 26.0) < 0.5   # 默认凹陷约 -26 dB
        and outline_boost_for(0.0) == 1.0      # 强度 0 时不该有任何凸起
    )
    print(f"[9] 描边的量                默认强度凸起 {default_db:+.1f} dB / "
          f"拉满 {full_db:+.1f} dB   对照凹陷 {cut_default_db:+.1f} dB"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_mask_rasterization() -> bool:
    """图案栅格化：纵向必须翻转（row 0 = 最低频 = 图片底部）。"""
    # 8×8 图案，只有最上面一行是本体（图片顶部 = 高频）
    binary = np.zeros((8, 8), dtype=bool)
    binary[0, :] = True

    mask = build_binary_mask(binary, 8, 4)
    top_is_bottom = bool(mask[-1].all() and not mask[0].any())

    # 行数/列数不匹配时的拉伸
    wide = build_binary_mask(binary, 64, 3)
    shape_ok = wide.shape == (64, 3)

    ok = top_is_bottom and shape_ok
    print(f"[5] 图案栅格化              纵向翻转到高频行 = {top_is_bottom}  "
          f"拉伸形状 = {wide.shape}  ->  {'通过' if ok else '失败'}")
    return ok


def main() -> int:
    print("=" * 68)
    print("SpectrumTag 批量版 —— DSP 核心自检")
    print(f"采样率 {SR} Hz   FFT {FFT_SIZE}   频段 {NUM_BINS} bins")
    print("=" * 68)

    results = [
        check_identity(),
        check_impulse_alignment(),
        check_mask_effect(),
        check_loop_intervals(),
        check_draw_mode(),
        check_truncated_stamp_not_squashed(),
        check_mask_rasterization(),
        check_outline_peak_valley(),
        check_outline_boost_range(),
    ]

    passed = sum(results)
    print("-" * 68)
    print(f"结果：{passed}/{len(results)} 项通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
