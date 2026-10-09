"""图案层与渲染编排的自检。

直接运行：``python -m spectrumtag_batch.tests.test_pattern``

检查项：

1. 文字图案能渲染出非空字形，且被裁剪到内容边界；
2. 中文与英文都能渲染；
3. 图片二值化遵守 SpectrumTag 的 alpha / 亮度双模式规则；
4. ``plan_render`` 能把循环设置展开成正确的区间与掩码形状。
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from ..core.params import (
    DspSpec,
    EngraveMode,
    LoopSpec,
    PatternSource,
    PatternSpec,
    PlacementSpec,
    PositionMode,
    RenderJob,
)
from ..core.pattern import (
    _OUTLINE_WIDTH_RATIO,
    PatternError,
    build_outline,
    find_default_font,
    render_text_pattern,
)
from ..core.render import choose_num_cols, plan_render

SR = 44100
FFT_SIZE = 4096
NUM_BINS = FFT_SIZE // 2 + 1


def check_text_basic() -> bool:
    pattern = render_text_pattern("ABC", weight=0.25)
    h, w = pattern.shape
    duty = np.count_nonzero(pattern) / pattern.size
    # 裁剪到内容边界后，字形应当占据相当比例的像素
    ok = pattern.any() and w > h and 0.05 < duty < 0.9
    print(f"[1] 文字图案 (ABC)          尺寸 = {w}×{h}  覆盖率 = {duty:.0%}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_text_cjk() -> bool:
    """中文渲染依赖系统字体，缺字体时应给出清晰报错而不是空白图案。"""
    try:
        pattern = render_text_pattern("频谱水印", weight=0.25)
    except PatternError as exc:
        print(f"[2] 中文图案                无法渲染：{exc}  ->  跳过")
        return True

    h, w = pattern.shape
    duty = np.count_nonzero(pattern) / pattern.size
    ok = pattern.any() and w > h and duty > 0.02
    print(f"[2] 中文图案 (频谱水印)      尺寸 = {w}×{h}  覆盖率 = {duty:.0%}"
          f"  字体 = {find_default_font()}")
    return ok


def check_image_alpha_mode() -> bool:
    """透明底 + 不透明方块 → 走 alpha 通道，方块内为 True。"""
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(20, 80):
        for y in range(20, 80):
            img.putpixel((x, y), (255, 0, 0, 255))

    pattern = load_image_pattern_from(img)
    inside = bool(pattern[50, 50])
    outside = bool(pattern[5, 5])
    # 红色偏暗，若误走亮度模式也会判成"图案本体"，所以额外检查边界像素
    edge_ok = bool(pattern[25, 50]) and not bool(pattern[15, 50])
    ok = inside and not outside and edge_ok
    print(f"[3a] 图片 alpha 模式         中心 = {inside}  边角 = {outside}  "
          f"边界对齐 = {edge_ok}  ->  {'通过' if ok else '失败'}")
    return ok


def check_image_brightness_mode() -> bool:
    """不透明白底黑字 → 走亮度模式，暗像素为 True。"""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    for x in range(30, 70):
        for y in range(30, 70):
            img.putpixel((x, y), (0, 0, 0))

    pattern = load_image_pattern_from(img)
    inside = bool(pattern[50, 50])
    outside = bool(pattern[5, 5])
    ok = inside and not outside
    print(f"[3b] 图片亮度模式           黑块内 = {inside}  白底 = {outside}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_text_not_clipped() -> bool:
    """字形不能被画布裁掉 —— 曾经因为漏掉 bbox 偏移而切掉底部。

    判定看垂直方向：绘制原点必须减掉包围盒的 y 偏移，否则字形整体下移、底部
    被切平，上下留白会明显不对称。水平方向不作要求 —— bbox 宽度包含字符的
    advance（前进宽度），右侧本来就带边距，左右不等是字体的正常行为。
    """
    cases = ("WATERMARK", "AB", "水印", "Hg", "AAA\nBBB")
    details: list[str] = []
    ok = True

    for text in cases:
        pattern = render_text_pattern(text, weight=0.25)
        rows = np.any(pattern, axis=1)
        top = int(np.argmax(rows))
        bottom = int(len(rows) - 1 - np.argmax(rows[::-1]))
        pad_top = top
        pad_bottom = len(rows) - 1 - bottom
        details.append(f"{text.replace(chr(10), '/')}:{pad_top}/{pad_bottom}")
        if pad_top <= 0 or pad_bottom <= 0 or abs(pad_top - pad_bottom) > 2:
            ok = False

    print(f"[5] 文字不被裁切            上下留白 {' '.join(details)}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def load_image_pattern_from(img: Image.Image) -> np.ndarray:
    """绕过文件读取，直接测二值化。"""
    from ..core.pattern import binarize_image
    return binarize_image(img)


def check_plan_render() -> bool:
    """循环设置 → 区间列表 + 掩码形状。"""
    job = RenderJob(
        pattern=PatternSpec(source=PatternSource.TEXT, text="TEST"),
        dsp=DspSpec(fft_size=FFT_SIZE, mode=EngraveMode.CUT, strength=1.0),
        placement=PlacementSpec(
            freq_low_norm=0.10, freq_high_norm=0.40,
            start=0.0, duration=1.0,
            position_mode=PositionMode.ABSOLUTE,
        ),
        loop=LoopSpec(enabled=True, interval_sec=2.0),
    )

    total = 5.0
    plan = plan_render(job, SR, total)

    # 间隔量的是「上一次印完 → 下一次开始」：0.0~1.0 印完，隔 2 秒 → 3.0~4.0；
    # 再下一次要 6.0 才开始，已越过 5 秒的音频末尾
    starts = [round(iv.start_sec, 3) for iv in plan.intervals]
    expected = [0.0, 3.0]
    intervals_ok = len(starts) == len(expected) and all(
        abs(a - b) < 1e-6 for a, b in zip(starts, expected)
    )
    # 每次都是完整的 1 秒时长，没有被音频末尾挤短
    last_ok = abs(plan.intervals[-1].end_sec - 4.0) < 1e-6

    num_cols = choose_num_cols(plan.intervals, SR, FFT_SIZE)
    expected_cols = round(1.0 * SR / (FFT_SIZE // 4))
    cols_ok = num_cols == min(expected_cols, 4096)
    shape_ok = plan.main_band.mask.shape == (NUM_BINS, num_cols)

    # 默认开着图案描边：计划里该多挂一层 —— 频率范围与本体完全一致，
    # 印法朝反方向（衰减配凸起），掩码是另一张（外沿那一圈）
    ring = plan.outline_band
    outline_ok = (
        ring is not None
        and ring.mode is EngraveMode.BOOST
        and ring.freq_low_norm == plan.main_band.freq_low_norm
        and ring.freq_high_norm == plan.main_band.freq_high_norm
        and ring.mask.shape == plan.main_band.mask.shape
        and not np.array_equal(ring.mask, plan.main_band.mask)
        and not (ring.mask.astype(bool) & plan.main_band.mask.astype(bool)).any()
    )
    # 注入印法下描边反过来去衰减 —— 峰谷对不分用哪种印法
    draw_ring = plan_render(
        job.with_dsp(mode=EngraveMode.DRAW), SR, total
    ).outline_band
    inverted_ok = draw_ring is not None and draw_ring.mode is EngraveMode.CUT
    # 关掉描边就只剩本体一层
    single_ok = plan_render(
        job.with_pattern(outline=False), SR, total
    ).outline_band is None

    ok = (intervals_ok and last_ok and cols_ok and shape_ok
          and outline_ok and inverted_ok and single_ok)
    print(f"[4] 渲染计划                区间 = {starts}  掩码 = {plan.main_band.mask.shape}  "
          f"层数 = {len(plan.bands)}（外圈 {ring.mode.value if ring else '无'}）"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_outline_shape() -> bool:
    """描边就是本体外沿的一圈：不与本体重叠、恰好包住本体、宽度随图案缩放。

    宽度按**较短边**算 —— 那一边对应印章框的频率方向，也是描边最容易把
    图案吃掉的方向。
    """
    pattern = np.zeros((200, 400), dtype=bool)
    pattern[80:120, 100:300] = True                 # 中间一块实心矩形
    radius = max(1, round(min(pattern.shape) * _OUTLINE_WIDTH_RATIO))

    outline = build_outline(pattern)
    no_overlap = not (outline & pattern).any()

    rows = np.where(outline.any(axis=1))[0]
    cols = np.where(outline.any(axis=0))[0]
    # 外扩了 radius 圈，一圈不多一圈不少
    ring_rows = (int(rows.min()), int(rows.max())) == (80 - radius, 119 + radius)
    ring_cols = (int(cols.min()), int(cols.max())) == (100 - radius, 299 + radius)
    # 实心块的描边是个"回"字形，面积可以精确算出来
    expected = (
        (120 - 80 + 2 * radius) * (300 - 100 + 2 * radius) - (120 - 80) * (300 - 100)
    )
    count_ok = int(outline.sum()) == expected

    # 文字那种细笔画也必须描得出来
    text_outline = build_outline(render_text_pattern("AB", weight=0.25))
    text_ok = bool(text_outline.any())

    ok = no_overlap and ring_rows and ring_cols and count_ok and text_ok
    print(f"[6] 描边形状                半径 = {radius}px  外框 = "
          f"{rows.min()}~{rows.max()} × {cols.min()}~{cols.max()}  像素 = "
          f"{int(outline.sum())}（期望 {expected}）  文字可描 = {text_ok}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def main() -> int:
    print("=" * 68)
    print("图案层与渲染编排 —— 自检")
    print("=" * 68)

    results = [
        check_text_basic(),
        check_text_cjk(),
        check_image_alpha_mode(),
        check_image_brightness_mode(),
        check_plan_render(),
        check_text_not_clipped(),
        check_outline_shape(),
    ]

    passed = sum(results)
    print("-" * 68)
    print(f"结果：{passed}/{len(results)} 项通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
