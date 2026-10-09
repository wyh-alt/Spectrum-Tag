"""生成演示图 —— 处理前后的时频图对比，用来肉眼确认水印真的印上去了。

直接运行：``python -m spectrumtag_batch.tests.make_demo``
输出：``%TEMP%/stbatch_demo/before_after.png``
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# 演示图只看清"印上去的是什么样"，所以循环显式关掉 —— 它是界面上的默认行为，
# 但在对比图里铺满整段会把三个面板糊成一片。图案描边保留：它也是默认行为，
# 而且正好把"本体与外圈反向"这件事一并显示出来。
_SINGLE_STAMP = dict(loop=LoopSpec(enabled=False))
from ..core.render import render_audio
from ..ui.spectrogram import compute_display_spectrogram, db_to_rgb

SR = 44100
DURATION = 3.0
OUT_DIR = os.path.join(tempfile.gettempdir(), "stbatch_demo")

# 水印覆盖的频率与时间范围
FREQ_LOW, FREQ_HIGH = 0.12, 0.62
STAMP_START, STAMP_DURATION = 0.35, 2.30


def _make_audio() -> np.ndarray:
    """用宽带噪声打底 —— 各频段都有能量，挖出来的文字轮廓最清楚。"""
    rng = np.random.default_rng(2024)
    frames = int(SR * DURATION)
    noise = rng.standard_normal(frames) * 0.18
    t = np.arange(frames) / SR
    hum = 0.06 * np.sin(2 * np.pi * 110 * t)
    mono = (noise + hum).astype(np.float32)
    return np.stack([mono, mono])


def _render_spec_image(db: np.ndarray, ceiling: float, width: int, height: int) -> Image.Image:
    """把 dB 矩阵渲染成一张热力图。横轴时间，纵轴频率（低频在下）。"""
    rgb = db_to_rgb(db, ceiling - 90.0, ceiling)
    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((width, height), Image.LANCZOS)
    # 数组行 0 是最低频，而屏幕上方应是高频 → 上下翻转
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def _label(text: str, size: tuple[int, int], font_path: str | None) -> Image.Image:
    img = Image.new("RGB", size, (18, 18, 22))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw.text((12, 8), text, fill=(235, 235, 240), font=font)
    return img


def _make_tonal_audio() -> np.ndarray:
    """有低频内容、高频**完全空白**的素材。

    噪声底必须真的低到零 —— 否则衰减也能靠"抹掉噪声底"显影，就分辨不出
    两种模式的本质差别了。
    """
    rng = np.random.default_rng(7)
    frames = int(SR * DURATION)
    t = np.arange(frames) / SR

    # 噪声底经低通后只留在 4 kHz 以下
    noise = rng.standard_normal(frames)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(frames, 1.0 / SR)
    spec[freqs > 4000.0] *= 1.0e-5
    noise = np.fft.irfft(spec, n=frames)

    content = (
        0.30 * np.sin(2 * np.pi * 150 * t)
        + 0.22 * np.sin(2 * np.pi * 420 * t)
        + 0.14 * np.sin(2 * np.pi * 980 * t)
        + 0.08 * np.sin(2 * np.pi * 2100 * t)
        + 0.010 * noise
    )
    mono = content.astype(np.float32)
    return np.stack([mono, mono])


def make_draw_vs_cut() -> str:
    """三面板对比：原始 / 衰减 / 注入，水印落在原本空白的超高频。"""
    from ..core.pattern import find_default_font
    from ..core.params import EngraveMode as _Mode

    audio = _make_tonal_audio()
    common = dict(
        pattern=PatternSpec(
            source=PatternSource.TEXT, text="水印", weight=0.35
        ),
        placement=PlacementSpec(
            # 归一化 0.62~0.95 ≈ 13.7k~21 kHz，这里原本什么都没有
            freq_low_norm=0.62, freq_high_norm=0.95,
            start=STAMP_START, duration=STAMP_DURATION,
            position_mode=PositionMode.ABSOLUTE,
        ),
    )

    cut_audio = render_audio(
        audio, SR,
        RenderJob(
            dsp=DspSpec(fft_size=4096, mode=_Mode.CUT, strength=1.0),
            **common, **_SINGLE_STAMP,
        ),
    )
    draw_audio = render_audio(
        audio, SR,
        RenderJob(
            dsp=DspSpec(fft_size=4096, mode=_Mode.DRAW, strength=0.85),
            **common, **_SINGLE_STAMP,
        ),
    )

    shows = [audio, cut_audio, draw_audio]
    dbs = [compute_display_spectrogram(s, SR) for s in shows]
    ceiling = min(0.0, float(np.percentile(np.concatenate([d.ravel() for d in dbs]), 99.9)))

    width, height = 1500, 380
    font_path = find_default_font()
    panels = []
    for title, db in zip(
        ("原始素材　（高频一片空白）",
         "衰减模式　（该区域本来就没内容，什么也挖不出来）",
         "注入模式　（凭空画出图案）"),
        dbs,
    ):
        panels.append(_label(title, (width, 40), font_path))
        panels.append(_render_spec_image(db, ceiling, width, height))

    total_h = sum(p.height for p in panels) + 10 * (len(panels) - 1)
    canvas = Image.new("RGB", (width, total_h), (18, 18, 22))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + 10

    out_path = os.path.join(OUT_DIR, "draw_vs_cut.png")
    canvas.save(out_path)
    return out_path


def main() -> int:
    from ..core.pattern import find_default_font

    os.makedirs(OUT_DIR, exist_ok=True)

    audio = _make_audio()
    job = RenderJob(
        pattern=PatternSpec(
            source=PatternSource.TEXT,
            text="水印",
            weight=0.35,
        ),
        dsp=DspSpec(fft_size=4096, mode=EngraveMode.CUT, strength=1.0),
        placement=PlacementSpec(
            freq_low_norm=FREQ_LOW,
            freq_high_norm=FREQ_HIGH,
            start=STAMP_START,
            duration=STAMP_DURATION,
            position_mode=PositionMode.ABSOLUTE,
        ),
        **_SINGLE_STAMP,
    )

    processed = render_audio(audio, SR, job)

    # 两张图用同一个显示上限，亮度才可比
    db_before = compute_display_spectrogram(audio, SR)
    db_after = compute_display_spectrogram(processed, SR)
    ceiling = float(np.percentile(np.concatenate([db_before.ravel(), db_after.ravel()]), 99.9))
    ceiling = min(0.0, ceiling)

    width, height = 1500, 420
    font_path = find_default_font()

    panels = [
        _label("处理前", (width, 40), font_path),
        _render_spec_image(db_before, ceiling, width, height),
        _label("处理后  ——  水印作为频域掩码印在频谱上", (width, 40), font_path),
        _render_spec_image(db_after, ceiling, width, height),
    ]

    total_h = sum(p.height for p in panels) + 3 * 10
    canvas = Image.new("RGB", (width, total_h), (18, 18, 22))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + 10

    out_path = os.path.join(OUT_DIR, "before_after.png")
    canvas.save(out_path)
    print("演示图已生成:", out_path)
    print(f"  音频 {DURATION}s @ {SR}Hz   水印频段 {FREQ_LOW}-{FREQ_HIGH}（归一化，约 "
          f"{FREQ_LOW * SR / 2:.0f}-{FREQ_HIGH * SR / 2:.0f} Hz）")
    print(f"  时间 {STAMP_START}-{STAMP_START + STAMP_DURATION}s")

    draw_path = make_draw_vs_cut()
    print("演示图已生成:", draw_path)
    print("  衰减 vs 注入 对比（水印落在原本空白的超高频 13.7k-21kHz）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
