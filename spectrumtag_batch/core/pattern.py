"""图案层 —— 把图片或文字变成 DSP 用的二值图案。

二值化规则逐条对应 `SpectrumTag/Standalone/SharedUI.cpp::preprocessImage`：

* 先把最长边缩到不超过 2048（避免超大图拖慢栅格化）；
* 统计透明像素占比，**超过 5% 就走 alpha 通道**（不透明像素 = 图案本体），
  否则走亮度，**暗于全图平均亮度的像素 = 图案本体**（即黑字白底）；
* 亮度用 JUCE 的加权公式 ``0.299R + 0.587G + 0.114B``。

文字图案额外做一件事：渲染后**裁剪到内容边界**，否则四周的空白会让文字
在掩码里缩成一小块。
"""

from __future__ import annotations

import os
import platform
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .params import PatternSource, PatternSpec

# 与 SpectrumTag 一致的缩放上限
_MAX_SIDE = 2048

# 图片二值化的透明像素阈值：透明占比 > 1/20 时改用 alpha 通道
_ALPHA_MODE_DIVISOR = 20

# 文字渲染的目标尺寸（最长边像素）。图案最终会被拉伸铺满掩码网格，
# 先在足够大的画布上渲染，栅格化时才不会丢细节。
_TEXT_TARGET_PX = 2048
_TEXT_MIN_SIZE = 24
_TEXT_MAX_SIZE = 4096

# 各平台常见中文字体，按优先级排列
_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Windows": (
        r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",     # 微软雅黑 Bold
        r"C:\Windows\Fonts\simhei.ttf",     # 黑体
        r"C:\Windows\Fonts\simsun.ttc",     # 宋体
        r"C:\Windows\Fonts\Deng.ttf",       # 等线
        r"C:\Windows\Fonts\arial.ttf",
    ),
    "Darwin": (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
    "Linux": (
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
}


class PatternError(RuntimeError):
    """图案无法生成（图片损坏、字体缺失等）。"""


@lru_cache(maxsize=1)
def find_default_font() -> Optional[str]:
    """挑一个能显示中文的系统字体；找不到返回 None（由 PIL 回退到默认位图字体）。"""
    for candidate in _FONT_CANDIDATES.get(platform.system(), ()):
        if os.path.exists(candidate):
            return candidate
    return None


def _fit_max_side(img: Image.Image, max_side: int = _MAX_SIDE) -> Image.Image:
    """等比缩放到最长边不超过 max_side。"""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side or longest == 0:
        return img
    scale = max_side / float(longest)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def binarize_image(img: Image.Image, threshold: Optional[float] = None) -> np.ndarray:
    """把 PIL 图像转成布尔图案数组（True = 图案本体）。

    对应 SpectrumTag 的 ``preprocessImage``。``threshold`` 为 None 时按上述
    规则自动选择判定方式与阈值。
    """
    img = _fit_max_side(img.convert("RGBA"))
    arr = np.asarray(img, dtype=np.uint8)
    if arr.size == 0:
        raise PatternError("图像为空")

    alpha = arr[..., 3].astype(np.float32)
    rgb = arr[..., :3].astype(np.float32)
    # JUCE Colour::getBrightness()：0.299R + 0.587G + 0.114B，分量归一化到 [0,1]
    brightness = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0

    total_px = arr.shape[0] * arr.shape[1]
    transparent = int(np.count_nonzero(alpha < 128))
    use_alpha_mode = transparent > total_px // _ALPHA_MODE_DIVISOR

    if use_alpha_mode:
        return alpha >= 128.0

    if threshold is None:
        threshold = float(brightness.mean())
    return brightness < threshold


def load_image_pattern(path: str, threshold: Optional[float] = None) -> np.ndarray:
    """从图片文件生成图案。"""
    if not os.path.exists(path):
        raise PatternError(f"图片不存在：{path}")
    try:
        with Image.open(path) as img:
            img.load()          # GIF 等多帧格式：只要第一帧
            return binarize_image(img, threshold)
    except PatternError:
        raise
    except Exception as exc:    # noqa: BLE001 - 交给调用方统一汇报
        raise PatternError(f"无法读取图片 {os.path.basename(path)}：{exc}") from exc


def binarize_alpha(img: Image.Image, threshold: float = 128.0) -> np.ndarray:
    """把一张灰度/alpha 图当作"透明底 + 不透明图案"来二值化。

    文字渲染走这条路：背景 alpha=0，字形 alpha=255。
    """
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8) >= threshold


def _measure_text(content: str, font, stroke: int) -> tuple[int, int, int, int]:
    """量出文字（含描边）的包围盒 ``(x0, y0, x1, y1)``。

    注意返回的是**带偏移**的原始 bbox：绘制原点 (0,0) 到字形左上角之间通常还有
    一段间隙（em 框的留白），所以贴图时必须减掉这个偏移，否则字形会整体下移，
    底部被画布裁掉。
    """
    probe = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(probe)
    return draw.multiline_textbbox(
        (0, 0), content, font=font, stroke_width=stroke, align="left"
    )


def render_text_pattern(
    text: str,
    font_path: Optional[str] = None,
    weight: float = 0.25,
) -> np.ndarray:
    """把文字渲染成布尔图案（True = 字形）。

    两点设计：

    * **字号由内部决定** —— 图案最终会被拉伸铺满掩码网格，字号只影响渲染精度，
      对成品没有可见影响。所以先用小字号试量一次，再线性推算出能把文字撑到
      目标尺寸的字号，保证栅格化时有足够细节。
    * **粗细用描边模拟** —— 中文字体大多没有独立的粗体文件，靠 ``stroke_width``
      描边可以得到连续的粗细，比在 regular/bold 两个文件之间二选一灵活得多。

    渲染后裁剪到内容边界，否则四周空白会在栅格化时把文字压得很小。
    多行文字用 ``\\n`` 分隔。
    """
    content = text if text.strip() else " "
    font_path = font_path or find_default_font()
    weight = max(0.0, min(1.0, float(weight)))

    def _load(size: int):
        try:
            return (
                ImageFont.truetype(font_path, size)
                if font_path
                else ImageFont.load_default()
            )
        except Exception as exc:    # noqa: BLE001
            raise PatternError(f"无法加载字体 {font_path}：{exc}") from exc

    def _stroke_for(size: int) -> int:
        # 最粗时描边约为字号的 1/20。再往上笔画就会互相粘连 —— 中文的
        # 横竖本来就密，1/16 时"水印"这种字已经开始糊成一团了。
        return int(round(weight * size / 20.0))

    # 试量一次，线性推算目标字号（文字尺寸与字号近似成正比）
    probe_size = 100
    probe_font = _load(probe_size)
    try:
        probe_box = _measure_text(content, probe_font, _stroke_for(probe_size))
    except Exception as exc:    # noqa: BLE001
        raise PatternError(f"无法渲染文字：{exc}") from exc

    longest = max(1, probe_box[2] - probe_box[0], probe_box[3] - probe_box[1])
    font_size = int(round(probe_size * _TEXT_TARGET_PX / longest))
    font_size = max(_TEXT_MIN_SIZE, min(_TEXT_MAX_SIZE, font_size))

    font = _load(font_size)
    stroke = _stroke_for(font_size)

    try:
        box = _measure_text(content, font, stroke)
    except Exception as exc:    # noqa: BLE001
        raise PatternError(f"无法渲染文字：{exc}") from exc

    text_w = max(1, box[2] - box[0])
    text_h = max(1, box[3] - box[1])
    pad = max(2, font_size // 16)
    canvas = Image.new("L", (text_w + pad * 2, text_h + pad * 2), 0)
    draw = ImageDraw.Draw(canvas)
    # 减掉 bbox 偏移，字形才会正好落在 pad 边距内
    draw.multiline_text(
        (pad - box[0], pad - box[1]),
        content,
        font=font,
        fill=255,
        stroke_width=stroke,
        stroke_fill=255,
        align="left",
    )

    pattern = binarize_alpha(canvas)
    if not pattern.any():
        raise PatternError("文字渲染后为空，请检查内容或字体是否支持这些字符")
    return pattern


def build_pattern(spec: PatternSpec) -> np.ndarray:
    """按 :class:`PatternSpec` 生成布尔图案（True = 图案本体）。

    这是图案层的统一入口，UI 与批处理都只走这里。
    """
    if spec.source is PatternSource.IMAGE:
        if not spec.image_path:
            raise PatternError("未选择图片")
        pattern = load_image_pattern(spec.image_path, spec.binary_threshold)
    else:
        pattern = render_text_pattern(spec.text, spec.font_path, spec.weight)

    if spec.invert_pattern:
        pattern = ~pattern

    if not pattern.any():
        raise PatternError("图案为空：二值化后没有任何有效像素，请调整阈值或换张图")
    return pattern


