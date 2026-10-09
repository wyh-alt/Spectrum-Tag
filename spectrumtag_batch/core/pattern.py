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

import math
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

# 图案描边的宽度，占图案**较短边**的比例。
#
# 取短边而不是最长边：图案铺满印章框后，短边对应的是频率方向 —— 那一头
# 范围有限，也是描边最容易把图案吃掉的方位。按最长边定宽时，宽扁的单行
# 文字（"WATERMARK" 高只有 285 像素）会算出 41 像素的半径，比字高的一半
# 还多，字母之间的空隙全被填满，描边反而盖过本体，水印看起来成了负形。
_OUTLINE_WIDTH_RATIO = 0.03

# 算描边时把图案缩到的最长边。描边只是一圈轮廓，没有细节可言，而膨胀的
# 开销与像素数成正比 —— 大图案上缩着算能快一个数量级，误差不到一个采样格。
_OUTLINE_WORK_MAX = 512

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


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """二值图向外扩张 radius 个像素（方形结构元）。

    两个方向各扫一遍，每遍在前缀和上判断"窗口里有没有本体" —— 逐轮移位
    要跑 radius 遍全图，2048 宽的图案上描边有几十像素，那样太慢。
    """
    if radius <= 0:
        return mask
    grown = mask
    for axis in (0, 1):
        n = grown.shape[axis]
        if n <= 1:
            continue
        # c[i] = 前 i 个元素里本体的个数；窗口和 > 0 即窗口内存在本体
        prefix = np.concatenate(
            [
                np.zeros_like(grown.take([0], axis=axis), dtype=np.int32),
                np.cumsum(grown, axis=axis, dtype=np.int32),
            ],
            axis=axis,
        )
        lo = np.maximum(np.arange(n) - radius, 0)
        hi = np.minimum(np.arange(n) + radius + 1, n)
        grown = (prefix.take(hi, axis=axis) - prefix.take(lo, axis=axis)) > 0
    return grown


def _pool_or(mask: np.ndarray, step: int) -> np.ndarray:
    """把 ``step×step`` 的块压成一个像素：块内有本体就算 True。

    用"有没有"而不是平均值 —— 细笔画（一两个像素宽的横竖）取平均会直接
    消失，取"有没有"最多让它粗一点。
    """
    h, w = mask.shape
    pad_h, pad_w = (-h) % step, (-w) % step
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    hh, ww = mask.shape
    return mask.reshape(hh // step, step, ww // step, step).any(axis=(1, 3))


def build_outline(pattern: np.ndarray) -> np.ndarray:
    """图案的描边带 —— 本体向外扩一圈之后挖掉本体。

    这一圈就是"峰谷对"里的另一半：本体被削弱（或注入）时，它反向处理，
    于是想抹平图案就得连这圈一起动，而一动这圈图案又露出来。

    宽度取图案**较短边**的 :data:`_OUTLINE_WIDTH_RATIO`（见那里的说明）：图案会被
    拉伸铺满印章框，所以按比例定宽，描边跟着一起缩放，粗细在成品里是恒定的
    —— 始终等于印章框频率跨度的 3%。

    大图案先缩到 :data:`_OUTLINE_WORK_MAX` 再膨胀。膨胀的开销与像素数成正比，
    2048² 的全分辨率上要近百毫秒，而它挂在每次按键与滑杆拖动上 —— 描边只是
    一圈轮廓、本来就没有细节，缩着算的误差不超过一个采样格。
    """
    if not pattern.any():
        return pattern

    h, w = pattern.shape
    longest, shortest = max(h, w), min(h, w)
    radius = max(1, int(round(shortest * _OUTLINE_WIDTH_RATIO)))

    step = max(1, int(math.ceil(longest / _OUTLINE_WORK_MAX)))
    # 半径本来就不大的话，缩得越狠、量化误差占的比例越大（9 像素的描边
    # 缩 5 倍只剩不到 2 像素，回来就面目全非了）—— 宁可多算一点
    step = min(step, max(1, radius // 3))
    if step == 1:
        return _dilate(pattern, radius) & ~pattern

    # 缩到工作尺寸上膨胀，再原样放大回去。池化本身已经让图案胖了最多
    # step-1 个像素，算半径时把这一圈扣掉。
    small = _pool_or(pattern, step)
    small_radius = max(1, int(round(radius / step)) - 1)
    grown = _dilate(small, small_radius)
    grown = np.repeat(np.repeat(grown, step, axis=0), step, axis=1)[:h, :w]
    return grown & ~pattern


