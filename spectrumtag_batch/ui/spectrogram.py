"""频谱视图 —— 显示整段音频的时频图，并叠一个可拖动/缩放的印章框。

横轴 = 时间（秒），纵轴 = 频率（Hz，线性到 Nyquist）。印章框直接对应 DSP 的
两个落点参数：框的上下边 → 归一化频率范围，框的左右边 → 时间区间。

时频图用与 DSP 完全相同的窗与 FFT 尺寸计算，保证"看到的就是印上去的"。
长音频会先降采样到显示所需的列数，避免占用过多内存。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsRectItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel

from ..core.dsp import hann_window
from ..core.params import (
    DEFAULT_CUT_STRENGTH,
    DRAW_LEVEL_MAX_DBFS,
    DRAW_LEVEL_MIN_DBFS,
)

# 印水印最容易影响听感的频率范围，写成"段的列表"是为了将来能拆成几段，
# 目前就是连续的一段。
#
# 下界压到 20 Hz：低频承载着力度与厚度，而音乐能量本来就集中在这一头，
# 衰减的绝对衰减量最大，最容易听出"变薄"。
# 上界停在 5 kHz 而不是更高：等响曲线确实在 2~5 kHz 最灵敏，再往上虽然
# 人耳仍能听见，但音乐里那一段的能量已经很少，改动的绝对影响有限。
SENSITIVE_BANDS: tuple[tuple[float, float], ...] = (
    (20.0, 5000.0),
)

# 推荐印刷区域 —— 与上面的敏感频段正好互补：低频不能碰（听见的都是它），
# 超高频又留不住（转个码就被低通切掉），中间这一段才是水印的落脚点。
# 和敏感频段一样只在拖动印章框时露面，两块提示一起出现、一起收起。
RECOMMENDED_BAND_HZ = (7000.0, 13000.0)

# 停手后提示再停留多久（毫秒）。拖动过程中每次变化都会重新计时。
_SENSITIVE_HIDE_DELAY_MS = 900

# 刻度文字比正文小一号 —— 边距要贴边，字就得收着点
_TICK_FONT_POINT_SIZE = 8

# 就地编辑框的高度：够放三行，又不至于把底下的频谱挡掉太多
_INLINE_EDIT_HEIGHT = 64

# 循环影子的绘制上限。它**只影响预览** —— 真正印多少次由 LoopSpec 决定，
# "无限制"就是一直印到音频结束，跟这个数没有关系。
# 定得足够宽，正常素材碰不到；真碰到了也只是少画几个影子，不影响导出结果。
_MAX_LOOP_GHOSTS = 96

# 叠加图案的不透明度区间：强度 0 时淡到几乎只剩个轮廓（确实也没什么效果），
# 强度 1 时最实。第一个框和循环影子用同一套，看起来才是一回事。
_OVERLAY_ALPHA_MIN = 32
_OVERLAY_ALPHA_MAX = 215

# 衰减模式预览用的暗色 —— 那一片的能量会被抹掉，在图上就是塌下去
_OVERLAY_CUT_RGB = (14, 12, 20)

# 时频图的显示区间（dBFS）：0 = 满量程。用绝对参考而不是相对峰值，
# 否则噪声类素材会把整个画面刷成同一种亮色。
_DB_CEILING = 0.0
_DB_FLOOR = -90.0

# 显示用的最大列数 —— 超过这个宽度屏幕上也看不出来，纯属浪费
_MAX_DISPLAY_COLS = 2000

# 印章框的最小尺寸（占整体的比例），防止拖成一条线后 DSP 无法工作
_MIN_FREQ_SPAN = 0.005
_MIN_TIME_SPAN_SEC = 0.01

# 热力图色标：黑 → 深紫 → 橙红 → 黄（与 SpectrumTag 的观感一致）。
# 位置 + R/G/B，取值均为 0~1。
_HEAT_STOPS = np.array([
    [0.00, 0.00, 0.00, 0.00],
    [0.22, 0.10, 0.03, 0.26],
    [0.50, 0.55, 0.13, 0.22],
    [0.78, 0.93, 0.52, 0.15],
    [1.00, 1.00, 0.95, 0.68],
], dtype=np.float64)


def _build_heat_lut(size: int = 256) -> np.ndarray:
    """自己插值出 ``(size, 3)`` 的 uint8 色标表。

    不走 ``pyqtgraph.ColorMap`` —— 该 API 在不同版本间对颜色取值的约定不一致
    （0~1 还是 0~255），实测会退化成单色。自己插值最稳。
    """
    xs = np.linspace(0.0, 1.0, size)
    lut = np.empty((size, 3), dtype=np.uint8)
    for channel in range(3):
        values = np.interp(xs, _HEAT_STOPS[:, 0], _HEAT_STOPS[:, channel + 1])
        lut[:, channel] = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
    return lut


HEAT_LUT = _build_heat_lut()


def _time_tick_strings(values, scale, _spacing) -> list[str]:
    """时间刻度直接带单位（``50s``、``100s``）。

    单位写在刻度上，轴外侧就不必再挂一行"时间 (s)"，省下一条边距。
    """
    return [f"{value * scale:g}s" for value in values]


def _freq_tick_strings(values, scale, _spacing) -> list[str]:
    """频率刻度写成 ``5k`` / ``15k`` 这种紧凑形式。

    写全 "5kHz" 要多占十几像素，左侧边距就压不下去、也没法跟底边对齐。
    频率轴就贴在频谱图旁边，单位不言自明 —— 省掉它换来更宽的图形区。
    """
    labels: list[str] = []
    for value in values:
        hz = value * scale
        labels.append(f"{hz / 1000:g}k" if hz >= 1000 else f"{hz:g}")
    return labels


def _heat_color_at(norm: float) -> tuple[int, int, int]:
    """从时频图的热力色标上取某个位置的颜色（输入 0~1，输出 0~255）。"""
    norm = max(0.0, min(1.0, norm))
    stops = _HEAT_STOPS
    for index in range(len(stops) - 1):
        p0, r0, g0, b0 = stops[index]
        p1, r1, g1, b1 = stops[index + 1]
        if p0 <= norm <= p1:
            t = (norm - p0) / max(1e-9, p1 - p0)
            return (
                int(round((r0 + (r1 - r0) * t) * 255)),
                int(round((g0 + (g1 - g0) * t) * 255)),
                int(round((b0 + (b1 - b0) * t) * 255)),
            )
    return (0, 0, 0)


def _draw_level_norm(strength: float) -> float:
    """把注入强度换算到频谱显示范围里的归一化位置。

    注入强度本就对应一个 dBFS 目标电平（见 params.DRAW_LEVEL_*），
    再按显示区间映射一下，就是它在热力图上的亮度位置。
    """
    dbfs = DRAW_LEVEL_MIN_DBFS + (
        DRAW_LEVEL_MAX_DBFS - DRAW_LEVEL_MIN_DBFS
    ) * max(0.0, min(1.0, strength))
    return (dbfs - _DB_FLOOR) / max(1e-6, _DB_CEILING - _DB_FLOOR)


def _build_overlay_lut(is_cut: bool, strength: float) -> np.ndarray:
    """图案叠加层用的色标：0 → 全透明，1 → 半透明。

    两处都跟着参数走，为的是"所见即所得"：

    * **颜色按印法算** —— 衰减是把这一片抹掉，画成暗色；注入是往这儿加内容，
      颜色直接取它在热力色标上对应的位置。以前注入一律画成白色，预览看着
      挺亮，实际印出来却是暗橙色，容易把强度估错。
    * **不透明度按强度算** —— 强度本来就是个抽象的 0~1，让它直接决定预览的
      虚实，拖滑杆时图案随之变浓变淡，比盯着数字直观得多。
    """
    ratio = max(0.0, min(1.0, float(strength)))
    alpha = int(round(
        _OVERLAY_ALPHA_MIN + (_OVERLAY_ALPHA_MAX - _OVERLAY_ALPHA_MIN) * ratio
    ))
    red, green, blue = (
        _OVERLAY_CUT_RGB if is_cut else _heat_color_at(_draw_level_norm(ratio))
    )

    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[1:, 0] = red
    lut[1:, 1] = green
    lut[1:, 2] = blue
    lut[1:, 3] = alpha
    return lut


def _build_outline_lut(is_cut: bool, strength: float) -> np.ndarray:
    """图案描边用的色标 —— 取本体**相反**那一套。

    描边是峰谷对里的另一半：本体在衰减（画成暗块）时它就在凸起（画成亮色），
    本体在注入（亮色）时它就在衰减（暗块）。用相反的配色，一眼就能看出
    "这一圈跟图案本体反着来"。
    """
    return _build_overlay_lut(not is_cut, strength)


def db_to_rgb(db: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """把 dB 矩阵映射成 ``(H, W, 3)`` 的 RGB 图。

    直接产出 RGB 而不是依赖 ImageItem 的 lookupTable，省掉一整类版本兼容问题。
    """
    span = max(1e-6, float(vmax) - float(vmin))
    norm = np.clip((db - float(vmin)) / span, 0.0, 1.0)
    indices = (norm * (HEAT_LUT.shape[0] - 1)).astype(np.uint8)
    return HEAT_LUT[indices]


def compute_display_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    fft_size: int = 4096,
    max_cols: int = _MAX_DISPLAY_COLS,
) -> np.ndarray:
    """算出显示用的幅度谱，返回 ``(freq_bins, cols)`` 的 dB 值。

    多声道会先混成单声道 —— 显示用，不影响处理。
    """
    mono = samples if samples.ndim == 1 else samples.mean(axis=0)
    mono = np.asarray(mono, dtype=np.float32)

    n = int(fft_size)
    hop = n // 4
    window = hann_window(n)

    total = mono.size
    if total < n:
        # 太短就补零成一帧，至少让用户看到点东西
        mono = np.pad(mono, (0, n - total))
        total = n

    num_frames = max(1, (total - n) // hop + 1)
    # 列数超过屏幕宽度没意义，等间隔抽帧
    stride = max(1, int(np.ceil(num_frames / max_cols)))
    frame_ids = np.arange(0, num_frames, stride)

    offsets = (frame_ids * hop)[:, None] + np.arange(n, dtype=np.int64)[None, :]
    frames = mono[offsets] * window[None, :]

    spectrum = np.fft.rfft(frames, axis=-1)
    mags = np.abs(spectrum).T                     # (num_bins, cols)

    # 按**相干增益**归一，而不是按窗能量或峰值：
    # 这样"满量程正弦 = 0 dB"，不同素材的亮度可以直接对比。
    # 若改用峰值归一，噪声类素材会把整屏刷到顶格。
    coherent_gain = max(1e-12, float(np.sum(window)) * 0.5)
    mags /= coherent_gain

    db = 20.0 * np.log10(np.maximum(mags, 1e-9))
    return db.astype(np.float32)


class _InlineTextEdit(QTextEdit):
    """就地编辑用的输入框：回车换行，Esc 或点别处结束。

    用多行控件是为了跟右侧输入框一致 —— 图案本来就支持 `\\n`，
    在框里编辑时要是打不出换行，两边就对不上了。
    """

    finished = pyqtSignal()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.finished.emit()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        self.finished.emit()


class _SpectrogramViewBox(pg.ViewBox):
    """按缩放级别决定能不能拖动平移。

    * **全览时**（视图已经装下整段音频）拖动不平移 —— 那是误操作，只会让人
      找不着北。
    * **放大后**拖动可以平移，用来查看细节。

    缩放下限由 :meth:`set_full_extent` 里的 ``setLimits`` 卡住：视图范围不能
    超出数据本身，所以最多只能缩到"整段铺满窗口"，不会出现频谱越缩越小、
    四周全是空白的情况。滚轮缩放本身以指针位置为中心（pyqtgraph 默认行为）。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._full_x = 0.0
        self._full_y = 0.0

    def set_full_extent(self, x_span: float, y_span: float) -> None:
        """记录数据全长，并禁止视图超出这个范围。"""
        self._full_x = float(x_span)
        self._full_y = float(y_span)
        self.setLimits(xMin=0.0, xMax=self._full_x, yMin=0.0, yMax=self._full_y)

    def is_zoomed(self) -> bool:
        """当前是否处于放大状态（任一轴的可见范围小于全长）。"""
        if self._full_x <= 0.0 or self._full_y <= 0.0:
            return False
        (x0, x1), (y0, y1) = self.viewRange()
        return (x1 - x0) < self._full_x * 0.995 or (y1 - y0) < self._full_y * 0.995

    def mouseDragEvent(self, ev, axis=None) -> None:  # noqa: N802 - pyqtgraph 命名
        if self.is_zoomed():
            super().mouseDragEvent(ev, axis)
        else:
            ev.ignore()

    def wheelEvent(self, ev, axis=None) -> None:  # noqa: N802 - pyqtgraph 命名
        # 还没载入音频（没设过 extent）时不给缩放：空频谱缩放没有意义，
        # 而且会把视图留在某个奇怪的范围上，等下真载入文件还得先复位。
        if self._full_x <= 0.0 or self._full_y <= 0.0:
            ev.ignore()
            return
        super().wheelEvent(ev, axis)


class _BandOverlay:
    """一条横贯时间轴的提示带：半透明填充 + 上下两道边界线。

    刻意**不画左右两条边** —— 带子横跨整个时间轴，左右本来就没有边界，
    画出来只会是两端各一道扎眼的亮竖条。所以填充归矩形项、边界归一条
    **开放**的折线：画笔沿着它走，到两端就停，不会拐下去。
    """

    def __init__(
        self,
        plot: pg.PlotWidget,
        fill: QColor,
        edge: QColor,
        z: float,
    ) -> None:
        self._fill = QGraphicsRectItem()
        self._fill.setBrush(QBrush(fill))
        self._fill.setPen(QPen(Qt.PenStyle.NoPen))   # 只留填充，边框交给下面那条折线
        self._fill.setZValue(z)
        self._fill.setVisible(False)
        plot.addItem(self._fill)

        pen = QPen(edge)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)
        self._edges = QGraphicsPathItem()
        self._edges.setPen(pen)
        self._edges.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._edges.setZValue(z)
        self._edges.setVisible(False)
        plot.addItem(self._edges)

    def set_rect(self, rect: QRectF) -> None:
        """摆放带子。空矩形（宽或高为 0）表示这一段用不上。"""
        self._fill.setRect(rect)
        path = QPainterPath()
        if rect.width() > 0.0 and rect.height() > 0.0:
            for y in (rect.top(), rect.bottom()):
                path.moveTo(rect.left(), y)
                path.lineTo(rect.right(), y)
        self._edges.setPath(path)

    def set_visible(self, visible: bool) -> None:
        self._fill.setVisible(visible)
        self._edges.setVisible(visible)

    def is_visible(self) -> bool:
        return self._fill.isVisible()

    def rect(self) -> QRectF:
        return self._fill.rect()

    def z_value(self) -> float:
        return self._fill.zValue()


class SpectrogramView(QWidget):
    """时频图 + 印章框 + 图案叠加预览。"""

    regionChanged = pyqtSignal()
    filesDropped = pyqtSignal(list)       # 拖入的本地文件路径
    patternDoubleClicked = pyqtSignal()   # 双击印章框，想改水印内容
    inlineTextChanged = pyqtSignal(str)   # 框内就地编辑时，每敲一个字都会发
    viewResized = pyqtSignal()            # 绘图区尺寸变了（外部可能要重算时长）

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._sample_rate = 44100
        self._duration = 1.0
        self._max_freq = 22050.0
        self._has_audio = False
        self._updating = False          # 防止程序化改框触发回调风暴
        self._pattern: Optional[np.ndarray] = None
        self._outline: Optional[np.ndarray] = None

        # 循环印刷的影子框：只显示位置与图案，不参与交互。
        # 本体和描边各铺一套 —— 它们本来就是同一枚印章的两层。
        self._loop_items: list[pg.ImageItem] = []
        self._outline_loop_items: list[pg.ImageItem] = []
        self._loop_enabled = False
        self._loop_interval = 5.0

        # 就地编辑用的浮层输入框（没有在编辑时为 None）
        self._inline_edit: Optional[_InlineTextEdit] = None

        self._plot = pg.PlotWidget(background="#141418", viewBox=_SpectrogramViewBox())
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        # 单位直接写在刻度上（5k、50s），所以不挂轴标签。
        # 刻度线和文字都贴紧轴线，两侧边距才压得下去。
        self._tick_font = QFont(self.font())
        self._tick_font.setPointSize(_TICK_FONT_POINT_SIZE)
        for side, formatter in (
            ("bottom", _time_tick_strings),
            ("left", _freq_tick_strings),
        ):
            axis = self._plot.getAxis(side)
            axis.setLabel("")
            axis.enableAutoSIPrefix(False)      # 不然它会给刻度自动加 k / m 前缀
            axis.setStyle(tickFont=self._tick_font, tickLength=-4, tickTextOffset=0)
            axis.tickStrings = formatter
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        # 未载入音频时给一个像样的初始视野，否则坐标轴会显示 pyqtgraph
        # 的 ±500 默认值（还带 mHz / ms 这种莫名其妙的前缀）
        self._plot.setXRange(0.0, 1.0, padding=0.0)
        self._plot.setYRange(0.0, 22050.0, padding=0.0)

        self._image = pg.ImageItem()
        self._plot.addItem(self._image)

        # 图案叠加层：把二值图案半透明地画在框内，所见即所得
        # 初值只是占位，主窗口建好后会调 set_preview_style 按实际印法刷新
        self._overlay_lut = _build_overlay_lut(is_cut=True, strength=DEFAULT_CUT_STRENGTH)
        self._pattern_item = self._add_overlay_item(self._overlay_lut)

        # 描边叠加层：与本体同样的矩形，但配色相反（它朝反方向处理）
        self._outline_lut = _build_outline_lut(is_cut=True, strength=DEFAULT_CUT_STRENGTH)
        self._outline_item = self._add_overlay_item(self._outline_lut)

        self._build_recommended_band()
        self._build_sensitive_band()

        # 印章框：拖动框体平移，八个手柄调整尺寸。
        self._roi = pg.RectROI(
            [0.0, 0.0], [1.0, 1.0],
            rotatable=False,
            resizable=False,          # 默认手柄由 _install_handles 统一重建
            pen=pg.mkPen("#FFD24A", width=2),
            hoverPen=pg.mkPen("#FFF08A", width=2),
            handlePen=pg.mkPen("#FFD24A", width=2),
            handleHoverPen=pg.mkPen("#FFF08A", width=3),
        )
        self._roi.setZValue(10)
        self._install_handles()
        self._plot.addItem(self._roi)
        self._roi.setVisible(False)      # 载入音频之前不显示，也就无从拖动
        self._roi.sigRegionChanged.connect(self._on_region_changed)

        self._hint = BodyLabel("拖入音频文件后，这里会显示整段时频图", self)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 1)

        self._hint.setParent(self)
        self._hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 拖放：pyqtgraph 内部的 GraphicsView 会自己吃掉拖放事件，
        # 所以这里显式接管，再通过 filesDropped 交给上层处理。
        self.setAcceptDrops(True)
        self._plot.setAcceptDrops(False)
        viewport = self._plot.viewport()
        if viewport is not None:
            viewport.setAcceptDrops(False)
            # 双击不走 QGraphicsItem.mouseDoubleClickEvent —— pyqtgraph 的 ROI
            # 用的是自己那套 mouseClickEvent / mouseDragEvent 转发，双击根本
            # 到不了 item（实测调用 0 次）。所以在视口上直接拦。
            viewport.installEventFilter(self)

    def _add_overlay_item(self, lut: np.ndarray) -> pg.ImageItem:
        """新建一层图案叠层（图案本体与描边各一层）。"""
        item = pg.ImageItem()
        item.setZValue(5)
        item.setLookupTable(lut)
        item.setVisible(False)
        self._plot.addItem(item)
        return item

    def _build_recommended_band(self) -> None:
        """推荐印刷区域：一条横贯时间轴的淡绿色带 + 说明文字。

        和敏感频段一样，只在调整印章框时亮出来（见 :meth:`_show_guide_bands`）——
        常驻会一直压在频谱上，看久了反而碍事。

        层级刻意排过：色带压在频谱之上、**图案叠层之下** —— 它横贯整个 7~13 kHz，
        而水印通常就印在这一段，填充色绝不能把图案染绿。文字反过来提到最上层，
        否则会被图案盖掉。
        """
        self._recommended_band = _BandOverlay(
            self._plot,
            fill=QColor(96, 220, 140, 34),
            edge=QColor(96, 220, 140, 170),
            z=1,
        )

        low, high = RECOMMENDED_BAND_HZ
        self._recommended_text = pg.TextItem(
            anchor=(0.5, 0.0),      # 以顶部中点为锚，文字从区域上沿往下展开
            fill=pg.mkBrush(12, 32, 20, 190),
            border=pg.mkPen(96, 220, 140, 130),
        )
        self._recommended_text.setHtml(
            '<div style="text-align:center; line-height:150%;">'
            '<span style="color:#7CE8A8; font-size:13px; font-weight:600;">'
            f'推荐印刷区域 {low / 1000:g}–{high / 1000:g} kHz'
            '</span><br>'
            '<span style="color:#C8F0D8; font-size:11px;">'
            '该频范围内印制水印相对安全，请合理控制图案尺寸！'
            '</span>'
            '</div>'
        )
        self._recommended_text.setZValue(7)
        self._recommended_text.setVisible(False)
        self._plot.addItem(self._recommended_text)

    def _layout_recommended_band(self) -> bool:
        """按当前音频的实际范围摆放推荐区域；这一段摆不下时返回 False。

        采样率不够高时上沿会被 Nyquist 截断；整段都落在 Nyquist 之外
        （比如 22.05 kHz 素材只有 11 kHz 可用）时就没有可推荐的频段了。
        """
        low, high = RECOMMENDED_BAND_HZ
        top = min(high, self._max_freq)
        bottom = min(low, top)
        if not self._has_audio or top - bottom <= 0.0:
            self._recommended_band.set_rect(QRectF())
            return False
        self._recommended_band.set_rect(
            QRectF(0.0, bottom, self._duration, top - bottom)
        )
        self._recommended_text.setPos(self._duration * 0.5, top)
        return True

    def recommended_band_rect(self) -> QRectF:
        """推荐区域当前的矩形（供自检使用）。"""
        return self._recommended_band.rect()

    def recommended_band_visible(self) -> bool:
        """推荐区域当前是否可见（供自检使用）。"""
        return self._recommended_band.is_visible()

    def _build_sensitive_band(self) -> None:
        """敏感频段提示：几片横贯时间轴的半透明黄带 + 中央说明。

        默认隐藏，只在调整印章框时亮出来。放置层级刻意排过：色带在频谱之上、
        图案叠层之上（警告要看得见），而说明文字再高一层（不被色带压住）。

        和推荐区域共用 :class:`_BandOverlay` —— 两者只是配色、层级和位置不同。
        """
        # zValue 排在图案叠层（5）之上：否则色带会被水印图案从中间切断，
        # 看起来像两块无关的色块。填充只有 ~19% 不透明度，不会盖住图案本身。
        self._sensitive_bands: list[_BandOverlay] = [
            _BandOverlay(
                self._plot,
                fill=QColor(255, 214, 90, 48),
                edge=QColor(255, 214, 90, 170),
                z=6,
            )
            for _ in SENSITIVE_BANDS
        ]

        # 频谱底色本身是暖色，纯黄文字对比度不够，垫一层深色底
        self._sensitive_text = pg.TextItem(
            anchor=(0.5, 0.5),
            fill=pg.mkBrush(18, 16, 10, 170),
            border=pg.mkPen(255, 214, 90, 130),
        )
        self._sensitive_text.setHtml(
            '<div style="text-align:center; line-height:150%;">'
            '<span style="color:#FFE066; font-size:15px; font-weight:600;">'
            '敏感频段区域</span><br>'
            '<span style="color:#F2D488; font-size:11px;">'
            '该范围内印制水印，可能会影响音频听感！</span>'
            '</div>'
        )
        self._sensitive_text.setZValue(7)
        self._sensitive_text.setVisible(False)
        self._plot.addItem(self._sensitive_text)

        # 用去抖定时器判断"停手"：pyqtgraph 的 sigRegionChangeFinished 每次
        # 变化都会发（框体平移更是连 started 都没有），没法当作拖动结束的信号。
        self._sensitive_timer = QTimer(self)
        self._sensitive_timer.setSingleShot(True)
        self._sensitive_timer.setInterval(_SENSITIVE_HIDE_DELAY_MS)
        self._sensitive_timer.timeout.connect(self._hide_guide_bands)

    def _sync_axis_margins(self) -> None:
        """把两侧边距设成同一个值：刚好放得下最宽的刻度。

        不用 ``max(left.width(), bottom.height())`` 那种"先让它们自适应再取大者"
        的做法 —— ``setWidth()`` 会把轴的 ``fixedWidth`` 锁死，之后它再也不按内容
        重新量尺寸，刻度一变宽就会被截断。这里直接算出需要的值，可预期也更稳。
        """
        metrics = QFontMetrics(self._tick_font)
        # 最宽的刻度文字（"20k"）加上一点留白
        margin = max(metrics.height(), metrics.horizontalAdvance("20k")) + 4

        self._plot.getAxis("left").setWidth(margin)
        self._plot.getAxis("bottom").setHeight(margin)

    def _queue_axis_margin_sync(self) -> None:
        """等这一轮布局跑完再对齐 —— 轴的宽高要重绘后才更新到位。"""
        QTimer.singleShot(0, self._sync_axis_margins)

    def _layout_sensitive_band(self) -> None:
        """按当前音频的实际范围摆放提示带。

        采样率不够高时，靠上的频段可能整个落在 Nyquist 之外，那一片就没什么
        可提醒的，直接跳过（矩形置空）。参考文字跟着最后一片可见的色带居中。
        """
        anchor: Optional[tuple[float, float]] = None
        for band, (low, high) in zip(self._sensitive_bands, SENSITIVE_BANDS):
            top = min(high, self._max_freq)
            bottom = min(low, top)
            if top - bottom <= 0.0:
                band.set_rect(QRectF())
                continue
            band.set_rect(QRectF(0.0, bottom, self._duration, top - bottom))
            anchor = (bottom, top)

        if anchor is None:
            self._sensitive_text.setVisible(False)
            return
        self._sensitive_text.setPos(
            self._duration * 0.5, anchor[0] + (anchor[1] - anchor[0]) * 0.5
        )

    def _show_guide_bands(self) -> None:
        """亮出推荐区与敏感区，并重新计时。

        两块提示是同一个动作的两面 —— "该印哪儿"和"别印哪儿" —— 所以一起
        出现、一起收起。拖动期间这个函数会被反复调用，提示因此不会中途消失。
        """
        if not self._has_audio:
            return
        self._layout_sensitive_band()
        for band in self._sensitive_bands:
            band.set_visible(True)
        self._sensitive_text.setVisible(True)

        if self._layout_recommended_band():
            self._recommended_band.set_visible(True)
            self._recommended_text.setVisible(True)

        self._sensitive_timer.start()

    def _hide_guide_bands(self) -> None:
        for band in self._sensitive_bands:
            band.set_visible(False)
        self._recommended_band.set_visible(False)
        self._sensitive_text.setVisible(False)
        self._recommended_text.setVisible(False)

    def sensitive_band_visible(self) -> bool:
        """敏感提示当前是否可见（供自检使用）。"""
        return bool(self._sensitive_bands) and self._sensitive_bands[0].is_visible()

    def sensitive_band_rects(self) -> list[QRectF]:
        """各片敏感提示带当前的矩形（供自检使用）。"""
        return [band.rect() for band in self._sensitive_bands]

    def guide_bands_visible(self) -> bool:
        """两块提示当前是否都亮着（供自检使用）。"""
        return self.sensitive_band_visible() and self.recommended_band_visible()

    def _install_handles(self) -> None:
        """装上八个缩放手柄。

        * **四角** —— 等比缩放：``lockAspect=True`` 让宽高按同一比例变化，
          图案不会被拉变形。锚点取对角，所以拖哪个角，对面就固定不动。
        * **四边** —— 单向缩放：``addScaleHandle`` 的规则是"手柄位置与锚点在
          某一轴上相同时，该轴禁用缩放"。边中点手柄的 y（或 x）与锚点一致，
          于是自然只改变宽度或高度。

        RectROI 在 ``resizable=False`` 时不自建手柄，这里从零装一套，避免默认
        那个右下角手柄混进来。
        """
        for handle in list(self._roi.handles):
            self._roi.removeHandle(handle["item"])

        # 四角：等比缩放
        self._roi.addScaleHandle([0.0, 0.0], [1.0, 1.0], lockAspect=True)
        self._roi.addScaleHandle([1.0, 0.0], [0.0, 1.0], lockAspect=True)
        self._roi.addScaleHandle([0.0, 1.0], [1.0, 0.0], lockAspect=True)
        self._roi.addScaleHandle([1.0, 1.0], [0.0, 0.0], lockAspect=True)

        # 四边：分别只改变宽度或高度
        self._roi.addScaleHandle([0.5, 0.0], [0.5, 1.0])   # 下边 → 高度
        self._roi.addScaleHandle([0.5, 1.0], [0.5, 0.0])   # 上边 → 高度
        self._roi.addScaleHandle([0.0, 0.5], [1.0, 0.5])   # 左边 → 宽度
        self._roi.addScaleHandle([1.0, 0.5], [0.0, 0.5])   # 右边 → 宽度

    # ---------------------------------------------------------------- 数据

    def set_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        """载入音频并重画时频图。"""
        self._sample_rate = int(sample_rate)
        self._max_freq = self._sample_rate / 2.0
        self._duration = samples.shape[-1] / float(sample_rate) if samples.size else 1.0
        self._has_audio = samples.size > 0

        db = compute_display_spectrogram(samples, self._sample_rate)

        # 素材整体偏轻时把上限压下来一点，避免画面过暗；但绝不超过满量程
        peak = float(np.percentile(db, 99.9)) if db.size else _DB_CEILING
        ceiling = min(_DB_CEILING, max(peak + 12.0, _DB_CEILING - 24.0))
        rgb = db_to_rgb(db, ceiling + _DB_FLOOR, ceiling)
        self._image.setImage(rgb, axisOrder="row-major", autoLevels=False)
        # 图像铺满 [0, duration] × [0, max_freq]
        self._image.setRect(QRectF(0.0, 0.0, self._duration, self._max_freq))

        # 全览：铺满窗口且不超出数据范围，这样"缩到最小"就停在这里
        view_box = self._plot.getViewBox()
        view_box.set_full_extent(self._duration, self._max_freq)
        view_box.setRange(
            xRange=(0.0, self._duration),
            yRange=(0.0, self._max_freq),
            padding=0.0,
        )
        self._hint.setVisible(False)
        self._layout_sensitive_band()
        self._layout_recommended_band()
        self._queue_axis_margin_sync()   # 刻度文字随音频时长变化，边距要重新对齐

        # 有音频了才让印章框出现
        self._roi.setVisible(True)
        self._refresh_overlays()

    def clear(self) -> None:
        self._image.clear()
        self._has_audio = False
        self._pattern_item.setVisible(False)
        self._outline_item.setVisible(False)
        self._clear_loop_ghosts()
        self._roi.setVisible(False)
        self._sensitive_timer.stop()
        self._hide_guide_bands()
        self._layout_recommended_band()   # 把矩形也清空，下次载入时不会闪一下旧的
        # 抹掉 extent，滚轮缩放会随之失效
        self._plot.getViewBox().set_full_extent(0.0, 0.0)
        self._hint.setVisible(True)

    def set_pattern(
        self,
        pattern: Optional[np.ndarray],
        outline: Optional[np.ndarray] = None,
    ) -> None:
        """设置框内叠加显示的图案（布尔数组，True = 图案本体）。

        显示时纵向翻转 —— 图片顶部对应高频，与掩码栅格化的方向一致。
        ``outline`` 是图案外沿那一圈（见 ``pattern.build_outline``），
        与本体铺在同一个矩形里，只是配色相反。
        """
        if pattern is None or pattern.size == 0:
            self._pattern = None
            self._outline = None
            self._pattern_item.setVisible(False)
            self._outline_item.setVisible(False)
            self._clear_loop_ghosts()
            return

        self._pattern = pattern
        self._outline = outline if (outline is not None and outline.size) else None
        self._pattern_item.setImage(
            self._flipped(pattern),
            axisOrder="row-major", autoLevels=False, levels=(0, 1),
        )
        if self._outline is not None:
            self._outline_item.setImage(
                self._flipped(self._outline),
                axisOrder="row-major", autoLevels=False, levels=(0, 1),
            )
        # 影子的贴图得跟着换，撤掉重建最省事（图案变化本来就不频繁）
        self._clear_loop_ghosts()
        self._update_pattern_rect()      # 可见性由它统一决定
        self._refresh_loop_ghosts()

    def pattern_shape(self) -> Optional[tuple[int, int]]:
        """当前图案的 ``(高, 宽)``；还没有图案时 None。

        「保持原始水印比例」要靠它拿到图案的长宽比。
        """
        if self._pattern is None or self._pattern.size == 0:
            return None
        return int(self._pattern.shape[0]), int(self._pattern.shape[1])

    def viewport_aspect(self) -> float:
        """频谱绘图区的高宽比（高 / 宽）。

        「保持原始水印比例」用它把频率跨度折成时间跨度：图案在频谱图上
        看起来不变形，靠的就是这个比例。
        """
        rect = self._plot.getViewBox().geometry()
        width, height = float(rect.width()), float(rect.height())
        if width <= 1.0 or height <= 1.0:      # 还没布局完，退回整个视口
            viewport = self._plot.viewport()
            width, height = float(viewport.width()), float(viewport.height())
        return max(1e-3, height / max(1.0, width))

    def _update_pattern_rect(self) -> None:
        """让图案叠层跟着印章框走。

        没有音频时不显示：此时印章框还没有有效的尺寸，叠层会被拉到默认的
        0~1 范围上，在空频谱里铺满一整屏。

        描边层与本体同频段、同时间，所以矩形完全一致 —— 它俩本来就是同一枚
        印章的两层，只是处理方向相反。
        """
        visible = self._pattern is not None and self._has_audio
        self._pattern_item.setVisible(visible)
        if not visible:
            self._outline_item.setVisible(False)
            return
        pos = self._roi.pos()
        size = self._roi.size()
        rect = QRectF(
            float(pos.x()), float(pos.y()), float(size.x()), float(size.y())
        )
        self._pattern_item.setRect(rect)
        self._outline_item.setVisible(self._outline is not None)
        if self._outline is not None:
            self._outline_item.setRect(rect)

    # ------------------------------------------------------------ 循环预览

    def set_preview_style(self, is_cut: bool, strength: float) -> None:
        """按印法与强度刷新预览配色。

        第一个框和循环影子共用同一套色标 —— 它们本来就是同一个印章的多次印刷，
        深浅不一样只会让人以为哪里出了岔子。"哪个能调"靠边框和手柄区分就够了。

        描边层取的是相反那一套：本体画成暗块时它亮，本体亮时它暗。
        """
        self._overlay_lut = _build_overlay_lut(bool(is_cut), float(strength))
        self._outline_lut = _build_outline_lut(bool(is_cut), float(strength))
        self._pattern_item.setLookupTable(self._overlay_lut)
        self._outline_item.setLookupTable(self._outline_lut)
        for item in self._loop_items:
            item.setLookupTable(self._overlay_lut)
        for item in self._outline_loop_items:
            item.setLookupTable(self._outline_lut)

    def set_loop_preview(self, enabled: bool, interval_sec: float) -> None:
        """告诉视图当前的循环设置，它会据此铺出一串只读的影子框。"""
        self._loop_enabled = bool(enabled)
        self._loop_interval = max(0.0, float(interval_sec))
        self._refresh_loop_ghosts()

    def loop_ghost_count(self) -> int:
        """当前画了几个循环影子（供自检使用）。"""
        return len(self._loop_items)

    def outline_ghost_count(self) -> int:
        """描边层画了几个循环影子（供自检使用）。"""
        return len(self._outline_loop_items)

    def outline_visible(self) -> bool:
        """描边叠层是否可见（供自检使用）。"""
        return self._outline_item.isVisible()

    def _loop_positions(self) -> list[float]:
        """算出后续各次印刷的起始时间（秒）。

        位置完全由第一个框和间隔推出来 —— 界面上只让用户调第一个框，
        "调一个等于调全部"的关系才清楚。步进是「框宽 + 间隔」：间隔量的是
        上一个印章的右边框到下一个印章的左边框，与 DSP 层保持一致。
        """
        if not (self._has_audio and self._loop_enabled):
            return []
        if self._loop_interval <= 0.0:
            return []
        width = float(self._roi.size().x())
        if width <= 0.0:
            return []

        step = width + self._loop_interval
        positions: list[float] = []
        x = float(self._roi.pos().x()) + step
        while x < self._duration and len(positions) < _MAX_LOOP_GHOSTS:
            positions.append(x)
            x += step
        return positions

    @staticmethod
    def _flipped(pattern: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """行 0 = 图片顶部 = 最高频，而图像数组行 0 在最下方，故翻转。"""
        if pattern is None or not pattern.size:
            return None
        return np.ascontiguousarray(pattern[::-1], dtype=np.uint8)

    def _ghost_image_data(self) -> Optional[np.ndarray]:
        return self._flipped(self._pattern)

    def _clear_loop_ghosts(self) -> None:
        for items in (self._loop_items, self._outline_loop_items):
            for item in items:
                self._plot.removeItem(item)
            items.clear()

    def _sync_ghost_row(
        self,
        items: list[pg.ImageItem],
        positions: list[float],
        data: np.ndarray,
        lut: np.ndarray,
        rect: QRectF,
    ) -> None:
        """把一排影子框铺到给定位置上（矩形由调用方算好，只挪 x）。

        已有条目复用，只更新矩形，避免每次拖动都重传一次纹理。
        """
        while len(items) > len(positions):
            self._plot.removeItem(items.pop())

        while len(items) < len(positions):
            item = pg.ImageItem()
            # 与对应的印章共用色标，不另外调 opacity —— 深浅该由强度决定，
            # 不该因为"这是循环的影子"就无端淡一半
            item.setLookupTable(lut)
            item.setZValue(5)
            item.setImage(data, axisOrder="row-major", autoLevels=False, levels=(0, 1))
            self._plot.addItem(item)
            items.append(item)

        for item, x in zip(items, positions):
            item.setLookupTable(lut)
            item.setRect(QRectF(x, rect.y(), rect.width(), rect.height()))

    def _refresh_loop_ghosts(self) -> None:
        """按循环间隔铺一串影子框 —— 图案本体和描边各铺一套。

        影子只反映位置和图案，拖不动 —— 它们跟着第一个框和间隔走。
        """
        positions = self._loop_positions()
        data = self._ghost_image_data()

        if data is None or not positions:
            self._clear_loop_ghosts()
            return

        pos = self._roi.pos()
        size = self._roi.size()
        rect = QRectF(0.0, float(pos.y()), float(size.x()), float(size.y()))
        self._sync_ghost_row(
            self._loop_items, positions, data, self._overlay_lut, rect,
        )

        outline_data = self._flipped(self._outline)
        self._sync_ghost_row(
            self._outline_loop_items,
            positions if outline_data is not None else [],
            outline_data if outline_data is not None else data,
            self._outline_lut,
            rect,
        )

    # ---------------------------------------------------------------- 印章框

    def set_freq_range(self, low_norm: float, high_norm: float) -> None:
        """按归一化频率设置框的上下边。"""
        if not self._has_audio:
            return
        y = low_norm * self._max_freq
        h = max(_MIN_FREQ_SPAN * self._max_freq, (high_norm - low_norm) * self._max_freq)
        self._apply_region(self._roi.pos().x(), y, self._roi.size().x(), h)

    def set_time_range(self, start_sec: float, duration_sec: float) -> None:
        """按秒设置框的左右边。"""
        self._apply_region(start_sec, self._roi.pos().y(), duration_sec, self._roi.size().y())

    def freq_range_norm(self) -> tuple[float, float]:
        """当前框对应的归一化频率范围。"""
        y = float(self._roi.pos().y())
        h = float(self._roi.size().y())
        low = max(0.0, min(1.0, y / self._max_freq))
        high = max(0.0, min(1.0, (y + h) / self._max_freq))
        if high - low < _MIN_FREQ_SPAN:
            high = min(1.0, low + _MIN_FREQ_SPAN)
        return low, high

    def time_range_sec(self) -> tuple[float, float]:
        """当前框对应的时间区间（秒）。"""
        x = max(0.0, float(self._roi.pos().x()))
        w = max(_MIN_TIME_SPAN_SEC, float(self._roi.size().x()))
        return x, w

    def duration_sec(self) -> float:
        return self._duration

    def has_audio(self) -> bool:
        return self._has_audio

    def _sync_handles(self) -> None:
        """把八个手柄重新摆到 ``pos × size`` 上。

        pyqtgraph 是在 ``stateChanged()`` 里做这件事的，而且**只在它自己检测到
        state 有变化时才执行**。程序化设几何、clamp 修正、以及若干拖动路径都可能
        绕过那一步，结果就是手柄停在旧位置、看起来从边框上掉了下来。

        这里不依赖它的判断，直接同步一次 —— 八个手柄的坐标写入，开销可以忽略。
        """
        size = self._roi.state["size"]
        for handle in self._roi.handles:
            handle["item"].setPos(handle["pos"] * size)

    def _refresh_overlays(self) -> None:
        """几何变化后的统一收尾：手柄、图案叠层、循环影子都跟上框。"""
        self._sync_handles()
        self._update_pattern_rect()
        self._refresh_loop_ghosts()
        self._position_inline_edit()

    def _set_roi_geometry(self, x: float, y: float, w: float, h: float) -> None:
        """程序化设置印章框的几何。

        仍然走默认的 ``update=True``，让 pyqtgraph 自己那一套也正常运转；
        回调由 ``_updating`` 标志挡住。收尾再统一同步一次手柄。
        """
        was_updating = self._updating
        self._updating = True
        try:
            self._roi.setPos([x, y])
            self._roi.setSize([max(w, 1e-6), max(h, 1e-6)])
        finally:
            self._updating = was_updating
        self._refresh_overlays()

    def _apply_region(self, x: float, y: float, w: float, h: float) -> None:
        self._set_roi_geometry(x, y, w, h)
        self._clamp_region()
        self._refresh_overlays()

    def _on_region_changed(self) -> None:
        # 拖动、程序化设值都汇到这里：先把手柄与图案叠层对齐，再处理业务
        self._refresh_overlays()
        if self._updating:
            # 程序化设几何（含载入文件时的初始化）不弹敏感频段提示 ——
            # 那个提示是给"手动拖框"这个动作配的。
            return
        self._show_guide_bands()
        self._clamp_region()
        self.regionChanged.emit()

    def _clamp_region(self) -> None:
        """把框锁在音频范围内，并保证不退化成一个点。

        频率方向锁在 ``[0, max_freq]``、时间方向锁在 ``[0, duration]`` ——
        拖到边缘就停住，不会拖到频谱外面去。
        """
        if not self._has_audio:
            return

        pos = self._roi.pos()
        size = self._roi.size()
        x, y = float(pos.x()), float(pos.y())
        w, h = float(size.x()), float(size.y())

        min_h = _MIN_FREQ_SPAN * self._max_freq
        min_w = _MIN_TIME_SPAN_SEC

        # 先夹尺寸再夹位置，顺序反了会让框贴不住边
        h = max(min_h, min(h, self._max_freq))
        y = max(0.0, min(y, self._max_freq - h))

        w = max(min_w, min(w, self._duration))
        x = max(0.0, min(x, self._duration - w))

        changed = (
            abs(x - float(pos.x())) > 1e-9
            or abs(y - float(pos.y())) > 1e-9
            or abs(w - float(size.x())) > 1e-9
            or abs(h - float(size.y())) > 1e-9
        )
        if not changed:
            return

        self._set_roi_geometry(x, y, w, h)
        # clamp 后需要通知外部，否则界面与数据会悄悄不一致
        if not self._updating:
            self.regionChanged.emit()

    # ---------------------------------------------------------------- 布局

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._hint.setGeometry(self.rect())
        self._position_inline_edit()
        self._queue_axis_margin_sync()
        # 视口高宽比变了，「保持原始水印比例」算出来的时长要跟着重算
        self.viewResized.emit()

    # ------------------------------------------------------------ 就地编辑

    def begin_inline_edit(self, initial_text: str) -> None:
        """在印章框上浮出一个输入框，就地改水印文字。

        它是个盖在绘图区之上的普通子控件 —— 比往场景里塞
        QGraphicsProxyWidget 简单得多，也不用担心随视图缩放被拉伸变形。
        每敲一个字都会发 inlineTextChanged，图案实时重绘。
        """
        self.end_inline_edit()
        if not self._has_audio:
            return

        editor = _InlineTextEdit(self)
        editor.setPlainText(initial_text)
        editor.setPlaceholderText("输入水印文字")
        editor.setStyleSheet(
            "QTextEdit {"
            " background: rgba(20, 20, 26, 235);"
            " color: #FFE066;"
            " border: 2px solid #FFD24A;"
            " border-radius: 4px;"
            " padding: 2px 6px;"
            " font-size: 14px;"
            "}"
        )
        editor.textChanged.connect(
            lambda: self.inlineTextChanged.emit(editor.toPlainText())
        )
        editor.finished.connect(self.end_inline_edit)
        self._inline_edit = editor

        self._position_inline_edit()
        editor.show()
        editor.setFocus()
        editor.selectAll()

    def end_inline_edit(self) -> None:
        editor = self._inline_edit
        if editor is None:
            return
        self._inline_edit = None
        editor.hide()
        editor.deleteLater()

    def inline_edit_active(self) -> bool:
        """就地编辑框是否开着（供自检使用）。"""
        return self._inline_edit is not None

    def _position_inline_edit(self) -> None:
        """把就地输入框摆到印章框的中心。"""
        editor = self._inline_edit
        if editor is None:
            return

        pos = self._roi.pos()
        size = self._roi.size()
        view_box = self._plot.getViewBox()
        viewport = self._plot.viewport()

        top_left = viewport.mapTo(
            self, self._plot.mapFromScene(view_box.mapViewToScene(pos))
        )
        bottom_right = viewport.mapTo(
            self,
            self._plot.mapFromScene(
                view_box.mapViewToScene(QPointF(pos.x() + size.x(), pos.y() + size.y()))
            ),
        )

        span = QRect(top_left, bottom_right).normalized()
        width = max(160, min(span.width() - 8, 420))
        height = _INLINE_EDIT_HEIGHT
        editor.setGeometry(
            span.center().x() - width // 2,
            span.center().y() - height // 2,
            width,
            height,
        )

    # ---------------------------------------------------------------- 事件

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt 命名
        """在视口上拦截双击：落进印章框就请上层去编辑水印内容。"""
        if event.type() == QEvent.Type.MouseButtonDblClick and self._hit_roi(
            event.position().toPoint()
        ):
            self.patternDoubleClicked.emit()
            return True
        return super().eventFilter(obj, event)

    def _hit_roi(self, viewport_pos) -> bool:
        """视口坐标是否落在印章框内。"""
        if not (self._has_audio and self._roi.isVisible()):
            return False
        scene_pos = self._plot.mapToScene(viewport_pos)
        data_pos = self._plot.getViewBox().mapSceneToView(scene_pos)
        pos = self._roi.pos()
        size = self._roi.size()
        return (
            pos.x() <= data_pos.x() <= pos.x() + size.x()
            and pos.y() <= data_pos.y() <= pos.y() + size.y()
        )

    # ---------------------------------------------------------------- 拖放

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
