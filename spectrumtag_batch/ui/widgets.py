"""通用界面组件 —— 拖拽输入框、细进度条、图案预览、紧凑控件工厂。"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    DoubleSpinBox,
    LineEdit,
    ProgressBar,
    PushButton,
    Slider,
    SpinBox,
)

# 同行并排的紧凑控件统一高度，避免一行里控件高矮不齐
COMPACT_CONTROL_HEIGHT = 28


class DragLineEdit(LineEdit):
    """支持拖入文件/文件夹的路径输入框。

    拖入多个条目时只取第一个，其余由外部提示（批量列表另有专门的拖放区）。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setClearButtonEnabled(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 命名
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
        urls = event.mimeData().urls()
        if urls:
            self.setText(urls[0].toLocalFile())
            event.acceptProposedAction()
        else:
            event.ignore()

    def paths(self) -> list[str]:
        """按换行/分号切分，返回其中的路径列表。"""
        raw = self.text().replace(";", "\n")
        return [line.strip().strip('"') for line in raw.splitlines() if line.strip()]


def browse_button_for(
    target: DragLineEdit,
    *,
    directory: bool = False,
    title: str = "选择",
    name_filter: str = "所有文件 (*)",
) -> PushButton:
    """给路径输入框配一个「浏览」按钮。

    选文件时先弹文件选择框，用户取消则退回到目录选择 —— 这样同一个按钮
    既能选文件也能选文件夹。
    """

    def on_click() -> None:
        if directory:
            chosen = QFileDialog.getExistingDirectory(None, title, target.text())
            if chosen:
                target.setText(chosen)
            return

        chosen, _ = QFileDialog.getOpenFileName(None, title, target.text(), name_filter)
        if chosen:
            target.setText(chosen)
            return
        # 取消后给一次选目录的机会（用户想选整个文件夹的场景）
        fallback = QFileDialog.getExistingDirectory(None, f"{title}（或选择文件夹）")
        if fallback:
            target.setText(fallback)

    button = PushButton("浏览")
    button.setFixedHeight(COMPACT_CONTROL_HEIGHT)
    button.clicked.connect(on_click)
    return button


def path_row(
    target: DragLineEdit,
    *,
    directory: bool = False,
    title: str = "选择",
    name_filter: str = "所有文件 (*)",
) -> QWidget:
    """``[输入框占满] + [浏览]`` 的标准路径行。"""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(target, 1)
    layout.addWidget(browse_button_for(
        target, directory=directory, title=title, name_filter=name_filter
    ))
    return row


class BatchProgressPanel(QWidget):
    """批量任务的进度展示：6px 细条 + 一行状态文字。默认隐藏。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._bar = ProgressBar(self)
        self._bar.setFixedHeight(6)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)

        self._label = CaptionLabel("准备就绪", self)

        layout.addWidget(self._bar)
        layout.addWidget(self._label)
        self.setVisible(False)

    def start(self, message: str = "正在处理…") -> None:
        self._bar.setValue(0)
        self._label.setText(message)
        self.setVisible(True)

    def update_progress(self, value: int, message: str = "") -> None:
        self._bar.setValue(max(0, min(100, int(value))))
        if message:
            self._label.setText(message)

    def finish(self, message: str = "已完成") -> None:
        self._bar.setValue(100)
        self._label.setText(message)
        self.setVisible(False)

    def fail(self, message: str) -> None:
        self._label.setText(message)
        self.setVisible(True)


class _SpinDragFilter(QObject):
    """让数值框支持「按住左右拖动调值」。

    实现成事件过滤器而不是子类，这样工厂函数造出来的框全都能用上，
    也不用去掺和 qfluentwidgets 自己的 SpinBox 继承链。

    两种操作互不干扰：单击（横向位移不到阈值）完全放行，输入框照常编辑；
    一旦拖动起来就把后续移动事件截走，免得输入框把它当成选择文本。
    """

    _THRESHOLD_PX = 3        # 超过这个位移才算拖动，避免手抖误触发
    _PIXELS_PER_STEP = 3     # 每拖这么多像素走一步

    def __init__(self, spin: QAbstractSpinBox) -> None:
        super().__init__(spin)          # 以 spin 为 parent，随它一起销毁
        self._spin = spin
        self._origin_x: Optional[int] = None
        self._base_value: float = 0.0
        self._dragging = False

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt 命名
        kind = event.type()

        if kind == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._origin_x = int(event.position().x())
                self._base_value = self._spin.value()
                self._dragging = False
            return False

        if kind == QEvent.Type.MouseMove:
            if self._origin_x is None:
                return False
            delta = int(event.position().x()) - self._origin_x
            if not self._dragging:
                if abs(delta) < self._THRESHOLD_PX:
                    return False
                self._dragging = True
            self._apply(delta)
            return True

        if kind == QEvent.Type.MouseButtonRelease:
            was_dragging = self._dragging
            self._origin_x = None
            self._dragging = False
            return was_dragging

        return False

    def _apply(self, delta_px: int) -> None:
        step = self._spin.singleStep() or 1
        # 以按下时的值为基准累加，避免一步步累积舍入误差
        self._spin.setValue(
            self._base_value + int(delta_px / self._PIXELS_PER_STEP) * step
        )


def enable_drag_adjust(spin: QAbstractSpinBox) -> QAbstractSpinBox:
    """给数值框装上「按住拖动调值」，并换成左右拖动光标作提示。"""
    editor = spin.lineEdit()
    if editor is not None:
        editor.setCursor(Qt.CursorShape.SizeHorCursor)
        editor.installEventFilter(_SpinDragFilter(spin))
    return spin


def create_compact_combo(items: list[str], *, width: int = 110) -> ComboBox:
    combo = ComboBox()
    combo.addItems(items)
    combo.setFixedHeight(COMPACT_CONTROL_HEIGHT)
    combo.setMinimumWidth(width)
    return combo


def create_compact_spinbox(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 2,
    step: float = 0.1,
    suffix: str = "",
    width: int = 110,
) -> DoubleSpinBox:
    box = DoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    if suffix:
        box.setSuffix(f" {suffix}")
    box.setFixedHeight(COMPACT_CONTROL_HEIGHT)
    box.setMinimumWidth(width)
    return enable_drag_adjust(box)


def create_compact_int_spin(
    minimum: int,
    maximum: int,
    value: int,
    *,
    suffix: str = "",
    width: int = 110,
) -> SpinBox:
    box = SpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    if suffix:
        box.setSuffix(f" {suffix}")
    box.setFixedHeight(COMPACT_CONTROL_HEIGHT)
    box.setMinimumWidth(width)
    return enable_drag_adjust(box)


class UnlimitedSpinBox(SpinBox):
    """整数值为 0 时显示成「无限制」的输入框。

    比在界面某个角落写一行「0 = 不限」更容易看懂 —— 用户直接看到的就是
    "无限制"两个字，不用再去对照说明。
    """

    def __init__(
        self,
        unlimited_text: str = "无限制",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._unlimited_text = unlimited_text
        enable_drag_adjust(self)

    def textFromValue(self, value: int) -> str:  # noqa: N802 - Qt 命名
        if value == 0:
            return self._unlimited_text
        return super().textFromValue(value)

    def valueFromText(self, text: str) -> int:  # noqa: N802
        if text.strip() == self._unlimited_text:
            return 0
        return super().valueFromText(text)


class SliderRow(QWidget):
    """一行：`标签 + 滑块 + 当前数值`。

    对外仍是 0~1 的浮点值，内部用整数刻度驱动 ``QSlider``（默认 1000 档，
    即分辨率 0.001，足够覆盖所有需要）。数值实时显示在右侧，拖动时能看清
    当前落在哪 —— 这是滑块相比数值输入框的主要好处。
    """

    valueChanged = pyqtSignal(float)

    _STEPS = 1000

    def __init__(
        self,
        label: str,
        value: float = 0.5,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        decimals: int = 2,
        label_width: int = 30,
        value_width: int = 40,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._decimals = int(decimals)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        name = BodyLabel(label, self)
        name.setFixedWidth(label_width)
        name.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(name)

        self._slider = Slider(self)
        self._slider.setOrientation(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._STEPS)
        self._slider.setSingleStep(5)
        self._slider.setPageStep(50)
        self._slider.setFixedHeight(COMPACT_CONTROL_HEIGHT)
        layout.addWidget(self._slider, 1)

        self._value_label = BodyLabel(self)
        self._value_label.setFixedWidth(value_width)
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self._value_label)

        self._slider.valueChanged.connect(self._on_slider_moved)
        self.set_value(value)

    def value(self) -> float:
        ratio = self._slider.value() / float(self._STEPS)
        return self._min + (self._max - self._min) * ratio

    def set_value(self, value: float) -> None:
        span = (self._max - self._min) or 1.0
        ratio = (float(value) - self._min) / span
        ticks = int(round(min(1.0, max(0.0, ratio)) * self._STEPS))
        self._slider.setValue(ticks)
        self._refresh_label()

    def _on_slider_moved(self) -> None:
        self._refresh_label()
        self.valueChanged.emit(self.value())

    def _refresh_label(self) -> None:
        self._value_label.setText(f"{self.value():.{self._decimals}f}")


def labeled_row(*pairs: tuple[str, QWidget], spacing: int = 16) -> QWidget:
    """把若干 ``(标签, 控件)`` 并排排成一行，行尾用弹簧把内容推向左侧。"""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for text, widget in pairs:
        label = BodyLabel(text)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(label)
        layout.addWidget(widget)
    layout.addStretch(1)
    return row


class OptionRow(QWidget):
    """一行并排的复选框，行尾用弹簧把内容推向左侧。"""

    def __init__(self, *checks: QWidget, spacing: int = 16, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        for check in checks:
            layout.addWidget(check)
        layout.addStretch(1)
