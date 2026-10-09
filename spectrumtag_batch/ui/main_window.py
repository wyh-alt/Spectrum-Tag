"""主界面 —— 单窗口完成「选文件 → 调水印 → 批量导出」的全流程。"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QListWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CheckBox,
    FluentTitleBar,
    InfoBar,
    InfoBarPosition,
    ListWidget,
    MessageBox,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    StrongBodyLabel,
    SwitchButton,
    TextEdit,
)

# 无导航栏的窗口基类只在这里导出，顶层没转出来
from qfluentwidgets.window.fluent_window import FluentWindowBase

from ..batch import (
    SUPPORTED_EXTENSIONS,
    BatchProgress,
    BatchResult,
    TextMode,
    collect_files,
    run_batch,
)
from ..core.params import (
    DEFAULT_CUT_STRENGTH,
    DEFAULT_DRAW_STRENGTH,
    DEFAULT_HIGH_FREQ_HZ,
    DEFAULT_LOW_FREQ_HZ,
    FFT_SIZE_CHOICES,
    AudioFormat,
    DspSpec,
    EngraveMode,
    LoopSpec,
    OutputSpec,
    PatternSource,
    PatternSpec,
    PlacementSpec,
    PositionMode,
    RenderJob,
    VideoFormat,
)
from ..core.pattern import PatternError, build_outline, build_pattern
from ..media.audio_io import MediaError, read_audio
from ..media.video_io import find_ffmpeg, is_video_file
from .spectrogram import SpectrogramView
from .widgets import (
    COMPACT_CONTROL_HEIGHT,
    BatchProgressPanel,
    DragLineEdit,
    OptionRow,
    SliderRow,
    UnlimitedSpinBox,
    create_compact_combo,
    create_compact_spinbox,
    labeled_row,
    path_row,
)

# 输入过滤器（用于文件选择对话框）
_INPUT_FILTER = (
    "媒体文件 (" + " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS) + ");;所有文件 (*)"
)

# 预览用的频谱列数上限
_PREVIEW_MAX_COLS = 2000


# 图片扩展名（与 SpectrumTag 的图片白名单一致）—— 用来分辨"拖错了地方"和"格式不支持"
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

# 印法建议的频率门槛。两种印法各有擅长的频段：
#   * 15 kHz 往上素材能量通常已很低，衰减缺乏可操作余量；
#   * 10 kHz 往下能量集中，注入合成内容对频谱包络的改变更明显。
ADVISE_SPARSE_BAND_HZ = 15000.0
ADVISE_RICH_BAND_HZ = 10000.0

# 框有多大比例落在该频带内才算"主要位于" —— 要求整个框都进去太苛刻了，
# 稍微超出边界一点就不提醒，反而漏掉本该提醒的情况
ADVICE_OVERLAP_RATIO = 0.5

# 拖动停下多久之后才给建议 —— 拖的过程中弹模态框会打断操作，而且会连弹好几次
_ADVICE_DELAY_MS = 600

# 窗口尺寸变化后等多久再重算「保持原始水印比例」的时长。
# 拖窗口边框会连发几十次 resize，逐次重算纯属浪费。
_ASPECT_DELAY_MS = 150

# 「保持原始水印比例」算出来的时长下限（占音频总长的比例）——
# 频率框压得极扁时比例会趋近 0，留一点余量免得印章短到印不出来
_MIN_ASPECT_DURATION = 0.002

# 内容区左右留白（每张卡片自己另有 16 的内边距）
_CONTENT_PADDING_X = 28

# 输出格式下拉项。顺序即下拉里的顺序，第一项都是"与源一致"。
_AUDIO_FORMATS = (
    ("与源一致", AudioFormat.SOURCE),
    (".wav", AudioFormat.WAV),
    (".mp3", AudioFormat.MP3),
    (".m4a", AudioFormat.M4A),
    (".flac", AudioFormat.FLAC),
    (".aif", AudioFormat.AIF),
    (".aiff", AudioFormat.AIFF),
    (".ogg", AudioFormat.OGG),
)
_VIDEO_FORMATS = (
    ("与源一致", VideoFormat.SOURCE),
    (".mp4", VideoFormat.MP4),
    (".mkv", VideoFormat.MKV),
    (".mov", VideoFormat.MOV),
    (".avi", VideoFormat.AVI),
    (".webm", VideoFormat.WEBM),
)


def _overlap_ratio(
    frame_low: float, frame_high: float, band_low: float, band_high: float
) -> float:
    """水印框与给定频带的重叠，占框自身跨度的比例。"""
    span = frame_high - frame_low
    if span <= 0.0:
        return 0.0
    overlap = max(0.0, min(frame_high, band_high) - max(frame_low, band_low))
    return overlap / span


def _shorten(text: str, limit: int = 42) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class PreviewWorker(QThread):
    """后台解码音频并计算用于显示的时频图。"""

    ready = pyqtSignal(object, int, str)     # samples, sample_rate, path
    failed = pyqtSignal(str, str)            # path, message

    def __init__(self, path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._path = path

    def run(self) -> None:  # noqa: D102
        try:
            if is_video_file(self._path):
                from ..media.video_io import extract_audio_track

                with extract_audio_track(self._path) as wav_path:
                    audio = read_audio(wav_path)
            else:
                audio = read_audio(self._path)
        except (MediaError, Exception) as exc:  # noqa: BLE001
            self.failed.emit(self._path, str(exc))
            return
        self.ready.emit(audio.samples, audio.sample_rate, self._path)


class BatchWorker(QThread):
    """后台跑批处理。"""

    progressed = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        files: list[str],
        job: RenderJob,
        output_dir: str,
        output: OutputSpec,
        text_mode: TextMode,
        pattern_cache: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._files = files
        self._job = job
        self._output_dir = output_dir
        self._output = output
        self._text_mode = text_mode
        self._pattern_cache = pattern_cache
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D102
        try:
            result = run_batch(
                self._files,
                self._job,
                self._output_dir,
                output=self._output,
                text_mode=self._text_mode,
                progress=self._on_progress,
                pattern_cache=self._pattern_cache,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.completed.emit(result)

    def _on_progress(self, progress: BatchProgress) -> bool:
        name = _shorten(os.path.basename(progress.file_path), 30)
        self.progressed.emit(
            progress.overall_percent,
            f"[{progress.file_index}/{progress.file_count}] {name} · {progress.stage}",
        )
        return not self._cancelled


class BatchPage(QWidget):
    """批量处理主页面 —— 也是这个窗口唯一的一页。"""

    helpRequested = pyqtSignal()        # 点了「使用说明」

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("batchPage")

        self._files: list[str] = []
        self._preview_path: Optional[str] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._preview_samples: Optional[np.ndarray] = None
        self._preview_rate: int = 44100
        self._batch_worker: Optional[BatchWorker] = None
        self._pattern_cache: dict = {}
        self._syncing = False
        self._aspect_applying = False    # 正在按比例改时长，别被自己的回灌打断

        # 印法建议：同一文件同一类建议只提一次，换文件后重新允许
        self._advised_for = ""
        self._advised_kinds: set[str] = set()
        self._advice_timer = QTimer(self)
        self._advice_timer.setSingleShot(True)
        self._advice_timer.setInterval(_ADVICE_DELAY_MS)
        self._advice_timer.timeout.connect(self._maybe_advise_mode)

        self.setAcceptDrops(True)       # 整个页面都能接文件，不只是输入框

        # 「保持原始水印比例」的时长重算去抖：视口比例变了才需要重来
        self._aspect_timer = QTimer(self)
        self._aspect_timer.setSingleShot(True)
        self._aspect_timer.setInterval(_ASPECT_DELAY_MS)
        self._aspect_timer.timeout.connect(self._apply_aspect_duration)

        self._build_ui()
        self._connect_signals()
        self._refresh_pattern()
        self._on_engrave_changed()      # 让预览配色与初始模式（衰减）一致
        # 循环默认是开的，得主动同步一次 —— setChecked 发生在连信号之前，
        # 那个 checkedChanged 没人接
        self._sync_loop_preview()
        # 合成视频默认是开的，得主动同步一次 —— setChecked 发生在连信号之前，
        # 那个 checkedChanged 没人接
        self._on_mux_toggled(self.mux_switch.isChecked())

    # ------------------------------------------------------------ 界面构建

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 不设页面标题 —— 窗口标题栏里已经写着程序名，页面里再重复一遍是多余的。
        # 左右两列贯穿到窗口底部：右侧整列都留给参数，左列从频谱到底部操作区
        # 保持同一条右边线，视觉上不会出现"下宽上窄"的错位。
        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(
            _CONTENT_PADDING_X, 16, _CONTENT_PADDING_X, 20
        )
        content_layout.setSpacing(16)

        left = QWidget(content)
        # 同理给左列兜底：文件列表头部那一排按钮（添加文件 / 添加文件夹 /
        # 移除选中 / 清空）挤到一定程度就会互相压住。
        left.setMinimumWidth(500)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(self._build_spectrum_card(left), 1)
        left_layout.addWidget(self._build_files_card(left))
        left_layout.addWidget(self._build_output_card(left))
        content_layout.addWidget(left, 62)

        # 右列：参数区（伸缩）+ 进度 + 操作按钮 —— 按钮紧贴它所依赖的参数下方
        right = QWidget(content)
        # 参数列压不得：里面的数值框和复选框挤到一定程度就会截断数字、切掉
        # 文字（横滚条又是关着的）。给它一个下限，窗口变窄时先收左列。
        right.setMinimumWidth(460)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(self._build_parameter_column(right), 1)

        self.progress_panel = BatchProgressPanel(right)
        right_layout.addWidget(self.progress_panel)
        right_layout.addWidget(self._build_action_row(right))

        content_layout.addWidget(right, 38)
        root.addWidget(content, 1)

    def _build_action_row(self, parent: QWidget) -> QWidget:
        """底部操作区：使用说明在最左，两个处理按钮靠右、主操作在最右。

        说明按钮占的正是"最不碍事"的那个角 —— 它不参与主流程，离主操作
        越远越好，免得跟"处理所有文件"抢注意力。
        """
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.help_button = PushButton("使用说明", row)
        self.help_button.setMinimumHeight(36)

        self.process_current_button = PushButton("只处理当前文件", row)
        self.process_current_button.setMinimumHeight(36)

        self.process_all_button = PrimaryPushButton("处理所有文件", row)
        self.process_all_button.setMinimumWidth(150)
        self.process_all_button.setMinimumHeight(36)

        layout.addWidget(self.help_button)
        layout.addStretch(1)
        layout.addWidget(self.process_current_button)
        layout.addWidget(self.process_all_button)
        return row

    def _build_spectrum_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        layout.addWidget(StrongBodyLabel("频谱预览", card))
        self.spectrum_hint = BodyLabel("在下方拖入文件后，这里显示选中文件的时频图", card)
        layout.addWidget(self.spectrum_hint)

        self.spectrum = SpectrogramView(card)
        layout.addWidget(self.spectrum, 1)
        return card

    def _build_parameter_column(self, parent: QWidget) -> QWidget:
        """参数列放进滚动区 —— 卡片总高度会超过窗口，不能任由布局压缩它们。"""
        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setStyleSheet("background: transparent;")

        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_pattern_card(column))
        layout.addWidget(self._build_placement_card(column))
        layout.addWidget(self._build_loop_card(column))
        layout.addWidget(self._build_advanced_card(column))
        layout.addStretch(1)

        scroll.setWidget(column)
        return scroll

    def _build_pattern_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        layout.addWidget(StrongBodyLabel("水印图案", card))

        self.pattern_segment = SegmentedWidget(card)
        self.pattern_segment.addItem("text", "文字")
        self.pattern_segment.addItem("image", "图片")
        self.pattern_segment.setCurrentItem("text")
        self.pattern_segment.setFixedHeight(34)
        layout.addWidget(self.pattern_segment)

        # --- 文字模式 ---
        self.text_row = QWidget(card)
        text_layout = QVBoxLayout(self.text_row)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(6)
        # 多行：图案本来就支持 \n 换行，输入框得跟得上
        self.text_edit = TextEdit(self.text_row)
        self.text_edit.setPlainText("WATERMARK")
        self.text_edit.setPlaceholderText("要印上去的文字，支持中文与换行")
        self.text_edit.setFixedHeight(62)
        text_layout.addWidget(self.text_edit)

        self.weight_slider = SliderRow("粗细", 0.25, parent=self.text_row)
        text_layout.addWidget(self.weight_slider)
        layout.addWidget(self.text_row)

        # --- 图片模式 ---
        self.image_row = QWidget(card)
        image_layout = QVBoxLayout(self.image_row)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(6)
        self.image_edit = DragLineEdit(self.image_row)
        self.image_edit.setPlaceholderText("拖拽或输入图片路径（png / jpg / bmp / gif）")
        image_layout.addWidget(
            path_row(
                self.image_edit,
                title="选择水印图片",
                name_filter="图片 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*)",
            )
        )
        layout.addWidget(self.image_row)
        self.image_row.setVisible(False)

        self.invert_pattern_check = CheckBox("反转图案明暗")
        self.invert_pattern_check.setToolTip("适配黑底 / 白底的素材")

        # 图案描边：外沿再描一圈反向处理，与图案本体配成峰谷对
        self.outline_check = CheckBox("图案描边")
        self.outline_check.setChecked(True)
        self.outline_check.setToolTip(
            "沿图案外沿描一圈，朝「反方向」处理，与本体配成峰谷对：\n"
            "衰减时外圈凸起，注入时外圈凹陷。\n"
            "想抹平图案就得连外圈一起动，而外圈一动图案又露出来 ——\n"
            "两头的代价互相牵制，大幅抬高去除成本。"
        )

        # 保持原始水印比例：时长不再由用户直接给，而是按图案长宽比反推
        self.keep_aspect_check = CheckBox("保持原始水印比例")
        self.keep_aspect_check.setChecked(True)
        self.keep_aspect_check.setToolTip(
            "时长按图案的长宽比自动调节，图案不被拉伸变形。\n"
            "关掉之后时长重新可以手调，代价是图案会被拉伸铺满整个框。"
        )

        self.options_row = OptionRow(
            self.invert_pattern_check,
            self.outline_check,
            self.keep_aspect_check,
            parent=card,
        )
        layout.addWidget(self.options_row)

        # 印发方式紧跟在图案之后 —— 这是选完图案后紧接着要做的决定
        self.engrave_segment = SegmentedWidget(card)
        self.engrave_segment.addItem("cut", "衰减")
        self.engrave_segment.addItem("draw", "注入")
        self.engrave_segment.setCurrentItem("cut")
        self.engrave_segment.setFixedHeight(34)
        layout.addWidget(self.engrave_segment)

        # 两种印法各记一份强度：衰减要更狠才听得出，注入则不宜过强。
        # 默认值取自 DSP 层的常量，避免两边各写一份、改一处漏一处。
        self._strength_by_mode: dict[EngraveMode, float] = {
            EngraveMode.CUT: DEFAULT_CUT_STRENGTH,
            EngraveMode.DRAW: DEFAULT_DRAW_STRENGTH,
        }
        self._current_engrave_mode = EngraveMode.CUT

        self.strength_slider = SliderRow(
            "强度", self._strength_by_mode[EngraveMode.CUT], parent=card
        )
        layout.addWidget(self.strength_slider)

        # 平时不露面，只在图案出错时顶上来当错误提示
        self.pattern_info = BodyLabel("", card)
        self.pattern_info.setWordWrap(True)
        self.pattern_info.setVisible(False)
        layout.addWidget(self.pattern_info)
        return card

    def _build_placement_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(StrongBodyLabel("位置与尺寸", card))

        # 频率框比其它数值框宽一点：要放得下 "10500 Hz" 这种五位数带单位
        self.low_freq_spin = create_compact_spinbox(
            0, 22050, DEFAULT_LOW_FREQ_HZ, decimals=0, step=50, suffix="Hz", width=116
        )
        self.high_freq_spin = create_compact_spinbox(
            0, 22050, DEFAULT_HIGH_FREQ_HZ, decimals=0, step=50, suffix="Hz", width=116
        )
        layout.addWidget(labeled_row(("低频", self.low_freq_spin), ("高频", self.high_freq_spin)))

        self.mode_combo = create_compact_combo(["按比例", "按秒"], width=96)
        layout.addWidget(labeled_row(("定位", self.mode_combo)))

        self.start_spin = create_compact_spinbox(0, 3600, 0.0, decimals=3, step=0.05, width=104)
        self.duration_spin = create_compact_spinbox(0.001, 3600, 0.30, decimals=3, step=0.05, width=104)
        # 保持比例时它由图案长宽比反推，手改也会被立刻覆盖 —— 索性锁上
        self.duration_spin.setEnabled(not self.keep_aspect_check.isChecked())
        self.duration_spin.setToolTip("勾选「保持原始水印比例」时，时长由图案比例自动决定")
        layout.addWidget(labeled_row(("起点", self.start_spin), ("时长", self.duration_spin)))
        return card

    def _build_loop_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        header = QWidget(card)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(StrongBodyLabel("循环印刷", header))
        header_layout.addStretch(1)
        self.loop_switch = SwitchButton(header)
        self.loop_switch.setOnText("")      # 开关自己就能表示状态，不必再标 On/Off
        self.loop_switch.setOffText("")
        self.loop_switch.setChecked(True)   # 默认就循环印满整段
        header_layout.addWidget(self.loop_switch)
        layout.addWidget(header)

        self.interval_spin = create_compact_spinbox(0.1, 600, 5.0, decimals=2, step=0.5, suffix="s", width=104)
        self.max_repeat_spin = UnlimitedSpinBox("无限制", card)
        self.max_repeat_spin.setRange(0, 9999)
        self.max_repeat_spin.setFixedHeight(COMPACT_CONTROL_HEIGHT)
        self.max_repeat_spin.setMinimumWidth(104)
        self.loop_row = labeled_row(
            ("间隔", self.interval_spin), ("上限", self.max_repeat_spin)
        )
        layout.addWidget(self.loop_row)
        return card

    def _build_advanced_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(StrongBodyLabel("高级参数", card))

        self.fft_combo = create_compact_combo(
            [str(size) for size in FFT_SIZE_CHOICES], width=104
        )
        self.fft_combo.setCurrentIndex(FFT_SIZE_CHOICES.index(4096))

        self.mux_switch = SwitchButton(card)
        self.mux_switch.setOnText("")
        self.mux_switch.setOffText("")
        self.mux_switch.setChecked(True)   # 视频素材默认合回视频，画面原样保留
        self.mux_switch.setToolTip(
            "开着：视频素材输出成视频（画面原样保留，只替换音轨）\n"
            "关掉：一律只导出音频"
        )
        layout.addWidget(labeled_row(("FFT", self.fft_combo), ("合成视频", self.mux_switch)))

        self.audio_format_combo = create_compact_combo(
            [label for label, _ in _AUDIO_FORMATS], width=112
        )
        layout.addWidget(labeled_row(("音频格式", self.audio_format_combo)))

        self.video_format_combo = create_compact_combo(
            [label for label, _ in _VIDEO_FORMATS], width=112
        )
        self.video_format_combo.setEnabled(False)      # 没开合成视频时它没意义
        layout.addWidget(labeled_row(("视频格式", self.video_format_combo)))
        return card

    def _build_files_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        card.setFixedHeight(158)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        header = QWidget(card)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(StrongBodyLabel("待处理文件", header))
        self.file_count_label = BodyLabel("（空）", header)
        header_layout.addWidget(self.file_count_label)
        header_layout.addStretch(1)

        add_files_btn = PushButton("添加文件", header)
        add_folder_btn = PushButton("添加文件夹", header)
        self.remove_btn = PushButton("移除选中", header)
        self.clear_btn = PushButton("清空", header)
        for btn in (add_files_btn, add_folder_btn, self.remove_btn, self.clear_btn):
            btn.setFixedHeight(COMPACT_CONTROL_HEIGHT)
            header_layout.addWidget(btn)
        layout.addWidget(header)

        self.file_list = ListWidget(card)
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.file_list.setAlternatingRowColors(False)
        # 不自己处理拖放，让事件冒泡到页面统一处理
        self.file_list.setAcceptDrops(False)
        layout.addWidget(self.file_list, 1)

        self._add_files_btn = add_files_btn
        self._add_folder_btn = add_folder_btn
        return card

    def _build_output_card(self, parent: QWidget) -> CardWidget:
        card = CardWidget(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(StrongBodyLabel("输出目录", card))

        self.output_edit = DragLineEdit(card)
        self.output_edit.setPlaceholderText("拖拽或输入输出文件夹")
        self.output_edit.setText(os.path.join(os.path.expanduser("~"), "SpectrumTagOutput"))
        layout.addWidget(
            path_row(self.output_edit, directory=True, title="选择输出目录")
        )
        return card

    # ------------------------------------------------------------ 信号连接

    def _connect_signals(self) -> None:
        self.pattern_segment.currentItemChanged.connect(self._on_pattern_source_changed)
        self.text_edit.textChanged.connect(self._refresh_pattern)
        self.weight_slider.valueChanged.connect(self._refresh_pattern)
        self.image_edit.textChanged.connect(self._refresh_pattern)
        self.invert_pattern_check.stateChanged.connect(self._refresh_pattern)

        self.outline_check.stateChanged.connect(self._refresh_pattern)
        self.keep_aspect_check.stateChanged.connect(self._on_keep_aspect_toggled)

        self.engrave_segment.currentItemChanged.connect(self._on_engrave_changed)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        self.mux_switch.checkedChanged.connect(self._on_mux_toggled)

        self.loop_switch.checkedChanged.connect(self._on_loop_toggled)
        self.interval_spin.valueChanged.connect(self._sync_loop_preview)
        self.mode_combo.currentIndexChanged.connect(self._on_position_mode_changed)

        self.spectrum.regionChanged.connect(self._on_region_changed)
        self.spectrum.viewResized.connect(self._aspect_timer.start)
        self.spectrum.filesDropped.connect(self._add_paths)
        self.spectrum.patternDoubleClicked.connect(self._on_pattern_double_clicked)
        self.spectrum.inlineTextChanged.connect(self._on_inline_text_changed)
        self.low_freq_spin.valueChanged.connect(self._on_freq_edited)
        self.high_freq_spin.valueChanged.connect(self._on_freq_edited)
        self.start_spin.valueChanged.connect(self._on_time_edited)
        self.duration_spin.valueChanged.connect(self._on_time_edited)
        self.fft_combo.currentIndexChanged.connect(self._on_fft_changed)

        self.help_button.clicked.connect(self.helpRequested)
        self._add_files_btn.clicked.connect(self._pick_files)
        self._add_folder_btn.clicked.connect(self._pick_folder)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear_files)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        self.process_current_button.clicked.connect(self._on_process_current)
        self.process_all_button.clicked.connect(self._on_process_all)

        self.output_edit.textChanged.connect(lambda _: self._refresh_pattern())

    # ------------------------------------------------------------ 图案

    def _pattern_spec(self) -> PatternSpec:
        """从界面读出图案设置。"""
        outline = self.outline_check.isChecked()
        if self.pattern_segment.currentRouteKey() == "image":
            return PatternSpec(
                source=PatternSource.IMAGE,
                image_path=self.image_edit.text().strip() or None,
                invert_pattern=self.invert_pattern_check.isChecked(),
                outline=outline,
            )
        return PatternSpec(
            source=PatternSource.TEXT,
            text=self.text_edit.toPlainText(),
            weight=self.weight_slider.value(),
            invert_pattern=self.invert_pattern_check.isChecked(),
            outline=outline,
        )

    def _on_pattern_source_changed(self) -> None:
        is_image = self.pattern_segment.currentRouteKey() == "image"
        self.text_row.setVisible(not is_image)
        self.image_row.setVisible(is_image)
        if is_image:
            self.spectrum.end_inline_edit()   # 切到图片时，没编辑完的文字框该收掉
        self._refresh_pattern()

    def _on_pattern_double_clicked(self) -> None:
        """双击水印框 → 就地在框上开输入框，边打边看图案变化。

        图片模式下没有"文字"可改，直接忽略 —— 不弹框，也不擅自把来源
        切回文字（用户可能只是没看清当前是哪个模式）。
        """
        if self.pattern_segment.currentRouteKey() != "text":
            return
        self.spectrum.begin_inline_edit(self.text_edit.toPlainText())

    def _on_inline_text_changed(self, text: str) -> None:
        """框内敲字 → 同步回右侧输入框，并重算图案。"""
        if self.text_edit.toPlainText() == text:
            return
        self.text_edit.blockSignals(True)     # 免得绕回来又触发一轮
        self.text_edit.setPlainText(text)
        self.text_edit.blockSignals(False)
        self._refresh_pattern()

    def _refresh_pattern(self) -> None:
        """重算图案。

        图案的实际样子直接叠在频谱框里显示 —— 那是它真正会印上去的位置和比例，
        比单独放一块预览图更说明问题，所以这里不另设预览控件。
        """
        spec = self._pattern_spec()
        try:
            pattern = build_pattern(spec)
            outline = build_outline(pattern) if spec.outline else None
        except PatternError as exc:
            self.spectrum.set_pattern(None)
            self.pattern_info.setText(f"⚠ {exc}")
            self.pattern_info.setVisible(True)
            return

        self._pattern_cache.clear()          # 图案变了，缓存作废
        self.spectrum.set_pattern(pattern, outline)
        self.pattern_info.setVisible(False)
        # 图案换了，长宽比跟着变 —— 保持比例时时长要重算
        self._apply_aspect_duration()

    def _on_keep_aspect_toggled(self) -> None:
        """勾上就锁住时长输入框 —— 它马上会被按比例算出的值覆盖，留着能改只会让人困惑。"""
        self.duration_spin.setEnabled(not self.keep_aspect_check.isChecked())
        self._apply_aspect_duration()

    def _apply_aspect_duration(self) -> None:
        """「保持原始水印比例」：按图案长宽比反推印章时长。

        图案最终会被拉伸铺满整个印章框，所以"比例"只可能是**显示坐标**下的
        比例 —— 让框在频谱图上的像素宽高比等于图案自身的宽高比，看起来才跟
        原图一致。频率跨度由用户定死，时长于是被唯一确定：

            时长/总时长 = (图案宽/图案高) × 频率跨度(归一化) × (绘图区高/宽)

        绘图区比例参与其中，所以窗口大小一变就得重算 —— 这正是"所见即所得"
        该有的代价：预览里看到的形状，就是导出后频谱上的形状。

        未勾选、没有音频、图案还没渲染出来，或已经在应用途中时什么都不做。
        """
        if (
            self._aspect_applying
            or not self.keep_aspect_check.isChecked()
            or not self.spectrum.has_audio()
        ):
            return

        shape = self.spectrum.pattern_shape()
        if shape is None:
            return
        height, width = shape
        low, high = self._current_freq_norm()
        span = high - low
        if height <= 0 or width <= 0 or span <= 0.0:
            return

        norm = (width / height) * span * self.spectrum.viewport_aspect()
        norm = max(_MIN_ASPECT_DURATION, min(1.0, norm))

        total = self._preview_total_seconds()
        is_absolute = self.mode_combo.currentIndex() == 1
        # 起点直接取框当前的位置，不走输入框 —— 输入框是四舍五入过的，
        # 拿它回写会让框一点点漂移，每漂一次又触发一轮回调，成了死循环
        start_sec = self.spectrum.time_range_sec()[0]

        self._aspect_applying = True
        try:
            self._syncing = True
            try:
                self.duration_spin.setValue(
                    round(norm * total if is_absolute else norm, 3)
                )
            finally:
                self._syncing = False
            self.spectrum.set_time_range(start_sec, norm * total)
        finally:
            self._aspect_applying = False

    def _engrave_mode(self) -> EngraveMode:
        return (
            EngraveMode.CUT
            if self.engrave_segment.currentRouteKey() == "cut"
            else EngraveMode.DRAW
        )

    def _collect_output(self) -> OutputSpec:
        """从界面读出导出形态。"""
        return OutputSpec(
            mux_video=self.mux_switch.isChecked(),
            audio_format=_AUDIO_FORMATS[self.audio_format_combo.currentIndex()][1],
            video_format=_VIDEO_FORMATS[self.video_format_combo.currentIndex()][1],
        )

    def _on_mux_toggled(self, checked: bool) -> None:
        """不合成视频时，视频格式这个下拉没有意义，置灰。"""
        self.video_format_combo.setEnabled(checked)

    def _on_strength_changed(self, _value: float) -> None:
        """记下当前印法下调到的强度，切走再切回来还能保持。

        顺带刷新预览 —— 强度的作用直接体现在图案的浓淡上，边拖边看才直观。
        """
        strength = self.strength_slider.value()
        self._strength_by_mode[self._current_engrave_mode] = strength
        self._sync_preview_style()

    def _sync_preview_style(self) -> None:
        self.spectrum.set_preview_style(
            self._current_engrave_mode is EngraveMode.CUT,
            self.strength_slider.value(),
        )

    def _on_engrave_changed(self) -> None:
        """切换衰减 / 注入。

        两种印法的强度各自独立：先存下旧模式的取值，再换成新模式的
        （可能是它自己的默认值，也可能是上次调过的）。预览配色也跟着换。
        """
        mode = self._engrave_mode()
        if mode is not self._current_engrave_mode:
            self._strength_by_mode[self._current_engrave_mode] = (
                self.strength_slider.value()
            )
            self._current_engrave_mode = mode
            self.strength_slider.set_value(self._strength_by_mode[mode])
        self._sync_preview_style()

    def _on_loop_toggled(self, _checked: bool) -> None:
        self._sync_loop_preview()

    def _sync_loop_preview(self) -> None:
        """把循环设置递给频谱视图，好让它画出后续印章的影子。"""
        self.spectrum.set_loop_preview(
            self.loop_switch.isChecked(), float(self.interval_spin.value())
        )

    def _on_position_mode_changed(self) -> None:
        """按比例 / 按秒 切换时，把当前值换算过去，避免数值突变。"""
        if self._syncing:
            return
        total = self._preview_total_seconds()
        is_absolute = self.mode_combo.currentIndex() == 1
        self._syncing = True
        try:
            if is_absolute:
                self.start_spin.setRange(0.0, max(total, 3600.0))
                self.start_spin.setValue(round(self.start_spin.value() * total, 3))
                self.duration_spin.setValue(round(self.duration_spin.value() * total, 3))
                self.start_spin.setSuffix(" s")
                self.duration_spin.setSuffix(" s")
            else:
                self.start_spin.setRange(0.0, 1.0)
                safe_total = total if total > 1e-6 else 1.0
                self.start_spin.setValue(round(min(1.0, self.start_spin.value() / safe_total), 4))
                self.duration_spin.setValue(
                    round(min(1.0, self.duration_spin.value() / safe_total), 4)
                )
                self.start_spin.setSuffix("")
                self.duration_spin.setSuffix("")
        finally:
            self._syncing = False
        # 换算后时长要按新单位重算（保持比例时它由图案比例定，不跟着换算走）
        self._apply_aspect_duration()

    def _on_fft_changed(self) -> None:
        if self._preview_samples is not None:
            self._render_preview(self._preview_samples, self._preview_rate, self._preview_path or "")

    # ------------------------------------------------------------ 位置同步

    def _preview_total_seconds(self) -> float:
        if self._preview_samples is None or self._preview_rate <= 0:
            return 1.0
        return self._preview_samples.shape[-1] / float(self._preview_rate)

    def _maybe_advise_mode(self) -> None:
        """按水印框的落点与当前印法，给一次切换建议。

        每个文件、每条建议只提一次 —— 用户已经知道的事不用反复说。
        切换文件时清空记录，这样下一首还会提醒。
        """
        path = self._preview_path
        if not path or not self.spectrum.has_audio():
            return

        low_hz, high_hz = self._frame_freq_hz()
        max_hz = self._preview_rate / 2.0
        mode = self._engrave_mode()

        sparse_ratio = _overlap_ratio(
            low_hz, high_hz, ADVISE_SPARSE_BAND_HZ, max_hz
        )
        rich_ratio = _overlap_ratio(low_hz, high_hz, 0.0, ADVISE_RICH_BAND_HZ)

        if mode is EngraveMode.CUT and sparse_ratio >= ADVICE_OVERLAP_RATIO:
            kind, title, body, target, target_label = (
                "sparse",
                "建议改用注入",
                f"水印框主要位于 {ADVISE_SPARSE_BAND_HZ / 1000:g} kHz 以上频段。"
                "该频段素材能量通常较低，衰减处理的可操作余量有限，"
                "图案在频谱图上不易辨识。\n\n"
                "改用注入可直接在该频段合成目标能量，不受源素材内容限制。",
                "draw",
                "注入",
            )
        elif mode is EngraveMode.DRAW and rich_ratio >= ADVICE_OVERLAP_RATIO:
            kind, title, body, target, target_label = (
                "rich",
                "建议改用衰减",
                f"水印框主要位于 {ADVISE_RICH_BAND_HZ / 1000:g} kHz 以下频段。"
                "该频段能量集中，注入合成信号会明显改变原有频谱包络，"
                "听感差异较易察觉。\n\n"
                "改用衰减作用于既有内容，对音色的影响更可控。",
                "cut",
                "衰减",
            )
        else:
            return

        if self._advised_for != path:
            self._advised_for = path
            self._advised_kinds.clear()
        if kind in self._advised_kinds:
            return
        self._advised_kinds.add(kind)

        box = MessageBox(title, body, self.window())
        box.yesButton.setText("立即切换")
        box.cancelButton.setText("暂不需要")
        if box.exec():
            self.engrave_segment.setCurrentItem(target)
            InfoBar.success(
                "已切换",
                f"现在是「{target_label}」模式",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=2500,
            )

    def _frame_freq_hz(self) -> tuple[float, float]:
        """水印框当前覆盖的频率范围（Hz）。"""
        low_norm, high_norm = self.spectrum.freq_range_norm()
        max_hz = self._preview_rate / 2.0
        return low_norm * max_hz, high_norm * max_hz

    def _on_region_changed(self) -> None:
        """印章框被拖动 → 回写频率与时间控件。"""
        if self._aspect_applying or self._syncing or not self.spectrum.has_audio():
            return
        low_norm, high_norm = self.spectrum.freq_range_norm()
        start_sec, duration_sec = self.spectrum.time_range_sec()
        total = self._preview_total_seconds()
        max_hz = self._preview_rate / 2.0

        self._syncing = True
        try:
            self.low_freq_spin.setValue(round(low_norm * max_hz))
            self.high_freq_spin.setValue(round(high_norm * max_hz))
            is_absolute = self.mode_combo.currentIndex() == 1
            if is_absolute:
                self.start_spin.setValue(round(start_sec, 3))
                self.duration_spin.setValue(round(duration_sec, 3))
            else:
                safe_total = total if total > 1e-6 else 1.0
                self.start_spin.setValue(round(min(1.0, start_sec / safe_total), 4))
                self.duration_spin.setValue(round(min(1.0, duration_sec / safe_total), 4))
        finally:
            self._syncing = False

        # 拖上下边改了频率跨度 → 保持比例时时长要跟着重算；
        # 拖左右边改了宽度 → 会被算出来的值顶回去，宽度于是拖不动
        self._apply_aspect_duration()

        # 等拖动停下来再判断要不要给建议 —— 拖动过程中弹模态框会打断操作
        self._advice_timer.start()

    def _on_freq_edited(self) -> None:
        if self._syncing or not self.spectrum.has_audio():
            return
        max_hz = self._preview_rate / 2.0
        if max_hz <= 0:
            return
        low = self.low_freq_spin.value() / max_hz
        high = self.high_freq_spin.value() / max_hz
        if high <= low:
            return
        self._syncing = True
        try:
            self.spectrum.set_freq_range(low, high)
        finally:
            self._syncing = False
        self._apply_aspect_duration()

    def _on_time_edited(self) -> None:
        if self._syncing or not self.spectrum.has_audio():
            return
        total = self._preview_total_seconds()
        is_absolute = self.mode_combo.currentIndex() == 1
        if is_absolute:
            start, duration = self.start_spin.value(), self.duration_spin.value()
        else:
            start = self.start_spin.value() * total
            duration = self.duration_spin.value() * total
        self._syncing = True
        try:
            self.spectrum.set_time_range(start, duration)
        finally:
            self._syncing = False

    # ------------------------------------------------------------ 拖放

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
            self._add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ------------------------------------------------------------ 文件管理

    def _pick_files(self) -> None:
        chosen, _ = QFileDialog.getOpenFileNames(
            self, "选择媒体文件", "", _INPUT_FILTER
        )
        if chosen:
            self._add_paths(chosen)

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self._add_paths([folder])

    def _add_paths(self, paths: list[str]) -> None:
        found = collect_files(paths)
        existing = {os.path.normcase(os.path.abspath(p)) for p in self._files}
        added = [
            p for p in found
            if os.path.normcase(os.path.abspath(p)) not in existing
        ]
        if not added:
            InfoBar.warning(
                "没有新增文件",
                self._no_media_hint(paths),
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3500,
            )
            return

        was_empty = not self._files
        self._files.extend(added)
        for path in added:
            item = QListWidgetItem(self._display_name(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.file_list.addItem(item)

        self.file_count_label.setText(f"（共 {len(self._files)} 个）")
        # 原本是空列表时自动选中并加载预览，让「拖进来就能看到」成立
        if was_empty or self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(0)

        InfoBar.success(
            "已添加",
            f"新增 {len(added)} 个文件",
            parent=self.window(),
            position=InfoBarPosition.TOP,
            duration=2000,
        )

    def _no_media_hint(self, paths: list[str]) -> str:
        """一个都没收下时，给一句到点上的说明。

        拖图片进来多半是想拿它当水印图案 —— 直接说清楚该往哪儿拖，
        比一句干巴巴的"没有可处理文件"有用。
        """
        dropped_images = any(
            os.path.isfile(path)
            and os.path.splitext(path)[1].lower() in _IMAGE_EXTENSIONS
            for path in paths
        )
        if dropped_images:
            return "图片请在「水印图案」里选，或切到图片模式后拖进那个路径框"
        return "这里只收音频和视频文件"

    def _display_name(self, path: str) -> str:
        prefix = "[视频] " if is_video_file(path) else ""
        return f"{prefix}{os.path.basename(path)}"

    def _remove_selected(self) -> None:
        rows = sorted(
            {index.row() for index in self.file_list.selectedIndexes()}, reverse=True
        )
        if not rows:
            return
        for row in rows:
            self.file_list.takeItem(row)
            del self._files[row]
        self.file_count_label.setText(
            f"（共 {len(self._files)} 个）" if self._files else "（空）"
        )
        if self._files and self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(0)
        elif not self._files:
            self._on_file_selected(-1)

    def _clear_files(self) -> None:
        self.file_list.clear()
        self._files.clear()
        self.file_count_label.setText("（空）")
        self._on_file_selected(-1)

    def _on_file_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._files):
            self._preview_path = None
            self._preview_samples = None
            self.spectrum.clear()
            self.spectrum_hint.setText("在下方拖入文件后，这里显示选中文件的时频图")
            self.spectrum_hint.setVisible(True)
            return

        path = self._files[row]
        self.spectrum_hint.setText(f"正在读取 {_shorten(os.path.basename(path), 50)} …")
        self.spectrum_hint.setVisible(True)

        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.wait(50)

        self._preview_worker = PreviewWorker(path, self)
        self._preview_worker.ready.connect(self._render_preview)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.start()

    def _on_preview_failed(self, path: str, message: str) -> None:
        self.spectrum_hint.setText(f"无法预览 {os.path.basename(path)}：{message}")
        self.spectrum_hint.setVisible(True)

    def _render_preview(self, samples: np.ndarray, sample_rate: int, path: str) -> None:
        self._preview_samples = samples
        self._preview_rate = int(sample_rate)
        self._preview_path = path
        self.spectrum.set_audio(samples, sample_rate)

        duration = samples.shape[-1] / float(sample_rate)
        self.spectrum_hint.setText(
            f"{os.path.basename(path)} · {duration:.2f} 秒 · "
            f"{sample_rate} Hz · {samples.shape[0]} 声道  |  拖动黄框可调整水印位置与尺寸"
        )
        self.spectrum_hint.setVisible(True)

        # 频率控件范围跟随实际采样率
        max_hz = sample_rate / 2.0
        self._syncing = True
        try:
            self.low_freq_spin.setRange(0.0, max_hz)
            self.high_freq_spin.setRange(0.0, max_hz)
        finally:
            self._syncing = False

        # 用当前参数把框摆到正确位置
        low_norm, high_norm = self._current_freq_norm()
        self.spectrum.set_freq_range(low_norm, high_norm)
        total = duration if duration > 1e-6 else 1.0
        is_absolute = self.mode_combo.currentIndex() == 1
        if is_absolute:
            start, dur = self.start_spin.value(), self.duration_spin.value()
        else:
            start, dur = self.start_spin.value() * total, self.duration_spin.value() * total
        self.spectrum.set_time_range(start, dur)

        # 换了素材：时长按新音频重新校准（保持比例时它就是新总长的比例值）
        self._aspect_timer.stop()        # 载入过程里的 resize 通知已经过时了
        self._apply_aspect_duration()

    def _current_freq_norm(self) -> tuple[float, float]:
        max_hz = self._preview_rate / 2.0 if self._preview_rate else 22050.0
        low = self.low_freq_spin.value() / max_hz
        high = self.high_freq_spin.value() / max_hz
        low = max(0.0, min(0.99, low))
        high = max(low + 0.01, min(1.0, high))
        return low, high

    # ------------------------------------------------------------ 执行

    def _collect_job(self) -> RenderJob:
        low, high = self._current_freq_norm()
        is_absolute = self.mode_combo.currentIndex() == 1
        return RenderJob(
            pattern=self._pattern_spec(),
            dsp=DspSpec(
                fft_size=FFT_SIZE_CHOICES[self.fft_combo.currentIndex()],
                mode=self._engrave_mode(),
                strength=self.strength_slider.value(),
            ),
            placement=PlacementSpec(
                freq_low_norm=low,
                freq_high_norm=high,
                start=float(self.start_spin.value()),
                duration=float(self.duration_spin.value()),
                position_mode=(
                    PositionMode.ABSOLUTE if is_absolute else PositionMode.RELATIVE
                ),
            ),
            loop=LoopSpec(
                enabled=self.loop_switch.isChecked(),
                interval_sec=float(self.interval_spin.value()),
                max_repeats=int(self.max_repeat_spin.value()),
            ),
        )

    def _on_process_current(self) -> None:
        """只处理列表里当前选中的那个文件。"""
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self._files):
            InfoBar.warning(
                "请先选中一个文件",
                "在下面的列表里点一下要处理的文件",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return
        self._start_batch([self._files[row]])

    def _on_process_all(self) -> None:
        """处理整批；正在跑的时候这个按钮变成取消。"""
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.cancel()
            self.process_all_button.setEnabled(False)
            self.progress_panel.update_progress(0, "正在取消…")
            return

        if not self._files:
            InfoBar.warning(
                "还没有待处理文件",
                "请先拖入音频/视频文件或文件夹",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return
        self._start_batch(list(self._files))

    def _start_batch(self, files: list[str]) -> None:
        """校验设置并在后台启动渲染。"""
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return                       # 已经在跑，忽略重复触发

        output_dir = self.output_edit.text().strip()
        if not output_dir:
            InfoBar.warning(
                "请先选择输出目录",
                "导出的 WAV 会写到这里",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return

        if any(is_video_file(p) for p in files) and not find_ffmpeg():
            InfoBar.warning(
                "缺少 ffmpeg",
                "待处理列表里有视频文件，但没找到 ffmpeg，视频会被跳过",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=5000,
            )

        # 先用当前设置生成一次图案，把配置错误挡在开工之前
        try:
            build_pattern(self._pattern_spec())
        except PatternError as exc:
            InfoBar.error(
                "水印图案无效",
                str(exc),
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=4000,
            )
            return

        self._pattern_cache.clear()
        label = "当前文件" if len(files) == 1 else f"{len(files)} 个文件"
        self.progress_panel.start(f"正在处理{label}…")
        self._set_running(True)

        worker = BatchWorker(
            files,
            self._collect_job(),
            output_dir,
            self._collect_output(),
            TextMode.FIXED,
            self._pattern_cache,
            self,
        )
        worker.progressed.connect(self.progress_panel.update_progress)
        worker.completed.connect(self._on_batch_completed)
        worker.failed.connect(self._on_batch_failed)
        worker.finished.connect(self._on_batch_finished)
        self._batch_worker = worker
        worker.start()

    def _set_running(self, running: bool) -> None:
        """任务期间锁住按钮，防止重复触发；主按钮借用为取消。"""
        self.process_all_button.setText("取消" if running else "处理所有文件")
        self.process_all_button.setEnabled(True)
        self.process_current_button.setEnabled(not running)

    def _on_batch_failed(self, message: str) -> None:
        self.progress_panel.fail(f"任务失败：{message}")

    def _on_batch_finished(self) -> None:
        self._set_running(False)

    def _on_batch_completed(self, result: BatchResult) -> None:
        self.progress_panel.finish(
            f"完成 {result.succeeded} 个"
            + (f"，失败 {len(result.failed)} 个" if result.failed else "")
        )

        lines = [
            f"成功 {result.succeeded} / {len(result.items)} 个文件",
            f"总耗时 {result.total_seconds:.1f} 秒",
        ]
        if result.cancelled:
            lines.append("（任务被取消，未处理的文件已跳过）")
        if result.failed:
            lines.append("")
            lines.append("失败明细：")
            for item in result.failed[:12]:
                lines.append(f"  · {item.name}：{_shorten(item.message, 70)}")
            if len(result.failed) > 12:
                lines.append(f"  … 另有 {len(result.failed) - 12} 个")

        box = MessageBox("批量处理结果", "\n".join(lines), self.window())
        box.yesButton.setText("打开输出目录")
        box.cancelButton.setText("关闭")
        if box.exec():
            self._open_output_dir()
        else:
            InfoBar.success(
                "已导出",
                "处理完成的 WAV 已写入输出目录",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )

    def _open_output_dir(self) -> None:
        output_dir = self.output_edit.text().strip()
        if not output_dir or not os.path.isdir(output_dir):
            return
        try:
            if sys.platform == "win32":
                os.startfile(output_dir)  # noqa: S606 - Windows 资源管理器
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])
        except OSError as exc:
            InfoBar.warning(
                "无法打开目录",
                f"{exc}",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )


class HelpContent(QWidget):
    """使用说明的内容本体 —— 装在 :class:`HelpDialog` 里。

    它曾经是侧边栏里的一页；侧边栏去掉之后，同一份内容换成弹窗，
    免得为它单独占一块常驻的地方。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpContent")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(14)

        layout.addWidget(self._card("三步上手", [
            "1. 将音频或视频文件（可整个文件夹）拖入窗口。",
            "2. 在频谱图上拖动印章框，确定水印的频率与时间范围。",
            "3. 点击「处理所有文件」批量导出，或「只处理当前文件」单独导出。",
        ]))
        layout.addWidget(self._card("两种印法", [
            "衰减：降低图案区域的能量，仅对素材已有内容有效。",
            "注入：在图案区域合成能量，空白频段也能形成图案。",
            "强度 0 为无效果、1 为最强；两者各自记忆，默认 0.95 / 0.6。",
        ]))
        layout.addWidget(self._card("默认开启的三项", [
            "图案描边：沿图案外沿附加一层反向处理，与本体重构为峰谷对，提高去除成本。",
            "保持原始水印比例：时长由图案宽高比反推，避免拉伸变形。",
            "循环印刷：按设定间隔重复印刷；间隔指上次结束到下次开始的距离。",
        ]))
        layout.addWidget(self._card("频率位置", [
            "默认 9–10.5 kHz。拖动印章框时显示两块参考区域：",
            "绿色为推荐印刷区域（7–13 kHz）；",
            "黄色为敏感频段（20 Hz–5 kHz），该频段内的改动最易被察觉。",
            "两者均为参考范围，不构成限制。",
        ]))
        layout.addWidget(self._card("支持的格式", [
            "音频：wav / mp3 / flac / aif / aiff / ogg",
            "视频：mp4 / mkv / mov / avi / webm 等，需安装 ffmpeg",
            "输出格式在「高级参数」中选择；启用「合成视频」时，视频素材将输出为视频。",
        ]))
        layout.addStretch(1)

    def _card(self, title: str, lines: list[str]) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        layout.addWidget(StrongBodyLabel(title, card))
        for line in lines:
            label = BodyLabel(line, card)
            label.setWordWrap(True)
            layout.addWidget(label)
        return card


class HelpDialog(QDialog):
    """使用说明弹窗：一个可滚动的窗口，内容就是 :class:`HelpContent`。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("使用说明")
        self.setMinimumSize(520, 420)
        # 默认开到刚好看全 —— 内容在 660 宽下要 727 高，留一点余量免得一上来
        # 就顶着滚动条。窗口仍可自由缩放。
        self.resize(700, 800)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setWidget(HelpContent(scroll))
        layout.addWidget(scroll, 1)


class MainWindow(FluentWindowBase):
    """应用主窗口。

    只有批量处理这一个功能，所以不留侧边导航栏 —— 直接让页面铺满窗口，
    省下那一条常驻的宽度。使用说明挪进了底部操作区的按钮里。
    """

    def __init__(self) -> None:
        super().__init__()
        # 顺序要紧：FluentTitleBar 只在初始化时接上 windowTitleChanged 信号，
        # 并不会回头读一次当前标题 —— 先设标题的话它拿到的是空字符串，标题栏
        # 就只剩一个图标。（窗口图标同理，好在 main.py 是之后才设的。）
        self.setTitleBar(FluentTitleBar(self))
        # 程序名仍要设：任务栏、Alt+Tab 都靠它。只是自绘标题栏里不再显示 ——
        # 左侧留空，只留右边那排窗口按钮，跟内容区那一片卡片放在一起更清爽。
        self.setWindowTitle("频谱水印生成")
        self.titleBar.iconLabel.hide()
        self.titleBar.titleLabel.hide()
        # 尺寸不在这里设 —— 见 apply_default_size()，它必须等窗口显示之后才生效

        self.batch_page = BatchPage(self)
        self.stackedWidget.addWidget(self.batch_page)
        self.stackedWidget.setCurrentWidget(self.batch_page)

        # 没走 FluentWindow 那条路，得自己把内容接进根布局：
        # 上边距 48 是给浮在上面的标题栏让位。
        self.hBoxLayout.setContentsMargins(0, 48, 0, 0)
        self.hBoxLayout.addWidget(self.stackedWidget)

        self._help_dialog: Optional[HelpDialog] = None
        self.batch_page.helpRequested.connect(self._show_help)

    def _show_help(self) -> None:
        """弹出使用说明；已经开着就带到前面来，不重复开一堆窗口。"""
        if self._help_dialog is None:
            self._help_dialog = HelpDialog(self)
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    def apply_default_size(self) -> None:
        """设成默认窗口尺寸；屏幕装不下时**等比**收窄。

        历史注记：``FluentWindow`` 的初始化里有延迟执行的部分，会在事件循环
        第一次转动时把窗口尺寸重置成 Qt 默认的 500x500，所以那时**必须**等
        ``show()`` 之后再设。换成不带导航栏的基类后这个行为没有了（实测
        ``show()`` 前 resize 也能留住），这个函数仍放在显示之后调用 —— 顺序
        没有坏处，也免得哪天换回带导航的窗口时又踩一遍。

        收窄要等比，宽高各自受限会导致比例失真 —— 高 DPI 缩放或小屏都会撞上。
        """
        width, height = 1440, 940
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            scale = min(
                1.0,
                area.width() * 0.92 / width,
                area.height() * 0.92 / height,
            )
            if scale < 1.0:
                width, height = int(width * scale), int(height * scale)
        # 最小尺寸也跟着收，否则窄屏上会被它顶回去。
        # 少了侧边栏那一条，下限可以比从前低一档（参数列另有 460 的下限兜着）。
        self.setMinimumSize(min(1020, width), min(680, height))
        self.resize(width, height)

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self.batch_page._batch_worker  # noqa: SLF001 - 关闭前先收尾后台任务
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)
        preview = self.batch_page._preview_worker  # noqa: SLF001
        if preview is not None and preview.isRunning():
            preview.wait(2000)
        super().closeEvent(event)
