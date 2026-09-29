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
    FluentIcon,
    FluentWindow,
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
    TitleLabel,
)

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
from ..core.pattern import PatternError, build_pattern
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
    """批量处理主页面。"""

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

        # 印法建议：同一文件同一类建议只提一次，换文件后重新允许
        self._advised_for = ""
        self._advised_kinds: set[str] = set()
        self._advice_timer = QTimer(self)
        self._advice_timer.setSingleShot(True)
        self._advice_timer.setInterval(_ADVICE_DELAY_MS)
        self._advice_timer.timeout.connect(self._maybe_advise_mode)

        self.setAcceptDrops(True)       # 整个页面都能接文件，不只是输入框

        self._build_ui()
        self._connect_signals()
        self._refresh_pattern()
        self._on_engrave_changed()      # 让预览配色与初始模式（衰减）一致
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
        content_layout.setContentsMargins(28, 16, 28, 20)
        content_layout.setSpacing(16)

        left = QWidget(content)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(self._build_spectrum_card(left), 1)
        left_layout.addWidget(self._build_files_card(left))
        left_layout.addWidget(self._build_output_card(left))
        content_layout.addWidget(left, 62)

        # 右列：参数区（伸缩）+ 进度 + 操作按钮 —— 按钮紧贴它所依赖的参数下方
        right = QWidget(content)
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
        """底部操作区：只处理当前选中文件 / 处理整批。两个都靠右，主操作在最右。"""
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.process_current_button = PushButton("只处理当前文件", row)
        self.process_current_button.setMinimumHeight(36)

        self.process_all_button = PrimaryPushButton("处理所有文件", row)
        self.process_all_button.setMinimumWidth(150)
        self.process_all_button.setMinimumHeight(36)

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
        self.options_row = OptionRow(self.invert_pattern_check, parent=card)
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

        self.low_freq_spin = create_compact_spinbox(0, 22050, 1100, decimals=0, step=50, suffix="Hz", width=104)
        self.high_freq_spin = create_compact_spinbox(0, 22050, 9900, decimals=0, step=50, suffix="Hz", width=104)
        layout.addWidget(labeled_row(("低频", self.low_freq_spin), ("高频", self.high_freq_spin)))

        self.mode_combo = create_compact_combo(["按比例", "按秒"], width=96)
        layout.addWidget(labeled_row(("定位", self.mode_combo)))

        self.start_spin = create_compact_spinbox(0, 3600, 0.10, decimals=3, step=0.05, width=104)
        self.duration_spin = create_compact_spinbox(0.001, 3600, 0.30, decimals=3, step=0.05, width=104)
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

        self.engrave_segment.currentItemChanged.connect(self._on_engrave_changed)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        self.mux_switch.checkedChanged.connect(self._on_mux_toggled)

        self.loop_switch.checkedChanged.connect(self._on_loop_toggled)
        self.interval_spin.valueChanged.connect(self._sync_loop_preview)
        self.mode_combo.currentIndexChanged.connect(self._on_position_mode_changed)

        self.spectrum.regionChanged.connect(self._on_region_changed)
        self.spectrum.filesDropped.connect(self._add_paths)
        self.spectrum.patternDoubleClicked.connect(self._on_pattern_double_clicked)
        self.spectrum.inlineTextChanged.connect(self._on_inline_text_changed)
        self.low_freq_spin.valueChanged.connect(self._on_freq_edited)
        self.high_freq_spin.valueChanged.connect(self._on_freq_edited)
        self.start_spin.valueChanged.connect(self._on_time_edited)
        self.duration_spin.valueChanged.connect(self._on_time_edited)
        self.fft_combo.currentIndexChanged.connect(self._on_fft_changed)

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
        if self.pattern_segment.currentRouteKey() == "image":
            return PatternSpec(
                source=PatternSource.IMAGE,
                image_path=self.image_edit.text().strip() or None,
                invert_pattern=self.invert_pattern_check.isChecked(),
            )
        return PatternSpec(
            source=PatternSource.TEXT,
            text=self.text_edit.toPlainText(),
            weight=self.weight_slider.value(),
            invert_pattern=self.invert_pattern_check.isChecked(),
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
        except PatternError as exc:
            self.spectrum.set_pattern(None)
            self.pattern_info.setText(f"⚠ {exc}")
            self.pattern_info.setVisible(True)
            return

        self._pattern_cache.clear()          # 图案变了，缓存作废
        self.spectrum.set_pattern(pattern)
        self.pattern_info.setVisible(False)

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
        """记下当前印法下调到的强度，切走再切回来还能保持。"""
        self._strength_by_mode[self._current_engrave_mode] = (
            self.strength_slider.value()
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
        self.spectrum.set_engrave_mode(mode is EngraveMode.CUT)

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
        if self._syncing or not self.spectrum.has_audio():
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


class HelpPage(QWidget):
    """使用说明页。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpPage")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(14)

        layout.addWidget(TitleLabel("使用说明", self))
        layout.addWidget(self._card("三步上手", [
            "1. 把音频（或视频）文件、文件夹拖到窗口任意位置，列表会列出所有可处理的文件。",
            "2. 在左侧频谱图上拖动黄色印章框：上下边决定水印落在哪些频率，左右边决定印在什么时间。"
            "框内会实时显示图案的实际形状；拖动时还会浮出 20 Hz–5 kHz 的敏感频段提示，"
            "提醒这一段印水印最容易影响听感。",
            "3. 点右下角「处理所有文件」导出整批；只想先试一个，就选中它再点"
            "「只处理当前文件」。导出的 WAV 沿用源文件的名字（例外：输出目录就是源目录、"
            "源又是 wav 时会退让成 _tagged，免得把原始素材覆盖掉）。",
        ]))
        layout.addWidget(self._card("图案与参数", [
            "图案来源可以是文字或图片。文字支持中文与多行，还能开启「按文件名生成」，"
            "让每个文件印上自己的文件名。",
            "衰减是削弱图案区域 —— 但它只能改变素材里本来就有的内容，"
            "空白频段乘任何系数仍然是空白。",
            "注入是在图案区域造出内容 —— 即使超高频那种原本一片空白的地方，也能"
            "画出图案。代价是宽频段叠加高强度容易削波，收窄频段或降低强度即可。",
            "强度在两种印法下都是 0 = 无效果、1 = 效果拉满，默认值各自独立"
            "（衰减 0.8、注入 0.6），切换时会取回自己那份。",
            "把框拖到 15 kHz 以上而当前是衰减、或拖到 10 kHz 以下而当前是注入时，"
            "程序会提醒一次并可以直接切换 —— 同一首曲目同一类建议只提一次。",
            "频率位置在所有文件上按比例应用，因此不同采样率的素材会有一点点 Hz 偏移。",
            "FFT 越大频率越细、时间越粗。印章时长建议明显大于 0.1 秒，太短会被 STFT 的帧边界稀释。",
        ]))
        layout.addWidget(self._card("循环印刷", [
            "打开开关后，水印会在同样的频段反复印下去，每次的持续长度等于"
            "「位置与尺寸」里的时长。「间隔」量的是**上一次印完到下一次开始**的"
            "距离，所以印章本身多长都不会挤占它。",
            "频谱上会随之铺出一串淡色的影子印章 —— 它们的位置由第一个框和间隔推出来，"
            "所以只需要调第一个框，后面的会自动跟着走。",
            "「上限」留作「无限制」就一直印到音频结束；末尾放不下的那一次会被音频长度截断。",
        ]))
        layout.addWidget(self._card("支持的格式", [
            "音频输入：wav / mp3 / flac / aif / aiff / ogg。",
            "视频输入：mp4 / mkv / mov / avi / webm 等，需要系统里装有 ffmpeg。",
            "输出格式在「高级参数」里选。开着「合成视频」时，视频素材会输出成视频 —— "
            "画面原样保留，只把处理过的音轨换进去；关掉则一律只出音频。",
            "wav / flac / aiff / ogg / mp3 由 libsndfile 直接编码，m4a 以及所有视频合成需要 ffmpeg。",
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


class MainWindow(FluentWindow):
    """应用主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("频谱水印生成")
        self.resize(1440, 940)
        self.setMinimumSize(1120, 720)

        self.batch_page = BatchPage(self)
        self.help_page = HelpPage(self)

        self.addSubInterface(self.batch_page, FluentIcon.MUSIC, "批量处理")
        self.addSubInterface(self.help_page, FluentIcon.HELP, "使用说明")

        self.navigationInterface.setExpandWidth(180)

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self.batch_page._batch_worker  # noqa: SLF001 - 关闭前先收尾后台任务
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)
        preview = self.batch_page._preview_worker  # noqa: SLF001
        if preview is not None and preview.isRunning():
            preview.wait(2000)
        super().closeEvent(event)
