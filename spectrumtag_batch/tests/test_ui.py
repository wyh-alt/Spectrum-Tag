"""界面自检 —— 真的把窗口开起来，走一遍工作流并截图。

直接运行：``python -m spectrumtag_batch.tests.test_ui``

截图输出到 ``%TEMP%/stbatch_ui/``，同时检查控件是否都建起来了、
拖入文件后频谱预览是否真的算出了图。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

SHOT_DIR = os.path.join(tempfile.gettempdir(), "stbatch_ui")


class _FakeWheelEvent:
    """够用的滚轮事件替身：只关心对方有没有把它 ignore 掉。

    PyQt6 不允许直接实例化 QGraphicsSceneWheelEvent，而这个分支的判断
    就在我们自己的 wheelEvent 里，根本走不到父类，所以替身足够。
    """

    def __init__(self) -> None:
        self.ignored = False

    def ignore(self) -> None:
        self.ignored = True

    def accept(self) -> None:
        self.ignored = False

    def delta(self) -> int:
        return 120

    def angleDelta(self):
        return QPoint(0, 120)

    def orientation(self):
        return Qt.Orientation.Vertical

    def modifiers(self):
        return Qt.KeyboardModifier.NoModifier


def _make_inputs(folder: str) -> list[str]:
    """造几个用于自检的音频文件。"""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(11)
    paths = []
    for name, seconds, rate, channels in (
        ("Sample_A", 3.0, 44100, 2),
        ("Sample_B", 2.0, 48000, 1),
    ):
        path = os.path.join(folder, f"{name}.wav")
        # 带音调结构的素材：频谱图上会有清晰的水平线，便于判断显示与印章效果
        frames = int(seconds * rate)
        t = np.arange(frames) / rate
        tone = (
            0.30 * np.sin(2 * np.pi * 220 * t)
            + 0.20 * np.sin(2 * np.pi * 880 * t)
            + 0.12 * np.sin(2 * np.pi * 3000 * t)
            + 0.06 * rng.standard_normal(frames)
        )
        data = np.repeat(tone[:, None], channels, axis=1).astype(np.float32)
        sf.write(path, data, rate, subtype="PCM_16")
        paths.append(path)
    return paths


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)

    from ..ui.main_window import MainWindow

    shutil.rmtree(SHOT_DIR, ignore_errors=True)
    os.makedirs(SHOT_DIR, exist_ok=True)

    workspace = tempfile.mkdtemp(prefix="stbatch_ui_in_")
    inputs = _make_inputs(os.path.join(workspace, "in"))

    window = MainWindow()
    window.resize(1440, 920)
    window.show()

    page = window.batch_page
    checks: list[tuple[str, bool]] = []
    shots: list[str] = []

    def snap(name: str) -> None:
        path = os.path.join(SHOT_DIR, f"{name}.png")
        window.grab().save(path)
        shots.append(path)

    def check(label: str, condition: bool) -> None:
        checks.append((label, condition))

    def step_empty() -> None:
        from ..core.params import DEFAULT_CUT_STRENGTH

        snap("01_empty")
        check("空状态显示提示", page.spectrum_hint.isVisible())
        check("文件列表为空", page.file_list.count() == 0)
        check(f"衰减强度默认为 {DEFAULT_CUT_STRENGTH:g}",
              abs(page.strength_slider.value() - DEFAULT_CUT_STRENGTH) < 1e-6)

        # 没有音频时：印章框不出现（也就拖不动），滚轮缩放也不响应
        view = page.spectrum
        check("未载入时不显示印章框", not view._roi.isVisible())  # noqa: SLF001
        event = _FakeWheelEvent()
        view._plot.getViewBox().wheelEvent(event)  # noqa: SLF001
        check("未载入时滚轮缩放被忽略", event.ignored)

    def step_add_files() -> None:
        page._add_paths(inputs)  # noqa: SLF001 - 自检直接调内部方法
        check("文件已加入列表", page.file_list.count() == 2)
        check("计数标签正确", "2" in page.file_count_label.text())

    def step_after_preview() -> None:
        snap("02_loaded")
        check("频谱已载入音频", page.spectrum.has_audio())
        check("预览采样率正确", page._preview_rate in (44100, 48000))  # noqa: SLF001
        check("图案已叠到频谱框上",
              page.spectrum._pattern_item.isVisible())  # noqa: SLF001
        check("无错误时不显示提示行", not page.pattern_info.isVisible())
        check("载入后印章框出现", page.spectrum._roi.isVisible())  # noqa: SLF001
        # extent 已建立 → wheelEvent 不再被拦，滚轮可以正常缩放
        view_box = page.spectrum._plot.getViewBox()  # noqa: SLF001
        check("载入后允许缩放", view_box._full_x > 0)  # noqa: SLF001

    def step_image_mode() -> None:
        page.pattern_segment.setCurrentItem("image")
        QApplication.processEvents()
        check("图片行已显示", page.image_row.isVisible())
        check("文字行已隐藏", not page.text_row.isVisible())
        snap("03_image_mode")
        page.pattern_segment.setCurrentItem("text")
        QApplication.processEvents()

    def step_loop_on() -> None:
        """循环参数常驻；开启后频谱里铺出后续印章的影子。"""
        # 还没开循环，参数行也应该在
        check("循环参数行常驻可见", page.loop_row.isVisible())
        check("上限显示为无限制", page.max_repeat_spin.text() == "无限制")

        # 一律通过控件摆框（用户实际会走的路径）——直接调 set_time_range 会绕过
        # 回写，控件与框就对不上了，后面的一致性检查也就失去意义
        duration = page.spectrum.duration_sec()
        page.interval_spin.setValue(0.6)
        page.start_spin.setValue(0.10)
        page.duration_spin.setValue(0.30)
        QApplication.processEvents()
        check("循环关闭时不画影子", page.spectrum.loop_ghost_count() == 0)

        pos_x = page.spectrum._roi.pos().x()  # noqa: SLF001
        check("控件与框位置保持同步",
              abs(page.start_spin.value() * duration - pos_x) < 0.01)

        page.loop_switch.setChecked(True)
        QApplication.processEvents()
        before = page.spectrum.loop_ghost_count()
        check("开启循环后出现影子框", before > 0)

        # 影子由第一个框 + 间隔推出：把框往后挪，能放下的次数自然变少
        page.start_spin.setValue(0.60)
        page.duration_spin.setValue(0.20)
        QApplication.processEvents()
        check("影子随第一个框位置变化",
              page.spectrum.loop_ghost_count() < before)

        page.loop_switch.setChecked(False)
        QApplication.processEvents()
        check("关闭循环后影子清空", page.spectrum.loop_ghost_count() == 0)

        page.loop_switch.setChecked(True)
        page.interval_spin.setValue(1.0)
        page.start_spin.setValue(0.08)
        page.duration_spin.setValue(0.30)
        QApplication.processEvents()
        snap("04_loop")

    def step_loop_consistency() -> None:
        """预览影子必须和 DSP 实际印刷的区间对得上。

        这两处是各自独立实现的（一处画图、一处算区间），很容易各改各的。
        """
        from ..core.params import resolve_intervals

        view = page.spectrum
        job = page._collect_job()  # noqa: SLF001
        total = view.duration_sec()

        intervals = resolve_intervals(job.placement, job.loop, total)
        # 第一个由用户摆，不在影子里；影子对应的是后续各次
        expected = [round(iv.start_sec, 2) for iv in intervals[1:]]

        pos, size = view._roi.pos(), view._roi.size()  # noqa: SLF001
        step = size.x() + job.loop.interval_sec
        actual = [
            round(pos.x() + step * (i + 1), 2)
            for i in range(view.loop_ghost_count())
        ]

        check("预览影子与实际印刷区间一致",
              expected[: len(actual)] == actual and actual)

    def step_region_bounds() -> None:
        """框拖到边缘就该停住，不能伸到频谱范围之外。

        这里刻意用 set_time_range / set_freq_range 做程序化越界测试，
        所以结束后要把控件重新同步一次，免得影响后续步骤。
        """
        view = page.spectrum
        roi = view._roi  # noqa: SLF001
        duration = view.duration_sec()
        max_freq = view._max_freq  # noqa: SLF001

        # 往右下角外面拖
        roi.setSize([duration * 0.6, max_freq * 0.6])
        roi.setPos([duration * 0.9, max_freq * 0.9])
        QApplication.processEvents()
        pos, size = roi.pos(), roi.size()
        check("右边界不超出音频末尾",
              pos.x() + size.x() <= duration + 1e-6)
        check("上边界不超出 Nyquist",
              pos.y() + size.y() <= max_freq + 1e-6)

        # 往左上角外面拖
        roi.setPos([-duration, -max_freq])
        QApplication.processEvents()
        pos = roi.pos()
        check("左边界不低于 0", pos.x() >= -1e-9)
        check("下边界不低于 0", pos.y() >= -1e-9)

        # 尺寸本身也不能超过整段音频
        roi.setPos([0.0, 0.0])
        roi.setSize([duration * 5.0, max_freq * 5.0])
        QApplication.processEvents()
        size = roi.size()
        check("框宽不超过音频总长", size.x() <= duration + 1e-6)
        check("框高不超过 Nyquist", size.y() <= max_freq + 1e-6)

        # 用控件复位，让框与控件重新对齐，别影响后续步骤
        page.low_freq_spin.setValue(0.05 * max_freq)
        page.high_freq_spin.setValue(0.45 * max_freq)
        page.start_spin.setValue(0.10)
        page.duration_spin.setValue(0.30)
        QApplication.processEvents()

    def step_help_page() -> None:
        window.switchTo(window.help_page)
        QApplication.processEvents()

    def step_help_snap() -> None:
        snap("05_help")

    def step_collect_job() -> None:
        job = page._collect_job()  # noqa: SLF001
        check("job 图案来源为文字", job.pattern.source.value == "text")
        check("job 循环已启用", job.loop.enabled)
        check("频率范围合法", job.placement.freq_range_valid)
        check("FFT 尺寸合法", job.dsp.fft_size in (1024, 2048, 4096, 8192))

        # 反向验证：改框会同步到控件
        page.spectrum.set_freq_range(0.2, 0.6)
        QApplication.processEvents()
        lo, hi = page.spectrum.freq_range_norm()
        check("频率框同步一致", abs(lo - 0.2) < 0.02 and abs(hi - 0.6) < 0.02)

    def step_zoom_behavior() -> None:
        """全览时拖动不平移；放大后允许平移；缩放下限锁在全览。"""
        import pyqtgraph as pg

        from ..ui.spectrogram import _SpectrogramViewBox

        view_box = page.spectrum._plot.getViewBox()  # noqa: SLF001
        check("频谱用的是自定义 ViewBox",
              isinstance(view_box, _SpectrogramViewBox))
        check("mouseDragEvent 已被覆盖",
              type(view_box).mouseDragEvent is not pg.ViewBox.mouseDragEvent)

        duration = page.spectrum.duration_sec()          # noqa: SLF001
        max_freq = page.spectrum._max_freq               # noqa: SLF001

        # 全览状态
        view_box.setRange(xRange=(0.0, duration), yRange=(0.0, max_freq), padding=0.0)
        QApplication.processEvents()
        check("全览时不算放大", not view_box.is_zoomed())

        # 放大后
        view_box.setRange(
            xRange=(0.0, duration * 0.4), yRange=(0.0, max_freq * 0.4), padding=0.0
        )
        QApplication.processEvents()
        check("放大后判定为已放大", view_box.is_zoomed())

        # 试图缩到比数据还大 —— 应被 limits 卡住
        view_box.setRange(
            xRange=(-1.0, duration + 1.0), yRange=(-100.0, max_freq + 100.0), padding=0.0
        )
        QApplication.processEvents()
        x_range, y_range = view_box.viewRange()
        check("缩放下限锁在全览（时间）",
              x_range[0] >= -1e-6 and x_range[1] <= duration + 1e-6)
        check("缩放下限锁在全览（频率）",
              y_range[0] >= -1e-6 and y_range[1] <= max_freq + 1e-6)

        # 恢复全览，免得影响后续截图
        view_box.setRange(xRange=(0.0, duration), yRange=(0.0, max_freq), padding=0.0)
        QApplication.processEvents()

    def step_pattern_overlay() -> None:
        """图案应当实时叠加在框里。"""
        item = page.spectrum._pattern_item  # noqa: SLF001
        check("图案叠加层可见", item.isVisible())
        check("图案数据非空", item.image is not None and item.image.size > 0)

        # 改文字后叠加层应随之更新
        page.text_edit.setText("AB")
        QApplication.processEvents()
        check("改文字后图案仍在", item.isVisible() and item.image.size > 0)

    def step_engraving_mode() -> None:
        """衰减 / 注入 切换要落到 DSP 的 mode 上。"""
        from ..core.params import EngraveMode

        window.switchTo(window.batch_page)      # 截图要回主页面
        QApplication.processEvents()

        page.engrave_segment.setCurrentItem("cut")
        QApplication.processEvents()
        job = page._collect_job()  # noqa: SLF001
        check("挖空 -> mode=CUT", job.dsp.mode is EngraveMode.CUT)
        check("衰减增益随强度变化", abs(job.dsp.cut_gain - (1 - job.dsp.strength)) < 1e-9)

        # 预览配色要跟着模式变：衰减是暗块（那片会被抹掉），注入是亮块（会加内容）
        dark_lut = page.spectrum._overlay_lut[1]  # noqa: SLF001
        check("衰减模式预览为暗色", dark_lut[0] < 100)

        page.engrave_segment.setCurrentItem("draw")
        QApplication.processEvents()
        job = page._collect_job()  # noqa: SLF001
        check("注入 -> mode=DRAW", job.dsp.mode is EngraveMode.DRAW)

        bright_lut = page.spectrum._overlay_lut[1]  # noqa: SLF001
        check("注入模式预览为亮色", bright_lut[0] > 200)

        # 两种印法的强度各自独立：切过去取默认，调过再切回来要记得
        from ..core.params import DEFAULT_CUT_STRENGTH, DEFAULT_DRAW_STRENGTH

        check("注入的默认强度独立",
              abs(page.strength_slider.value() - DEFAULT_DRAW_STRENGTH) < 1e-6)
        page.strength_slider.set_value(0.9)
        QApplication.processEvents()

        page.engrave_segment.setCurrentItem("cut")
        QApplication.processEvents()
        check("切回衰减取回它自己的强度",
              abs(page.strength_slider.value() - DEFAULT_CUT_STRENGTH) < 1e-6)

        page.engrave_segment.setCurrentItem("draw")
        QApplication.processEvents()
        check("再切回注入记得调过的值",
              abs(page.strength_slider.value() - 0.9) < 1e-6)
        check("注入幅度 = N/2 标定",
              abs(job.dsp.draw_amplitude - 10 ** (job.dsp.draw_dbfs / 20) * job.dsp.fft_size / 2) < 1e-6)

        # 强度滑块：0~1 的浮点值直通 DSP，数值标签同步显示
        page.strength_slider.set_value(0.5)
        QApplication.processEvents()
        check("强度滑块联动到 job",
              abs(page._collect_job().dsp.strength - 0.5) < 1e-6)  # noqa: SLF001

        # 粗细滑块同理
        page.weight_slider.set_value(0.75)
        QApplication.processEvents()
        check("粗细滑块联动到图案",
              abs(page._collect_job().pattern.weight - 0.75) < 1e-6)  # noqa: SLF001
        check("粗细滑块数值标签同步",
              page.weight_slider._value_label.text() == "0.75")  # noqa: SLF001

        snap("06_draw")
        page.engrave_segment.setCurrentItem("cut")
        page.strength_slider.set_value(1.0)
        page.weight_slider.set_value(0.25)
        QApplication.processEvents()

    def step_mode_advice() -> None:
        """按水印框落点与当前印法给一次建议，同一文件同类只提一次。"""
        import spectrumtag_batch.ui.main_window as mw

        shown: list[str] = []

        class _Btn:
            def setText(self, _text: str) -> None:
                pass

        class _FakeBox:
            """替身弹窗：记下标题，一律返回"暂不需要"。"""

            def __init__(self, title, _body, _parent=None):
                shown.append(title)
                self.yesButton = _Btn()
                self.cancelButton = _Btn()

            def exec(self) -> int:
                return 0

        class _YesBox(_FakeBox):
            def exec(self) -> int:
                return 1        # 选"立即切换"

        original = mw.MessageBox
        try:
            def fire(low_hz: float, high_hz: float, mode_key: str):
                page._advised_kinds.clear()          # noqa: SLF001
                page.engrave_segment.setCurrentItem(mode_key)
                page.low_freq_spin.setValue(low_hz)
                page.high_freq_spin.setValue(high_hz)
                QApplication.processEvents()
                shown.clear()
                page._maybe_advise_mode()            # noqa: SLF001
                return shown[0] if shown else None

            mw.MessageBox = _FakeBox
            check("衰减 + 框全在 15kHz 以上 → 建议注入",
                  fire(15500, 20000, "cut") == "建议改用注入")
            check("衰减 + 大部分在 15kHz 以上也提醒",
                  fire(14000, 19000, "cut") == "建议改用注入")
            check("衰减 + 只有小部分越界 → 不打扰",
                  fire(8000, 18000, "cut") is None)
            check("注入 + 框全在 10kHz 以下 → 建议衰减",
                  fire(500, 8000, "draw") == "建议改用衰减")
            check("注入 + 大部分在 10kHz 以下也提醒",
                  fire(3000, 12000, "draw") == "建议改用衰减")
            check("注入 + 只有小部分越界 → 不打扰",
                  fire(8000, 18000, "draw") is None)

            # 同一文件、同一类建议只提一次
            shown.clear()
            page._maybe_advise_mode()                # noqa: SLF001
            check("同类建议不重复弹", not shown)

            # 选"立即切换"要真的换过去
            mw.MessageBox = _YesBox
            page._advised_kinds.clear()              # noqa: SLF001
            page.engrave_segment.setCurrentItem("cut")
            page.low_freq_spin.setValue(15500)
            page.high_freq_spin.setValue(20000)
            QApplication.processEvents()
            page._maybe_advise_mode()                # noqa: SLF001
            check("选「立即切换」后印法已变",
                  page.engrave_segment.currentRouteKey() == "draw")
        finally:
            mw.MessageBox = original
            page.engrave_segment.setCurrentItem("cut")
            page._advised_kinds.clear()              # noqa: SLF001
            page.low_freq_spin.setValue(1100)
            page.high_freq_spin.setValue(9900)
            QApplication.processEvents()

    def step_text_multiline_and_dblclick() -> None:
        """文字支持换行；双击水印框能直接落到输入上。"""
        page.text_edit.setPlainText("AB")
        QApplication.processEvents()
        single_line = page.spectrum._pattern.shape[0]  # noqa: SLF001

        page.text_edit.setPlainText("AB\nCD")
        QApplication.processEvents()
        two_lines = page.spectrum._pattern.shape[0]  # noqa: SLF001
        check("文字支持换行（图案变高）", two_lines > single_line)

        # 双击水印框 → 切回文字模式并全选内容。
        # 这里打的是真实双击事件：pyqtgraph 的 ROI 不走 Qt 的
        # mouseDoubleClickEvent，直接 emit 信号会让这条测试形同虚设。
        view = page.spectrum
        roi = view._roi  # noqa: SLF001
        center = roi.pos() + roi.size() / 2
        inside = view._plot.mapFromScene(  # noqa: SLF001
            view._plot.getViewBox().mapViewToScene(center)  # noqa: SLF001
        )

        # 图片模式下双击不该弹文字框，也不该擅自切回文字模式
        page.pattern_segment.setCurrentItem("image")
        QApplication.processEvents()
        check("双击前没有就地编辑框", not view.inline_edit_active())
        QTest.mouseDClick(
            view._plot.viewport(),  # noqa: SLF001
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, inside,
        )
        QApplication.processEvents()
        check("图片模式下双击不弹文字框", not view.inline_edit_active())
        check("图片模式保持不变",
              page.pattern_segment.currentRouteKey() == "image")

        # 文字模式下才开框内编辑
        page.pattern_segment.setCurrentItem("text")
        QApplication.processEvents()
        QTest.mouseDClick(
            view._plot.viewport(),  # noqa: SLF001
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, inside,
        )
        QApplication.processEvents()
        check("文字模式下双击弹出编辑框", view.inline_edit_active())

        # 框内改字要同步回右侧输入框，并重算图案
        editor = view._inline_edit  # noqa: SLF001
        editor.setPlainText("框内改的")
        QApplication.processEvents()
        check("框内输入同步到右侧",
              page.text_edit.toPlainText() == "框内改的")

        QTest.keyClick(editor, Qt.Key.Key_Escape)
        QApplication.processEvents()
        check("Esc 后编辑框关闭", not view.inline_edit_active())
        check("结束后内容保留", page.text_edit.toPlainText() == "框内改的")

        # 框外双击不该有反应
        QTest.mouseDClick(
            view._plot.viewport(),  # noqa: SLF001
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            inside + QPoint(260, 130),
        )
        QApplication.processEvents()
        check("框外双击不弹编辑框", not view.inline_edit_active())

        # 「按文件名生成」已经撤掉
        check("不再有「按文件名生成」开关",
              not hasattr(page, "from_filename_check"))

        page.pattern_segment.setCurrentItem("text")
        page.text_edit.setPlainText("WATERMARK")
        QApplication.processEvents()

    def step_drag_spinbox() -> None:
        """位置与尺寸里的数值框：按住左右拖动即可调值。"""
        box = page.low_freq_spin
        editor = box.lineEdit()
        original = box.value()
        step = box.singleStep()

        def send(kind: QEvent.Type, x: float) -> None:
            event = QMouseEvent(
                kind, QPointF(x, 14.0), QPointF(x, 14.0),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(editor, event)

        check("数值框用的是左右拖动光标",
              editor.cursor().shape() == Qt.CursorShape.SizeHorCursor)

        # 右移 30px → 10 步
        send(QEvent.Type.MouseButtonPress, 50.0)
        send(QEvent.Type.MouseMove, 80.0)
        QApplication.processEvents()
        moved = box.value() - original
        send(QEvent.Type.MouseButtonRelease, 80.0)
        check("右拖数值框可增大", abs(moved - 10 * step) < 1e-6)

        # 阈值内的微小位移不该改值（否则想点进去输入时会误触）
        settled = box.value()
        send(QEvent.Type.MouseButtonPress, 50.0)
        send(QEvent.Type.MouseMove, 51.0)
        QApplication.processEvents()
        check("微小位移不改变数值", box.value() == settled)
        send(QEvent.Type.MouseButtonRelease, 51.0)

        box.setValue(original)          # 复位，别影响后续步骤
        QApplication.processEvents()

    def step_sensitive_band() -> None:
        """拖动印章框时提示 4k~16kHz 敏感频段，停手后自动收起。"""
        from ..ui.spectrogram import _SENSITIVE_HIDE_DELAY_MS, SENSITIVE_BANDS

        view = page.spectrum
        roi = view._roi  # noqa: SLF001

        check("初始不显示敏感频段提示", not view.sensitive_band_visible())

        # 模拟拖动手柄（框体平移不会发 sigRegionChangeStarted，也一并覆盖）
        roi.handleMoveStarted()
        handle = [h for h in roi.handles
                  if h["pos"].x() == 1.0 and h["pos"].y() == 0.5][0]
        origin = roi.mapToParent(handle["pos"] * roi.state["size"])
        roi.movePoint(handle["item"], origin + QPointF(30, 0), finish=False)
        QApplication.processEvents()

        check("拖动时提示出现", view.sensitive_band_visible())

        # 每一片色带都要落在设定的频率范围内，且横贯整个时间轴。
        # 低频那片过去是漏掉的 —— 它才是挖空最容易听出差别的地方。
        rects = view.sensitive_band_rects()
        check("敏感频段片数与设定一致", len(rects) == len(SENSITIVE_BANDS))

        spanned = True
        for rect, (low, high) in zip(rects, SENSITIVE_BANDS):
            if abs(rect.y() - low) > 1.0 or abs(rect.y() + rect.height() - high) > 1.0:
                spanned = False
        labels = "、".join(f"{int(lo)}~{int(hi)}Hz" for lo, hi in SENSITIVE_BANDS)
        check(f"色带覆盖 {labels}", spanned)
        check("色带横贯整个时间轴",
              all(abs(r.x()) < 1e-6
                  and abs(r.width() - view.duration_sec()) < 1e-6
                  for r in rects))
        # 低频那一头必须标进去 —— 挖空在低频的影响比高频更明显
        check("低频已纳入标记范围",
              any(lo <= 100.0 and hi >= 5000.0 for lo, hi in SENSITIVE_BANDS))

        QTest.qWait(_SENSITIVE_HIDE_DELAY_MS + 300)
        check("停手后提示自动收起", not view.sensitive_band_visible())

    def step_roi_handles() -> None:
        """八个手柄：四角等比、四边单向，且位置必须跟着框走。"""
        roi = page.spectrum._roi  # noqa: SLF001
        handles = roi.handles
        check("共 8 个缩放手柄", len(handles) == 8)

        corner = [h for h in handles if h["pos"].x() != 0.5 and h["pos"].y() != 0.5]
        edge = [h for h in handles if h["pos"].x() == 0.5 or h["pos"].y() == 0.5]
        check("四角手柄存在", len(corner) == 4)
        check("四边手柄存在", len(edge) == 4)
        check("四角锁定宽高比", all(h["lockAspect"] for h in corner))
        # 边手柄靠 xoff/yoff 只放开一个轴
        check("边手柄为单向缩放",
              all(h.get("xoff") or h.get("yoff") for h in edge))

        # 手柄位置 = pos * size，用**相对容差**判定 —— 绝对容差在 x 方向
        # （框宽只有 1~2 秒）会宽到掩盖真实错位。
        def misaligned() -> int:
            size = roi.size()
            count = 0
            for h in handles:
                want = h["pos"] * size
                got = h["item"].pos()
                if (abs(got.x() - want.x()) > max(0.005, abs(want.x()) * 0.002)
                        or abs(got.y() - want.y()) > max(0.5, abs(want.y()) * 0.002)):
                    count += 1
            return count

        check("手柄位置跟随框尺寸", misaligned() == 0)

        # 调整起点 / 时长 / 频率之后必须依然贴合 —— 这是用户实际踩到的路径
        page.start_spin.setValue(0.40)
        QApplication.processEvents()
        check("改起点后手柄仍贴合", misaligned() == 0)

        page.duration_spin.setValue(0.45)
        QApplication.processEvents()
        check("改时长后手柄仍贴合", misaligned() == 0)

        page.low_freq_spin.setValue(500)
        page.high_freq_spin.setValue(15000)
        QApplication.processEvents()
        check("改频率后手柄仍贴合", misaligned() == 0)

        points = [h["item"].scenePos() for h in handles]
        min_gap = min(
            ((points[i].x() - points[j].x()) ** 2
             + (points[i].y() - points[j].y()) ** 2) ** 0.5
            for i in range(len(points)) for j in range(i + 1, len(points))
        )
        check("八个手柄互不重叠", min_gap > 5.0)

    def step_drop_files() -> None:
        """模拟把文件拖到窗口上。"""
        from PyQt6.QtCore import QMimeData, QPointF, QUrl
        from PyQt6.QtGui import QDropEvent

        page._clear_files()  # noqa: SLF001
        QApplication.processEvents()

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in inputs])
        event = QDropEvent(
            QPointF(40.0, 40.0),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        page.dropEvent(event)
        QApplication.processEvents()

        # 预览在后台线程解码，等它落地再断言
        worker = page._preview_worker  # noqa: SLF001
        if worker is not None and worker.isRunning():
            worker.wait(10000)
        QApplication.processEvents()

        check("拖放后文件进入列表", page.file_list.count() == 2)
        check("拖放后自动加载预览", page.spectrum.has_audio())

        # 拖到频谱区也应生效（走 filesDropped 信号）
        page._clear_files()  # noqa: SLF001
        QApplication.processEvents()
        page.spectrum.filesDropped.emit(list(inputs))
        QApplication.processEvents()
        check("频谱区拖放同样生效", page.file_list.count() == 2)

    def step_axis_units() -> None:
        """单位写在刻度上，不挂轴标签。"""
        plot = page.spectrum._plot  # noqa: SLF001
        bottom = plot.getAxis("bottom")
        left = plot.getAxis("left")

        check("底部没有轴标签", bottom.labelText == "")
        check("左侧没有轴标签", left.labelText == "")

        check("时间刻度带 s",
              bottom.tickStrings([0, 50, 100], 1.0, 50) == ["0s", "50s", "100s"])
        # 刻度写成紧凑形式（5k 而非 5kHz），左侧边距才压得下去
        check("频率刻度用紧凑写法",
              left.tickStrings([500, 5000, 15000], 1.0, 5000)
              == ["500", "5k", "15k"])
        # 顺带验一下 scale 不为 1 时也对：pyqtgraph 可能把 scale 设成 1000
        check("刻度格式化尊重 scale",
              left.tickStrings([0.5, 5, 15], 1000.0, 5000)
              == ["500", "5k", "15k"])

        # 左侧与底部的边距要一样宽，频谱区才是个规整矩形
        page.spectrum._sync_axis_margins()  # noqa: SLF001
        QTest.qWait(60)
        QApplication.processEvents()
        left_w = plot.getAxis("left").width()
        bottom_h = plot.getAxis("bottom").height()
        check(f"两侧边距等宽（{left_w:.0f} / {bottom_h:.0f}）",
              left_w > 0 and abs(left_w - bottom_h) <= 1.5)
        # 贴边：只够放刻度文字，不留多余空隙
        check(f"边距贴边（{left_w:.0f}px）", left_w <= 28)

    def step_output_options() -> None:
        """高级参数里的输出选项：合成视频开关 + 两种容器格式。"""
        from ..core.params import AudioFormat, VideoFormat

        check("合成视频默认开启", page.mux_switch.isChecked())
        check("开启时视频格式可用", page.video_format_combo.isEnabled())
        check("合成开关进到输出设置",
              page._collect_output().mux_video is True)  # noqa: SLF001

        page.mux_switch.setChecked(False)
        QApplication.processEvents()
        check("关掉后视频格式置灰",
              not page.video_format_combo.isEnabled())

        page.mux_switch.setChecked(True)
        QApplication.processEvents()
        check("重新开启后视频格式恢复",
              page.video_format_combo.isEnabled())

        page.audio_format_combo.setCurrentIndex(2)      # .mp3
        QApplication.processEvents()
        check("音频格式进到输出设置",
              page._collect_output().audio_format is AudioFormat.MP3)  # noqa: SLF001

        page.video_format_combo.setCurrentIndex(1)      # .mp4
        QApplication.processEvents()
        check("视频格式进到输出设置",
              page._collect_output().video_format is VideoFormat.MP4)  # noqa: SLF001

        # 复位，别影响后续步骤
        page.mux_switch.setChecked(True)
        page.audio_format_combo.setCurrentIndex(0)
        page.video_format_combo.setCurrentIndex(0)
        QApplication.processEvents()
        check("复位后回到与源一致",
              page._collect_output().audio_format is AudioFormat.SOURCE)  # noqa: SLF001

    def step_action_buttons() -> None:
        """两个操作按钮：只处理当前 / 处理所有。"""
        check("存在「只处理当前文件」按钮",
              page.process_current_button.text() == "只处理当前文件")
        check("存在「处理所有文件」按钮",
              page.process_all_button.text() == "处理所有文件")

        # 运行态：主按钮借用为取消，次要按钮锁住
        page._set_running(True)  # noqa: SLF001
        QApplication.processEvents()
        check("运行中主按钮变取消", page.process_all_button.text() == "取消")
        check("运行中禁用「只处理当前」",
              not page.process_current_button.isEnabled())

        page._set_running(False)  # noqa: SLF001
        QApplication.processEvents()
        check("结束后按钮复位",
              page.process_all_button.text() == "处理所有文件"
              and page.process_current_button.isEnabled())

        # 拦截 _start_batch，只验证它拿到的是哪些文件（不真的起后台任务）
        captured: list[list[str]] = []
        original = page._start_batch  # noqa: SLF001
        page._start_batch = lambda files: captured.append(list(files))  # noqa: SLF001
        try:
            page.file_list.setCurrentRow(1)
            QApplication.processEvents()
            page._on_process_current()  # noqa: SLF001
            check("「只处理当前」只交出一个文件",
                  captured == [[page._files[1]]])

            captured.clear()
            page._on_process_all()  # noqa: SLF001
            check("「处理所有」交出整批",
                  captured == [list(page._files)])
        finally:
            page._start_batch = original  # noqa: SLF001

    steps = [
        step_empty, step_add_files, step_after_preview, step_image_mode,
        step_loop_on, step_help_page, step_help_snap, step_collect_job,
        step_loop_consistency, step_region_bounds, step_zoom_behavior,
        step_pattern_overlay,
        step_engraving_mode, step_mode_advice, step_text_multiline_and_dblclick,
        step_drag_spinbox,
        step_sensitive_band, step_roi_handles, step_drop_files,
        step_axis_units, step_output_options, step_action_buttons,
    ]

    def run_next() -> None:
        if steps:
            fn = steps.pop(0)
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                checks.append((f"{fn.__name__} 抛异常: {exc}", False))
            QTimer.singleShot(700, run_next)
        else:
            app.quit()

    QTimer.singleShot(700, run_next)
    app.exec()

    shutil.rmtree(workspace, ignore_errors=True)

    print("=" * 68)
    print("界面自检")
    print("=" * 68)
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")
    print("-" * 68)
    print(f"截图 {len(shots)} 张 -> {SHOT_DIR}")
    for path in shots:
        print(f"    {os.path.basename(path)}")

    passed = sum(1 for _, ok in checks if ok)
    print(f"结果：{passed}/{len(checks)} 项通过")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
