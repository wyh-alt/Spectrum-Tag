"""程序入口。

启动方式：
    python -m spectrumtag_batch
    python run.py

打包成 exe 后走的是同一条路径，只是前面多了一层：PyInstaller 单文件模式要先
把内容解压到临时目录才能跑起 Python，这段时间归 bootloader 的启动画面管；
解释器就绪后由 Qt 的启动画面接替，两者视觉一致，交接处不留黑屏。
"""

from __future__ import annotations

import ctypes
import os
import sys
import time

# 启动画面的最短显示时长。加载太快时它一闪而过，看起来反倒像出了故障
_MIN_SPLASH_SECONDS = 0.7

_APP_NAME = "频谱水印生成"


def _set_taskbar_identity() -> None:
    """让 Windows 任务栏用本程序的图标，而不是 python.exe 的。

    必须在创建 QApplication 之前调用。
    """
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "spectrumtag.batch.app.1"
        )
    except Exception:  # noqa: BLE001 - 非 Windows 或受限环境，忽略即可
        pass


def _resource_path(name: str) -> str | None:
    """在开发态和打包态下都能找到随程序分发的资源。

    PyInstaller 会把资源解压到 ``sys._MEIPASS`` 指向的临时目录，那儿要优先找；
    开发态则直接看模块所在目录。
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, name)
    return candidate if os.path.exists(candidate) else None


_STARTUP_LOG = os.path.join(
    os.environ.get("TEMP", "."), "spectrumtag_startup.log"
)


def _reset_startup_log() -> None:
    """每次启动重写日志。

    打包后出问题（比如某个模块导不进来导致窗口根本没建起来）时，界面上
    什么都不会有 —— 这份日志是唯一能看出卡在哪一步的线索。
    """
    try:
        with open(_STARTUP_LOG, "w", encoding="utf-8") as fh:
            fh.write("main() 进入\n")
    except OSError:
        pass


def _log_startup(note: str) -> None:
    """追加一行启动记录。写不进去就算了，不能因为日志把程序拖垮。"""
    try:
        with open(_STARTUP_LOG, "a", encoding="utf-8") as fh:
            fh.write(note + "\n")
    except OSError:
        pass


def _release_pyi_splash() -> None:
    """关掉 PyInstaller 的 bootloader 启动画面。

    Qt 的启动画面已经顶上来了，留着那一张会一直盖在最上层、把主窗口整个挡住。
    先走官方的 ``pyi_splash``；它在某些打包配置下导不进来，那就退回 Win32 ——
    bootloader 那张图本质上是个标题为 "tk" 的窗口，找到关掉即可。
    """
    if _close_via_pyi_splash():
        return
    _close_tk_window()


def _close_via_pyi_splash() -> bool:
    try:
        import pyi_splash  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        _log_startup(f"pyi_splash 不可用：{exc!r}")
        return False
    try:
        if not pyi_splash.is_alive():
            _log_startup("pyi_splash 认为启动画面已关闭")
            return True
        pyi_splash.close()
        _log_startup("已通过 pyi_splash 关闭启动画面")
        return True
    except Exception as exc:  # noqa: BLE001
        _log_startup(f"pyi_splash.close() 失败：{exc!r}")
        return False


def _close_tk_window() -> None:
    """兜底：直接找到本进程里标题为 "tk" 的窗口并关掉。

    先按进程 id 过滤，免得误伤别的正在跑的 Tk 程序。
    """
    if os.name != "nt":
        return
    try:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        pid = kernel32.GetCurrentProcessId()
        found: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _visit(hwnd, _lparam):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid:
                length = user32.GetWindowTextLengthW(hwnd)
                if length:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if buf.value == "tk":
                        found.append(hwnd)
            return True

        user32.EnumWindows(_visit, 0)
        for hwnd in found:
            user32.PostMessageW(hwnd, 0x0010, 0, 0)      # WM_CLOSE
        _log_startup(f"Win32 兜底：找到并关闭 {len(found)} 个 tk 窗口")
    except Exception as exc:  # noqa: BLE001
        _log_startup(f"Win32 兜底失败：{exc!r}")


def main() -> int:
    _reset_startup_log()
    _log_startup(f"frozen={getattr(sys, 'frozen', False)}  exe={sys.executable}")

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QIcon, QPixmap
    from PyQt6.QtWidgets import QApplication, QSplashScreen

    # 高 DPI：必须在 QApplication 之前设置，否则缩放会取整导致模糊
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    _set_taskbar_identity()

    app = QApplication(sys.argv)
    app.setApplicationName(_APP_NAME)
    app.setOrganizationName("SpectrumTagBatch")

    from qfluentwidgets import Theme, setTheme

    setTheme(Theme.AUTO)

    icon_path = _resource_path("icon.ico") or _resource_path("icon.png")
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    _log_startup("QApplication 就绪，准备启动画面")
    started = time.monotonic()
    splash = _build_splash(QSplashScreen, QPixmap, QColor, Qt)
    _log_startup(f"Qt 启动画面: {'已显示' if splash else '未创建'}")
    _advance_splash(splash, 0.12, "正在加载界面模块…", app)
    _release_pyi_splash()

    _log_startup("开始导入主窗口模块")
    from .ui.main_window import MainWindow

    _log_startup("主窗口模块已导入，开始构建")
    _advance_splash(splash, 0.5, "正在构建频谱视图…", app)

    window = MainWindow()
    _log_startup("主窗口构建完成")
    _advance_splash(splash, 0.92, "即将就绪…", app)
    if icon_path:
        window.setWindowIcon(QIcon(icon_path))

    remaining = _MIN_SPLASH_SECONDS - (time.monotonic() - started)
    if splash is not None and remaining > 0:
        time.sleep(remaining)

    _center_on_screen(window)
    _log_startup("准备 show()")
    window.show()
    _log_startup(
        f"show() 完成 isVisible={window.isVisible()} "
        f"size={window.width()}x{window.height()}"
    )
    if splash is not None:
        splash.finish(window)      # 主窗口露脸之后才收起启动画面
        _log_startup("启动画面已收起")
    _log_startup("进入事件循环")
    return app.exec()


def _center_on_screen(window) -> None:
    """把主窗口摆到主屏正中。

    Qt 默认让窗口管理器决定初始位置，多屏或某些 Windows 配置下会落到右下角。

    这里固定用主屏，而不是 ``window.screen()``：窗口刚建出来时位置还没定，
    跟着"当前所在屏"走会落到哪块屏全看窗口管理器的心情 —— 实测多屏环境下
    它可能指到左边那块副屏（x 为负），结果窗口跑到屏幕外。
    用的是"可用区域"（已扣掉任务栏）而不是整块屏幕。
    """
    from PyQt6.QtWidgets import QApplication

    screen = QApplication.primaryScreen()
    if screen is None:
        return
    area = screen.availableGeometry()
    size = window.size()
    window.move(
        area.x() + max(0, (area.width() - size.width()) // 2),
        area.y() + max(0, (area.height() - size.height()) // 2),
    )


def _build_splash(splash_cls, pixmap_cls, color_cls, qt):
    """搭出 Qt 这一层的启动画面；图缺失就返回 None，不影响启动。

    底图（spec 里叫 ``splash_ui.png``）已经画好 logo、标题、描述和进度槽，
    这里只往上叠进度与当前加载项 —— 那两样是动态的，进不了静态图。

    打包态读改名后的图，开发态直接读源文件。
    """
    path = _resource_path("splash_ui.png") or _resource_path("splash.png")
    if not path:
        return None
    pixmap = pixmap_cls(path)
    if pixmap.isNull():
        return None

    class _StartupSplash(splash_cls):
        """带进度条和当前加载项的启动画面。"""

        # 与底图里画好的元素对齐
        _MARGIN = 30
        _TRACK_Y = 120
        _TRACK_H = 4
        _STAGE_Y = 132

        def __init__(self, pix):
            super().__init__(pix)
            self._progress = 0.0
            self._stage = "正在启动…"
            # 底图是圆角的，去掉窗口背景才不会有方块底
            self.setAttribute(qt.WidgetAttribute.WA_TranslucentBackground)

        def set_progress(self, value: float, stage: str) -> None:
            self._progress = max(0.0, min(1.0, float(value)))
            self._stage = stage
            self.repaint()

        def drawContents(self, painter) -> None:
            super().drawContents(painter)
            left = self._MARGIN
            right = self.width() - self._MARGIN

            # 进度：底槽由底图提供，这里只画已经走完的那一段
            filled = left + int((right - left) * self._progress)
            if filled > left:
                painter.setPen(qt.PenStyle.NoPen)
                painter.setBrush(color_cls(0, 178, 179))
                painter.drawRoundedRect(
                    left, self._TRACK_Y, filled - left, self._TRACK_H, 2, 2
                )

            painter.setPen(color_cls(130, 130, 142))
            painter.drawText(
                left, self._STAGE_Y, right - left, 16,
                qt.AlignmentFlag.AlignLeft, self._stage,
            )

    splash = _StartupSplash(pixmap)
    splash.show()
    splash.repaint()
    return splash


def _advance_splash(splash, value: float, stage: str, app) -> None:
    """推进启动画面。

    顺手跑一轮事件 —— 不跑的话画面只是改了数据，屏幕上还是上一帧。
    """
    if splash is None:
        return
    splash.set_progress(value, stage)
    app.processEvents()


if __name__ == "__main__":
    sys.exit(main())
