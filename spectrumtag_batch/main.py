"""程序入口。

启动方式：
    python -m spectrumtag_batch
    python run.py
"""

from __future__ import annotations

import ctypes
import os
import sys


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


def _icon_path() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("icon.ico", "icon.png"):
        candidate = os.path.join(here, name)
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    # 高 DPI：必须在 QApplication 之前设置，否则缩放会取整导致模糊
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    _set_taskbar_identity()

    app = QApplication(sys.argv)
    app.setApplicationName("频谱水印生成")
    app.setOrganizationName("SpectrumTagBatch")

    from qfluentwidgets import Theme, setTheme

    setTheme(Theme.AUTO)

    icon_path = _icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    from .ui.main_window import MainWindow

    window = MainWindow()
    if icon_path:
        window.setWindowIcon(QIcon(icon_path))
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
