# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：单文件 exe，连带 ffmpeg 一起分发。

用法：
    python -m PyInstaller build_exe.spec --noconfirm

几个关键点：

* **原生依赖显式收集** —— soundfile 的 libsndfile、numpy/scipy 的 MKL/OpenBLAS
  都躺在包内的旁系目录里，PyInstaller 的静态分析找不到，必须 collect_all。
* **ffmpeg 走 vendor 路线** —— 不信任构建机 PATH。包管理器（Chocolatey、Scoop）
  装出来的是转发 shim，复制到别处就失效；这里只接受能独立跑起来的真实二进制。
* **启动画面** —— bootloader 这层用 PyInstaller 的 Splash（在 Python 解释器
  起来之前就能显示），Qt 起来后由程序内的 QSplashScreen 接替，两者用同一张图。
"""

import os
import shutil
import subprocess

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

PROJECT_ROOT = os.path.abspath(os.path.dirname(SPEC))


# --------------------------------------------------------------------------
#  原生依赖
# --------------------------------------------------------------------------
# 这里刻意用 collect_data_files + collect_dynamic_libs 而不是 collect_all：
# 后者会顺着依赖链把用到用不到的一并拖进来 —— 实测拉进了 torch、onnxruntime、
# numba、pandas、PySide6 等一大批，解压后从 250MB 涨到 900MB。
# 我们要的只是"这个包自己的数据文件和它自带的 DLL"。
datas = []
binaries = []
hiddenimports = []

for package in (
    "soundfile",      # libsndfile 的 DLL 在 _soundfile_data 下
    "numpy",
    "scipy",
    "PIL",
    "pyqtgraph",
    "qfluentwidgets",
):
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)

# 启动画面要出两份：EXE 的 Splash 用原名，datas 里那份必须换个名字 ——
# 同名的话 PyInstaller 只保留给 bootloader 的，程序内那层 QSplashScreen
# 就找不到图。datas 的第二项是"目标目录"而非"目标文件名"，所以这里先在
# 构建目录里复制出一个改名副本，再按普通资源放进去。
_splash_src = os.path.join(PROJECT_ROOT, "spectrumtag_batch", "splash.png")
_splash_ui = os.path.join(PROJECT_ROOT, "build", "splash_ui.png")
os.makedirs(os.path.dirname(_splash_ui), exist_ok=True)
shutil.copyfile(_splash_src, _splash_ui)

datas += [
    (_splash_ui, "."),
    (os.path.join("spectrumtag_batch", "icon.ico"), "."),
]

# tkinter 是 PyInstaller 启动画面的实现基础 —— 它的 bootloader 用 Tcl/Tk 画那张图。
# 我们的代码并不 import 它，静态分析因此不会收集 Tcl/Tk 的数据文件，
# 结果就是启动时弹 "could not find requirement _tk_data\tk\tcl"。这里显式列上。
#
# 注意别把 pyi_splash 加进来：启用 Splash 时 PyInstaller 会自动提供它，
# 手动指定反而会把这个 stub 也打进**没开 Splash 的构建**里，而它一加载就要
# 连 IPC，连不上就抛 KeyError: '_PYI_SPLASH_IPC'。
hiddenimports += ["tkinter"]


# --------------------------------------------------------------------------
#  ffmpeg：必须挑一个能独立运行的
# --------------------------------------------------------------------------
def _pick_ffmpeg():
    """找一个真的能跑的 ffmpeg。

    包管理器造的 shim 复制出去就找不到目标文件了，所以这里在临时目录里
    实际执行一次 -version 验证，验不过就换下一个候选。
    """
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = []

    env = os.environ.get("SPECTRUMTAG_FFMPEG")
    if env:
        candidates.append(env)

    bundled = os.path.join(PROJECT_ROOT, "spectrumtag_batch", "vendor", "ffmpeg", name)
    candidates.append(bundled)

    on_path = shutil.which("ffmpeg")
    if on_path and not any(
        marker in on_path.lower()
        for marker in ("chocolatey", "scoop", "winget", "msys64")
    ):
        candidates.append(on_path)

    candidates += [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]

    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            result = subprocess.run(
                [candidate, "-version"], capture_output=True, timeout=20, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and b"ffmpeg" in result.stdout.lower():
            return candidate
    return None


_ffmpeg = _pick_ffmpeg()
if _ffmpeg:
    # 落到 _MEIPASS/ffmpeg/ 下，与 video_io._bundled_candidate 的查找路径对应
    binaries.append((_ffmpeg, "ffmpeg"))
    print(f"[spec] 打包 ffmpeg: {_ffmpeg} ({os.path.getsize(_ffmpeg) / 1048576:.0f} MB)")
else:
    print("[spec] 警告：没找到可用的 ffmpeg，视频相关功能在打包后将不可用")


# --------------------------------------------------------------------------
#  构建
# --------------------------------------------------------------------------
a = Analysis(
    ["run.py"],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 这些一个都用不到，但可能被某些依赖间接牵进来。列出来是为了让构建
    # 直接把它们掐掉，而不是等打完了再发现包大了三倍。
    #
    # 两个**不能排除**的：
    #   * tkinter —— PyInstaller 的启动画面就是用 Tk 画的，排掉它 bootloader
    #     会报 "could not find requirement _tk_data" 而启动画面直接失效；
    #   * win32 / win32com —— qfluentwidgets 依赖的 qframelesswindow 拿它做
    #     Win32 窗口操作，排掉打包后一启动就 ModuleNotFoundError。
    excludes=[
        "matplotlib", "IPython", "pytest", "setuptools",
        "torch", "onnx", "onnxruntime", "numba", "llvmlite",
        "pandas", "pydantic", "cryptography", "aiohttp", "multidict",
        "google",
        "PySide6", "shiboken6", "pyside6_fluent_widgets",
        "librosa", "sklearn", "sympy", "networkx", "Cython",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# bootloader 层的启动画面：解释器就绪前就能显示，遮住解压那段时间
splash = Splash(
    os.path.join("spectrumtag_batch", "splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,          # 文字交给 Qt 那层显示中文，这里只出图
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    [],
    name="频谱水印生成",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX 对 Qt 的 DLL 有副作用，且拖慢构建
    runtime_tmpdir=None,
    console=False,          # 桌面程序，不要黑框
    disable_windowed_traceback=False,
    icon=os.path.join("spectrumtag_batch", "icon.ico"),
)
