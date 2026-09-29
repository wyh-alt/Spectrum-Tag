"""SpectrumTag 批量版 —— 把图案/文字以频谱水印的形式批量印到音频上。

模块划分：

* :mod:`spectrumtag_batch.core`   —— 与 UI 无关的算法核心（STFT/OLA、图案栅格化）
* :mod:`spectrumtag_batch.media`  —— 音频/视频的读写与 ffmpeg 调用
* :mod:`spectrumtag_batch.batch`  —— 批量任务调度与进度汇报
* :mod:`spectrumtag_batch.ui`     —— PyQt6 界面
"""

__version__ = "0.1.0"
