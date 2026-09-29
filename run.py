"""启动频谱水印批量处理程序。

    python run.py

也可以直接用模块方式启动：``python -m spectrumtag_batch``
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectrumtag_batch.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
