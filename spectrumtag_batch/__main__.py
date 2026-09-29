"""支持 ``python -m spectrumtag_batch`` 直接启动。"""

import sys

from .main import main

sys.exit(main())
