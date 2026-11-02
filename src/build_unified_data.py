"""统一数据管线的入口脚本（与仓库既有 `python src/step0_*.py` 的调用习惯保持一致）。

用法::

    python src/build_unified_data.py download
    python src/build_unified_data.py build
    python src/build_unified_data.py qa
    python src/build_unified_data.py verify
    python src/build_unified_data.py all

等价的模块调用方式（需在项目根目录执行）::

    python -m src.data.pipeline build
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接以脚本方式运行时，sys.path[0] 是 src/，据此把 src 加入路径以导入 data 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
