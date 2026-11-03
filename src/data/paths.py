"""项目内所有数据路径的唯一来源。

刻意不 import `src/config.py`：本包既可以用 `python -m src.data.pipeline` 运行，
也可以用 `python src/build_unified_data.py` 运行，两种方式下 sys.path 不同，
因此路径一律从 __file__ 反推，不依赖调用方的工作目录。
"""

from __future__ import annotations

from pathlib import Path

# src/data/paths.py -> src/data -> src -> 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# 原始数据按数据集分目录存放，避免历史文件与新下载文件互相覆盖
RAW_YELPCHI_DIR: Path = RAW_DIR / "yelpchi"
RAW_YELPNYC_DIR: Path = RAW_DIR / "yelpnyc"
RAW_YELPZIP_DIR: Path = RAW_DIR / "yelpzip"
RAW_FRAUDSQUAD_DIR: Path = RAW_DIR / "fraudsquad"

# 下载中转目录（可安全删除，不参与构建）
DOWNLOAD_CACHE_DIR: Path = RAW_DIR / "_downloads"

# 统一格式输出目录
UNIFIED_DIR: Path = PROCESSED_DIR / "unified"

# ---- 历史文件（保持不动，兼容 step0~step3 与 B/C/D 的既有脚本）----
LEGACY_YELPCHI_METADATA: Path = RAW_DIR / "metadata.txt"
LEGACY_YELPCHI_REVIEWS_CSV: Path = RAW_DIR / "YelpChi_reviews.csv"
LEGACY_INJECTION_LOG: Path = PROCESSED_DIR / "injection_log_v2.json"
LEGACY_FEATURE_TABLE: Path = PROCESSED_DIR / "feature_table.csv"


def ensure_dir(path: Path) -> Path:
    """创建目录（含父目录）并原样返回，方便链式调用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
