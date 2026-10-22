from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据路径
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 文件路径
MAT_PATH = RAW_DIR / "YelpChi.mat"
RAW_METADATA_PATH = RAW_DIR / "metadata.txt"
RAW_CSV_PATH = RAW_DIR / "YelpChi_reviews.csv"