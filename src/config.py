from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 目录定义
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# 确保目录存在
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 路径配置
MAT_PATH = RAW_DIR / "YelpChi.mat"
REVIEW_PARQUET = PROCESSED_DIR / "review_level.parquet"    # 中间产物：评论级
ACCOUNT_PARQUET = PROCESSED_DIR / "account_level.parquet"  # 中间产物：账号级
SAMPLE_CSV = PROCESSED_DIR / "yelpchi_1000_accounts.csv"   # 最终交付物：1000条样本