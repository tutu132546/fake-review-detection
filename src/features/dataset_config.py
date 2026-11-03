# -*- coding: utf-8 -*-
"""数据集配置：把“每个数据集怎么读、怎么算、写到哪”集中成显式声明。

设计约束（方案 E-3 / F）：
  - 下游脚本不得再出现写死的 YelpChi 路径；
  - 每个数据集单独 fit scaler，禁止跨数据集联合 fit；
  - 输出列顺序、文件名、manifest 路径统一在此声明；
  - YelpNYC 缺失，只产出 MISSING 占位，不造假表。
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PROCESSED_DIR, RESULTS_DIR

FEATURES_DIR = PROCESSED_DIR / "features"
MANIFEST_DIR = RESULTS_DIR / "feature_manifests"
UNIFIED_DIR = PROCESSED_DIR / "unified"

FEATURE_LIST = ['n_reviews', 'active_days', 'daily_avg_reviews', 'max_daily_reviews',
                'burstiness', 'rating_deviation', 'extreme_ratio']

# 输出列顺序固定（F.2）；5 张表不得按数据集增减列
OUTPUT_COLUMNS = ['dataset', 'global_user_id', 'user_id', 'account_is_fake', 'cohort',
                  'is_injected', 'text_available', 'burstiness_missing'] \
                 + FEATURE_LIST + [f'{c}_z' for c in FEATURE_LIST]

# 模型输入白名单（F.8）：7 个 _z + burstiness_missing
MODEL_INPUT_COLUMNS = [f'{c}_z' for c in FEATURE_LIST] + ['burstiness_missing']


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    mode: str                      # account_base（YelpChi 自注入）| unified_reviews | missing
    input_path: Optional[Path]
    output_path: Path
    scaler_path: Path
    manifest_path: Path
    rating_min: float
    rating_max: float
    text_available: int
    scaler_fit_population: str
    expected_rows: Optional[int] = None
    expected_fake_review_rate: Optional[float] = None
    fake_raw_values: Optional[Tuple[str, ...]] = None   # 上游原始 label 中"假评"的取值
    real_raw_values: Optional[Tuple[str, ...]] = None    # 上游原始 label 中"真评"的取值
    require_label_source: bool = False
    new_account_cohort: str = 'native'   # 有生成评论的账号人群名（FraudSquad=hijacked_llm）
    missing_reason: Optional[str] = None

    @property
    def expected_fake_rate(self):
        """别名（B1 字段命名）：与 expected_fake_review_rate 同义。"""
        return self.expected_fake_review_rate

    @property
    def label_direction_verified(self) -> bool:
        """三要素齐全才算方向已声明；FraudSquad 未声明时恒为 False。"""
        return (self.fake_raw_values is not None and self.real_raw_values is not None
                and self.expected_fake_review_rate is not None)

    def model_input_columns(self) -> Tuple[str, ...]:
        return tuple(MODEL_INPUT_COLUMNS)


def _spec(dataset_id: str, display_name: str, mode: str, input_path: Optional[Path],
          text_available: int, fit_population: str, *, rating_min=1.0, rating_max=5.0,
          expected_rows=None, expected_fake_review_rate=None,
          fake_raw_values=None, real_raw_values=None,
          require_label_source=False, new_account_cohort='native',
          missing_reason=None) -> DatasetSpec:
    name = f'feature_table_{dataset_id}'
    return DatasetSpec(
        dataset_id=dataset_id,
        display_name=display_name,
        mode=mode,
        input_path=input_path,
        output_path=FEATURES_DIR / f'{name}.csv',
        scaler_path=FEATURES_DIR / f'scaler_{dataset_id}.json',
        manifest_path=MANIFEST_DIR / f'{name}.json',
        rating_min=rating_min,
        rating_max=rating_max,
        text_available=text_available,
        scaler_fit_population=fit_population,
        expected_rows=expected_rows,
        expected_fake_review_rate=expected_fake_review_rate,
        fake_raw_values=fake_raw_values,
        real_raw_values=real_raw_values,
        require_label_source=require_label_source,
        new_account_cohort=new_account_cohort,
        missing_reason=missing_reason,
    )


DATASETS = {s.dataset_id: s for s in [
    # YelpChi 自注入实验：直接消费账号级 augmented 表（方案 11.5 兜底路径），不依赖 review 级数据
    _spec('yelpchi', 'YelpChi（metadata 版，含 300 自注入）', 'account_base',
          PROCESSED_DIR / 'augmented_accounts_v2.parquet', 0, 'clean_native',
          expected_rows=38063),
    _spec('yelpzip', 'YelpZip', 'unified_reviews', UNIFIED_DIR / 'yelpzip.parquet', 1,
          'all_accounts', expected_fake_review_rate=0.1322,
          fake_raw_values=('-1',), real_raw_values=('1',)),
    _spec('fraudsquad_llama3', 'FraudSquad-Amazon-Llama3', 'unified_reviews',
          UNIFIED_DIR / 'fraudsquad_llama3.parquet', 1, 'all_accounts',
          require_label_source=True, new_account_cohort='hijacked_llm'),
    _spec('fraudsquad_qwen2', 'FraudSquad-Amazon-Qwen2', 'unified_reviews',
          UNIFIED_DIR / 'fraudsquad_qwen2.parquet', 1, 'all_accounts',
          require_label_source=True, new_account_cohort='hijacked_llm'),
    _spec('fraudsquad_qwen_dsr1', 'FraudSquad-Amazon-Qwen-DSR1', 'unified_reviews',
          UNIFIED_DIR / 'fraudsquad_qwen_dsr1.parquet', 1, 'all_accounts',
          require_label_source=True, new_account_cohort='hijacked_llm'),
    _spec('yelpnyc', 'YelpNYC', 'missing', None, 1, 'all_accounts',
          missing_reason='官方数据不可得（叶歆澜已确认缺失），不构造替代数据集'),
]}

DATASET_IDS = list(DATASETS.keys())


def get_spec(dataset_id: str, input_path: Optional[str] = None) -> DatasetSpec:
    if dataset_id not in DATASETS:
        raise KeyError(f"未知 dataset_id={dataset_id!r}，可选：{', '.join(DATASET_IDS)}")
    spec = DATASETS[dataset_id]
    if input_path is not None:
        spec = DatasetSpec(**{**spec.__dict__, 'input_path': Path(input_path)})
    return spec


def ensure_dirs() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
