"""统一数据层：把 YelpChi / YelpNYC / YelpZip / FraudSquad（三模型各自成表）四份公开数据
与自建注入集合成同一套评论级 schema，供特征、检测、图方法、评估四条下游链路共用。

本包只负责「原始数据 -> 统一格式」，不包含任何检测模型或特征工程逻辑。
论文口径的 6 张表（Yelp 三套 + FraudSquad 三模型）均已到位；唯一缺口是 YelpChi 无评论文本。
数据血缘与核验过程见 docs/data/provenance.md。
"""

from .schema import (
    ACCOUNT_COLUMNS,
    ACCOUNT_LABEL_BASIS,
    CANONICAL_COLUMNS,
    DATASET_SPECS,
    LABEL_SOURCE_VALUES,
    PUBLIC_DATASETS,
    VIEW_ONLY_DATASETS,
    DatasetSpec,
    SchemaError,
    build_account_table,
    empty_reviews_frame,
    label_from_yelp,
    validate_accounts,
    validate_reviews,
)
from .readers import read_yelp_pair
from .load import (
    iter_reviews,
    list_accounts,
    list_datasets,
    load_accounts,
    load_all,
    load_dataset,
)

__all__ = [
    "ACCOUNT_COLUMNS",
    "ACCOUNT_LABEL_BASIS",
    "CANONICAL_COLUMNS",
    "DATASET_SPECS",
    "LABEL_SOURCE_VALUES",
    "PUBLIC_DATASETS",
    "VIEW_ONLY_DATASETS",
    "DatasetSpec",
    "SchemaError",
    "build_account_table",
    "empty_reviews_frame",
    "label_from_yelp",
    "read_yelp_pair",
    "validate_accounts",
    "validate_reviews",
    "load_dataset",
    "load_all",
    "load_accounts",
    "list_datasets",
    "list_accounts",
    "iter_reviews",
]
