"""统一数据层：把 YelpChi / YelpNYC / YelpZip / FraudSquad-Amazon 四份公开数据
整理成同一套评论级 schema，供特征、检测、图方法、评估四条下游链路共用。

本包只负责「原始数据 -> 统一格式」，不包含任何检测模型或特征工程逻辑。
"""

from .schema import (
    CANONICAL_COLUMNS,
    DATASET_SPECS,
    LABEL_SOURCE_VALUES,
    DatasetSpec,
    SchemaError,
    empty_reviews_frame,
    validate_reviews,
)
from .load import iter_reviews, list_datasets, load_all, load_dataset

__all__ = [
    "CANONICAL_COLUMNS",
    "DATASET_SPECS",
    "LABEL_SOURCE_VALUES",
    "DatasetSpec",
    "SchemaError",
    "empty_reviews_frame",
    "validate_reviews",
    "load_dataset",
    "load_all",
    "list_datasets",
    "iter_reviews",
]
