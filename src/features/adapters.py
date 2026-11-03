# -*- coding: utf-8 -*-
"""统一数据层读取与标签守卫（方案 B.5 / E-3 / E-5）。

只做三件事：
  1. 把不同来源的数据读成统一 schema：user_id(str), item_id(str), rating, date, label∈{0,1}, label_source；
  2. 在入口断言标签编码合法，不合法直接报错（不静默翻转）；
  3. 返回特征计算所需的字段，label / label_source 只在账号级聚合时使用，绝不进入特征函数。
"""
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PROCESSED_DIR
from features.dataset_config import DatasetSpec, FEATURE_LIST
from features.manifest import sha256_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

UNIFIED_REQUIRED = ['user_id', 'item_id', 'rating', 'date', 'label']
LEGAL_LABEL_SOURCES = {'platform_filter', 'real_baseline', 'llm_generated', 'injected'}
MAX_BAD_RATIO = 0.005          # date / rating 解析失败上限（F.3）
MAX_ITEM_MISSING_RATIO = 0.001 # item_id 缺失上限（F.3）
LABEL_DIRECTION_UNVERIFIED = 'label_direction_unverified'
LABEL_RATE_TOLERANCE = 0.02    # 与基准假评率的容许偏差


def check_label_direction(raw_label, spec, tolerance=LABEL_RATE_TOLERANCE):
    """B1 标签方向守卫：原始取值为声明子集，且假率不得 > 50% 或明显偏离基准。

    返回 'passed'；方向未声明（FraudSquad）返回 'unverified'；反向或越界直接抛错。
    """
    if not spec.label_direction_verified:
        return LABEL_DIRECTION_UNVERIFIED
    values = pd.Series(raw_label).dropna().astype(str)
    observed = set(values.unique())
    declared = set(spec.fake_raw_values) | set(spec.real_raw_values)
    unknown = observed - declared
    if unknown:
        raise ValueError(f"{spec.dataset_id} 原始标签取值 {sorted(unknown)[:10]} 不在声明集合 "
                         f"{sorted(declared)} 中，先确认编码口径")
    fake_rate = float(values.isin(set(spec.fake_raw_values)).mean())
    if fake_rate > 0.5:
        raise ValueError(f"{spec.dataset_id} 标签方向疑似反转：按声明假评率达 {fake_rate:.2%} > 50%")
    if abs(fake_rate - spec.expected_fake_review_rate) > tolerance:
        raise ValueError(f"{spec.dataset_id} 假评率 {fake_rate:.2%} 偏离基准 "
                         f"{spec.expected_fake_review_rate:.2%}（容差 {tolerance:.0%}），拒绝构建")
    return 'passed'


@dataclass
class LoadedDataset:
    dataset_id: str
    kind: str                              # account_base | unified_reviews
    base: Optional[pd.DataFrame] = None    # 账号级：meta + 7 原始特征（burstiness 允许 NaN）
    reviews: Optional[pd.DataFrame] = None # 评论级：含 label，供标签聚合
    fit_features: Optional[pd.DataFrame] = None  # scaler 拟合人群的 7 原始特征（含 user_id）
    fit_user_ids: Optional[pd.Series] = None
    source_files: list = field(default_factory=list)
    label_stats: dict = field(default_factory=dict)
    label_encoding_check: str = 'passed'
    known_issues: list = field(default_factory=list)


def _read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少数据集文件：{path}")
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in ('.csv', '.txt'):
        return pd.read_csv(path)
    raise ValueError(f"不支持的文件类型：{path.suffix}")


def _assert_label_encoding(label, label_source=None, require_source=False):
    """E-5 标签守卫：label 必须 ∈ {0,1}；label_source 必须合法。"""
    label = pd.Series(label).dropna()
    uniq = set(label.unique().tolist())
    if not uniq <= {0, 1}:
        raise ValueError(
            f"标签编码非法：label 取值 {sorted(uniq)[:10]}，只允许 0/1。"
            f"若原始是 -1/1，请在统一层完成映射后再进入特征表（禁止在特征脚本里翻转）。")
    if label_source is not None:
        bad = set(pd.Series(label_source).dropna().unique()) - LEGAL_LABEL_SOURCES
        if bad:
            raise ValueError(f"label_source 非法：{sorted(bad)}，允许 {sorted(LEGAL_LABEL_SOURCES)}")
    elif require_source:
        raise ValueError("构造类数据集（FraudSquad）必须提供 label_source=real_baseline/llm_generated")
    return 'passed'


def load_unified(spec: DatasetSpec) -> LoadedDataset:
    """读取统一评论级数据（叶歆澜统一数据层）。"""
    path = Path(spec.input_path)
    df = _read_table(path)
    missing = set(UNIFIED_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少统一 schema 字段：{sorted(missing)}")

    n_raw = len(df)
    df = df.copy()
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    bad_date = int(df['date'].isna().sum())
    bad_rating = int(df['rating'].isna().sum())
    item_missing = int(df['item_id'].isna().sum())
    for name, n in (('date', bad_date), ('rating', bad_rating)):
        if n / max(n_raw, 1) > MAX_BAD_RATIO:
            raise ValueError(f"{spec.dataset_id} {name} 解析失败 {n}/{n_raw}，超过 0.5% 上限，先修 parser")
    if item_missing / max(n_raw, 1) > MAX_ITEM_MISSING_RATIO:
        raise ValueError(f"{spec.dataset_id} item_id 缺失 {item_missing}/{n_raw}，超过 0.1% 上限")
    df = df.dropna(subset=['date', 'rating', 'item_id'])

    label_source = df['label_source'] if 'label_source' in df.columns else None
    label_encoding_check = _assert_label_encoding(df['label'], label_source, spec.require_label_source)
    if label_source is None:
        df['label_source'] = 'platform_filter'

    fake_rate = float(df['label'].mean())
    if spec.expected_fake_review_rate is not None:
        if fake_rate > 0.5:
            raise ValueError(f"{spec.dataset_id} 假评率 {fake_rate:.2%} > 50%，标签方向疑似反转，拒绝构建")
        logging.info(f"{spec.dataset_id} 假评率 {fake_rate:.2%}（基准 {spec.expected_fake_review_rate:.2%}）")
    known_issues = [] if spec.label_direction_verified else [LABEL_DIRECTION_UNVERIFIED]

    if spec.expected_rows and len(df) != spec.expected_rows:
        logging.warning(f"{spec.dataset_id} 评论行数 {len(df)} != 期望 {spec.expected_rows}")

    source = [{'path': str(path), 'sha256': sha256_file(path), 'rows': int(len(df))}]
    logging.info(f"{spec.dataset_id}: 评论 {len(df)} 行（原始 {n_raw}），账号 {df['user_id'].nunique()} 个")
    return LoadedDataset(dataset_id=spec.dataset_id, kind='unified_reviews', reviews=df,
                         source_files=source,
                         label_stats={'rows': int(len(df)),
                                      'label_counts': {str(k): int(v) for k, v in
                                                       df['label'].value_counts().sort_index().items()}},
                         label_encoding_check=label_encoding_check,
                         known_issues=known_issues)


def aggregate_account_labels(reviews: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """账号级标签与人群：account_is_fake = max(label)；有生成评论的账号 = hijacked_llm。"""
    label = reviews.groupby('user_id', sort=True)['label'].max().astype('int8')
    is_injected = pd.Series(0, index=label.index, dtype='int8')
    cohort = pd.Series(np.where(label == 1, spec.new_account_cohort, 'native'),
                       index=label.index, dtype='object')
    return pd.DataFrame({'account_is_fake': label, 'is_injected': is_injected, 'cohort': cohort})


def _issue3_known_issues():
    """读取注入日志（优先 v3）判断 Issue #3 是否仍未修复。"""
    issues = []
    v3 = PROCESSED_DIR / 'injection_log_v3.json'
    v2 = PROCESSED_DIR / 'injection_log_v2.json'
    path = v3 if v3.exists() else (v2 if v2.exists() else None)
    if path is None:
        return ['issue3_unfixed', 'review_level_events_missing']
    with open(path, encoding='utf-8') as f:
        log = json.load(f)
    statuses = {r.get('issue3_status') for r in log}
    if statuses == {'recomputed'}:
        pass
    else:
        issues.append('issue3_unfixed')
    if sum(len(r.get('events_added', [])) for r in log) == 0:
        issues.append('review_level_events_missing')
    return issues


def load_yelpchi_account_base(spec: DatasetSpec) -> LoadedDataset:
    """YelpChi 自注入路径：直接消费账号级 augmented 表（不依赖 review 级数据）。"""
    aug_path = Path(spec.input_path)
    clean_path = PROCESSED_DIR / 'clean_accounts.parquet'
    log_path = PROCESSED_DIR / 'injection_log_v3.json'
    if not log_path.exists():
        log_path = PROCESSED_DIR / 'injection_log_v2.json'
    aug = _read_table(aug_path)
    clean = _read_table(clean_path)
    with open(log_path, encoding='utf-8') as f:
        log = json.load(f)

    if spec.expected_rows and len(aug) != spec.expected_rows:
        raise ValueError(f"YelpChi 行数 {len(aug)} != 期望 {spec.expected_rows}（38,063）")
    if int(aug['is_injected'].sum()) != 300:
        raise ValueError(f"注入账号数 {int(aug['is_injected'].sum())} != 300")
    if not len(log) == 300:
        raise ValueError(f"注入日志条数 {len(log)} != 300")

    label_encoding_check = _assert_label_encoding(aug['account_is_fake'])
    if not aug.loc[aug['is_injected'] == 1, 'account_is_fake'].eq(1).all():
        raise ValueError("注入账号中存在 account_is_fake=0，标签聚合有误")

    base = pd.DataFrame({
        'user_id': aug['user_id'].astype(str),
        'account_is_fake': aug['account_is_fake'].astype('int8'),
        'cohort': np.where(aug['is_injected'] == 1, 'self_injected', 'native'),
        'is_injected': aug['is_injected'].astype('int8'),
        'burstiness_missing': aug['burstiness'].isna().astype('int8'),
    })
    for col in FEATURE_LIST:
        base[col] = aug[col].astype('int32') if col in ('n_reviews', 'active_days',
                                                        'max_daily_reviews') else aug[col].astype('float64')
    base['user_id'] = base['user_id'].astype(str)

    source = [
        {'path': str(aug_path), 'sha256': sha256_file(aug_path), 'rows': int(len(aug))},
        {'path': str(clean_path), 'sha256': sha256_file(clean_path), 'rows': int(len(clean))},
        {'path': str(log_path), 'sha256': sha256_file(log_path), 'rows': int(len(log))},
    ]
    # scaler 拟合人群 = 注入前 clean 底表全体账号（含原生真假，不按标签筛）
    fit_features = clean[['user_id'] + FEATURE_LIST].copy()
    fit_features['user_id'] = fit_features['user_id'].astype(str)
    for col in FEATURE_LIST:
        fit_features[col] = fit_features[col].astype('float64')
    logging.info(f"YelpChi: 账号 {len(base)}，假 {int(base['account_is_fake'].sum())}，"
                 f"注入 {int(base['is_injected'].sum())}，burstiness 缺失 {int(base['burstiness_missing'].sum())}")
    return LoadedDataset(
        dataset_id=spec.dataset_id, kind='account_base', base=base,
        fit_features=fit_features,
        fit_user_ids=clean['user_id'].astype(str),
        source_files=source,
        label_stats={'label_counts': {str(k): int(v) for k, v in
                                      base['account_is_fake'].value_counts().sort_index().items()}},
        label_encoding_check=label_encoding_check,
        known_issues=_issue3_known_issues(),
    )


def load_dataset(spec: DatasetSpec) -> LoadedDataset:
    if spec.mode == 'missing':
        raise ValueError(f"{spec.dataset_id} 数据缺失：{spec.missing_reason}")
    if spec.mode == 'account_base':
        return load_yelpchi_account_base(spec)
    return load_unified(spec)
