# -*- coding: utf-8 -*-
"""B 成员交付：统一 schema → 纯函数特征计算 → 每数据集一张账号级特征表。

职责边界（特征口径的唯一出口）：
  1. 7 个公共行为特征按同一公式计算（不读 label / is_injected / cohort）；
  2. burstiness 缺失以"该数据集拟合人群"的观测中位数填充，并保留 burstiness_missing；
  3. 每个数据集单独 fit scaler，禁止跨数据集联合 fit（方案 F.4）；
  4. 输出列顺序固定（方案 F.2），并生成 sidecar manifest 与 QA 报告。

用法：
  python src/features/build_feature_table.py                       # 默认 YelpChi
  python -m src.features.build_feature_table --dataset yelpchi \
      --input data/processed/augmented_accounts_v2.parquet \
      --output data/processed/features/feature_table_yelpchi.csv \
      --scaler data/processed/features/scaler_yelpchi.json \
      --manifest results/feature_manifests/feature_table_yelpchi.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import PROCESSED_DIR, RESULTS_DIR
from features.adapters import LoadedDataset, aggregate_account_labels, load_dataset
from features.dataset_config import (FEATURE_LIST, OUTPUT_COLUMNS, DatasetSpec,
                                     ensure_dirs, get_spec)
from features.manifest import build_manifest, sha256_file, write_manifest, write_qa_report
from features.temporal import time_features_by_user

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

BANNED_FEATURE_INPUT = {'label', 'label_source', 'account_is_fake', 'is_injected',
                        'cohort', 'dataset', 'global_user_id'}

STRING_COLUMNS = ['dataset', 'global_user_id', 'user_id', 'cohort']
INT_COLUMNS = ['account_is_fake', 'is_injected', 'text_available', 'burstiness_missing',
               'n_reviews', 'active_days', 'max_daily_reviews']
FLOAT_COLUMNS = ['daily_avg_reviews', 'burstiness', 'rating_deviation', 'extreme_ratio'] \
                + [f'{c}_z' for c in FEATURE_LIST]


def check_schema(out):
    """F.7 schema/类型自检：返回违规条目（不抛异常），供 QA 统计 schema_violations。"""
    violations = []
    if list(out.columns) != OUTPUT_COLUMNS:
        return [f"列名/顺序与 F.2 不一致：{list(out.columns)}"]
    for col in STRING_COLUMNS:
        if not (pd.api.types.is_object_dtype(out[col])
                or pd.api.types.is_string_dtype(out[col])):
            violations.append(f"{col} 应为 str/object，实际 {out[col].dtype}")
    for col in INT_COLUMNS:
        if not pd.api.types.is_integer_dtype(out[col]):
            violations.append(f"{col} 应为整数类型，实际 {out[col].dtype}")
    for col in FLOAT_COLUMNS:
        if not pd.api.types.is_float_dtype(out[col]):
            violations.append(f"{col} 应为浮点类型，实际 {out[col].dtype}")
    if out[['global_user_id', 'user_id']].isna().any().any():
        violations.append("主键列 global_user_id / user_id 存在缺失")
    return violations


def read_feature_table(path):
    """规范读回：标识列强制为 str(object)，避免纯数字 user_id 被推断成 int64。"""
    head = pd.read_csv(path, nrows=0, encoding='utf-8-sig')
    df = pd.read_csv(path, dtype={c: str for c in STRING_COLUMNS if c in head.columns},
                     encoding='utf-8-sig')
    for c in STRING_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype(object)
    return df



def compute_account_features(reviews):
    """纯函数：统一评论级 schema → 账号级 7 原始特征 + burstiness_missing。

    输入必须不含 label 等评估列（红线：特征计算不得读标签）。
    """
    if BANNED_FEATURE_INPUT & set(reviews.columns):
        raise ValueError(f"特征函数收到评估列：{sorted(BANNED_FEATURE_INPUT & set(reviews.columns))}，"
                         f"标签列不得进入特征计算")
    req = ['user_id', 'item_id', 'rating', 'date']
    missing = set(req) - set(reviews.columns)
    if missing:
        raise ValueError(f"特征计算缺少字段：{sorted(missing)}")

    df = reviews[req].copy()
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    g = df.groupby('user_id', sort=True)

    out = pd.DataFrame(index=g.size().index)
    out.index.name = 'user_id'
    out['n_reviews'] = g.size().astype('int32')
    out = out.join(time_features_by_user(df))
    out['daily_avg_reviews'] = (out['n_reviews'] / out['active_days']).astype('float64')

    # 商品均分用全量评论（不按标签过滤，属"无监督但含假评污染"，manifest 需记录）
    item_mean = df.groupby('item_id')['rating'].transform('mean')
    out['rating_deviation'] = (df['rating'] - item_mean).groupby(df['user_id']).mean() \
        .reindex(out.index).astype('float64')
    out['extreme_ratio'] = g['rating'].apply(lambda s: float(((s == 1) | (s == 5)).mean()))
    out['burstiness_missing'] = out['burstiness'].isna().astype('int8')
    return out.reset_index()


def fit_scaler(fit_df, features, fit_population):
    """在指定人群上拟合 mean/std；std < 1e-12 置 1.0。"""
    params = {}
    for col in features:
        vals = pd.to_numeric(fit_df[col], errors='coerce').astype('float64')
        mean = float(vals.mean())
        std = float(vals.std())
        degenerate = False
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
            degenerate = True
        params[col] = {'mean': mean, 'std': std, 'degenerate': degenerate}
    return {
        'feature_version': '1.0.0',
        'fit_population': fit_population,
        'n_fit': int(len(fit_df)),
        'features': params,
    }


def apply_scaler(df, scaler):
    for col, p in scaler['features'].items():
        df[f'{col}_z'] = (df[col] - p['mean']) / p['std']
    return df


def fill_burstiness(df, median):
    """每数据集独立填充；填充前已生成 burstiness_missing。"""
    df = df.copy()
    if 'burstiness_missing' not in df.columns:
        df['burstiness_missing'] = df['burstiness'].isna().astype('int8')
    df['burstiness'] = df['burstiness'].fillna(float(median))
    return df


def validate_table(out, spec, scaler, fit_df):
    """F.7 可执行自检；返回 qa 摘要，有硬性违规直接报错。"""
    # schema/类型违规单独计数：既进 QA 报告，也作为硬性失败原因（不再恒为 0）
    schema_violations = check_schema(out)
    if list(out.columns) != OUTPUT_COLUMNS:
        raise ValueError("特征表自检失败：\n- " + "\n- ".join(schema_violations))
    problems = list(schema_violations)
    if out['global_user_id'].duplicated().any():
        problems.append(f"global_user_id 重复 {int(out['global_user_id'].duplicated().sum())} 行")
    if not set(out['account_is_fake'].unique()) <= {0, 1}:
        problems.append("account_is_fake 取值超出 {0,1}")
    nan_cols = [c for c in FEATURE_LIST + [f'{c}_z' for c in FEATURE_LIST] if out[c].isna().any()]
    if nan_cols:
        problems.append(f"存在 NaN 列：{nan_cols}")
    if not (out['n_reviews'] >= 1).all() or not (out['active_days'] >= 1).all():
        problems.append("n_reviews/active_days 存在 <1 的取值")

    daily_diff = (out['daily_avg_reviews'] - out['n_reviews'] / out['active_days']).abs().max()
    if daily_diff > 1e-9:
        problems.append(f"daily_avg_reviews 与 n_reviews/active_days 不一致（max={daily_diff:.3e}）")
    violations = int(((out['n_reviews'] - out['active_days'] * out['max_daily_reviews']) > 0).sum())
    if violations:
        problems.append(f"违反 n_reviews <= active_days*max_daily_reviews：{violations} 行")
    bound = spec.rating_max - spec.rating_min
    if not out['rating_deviation'].between(-bound, bound).all():
        problems.append(f"rating_deviation 越界（允许 ±{bound}）")
    if not out['extreme_ratio'].between(0, 1).all():
        problems.append("extreme_ratio 越界 [0,1]")

    # z 分在拟合人群上必须 mean≈0 / std≈1（用拟合人群自身数值复算，不用表内注入行）
    z_mean_max = 0.0
    z_std_max_dev = 0.0
    degenerate = []
    for col, p in scaler['features'].items():
        z_fit = (pd.to_numeric(fit_df[col], errors='coerce').astype('float64') - p['mean']) / p['std']
        z_mean_max = max(z_mean_max, abs(float(z_fit.mean())))
        if p.get('degenerate'):
            degenerate.append(col)   # 拟合人群内该列方差为 0，std 被兜底为 1，std≈1 检验不适用
            continue
        z_std_max_dev = max(z_std_max_dev, abs(float(z_fit.std()) - 1.0))
    if z_mean_max > 1e-6 or z_std_max_dev > 1e-6:
        problems.append(f"拟合人群 z 分未归零（|mean|={z_mean_max:.2e}, |std-1|={z_std_max_dev:.2e}）")

    qa = {
        'schema_violations': len(schema_violations),
        'constraint_violations': violations,
        'nan_feature_cells': int(out[FEATURE_LIST + [f'{c}_z' for c in FEATURE_LIST]].isna().sum().sum()),
        'daily_avg_max_abs_diff': round(float(daily_diff), 12),
        'z_mean_max_abs': round(float(z_mean_max), 12),
        'z_std_max_dev': round(float(z_std_max_dev), 12),
        'degenerate_features': degenerate,
        'rows': int(len(out)),
    }
    if problems:
        raise ValueError("特征表自检失败：\n- " + "\n- ".join(problems))
    return qa


def _write_missing(spec):
    """YelpNYC 等缺失数据集：只写状态，不造假表。"""
    ensure_dirs()
    payload = {
        'feature_version': '1.0.0',
        'dataset_id': spec.dataset_id,
        'status': 'missing',
        'reason': spec.missing_reason,
        'text_available': spec.text_available,
        'table_path': None,
    }
    path = spec.output_path.with_suffix('.MISSING.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.warning(f"{spec.dataset_id} 缺失，已写占位：{path}")
    return path


def build_dataset(spec: DatasetSpec, input_path=None, output_path=None,
                  scaler_path=None, manifest_path=None):
    """构建单个数据集的特征表 + scaler + manifest + QA 报告。"""
    if spec.mode == 'missing':
        return _write_missing(spec)
    ensure_dirs()
    spec = DatasetSpec(**{**spec.__dict__,
                          'input_path': Path(input_path) if input_path else spec.input_path,
                          'output_path': Path(output_path) if output_path else spec.output_path,
                          'scaler_path': Path(scaler_path) if scaler_path else spec.scaler_path,
                          'manifest_path': Path(manifest_path) if manifest_path else spec.manifest_path})
    logging.info(f"[{spec.dataset_id}] 1. 读取统一数据 ...")
    loaded: LoadedDataset = load_dataset(spec)

    if loaded.kind == 'unified_reviews':
        logging.info(f"[{spec.dataset_id}] 2. 计算账号级行为特征 ...")
        # 只传行为字段：label/label_source 不进入特征函数（红线：特征计算不得读标签）
        feats = compute_account_features(
            loaded.reviews[['user_id', 'item_id', 'rating', 'date']])
        labels = aggregate_account_labels(loaded.reviews, spec)
        base = feats.merge(labels, left_on='user_id', right_index=True, how='left')
    else:
        logging.info(f"[{spec.dataset_id}] 2. 使用账号级底表（YelpChi 自注入路径） ...")
        base = loaded.base.copy()

    base['dataset'] = spec.dataset_id
    base['global_user_id'] = spec.dataset_id + ':' + base['user_id'].astype(str)
    base['text_available'] = np.int8(spec.text_available)
    if 'burstiness_missing' not in base.columns:
        base['burstiness_missing'] = base['burstiness'].isna().astype('int8')

    # 拟合人群：YelpChi 用注入前 clean 底表（含原生真假、不按标签筛）；其他用该份全部账号
    fit_df = loaded.fit_features if loaded.kind == 'account_base' else base[FEATURE_LIST]
    missing_rates = {c: float(base[c].isna().mean()) for c in FEATURE_LIST}
    median = float(fit_df['burstiness'].dropna().median())
    logging.info(f"[{spec.dataset_id}] 3. burstiness 缺失 {missing_rates['burstiness']:.2%}，"
                 f"用拟合人群中位数 {median:.6f} 填充")

    base = fill_burstiness(base, median)
    scaler = fit_scaler(fit_df, FEATURE_LIST, spec.scaler_fit_population)
    base = apply_scaler(base, scaler)

    out = base[OUTPUT_COLUMNS].copy()
    out['account_is_fake'] = out['account_is_fake'].astype('int8')
    out['is_injected'] = out['is_injected'].astype('int8')
    out['cohort'] = out['cohort'].astype(str)
    out['text_available'] = out['text_available'].astype('int8')
    out['burstiness_missing'] = out['burstiness_missing'].astype('int8')
    for c in ('n_reviews', 'active_days', 'max_daily_reviews'):
        out[c] = out[c].astype('int32')
    for c in ('daily_avg_reviews', 'burstiness', 'rating_deviation', 'extreme_ratio'):
        out[c] = out[c].astype('float64')
    out = out.sort_values('global_user_id', kind='mergesort').reset_index(drop=True)

    logging.info(f"[{spec.dataset_id}] 4. 自检 ...")
    qa = validate_table(out, spec, scaler, fit_df)

    if spec.expected_rows and len(out) != spec.expected_rows:
        raise ValueError(f"{spec.dataset_id} 行数 {len(out)} != 期望 {spec.expected_rows}")

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(spec.output_path, index=False, encoding='utf-8-sig')
    with open(spec.scaler_path, 'w', encoding='utf-8') as f:
        json.dump(scaler, f, indent=2, ensure_ascii=False)

    manifest = build_manifest(
        spec, out, scaler, missing_rates, loaded.source_files,
        fit_population=spec.scaler_fit_population,
        label_encoding_check=loaded.label_encoding_check,
        qa=qa,
        known_issues=loaded.known_issues,
        unified_manifest_sha256=sha256_file(spec.input_path)
        if loaded.kind == 'unified_reviews' else None,
    )
    write_manifest(spec.manifest_path, manifest)
    qa_path = RESULTS_DIR / f'feature_qa_{spec.dataset_id}.md'
    write_qa_report(qa_path, manifest, spec)

    # YelpChi 兼容旧下游：保留一份 data/processed/feature_table.csv 共享副本
    if spec.dataset_id == 'yelpchi' and Path(output_path or spec.output_path) == spec.output_path:
        out.to_csv(PROCESSED_DIR / 'feature_table.csv', index=False, encoding='utf-8-sig')

    _report_discrimination(out)
    logging.info(f"✅ [{spec.dataset_id}] {len(out)} 行 × {out.shape[1]} 列 -> {spec.output_path}")
    logging.info(f"   scaler -> {spec.scaler_path} | manifest -> {spec.manifest_path} | QA -> {qa_path}")
    return manifest


def _report_discrimination(out):
    """单特征 ROC-AUC（供论文第 4 章引用）；注入组只作描述，不作主结论。"""
    y = out['account_is_fake'].values
    if len(set(y)) < 2:
        logging.info("单特征区分度跳过：标签只有单一类别")
        return
    native = (out['is_injected'] == 0).values
    z_cols = [f'{c}_z' for c in FEATURE_LIST] + ['burstiness_missing']
    logging.info("5. 单特征区分度 ROC-AUC（vs account_is_fake；<0.5 表示低值方向偏水军）:")
    for c in z_cols:
        auc_all = roc_auc_score(y, out[c])
        auc_nat = roc_auc_score(y[native], out.loc[native, c]) if len(set(y[native])) > 1 else float('nan')
        logging.info(f"   {c:22s} 全表 {auc_all:.3f} | 仅原生 {auc_nat:.3f}")


def build_feature_table():
    """兼容旧调用：等价于 python -m src.features.build_feature_table --dataset yelpchi。"""
    return build_dataset(get_spec('yelpchi'))


def main():
    parser = argparse.ArgumentParser(description="构建单数据集账号级特征表")
    parser.add_argument('--dataset', default='yelpchi',
                        help='数据集 id：yelpchi / yelpzip / fraudsquad_llama3 / '
                             'fraudsquad_qwen2 / fraudsquad_qwen_dsr1 / yelpnyc')
    parser.add_argument('--input', default=None, help='统一数据路径（默认取 dataset_config 声明）')
    parser.add_argument('--output', default=None, help='特征表输出路径')
    parser.add_argument('--scaler', default=None, help='scaler JSON 输出路径')
    parser.add_argument('--manifest', default=None, help='manifest JSON 输出路径')
    args = parser.parse_args()
    spec = get_spec(args.dataset, input_path=args.input)
    build_dataset(spec, output_path=args.output, scaler_path=args.scaler,
                  manifest_path=args.manifest)


if __name__ == '__main__':
    main()
