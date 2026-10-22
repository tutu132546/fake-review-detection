# -*- coding: utf-8 -*-
"""B 成员交付：从注入后账号底表生成面向建模的特征交接表 feature_table.csv

职责边界（特征口径的唯一出口）：
  1. burstiness NaN 处理：以 clean 底表观测值中位数填充，并附 burstiness_missing 标记列；
  2. 全部 _z 列统一按 scaler_params.json 重算（与 step1/step3 同口径，避免多处计算漂移）；
  3. 输出 raw 列（供 DCFO 反事实解释回译原始单位）与 _z 列（无监督模型输入）。

用法：python src/features/build_feature_table.py （任意工作目录均可）
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

FEATURES = ['n_reviews', 'active_days', 'daily_avg_reviews', 'max_daily_reviews',
            'burstiness', 'rating_deviation', 'extreme_ratio']


def build_feature_table():
    logging.info("1. 读取注入后底表与缩放参数...")
    df = pd.read_parquet(PROCESSED_DIR / "augmented_accounts_v2.parquet")
    clean = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    with open(PROCESSED_DIR / "scaler_params.json", encoding="utf-8") as f:
        scaler = json.load(f)

    # --- 2. burstiness 缺失处理 ---
    # 填充值取 clean 底表观测值中位数（不受注入分布污染，与 scaler 参数同源）
    observed = clean['burstiness'].dropna()
    fill_val = float(observed.median())
    n_nan = int(df['burstiness'].isna().sum())
    df['burstiness_missing'] = df['burstiness'].isna().astype(int)
    df['burstiness'] = df['burstiness'].fillna(fill_val)
    logging.info(f"2. burstiness NaN {n_nan}/{len(df)} ({n_nan / len(df) * 100:.1f}%)，"
                 f"以 clean 观测值中位数 {fill_val:.4f} 填充，并生成 burstiness_missing 标记列")

    # --- 3. 统一重算全部 _z 列（唯一口径） ---
    for col in FEATURES:
        p = scaler[col]
        df[f'{col}_z'] = (df[col] - p['mean']) / p['std']
    logging.info("3. 全部 _z 列已按 scaler_params.json 统一重算")

    # --- 4. 组装与自检 ---
    id_cols = ['user_id', 'account_is_fake', 'is_injected', 'burstiness_missing']
    raw_cols = FEATURES
    z_cols = [f'{c}_z' for c in FEATURES]
    out = df[id_cols + raw_cols + z_cols]

    assert out['user_id'].is_unique, "user_id 存在重复"
    assert out.notna().all().all(), "特征表仍存在 NaN，禁止交给下游"
    assert (out['daily_avg_reviews'] - out['n_reviews'] / out['active_days']).abs().max() < 1e-6, \
        "daily_avg_reviews 与 n_reviews/active_days 不一致"
    assert (out['n_reviews'] >= out['max_daily_reviews']).all(), "n_reviews < max_daily_reviews"
    assert out['extreme_ratio'].between(0, 1).all(), "extreme_ratio 越界"
    assert out['rating_deviation'].between(-4, 4).all(), "rating_deviation 越界"
    logging.info("4. 自检通过（无 NaN、量纲范围、内部一致性）")

    # --- 5. 落盘（processed 为规范位置；根目录共享副本由队伍同步时另行复制） ---
    out_path = PROCESSED_DIR / "feature_table.csv"
    out.to_csv(out_path, index=False, encoding='utf-8-sig')

    # --- 6. 单特征区分度（交付文档引用） ---
    y = out['account_is_fake'].values
    native = (out['is_injected'] == 0).values
    logging.info("5. 单特征区分度 ROC-AUC（vs account_is_fake；<0.5 表示低值方向偏水军）:")
    for c in z_cols + ['burstiness_missing']:
        auc_all = roc_auc_score(y, out[c])
        auc_nat = roc_auc_score(y[native], out.loc[native, c])
        logging.info(f"   {c:22s} 全表 {auc_all:.3f} | 仅原生 {auc_nat:.3f}")

    logging.info(f"✅ 特征表生成完毕：{out.shape[0]} 行 x {out.shape[1]} 列 -> {out_path}")


if __name__ == "__main__":
    build_feature_table()
