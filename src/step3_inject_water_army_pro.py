# -*- coding: utf-8 -*-
"""水军注入实验：模板 A（时间爆发）/ B（评分极化）/ C（隐蔽混合）。

本版修复两条已知 Issue：
  - Issue #6：模板 B 只加评论数却不改单日最大发评，导致
    `n_reviews <= active_days * max_daily_reviews` 对少数账号不成立。
    修法：required = ceil(new_n_reviews / active_days)，不足则补到 required 并写日志。
  - Issue #3：模板 A/C 原先把 burstiness 乘 0.3~0.6，方向与物理含义相反。
    修法：读取 `YelpChi_reviews.csv` 的真实时间戳，为新增评论构造日期，
    合并新旧时间戳后用 features/temporal.py 的 calc_burstiness 重算；
    同时补 `events_added`（解决 PR #10 的 review-level 复现阻塞）。
    若 review 级数据缺失，必须显式传 `--allow-issue3-blocked`，
    日志逐条标记 `issue3_status=blocked_by_review_level_data`，禁止用乘系数冒充修复。

随机流：抽样 random_state=42；注入随机数 default_rng(42)，各模板共用同一条随机流。
用法：
  python src/step3_inject_water_army_pro.py                  # 有 review 数据时完整修复
  python src/step3_inject_water_army_pro.py --allow-issue3-blocked   # 仅有账号级数据
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RAW_CSV_PATH
from features.temporal import calc_burstiness

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

LOG_PATH = PROCESSED_DIR / "injection_log_v3.json"
ISSUE3_OK = "recomputed"
ISSUE3_BLOCKED = "blocked_by_review_level_data"


def load_review_level(path):
    """读取 review 级时间戳；文件不存在时返回 None（由调用方决定是否 degraded）。"""
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    missing = {'user_id', 'product_id', 'rating', 'date'} - set(df.columns)
    if missing:
        raise ValueError(f"review 级数据缺少字段：{sorted(missing)}")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['user_id'] = df['user_id'].astype(str)
    return df


def build_burst_dates(rng, add_n, origin_dates):
    """为 add_n 条新增评论构造日期：优先落在原活跃期内，集中在 1~2 个自然日。"""
    lo, hi = origin_dates.min(), origin_dates.max()
    span = max(int((hi - lo).days), 0)
    base = lo + pd.Timedelta(days=int(rng.integers(0, span + 1))) if span > 0 else lo
    if add_n <= 5 or rng.random() < 0.5:
        return [base] * add_n
    second = base + pd.Timedelta(days=1)
    first_n = int(rng.integers(1, add_n))
    return [base] * first_n + [second] * (add_n - first_n)


def merge_time_features(origin_dates, new_dates, original_burstiness):
    """合并新旧时间戳后重算时间特征。

    原本 burstiness 为 NaN 的账号（n<3 或全部同日）保持 NaN，不因为注入被"补全"。
    """
    merged = pd.Series(list(origin_dates) + list(new_dates)).sort_values()
    new_active_days = int((merged.max() - merged.min()).days) + 1
    new_max_daily = int(merged.dt.normalize().value_counts().max())
    if np.isnan(original_burstiness):
        new_burst = np.nan
    else:
        new_burst = calc_burstiness(merged)
    return new_active_days, new_max_daily, new_burst, merged


def inject_water_army_pro(reviews_path=None, allow_issue3_blocked=False):
    df = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    with open(PROCESSED_DIR / "scaler_params.json", "r") as f:
        scaler = json.load(f)

    reviews = load_review_level(reviews_path or RAW_CSV_PATH)
    if reviews is None and not allow_issue3_blocked:
        raise SystemExit(
            "缺少 review 级时间戳（data/raw/YelpChi_reviews.csv），无法按真实时间序列修复 Issue #3。\n"
            "请补数据，或显式使用 --allow-issue3-blocked（日志会标记 issue3_status，不伪造 events_added）。"
        )
    issue3_status = ISSUE3_OK if reviews is not None else ISSUE3_BLOCKED
    if reviews is not None:
        reviews_by_user = {u: g['date'].sort_values() for u, g in reviews.groupby('user_id')}
        product_by_user = reviews.groupby('user_id')['product_id'].agg(
            lambda s: s.value_counts().index[0])
    else:
        reviews_by_user, product_by_user = {}, {}
    logging.warning(f"Issue #3 状态：{issue3_status}")

    # 核心约束：只从评论数 >= 5 的正常号中抽取，避免数据不够算
    valid_pool = df[(df['account_is_fake'] == 0) & (df['n_reviews'] >= 5)]
    logging.info(f"注入候选池（n_reviews>=5）共有 {len(valid_pool)} 个正常号。")

    normal_idx = valid_pool.sample(n=300, random_state=42).index
    rng = np.random.default_rng(seed=42)
    # 新增评论日期用独立随机流：保证 add_n / rating_deviation / burstiness 的特征随机流
    # 与修复前逐位一致（seed=42 复现），日期构造不污染该序列
    rng_dates = np.random.default_rng(seed=43)

    df['is_injected'] = 0
    logs = []
    n_issue6_fixed = 0

    for i, idx in enumerate(normal_idx):
        original = df.loc[idx].copy()
        uid = str(int(original['user_id']))

        # --- 1. 模拟水军"真实新增"的一批评论 ---
        add_n = int(rng.integers(5, 15))
        new_n_reviews = int(original['n_reviews']) + add_n
        # 先取随机数再分支，保证随机流与修复前一致（seed=42 可复现）
        new_dev = np.clip(original['rating_deviation'] + rng.uniform(0.5, 1.5), -4.0, 4.0)
        _burst_u = rng.uniform(0.3, 0.6)

        # 活跃天数：默认视为在原活跃期内突击刷评；有 review 级时间戳时按合并序列重算
        new_active_days = int(original['active_days'])
        new_max_daily = min(int(original['max_daily_reviews']) + add_n, new_n_reviews)
        new_burst = np.nan if np.isnan(original['burstiness']) else max(
            original['burstiness'] * _burst_u, 0.01)
        events_added = []

        # 所有模板真实新增的评论都落账（台账合计 = add_n 合计）；
        # 时间特征重算只对模板 A/C 生效：B 是评分侧模板，刻意不改评论时间戳
        if reviews is not None and uid in reviews_by_user:
            origin_dates = reviews_by_user[uid]
            new_dates = build_burst_dates(rng_dates, add_n, origin_dates)
            product = str(product_by_user[uid])
            events_added = [{"product_id": product, "rating": 5, "date": d.strftime('%Y-%m-%d')}
                            for d in new_dates]
            assert len(events_added) == add_n
            if not (100 <= i < 200):
                new_active_days, new_max_daily, new_burst, merged = merge_time_features(
                    origin_dates, new_dates, original['burstiness'])

        # 日均发评数随评论数与活跃期同步重算
        new_daily_avg = new_n_reviews / new_active_days

        # 极端评分比例：新增评论全部按 5 星计
        old_extreme_count = original['extreme_ratio'] * original['n_reviews']
        new_extreme = (old_extreme_count + add_n) / new_n_reviews

        # --- 2. 模板分离 ---
        if i < 100:
            template = "A_Time_Burst"
            df.loc[idx, ['n_reviews', 'active_days', 'max_daily_reviews', 'burstiness', 'extreme_ratio']] = \
                [new_n_reviews, new_active_days, new_max_daily, new_burst, new_extreme]
            mods = {
                "n_reviews": {"original": int(original['n_reviews']), "mutated": new_n_reviews},
                "max_daily_reviews": {"original": int(original['max_daily_reviews']), "mutated": new_max_daily},
                "burstiness": {"original": round(float(original['burstiness']), 4), "mutated": round(float(new_burst), 4)},
                "extreme_ratio": {"original": round(float(original['extreme_ratio']), 4), "mutated": round(float(new_extreme), 4)}
            }
        elif i < 200:
            template = "B_Rating_Polarization"
            # Issue #6：只加评论数时，单日最大发评必须补到 ceil(n/active_days)，否则违反物理约束
            required_max_daily = int(np.ceil(new_n_reviews / int(original['active_days'])))
            new_max_daily = max(int(original['max_daily_reviews']), required_max_daily)
            df.loc[idx, ['n_reviews', 'max_daily_reviews', 'rating_deviation', 'extreme_ratio']] = \
                [new_n_reviews, new_max_daily, new_dev, new_extreme]
            mods = {
                "n_reviews": {"original": int(original['n_reviews']), "mutated": new_n_reviews},
                "rating_deviation": {"original": round(float(original['rating_deviation']), 4), "mutated": round(float(new_dev), 4)},
                "extreme_ratio": {"original": round(float(original['extreme_ratio']), 4), "mutated": round(float(new_extreme), 4)}
            }
            if new_max_daily > int(original['max_daily_reviews']):
                mods["max_daily_reviews"] = {
                    "original": int(original['max_daily_reviews']), "mutated": new_max_daily}
                n_issue6_fixed += 1
        else:
            template = "C_Stealth_Mix"
            df.loc[idx, ['n_reviews', 'active_days', 'max_daily_reviews', 'burstiness',
                         'rating_deviation', 'extreme_ratio']] = \
                [new_n_reviews, new_active_days, new_max_daily, new_burst, new_dev, new_extreme]
            mods = {
                "n_reviews": {"original": int(original['n_reviews']), "mutated": new_n_reviews},
                "max_daily_reviews": {"original": int(original['max_daily_reviews']), "mutated": new_max_daily},
                "burstiness": {"original": round(float(original['burstiness']), 4), "mutated": round(float(new_burst), 4)},
                "rating_deviation": {"original": round(float(original['rating_deviation']), 4), "mutated": round(float(new_dev), 4)},
                "extreme_ratio": {"original": round(float(original['extreme_ratio']), 4), "mutated": round(float(new_extreme), 4)}
            }

        # 公共修正：评论总数增加后，日均发评数同步重算
        df.loc[idx, 'daily_avg_reviews'] = new_daily_avg
        mods["daily_avg_reviews"] = {
            "original": round(float(original['daily_avg_reviews']), 4),
            "mutated": round(float(new_daily_avg), 4)
        }
        if new_active_days != int(original['active_days']):
            mods["active_days"] = {
                "original": int(original['active_days']), "mutated": new_active_days}

        df.loc[idx, ['account_is_fake', 'is_injected']] = [1, 1]

        # --- 3. 动态同步更新 Z-score ---
        updated_cols = ['n_reviews', 'active_days', 'max_daily_reviews', 'burstiness',
                        'rating_deviation', 'extreme_ratio', 'daily_avg_reviews']
        for feat_name in updated_cols:
            val = df.loc[idx, feat_name]
            mean_val = scaler[feat_name]["mean"]
            std_val = scaler[feat_name]["std"]
            df.loc[idx, f'{feat_name}_z'] = (val - mean_val) / std_val

        logs.append({
            "user_id": int(original['user_id']),
            "injection_template": template,
            "add_n": add_n,
            "events_added": events_added,
            "issue3_status": issue3_status,
            "modified_features": mods,
        })

    # --- 4. 全表物理约束与日志台账自检 ---
    violated = df['n_reviews'] - df['active_days'] * df['max_daily_reviews']
    if (violated > 0).any():
        bad_rows = df.loc[violated > 0, ['user_id', 'n_reviews', 'active_days', 'max_daily_reviews']]
        raise AssertionError(
            f"仍有 {len(bad_rows)} 个账号违反 n_reviews <= active_days * max_daily_reviews：\n"
            + bad_rows.to_string(index=False))
    total_add_n = sum(r['add_n'] for r in logs)
    total_events = sum(len(r['events_added']) for r in logs)
    if reviews is not None:
        assert total_events == total_add_n, f"events_added 合计 {total_events} != add_n 合计 {total_add_n}"
    else:
        assert total_events == 0, "缺少 review 级数据时不得伪造 events_added"
    assert len(logs) == 300 and df['is_injected'].sum() == 300, "注入账号数必须为 300"

    df.to_parquet(PROCESSED_DIR / "augmented_accounts_v2.parquet", index=False)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

    logging.info(f"注入完成：300 个账号，add_n 合计 {total_add_n}，events_added 合计 {total_events}；"
                 f"Issue #6 补齐 {n_issue6_fixed} 个账号的 max_daily_reviews；"
                 f"日志 -> {LOG_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="水军注入实验（模板 A/B/C）")
    parser.add_argument("--reviews", default=None,
                        help="review 级 CSV（user_id,product_id,rating,date），默认 data/raw/YelpChi_reviews.csv")
    parser.add_argument("--allow-issue3-blocked", action="store_true",
                        help="无 review 级时间戳时仍生成注入表，但日志标记 issue3_status=blocked")
    args = parser.parse_args()
    inject_water_army_pro(reviews_path=args.reviews,
                          allow_issue3_blocked=args.allow_issue3_blocked)
