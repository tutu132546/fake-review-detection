# -*- coding: utf-8 -*-
"""step3_inject_water_army_pro.py — 事件层水军注入（v3.0）

对应 Issue：`src/issue1.txt`（爆发性方向反了）+ `src/issue2.txt`（模板 B 违反物理约束）。

设计要点
--------
1. **注入发生在事件层**：为每个注入账号生成 (日期, 评分, 商品) 三元组，
   再按 step1 的口径重算全部 7 个特征。不再"直接乘系数改特征值"，
   所以"单日发评数 > 总评论数""日均 != 总数 / 活跃天数"这类矛盾由构造消除。
2. **时间戳一律落在账号原有的活跃区间 [首条, 末条] 内**：
   - active_days（首末间隔 + 1）严格不变，不会出现"沉寂 N 年后突然复活"的痕迹；
   - burstiness / max_daily 由真实时间戳算出，方向天然正确（集中爆发 → 升高）。
3. **三个模板只控制"时间集中度"与"评分构成"**，特征差异由事件自然涌现：
   - A_Time_Burst          全部新增落在最后一个活跃日（单日爆发）
   - B_Rating_Polarization 均匀撒在活跃期内空闲的日子（单日频次不变）
   - C_Stealth_Mix         少量集中在末日 + 其余撒开（混合）
4. **商品互不重复且避开本人已评商品**：YelpChi 中一个用户对同一商品只有一条评论，
   重复评论是一眼假的痕迹（Issue #4 引入的回归缺陷）。
5. **每账号独立随机流**（SeedSequence([BASE_SEED, uid]).spawn(4)）：可复现，
   且单个账号能被单独重放——后续 DCFO"照解释改行为"的逃逸实验需要这一点。

输出（路径与字段名保持不变）：
  data/processed/augmented_accounts_v2.parquet
  data/processed/injection_log_v2.json

用法：python src/step3_inject_water_army_pro.py
"""
import json
import logging

import numpy as np
import pandas as pd

from config import RAW_CSV_PATH, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

BASE_SEED = 42
N_INJECT = 300
FEATURES = ["n_reviews", "active_days", "daily_avg_reviews", "max_daily_reviews",
            "burstiness", "rating_deviation", "extreme_ratio"]
TEMPLATES = ("A_Time_Burst", "B_Rating_Polarization", "C_Stealth_Mix")
# 新增评论中 5 星的比例：A 偏向纯刷好评，B/C 保留少量 1 星（极化）
P5 = {"A_Time_Burst": 0.95, "B_Rating_Polarization": 0.90, "C_Stealth_Mix": 0.90}
C_CLUSTER_SHARE = 0.40          # 模板 C 中集中在最后一个活跃日的比例


# --------------------------------------------------------------------------- #
# 1. Φ：与 step1_build_account_base.py 完全同口径的特征重算
# --------------------------------------------------------------------------- #
def calc_burstiness(ts):
    """逐字照搬 step1 的实现，保证同一口径（numpy std，ddof=0；间隔按秒）。"""
    if len(ts) < 3:
        return np.nan
    ts = np.sort(ts)
    gaps = np.diff(ts).astype("timedelta64[s]").astype(float)
    return gaps.std() / gaps.mean() if gaps.mean() > 0 else np.nan


def features_from_events(dates, ratings, products, prod_avg):
    """从一个账号的完整评论事件重算 7 个特征（与 step1 逐位一致）。

    dates     : np.datetime64[D] 数组（YelpChi 原数据为日粒度）
    ratings   : float 数组
    products  : int 数组
    prod_avg  : {product_id: 该商品全量评论均分}，与 step1 的 prod_avg_rating 同源

    口径：
      active_days      = (末条 - 首条).days + 1   ← 日历跨度，不是"有评论的天数"
      max_daily        = 同一天评论数的最大值
      burstiness       = 相邻间隔 std/mean（秒）；<3 条或全部同日 → NaN
      rating_deviation = mean(rating - 商品全量均分)
      extreme_ratio    = (1 星或 5 星) 占比
    """
    n = len(dates)
    if n == 0:
        return None
    days = pd.Series(dates)
    active_days = int((dates.max() - dates.min()) / np.timedelta64(1, "D")) + 1
    return {
        "n_reviews": int(n),
        "active_days": int(active_days),
        "daily_avg_reviews": float(n / active_days),
        "max_daily_reviews": int(days.value_counts().max()),
        "burstiness": float(calc_burstiness(dates)),
        "rating_deviation": float(np.mean(ratings - np.array([prod_avg[p] for p in products]))),
        "extreme_ratio": float(np.mean((ratings == 1) | (ratings == 5))),
    }


def verify_feature_pipeline(df_clean, user_events, prod_avg, n=300):
    """硬闸门：抽 n 个底表账号，用事件重算并与表里的值逐位比对。

    这是"注入号不会整体偏离人群"的唯一保证。任何一项不一致，说明事件源或口径
    与 step1 不同（例如 step1 过滤过原始数据），必须先对齐再注入。
    """
    pool = df_clean[df_clean["account_is_fake"] == 0]
    sample = pool.sample(n=min(n, len(pool)), random_state=BASE_SEED)
    worst = {k: 0.0 for k in FEATURES}
    missing = 0
    for _, row in sample.iterrows():
        ev = user_events.get(int(row["user_id"]))
        if ev is None:
            missing += 1
            continue
        f = features_from_events(*ev, prod_avg)
        for k in FEATURES:
            a, b = float(row[k]), float(f[k])
            if np.isnan(a) and np.isnan(b):
                continue
            worst[k] = max(worst[k], abs(a - b) if not (np.isnan(a) or np.isnan(b)) else np.inf)
    if missing or max(worst.values()) > 1e-9:
        raise AssertionError(f"特征口径与 step1 不一致：{worst}，缺事件账号 {missing} 个")
    logging.info("Φ 校验通过：与 clean_accounts 逐位一致（最大偏差 %.1e）", max(worst.values()))


# --------------------------------------------------------------------------- #
# 2. 事件构造：模板 -> (日期, 评分, 商品)
# --------------------------------------------------------------------------- #
def free_days_in_window(dates, k, rng):
    """在 [首条, 末条] 内随机取 k 个「当天没有任何评论」的日子。

    只取空闲日是为了让 max_daily_reviews 不变：每个新日期只挂 1 条评论，
    单日峰值仍是原值。活跃期之外的日期一律不取，从而保持 active_days 不变。
    """
    lo, hi = dates.min(), dates.max()
    span = int((hi - lo) / np.timedelta64(1, "D")) + 1
    all_days = lo + np.arange(span).astype("timedelta64[D]")
    free = np.setdiff1d(all_days, np.unique(dates))
    if len(free) == 0 or k <= 0:
        return np.array([], dtype="datetime64[D]")
    return np.sort(rng.choice(free, size=min(k, len(free)), replace=False))


def plan_events(dates, template, add_n, g_date):
    """按模板决定新增事件的日期；返回 (new_dates, n_free_used)。

    A：全部压在同一活跃日（模拟"一天之内集中刷评"）。
    B：只撒空闲日，单日频次不变（把时间维度的影响降到最小）。
    C：一部分集中在末日，其余撒开。
    空闲日不够（活跃期极短的账号）时，剩下的并入最后一个活跃日 —— 这是物理必然，
    并会被 check_physical 明确记录，不会掩盖。
    """
    last = dates.max()
    if template == "A_Time_Burst":
        return np.full(add_n, last), 0
    n_spread = add_n if template == "B_Rating_Polarization" \
        else add_n - max(1, int(round(add_n * C_CLUSTER_SHARE)))
    spread = free_days_in_window(dates, n_spread, g_date)
    rest = np.full(add_n - len(spread), last)
    return np.concatenate([spread, rest]), int(len(spread))


def pick_products(products, k, g_prod, prod_pool):
    """挑 k 个本人从未评过的商品，两两不重复（避免 (user, product) 重复评论）。"""
    used = set(int(p) for p in products)
    cand = prod_pool[~np.isin(prod_pool, list(used))] if used else prod_pool
    if len(cand) == 0:                      # 理论上不会发生：201 个商品，单账号最多 57 条
        raise RuntimeError("可用商品池为空")
    return g_prod.choice(cand, size=min(k, len(cand)), replace=False).astype(int)


# --------------------------------------------------------------------------- #
# 3. 单账号模拟 + 物理闸门
# --------------------------------------------------------------------------- #
def check_physical(uid, dates, old, new, new_dates, new_products, products):
    """注入后过闸：任何一条不过就中止，绝不写脏数据。"""
    assert new["n_reviews"] == old["n_reviews"] + len(new_dates), f"{uid} n_reviews"
    assert new["active_days"] == old["active_days"], f"{uid} 活跃期被意外改变"
    assert new["max_daily_reviews"] <= new["n_reviews"], f"{uid} 单日 > 总数"
    assert new["n_reviews"] <= new["active_days"] * new["max_daily_reviews"], f"{uid} 违反 Issue2 约束"
    assert abs(new["daily_avg_reviews"] - new["n_reviews"] / new["active_days"]) < 1e-12, \
        f"{uid} daily_avg 与 n/active_days 不一致"
    assert new_dates.min() >= dates.min() and new_dates.max() <= dates.max(), f"{uid} 时间越界"
    assert len(set(new_products.tolist())) == len(new_products), f"{uid} 商品重复"
    assert not (set(new_products.tolist()) & set(int(p) for p in products)), \
        f"{uid} 重复评论本人已评商品"
    assert np.isnan(old["burstiness"]) == np.isnan(new["burstiness"]), f"{uid} burstiness NaN 状态改变"
    assert 0.0 <= new["extreme_ratio"] <= 1.0, f"{uid} extreme_ratio 越界"
    assert -4.0 <= new["rating_deviation"] <= 4.0, f"{uid} rating_deviation 越界"


def simulate_account(uid, dates, ratings, products, template, prod_avg, prod_pool):
    """对一个账号做一次注入（纯函数）：返回 (old, new, new_dates, new_ratings, new_products, n_free)。

    随机流按 [BASE_SEED, uid] 派生并拆成四条子流（n_add / 日期 / 评分 / 商品），
    因此同 uid 重放结果完全一致，且改模板不会扰动其他维度的抽样。
    """
    g_add, g_date, g_rate, g_prod = (
        np.random.default_rng(s) for s in np.random.SeedSequence([BASE_SEED, uid]).spawn(4))
    add_n = int(g_add.integers(5, 15))          # 5~14 条，与旧版一致

    new_dates, n_free = plan_events(dates, template, add_n, g_date)
    new_products = pick_products(products, len(new_dates), g_prod, prod_pool)
    new_ratings = np.where(g_rate.random(len(new_products)) < P5[template], 5.0, 1.0)
    if len(new_products) != len(new_dates):     # 商品池不足时截断（数据上不会触发）
        new_dates = new_dates[:len(new_products)]

    old = features_from_events(dates, ratings, products, prod_avg)
    new = features_from_events(
        np.concatenate([dates, new_dates]),
        np.concatenate([ratings, new_ratings]),
        np.concatenate([products, new_products]),
        prod_avg)
    check_physical(uid, dates, old, new, new_dates, new_products, products)
    return old, new, new_dates, new_ratings, new_products, n_free


# --------------------------------------------------------------------------- #
# 4. 主流程
# --------------------------------------------------------------------------- #
def inject_water_army_pro(n_inject=N_INJECT, output_tag="v2"):
    logging.info("1. 读取底表、缩放参数与原始评论事件...")
    df = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    with open(PROCESSED_DIR / "scaler_params.json", encoding="utf-8") as fh:
        scaler = json.load(fh)

    raw = pd.read_csv(RAW_CSV_PATH)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values(["user_id", "date"])
    prod_avg = raw.groupby("product_id")["rating"].mean().to_dict()
    prod_pool = raw["product_id"].drop_duplicates().to_numpy()
    # 每个账号的完整事件序列（注入与重算共用同一份）
    user_events = {
        int(uid): (g["date"].values.astype("datetime64[D]"),
                   g["rating"].values.astype(float),
                   g["product_id"].values.astype(int))
        for uid, g in raw.groupby("user_id")
    }
    logging.info("   评论 %d 条 / 账号 %d 个 / 商品 %d 个", len(raw), len(user_events), len(prod_pool))

    verify_feature_pipeline(df, user_events, prod_avg)

    # 候选池与抽样：与旧版完全一致（同一批 300 个账号，便于和历史结果对照）
    valid_pool = df[(df["account_is_fake"] == 0) & (df["n_reviews"] >= 5)]
    if len(valid_pool) == 0:
        raise RuntimeError("候选池为空：检查 step1 产物")
    n_inject = int(min(n_inject, len(valid_pool)))
    normal_idx = valid_pool.sample(n=n_inject, random_state=BASE_SEED).index
    logging.info("2. 候选池 %d 个正常号，计划注入 %d 个", len(valid_pool), n_inject)

    df["is_injected"] = 0
    logs = []
    dz = {t: {k: [] for k in FEATURES} for t in TEMPLATES}

    for i, idx in enumerate(normal_idx):
        uid = int(df.loc[idx, "user_id"])
        dates, ratings, products = user_events[uid]
        template = TEMPLATES[min(i * 3 // max(n_inject, 1), 2)]

        old, new, new_dates, new_ratings, new_products, n_free = simulate_account(
            uid, dates, ratings, products, template, prod_avg, prod_pool)

        # --- 写回底表 + 同步 Z-score ---
        for k in FEATURES:
            df.loc[idx, k] = new[k]
            df.loc[idx, f"{k}_z"] = (new[k] - scaler[k]["mean"]) / scaler[k]["std"]
        df.loc[idx, ["account_is_fake", "is_injected"]] = [1, 1]

        # --- 日志：σ 单位的真实增量，供 DCFO 解释对齐 ---
        delta_sigma, modified = {}, {}
        for k in FEATURES:
            d = (new[k] - old[k]) / scaler[k]["std"]
            d = 0.0 if np.isnan(d) else float(d)
            delta_sigma[k] = round(d, 4)
            if abs(new[k] - old[k]) > 1e-12 or np.isnan(old[k]) != np.isnan(new[k]):
                modified[k] = {"original": None if np.isnan(old[k]) else round(float(old[k]), 4),
                               "mutated": None if np.isnan(new[k]) else round(float(new[k]), 4),
                               "delta_sigma": delta_sigma[k]}
            dz[template][k].append(d)

        logs.append({
            "user_id": uid,
            "injection_template": template,
            "policy": {
                "n_add": int(len(new_dates)),
                "temporal": {"A_Time_Burst": "burst_last_day",
                             "B_Rating_Polarization": "spread_free_days",
                             "C_Stealth_Mix": "cluster+spread"}[template],
                "cluster_size": int(len(new_dates) - n_free),
                "free_days_used": int(n_free),
                "p_five_star": P5[template],
            },
            "events_added": [
                {"date": str(pd.Timestamp(d).date()), "rating": float(r), "product_id": int(p)}
                for d, r, p in zip(new_dates, new_ratings, new_products)
            ],
            "modified_features": modified,
            "feature_delta_sigma": delta_sigma,
            "n_real": int(old["n_reviews"]),
            "n_fake": int(len(new_dates)),
            "physical_checks": "passed",
        })

    # --- 全表兜底断言：两条物理约束（Issue2 要求） ---
    bad1 = int((df["n_reviews"] > df["active_days"] * df["max_daily_reviews"]).sum())
    bad2 = int((df["max_daily_reviews"] > df["n_reviews"]).sum())
    assert bad1 == 0 and bad2 == 0, f"约束被破坏：n>active*max {bad1} 条，max>n {bad2} 条"
    logging.info("3. 全表物理约束校验通过：n_reviews <= active_days * max_daily_reviews（%d/%d 条）",
                 len(df), len(df))

    df.to_parquet(PROCESSED_DIR / f"augmented_accounts_{output_tag}.parquet", index=False)
    with open(PROCESSED_DIR / f"injection_log_{output_tag}.json", "w", encoding="utf-8") as fh:
        json.dump(logs, fh, indent=4, ensure_ascii=False, allow_nan=False)

    # --- 注入效应汇总（配对 dz，消除选池偏差） ---
    logging.info("4. 注入效应（同账号注入前后 dz = Δ/σ，模板内均值）：")
    logging.info("   %-22s %s", "feature", "".join(f"{t.split('_')[0]:>10}" for t in TEMPLATES))
    for k in FEATURES:
        logging.info("   %-22s %s", k,
                     "".join(f"{np.mean(dz[t][k]):>10.2f}" if dz[t][k] else f"{'-':>10}"
                             for t in TEMPLATES))
    for t in ("A_Time_Burst", "C_Stealth_Mix"):
        m = float(np.mean(dz[t]["burstiness"])) if dz[t]["burstiness"] else 0.0
        logging.info("   [判定] %s burstiness dz = %+.2f（Issue1 要求 > 0）%s",
                     t, m, "PASS" if m > 0 else "FAIL")

    logging.info("注入完成：%d 个账号，输出 augmented_accounts_%s.parquet / injection_log_%s.json",
                 len(logs), output_tag, output_tag)


if __name__ == "__main__":
    inject_water_army_pro()
