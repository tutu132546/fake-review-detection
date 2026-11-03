# -*- coding: utf-8 -*-
"""_smoke_test.py — step3_inject_water_army_pro.py 的自测

跑法（两种都行）：
    python src/_smoke_test.py
    cd src && python _smoke_test.py
通过标准：最后一行打印 ALL SMOKE TESTS PASSED

覆盖：
  1. Φ 与 step1 逐位一致（有真实数据时，抽 200 个账号比对 7 个特征）
  2. 三个模板的物理不变量：active_days 不变、时间不越界、商品不重复、
     n <= active_days * max_daily、burstiness 的 NaN 状态不改变
  3. 方向性：模板 A 的 burstiness 必须上升（Issue1）；模板 B 的 max_daily 必须不变
  4. 可复现性：同一 uid 重放结果逐位一致
  5. 边界：全部评论同一天的账号（active_days = 1，burstiness = NaN）仍自洽
  6. 真实数据端到端：300 个注入账号的配对 dz 方向与约束（Issue1 验收）
"""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import step3_inject_water_army_pro as m

FEATURES = m.FEATURES
TEMPLATES = m.TEMPLATES


def make_history(day_offsets, ratings=None, product_base=1000):
    dates = np.array(day_offsets, dtype="datetime64[D]")
    n = len(dates)
    ratings = np.array(ratings if ratings is not None else [4.0] * n, dtype=float)
    products = np.arange(product_base, product_base + n)
    return dates, ratings, products


PROD_AVG = {p: 4.2 for p in range(1000, 1400)}


def test_phi_against_table():
    """Φ 必须与 step1 产物逐位一致，否则注入号会整体偏离人群。"""
    import pandas as pd
    from config import RAW_CSV_PATH, PROCESSED_DIR

    if not (PROCESSED_DIR / "clean_accounts.parquet").exists():
        print("  [SKIP] 无 clean_accounts.parquet，跳过 Φ 一致性检查")
        return
    df = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    raw = pd.read_csv(RAW_CSV_PATH, parse_dates=["date"]).sort_values(["user_id", "date"])
    prod_avg = raw.groupby("product_id")["rating"].mean().to_dict()
    events = {int(u): (g["date"].values.astype("datetime64[D]"),
                       g["rating"].values.astype(float),
                       g["product_id"].values.astype(int))
              for u, g in raw.groupby("user_id")}
    pool = df[df["account_is_fake"] == 0]
    sample = pool.sample(n=min(200, len(pool)), random_state=0)
    worst = 0.0
    for _, row in sample.iterrows():
        f = m.features_from_events(*events[int(row["user_id"])], prod_avg)
        for k in FEATURES:
            a, b = float(row[k]), float(f[k])
            if np.isnan(a) and np.isnan(b):
                continue
            assert not (np.isnan(a) ^ np.isnan(b)), f"{row.user_id} {k} NaN 状态不一致"
            worst = max(worst, abs(a - b))
    assert worst < 1e-9, f"Φ 与底表偏差 {worst}"
    print("  phi vs step1: ok（200 个账号，最大偏差 %.1e）" % worst)


def test_templates_invariants():
    """三个模板都必须满足物理约束（Issue2 + 商品唯一性）。"""
    dates, ratings, products = make_history([0, 35, 80, 140, 210, 290])
    old = m.features_from_events(dates, ratings, products, PROD_AVG)
    for tpl in TEMPLATES:
        o, new, nd, nr, npd, n_free = m.simulate_account(
            12345, dates, ratings, products, tpl, PROD_AVG, np.arange(1000, 1400))
        assert new["active_days"] == o["active_days"], f"{tpl} 活跃期改变"
        assert new["n_reviews"] == o["n_reviews"] + len(nd)
        assert new["n_reviews"] <= new["active_days"] * new["max_daily_reviews"], f"{tpl} Issue2"
        assert nd.min() >= dates.min() and nd.max() <= dates.max(), f"{tpl} 时间越界"
        assert len(set(npd.tolist())) == len(npd), f"{tpl} 商品重复"
        assert not (set(npd.tolist()) & set(products.tolist())), f"{tpl} 打了本人已评商品"
        assert np.isnan(o["burstiness"]) == np.isnan(new["burstiness"]), f"{tpl} NaN 状态改变"
        print("  %-22s ok  n_fake=%2d max_daily %d->%d  burst %.3f->%.3f  free=%d"
              % (tpl, len(nd), o["max_daily_reviews"], new["max_daily_reviews"],
                 o["burstiness"], new["burstiness"], n_free))
    print("  template invariants: ok")


def test_direction():
    """Issue1：模板 A 必须让 burstiness 上升；B 必须保持 max_daily 不变。"""
    dates, ratings, products = make_history([0, 35, 80, 140, 210, 290])

    _, a_new, *_ = m.simulate_account(777, dates, ratings, products, "A_Time_Burst",
                                      PROD_AVG, np.arange(1000, 1400))
    b0 = m.features_from_events(dates, ratings, products, PROD_AVG)["burstiness"]
    assert a_new["burstiness"] > b0, f"A 的 burstiness 没有上升：{b0} -> {a_new['burstiness']}"

    _, b_new, *_ = m.simulate_account(777, dates, ratings, products, "B_Rating_Polarization",
                                      PROD_AVG, np.arange(1000, 1400))
    assert b_new["max_daily_reviews"] == 1, "B 改变了单日最大发评（空闲日足够时不应改变）"
    print("  direction: ok  A burst %.3f->%.3f (↑)  B max_daily 保持 %d"
          % (b0, a_new["burstiness"], b_new["max_daily_reviews"]))


def test_replay_determinism():
    dates, ratings, products = make_history([0, 12, 70, 120, 200, 260])
    a = m.simulate_account(999, dates, ratings, products, "C_Stealth_Mix", PROD_AVG, np.arange(1000, 1400))
    b = m.simulate_account(999, dates, ratings, products, "C_Stealth_Mix", PROD_AVG, np.arange(1000, 1400))
    assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3]) and np.array_equal(a[4], b[4])
    print("  replay determinism: ok（同 uid 两次生成逐位一致）")


def test_single_day_account():
    """全部评论同一天的账号：burstiness 必须保持 NaN，约束仍需成立。"""
    dates, ratings, products = make_history([10, 10, 10, 10, 10, 10])
    o = m.features_from_events(dates, ratings, products, PROD_AVG)
    assert np.isnan(o["burstiness"]) and o["active_days"] == 1
    for tpl in TEMPLATES:
        _, new, nd, nr, npd, _ = m.simulate_account(
            4321, dates, ratings, products, tpl, PROD_AVG, np.arange(1000, 1400))
        assert np.isnan(new["burstiness"]), f"{tpl} 把 NaN 变成数值"
        assert new["active_days"] == 1, f"{tpl} 越出了唯一活跃日"
        assert new["max_daily_reviews"] == new["n_reviews"]
        assert new["n_reviews"] <= new["active_days"] * new["max_daily_reviews"]
    print("  single-day account: ok（NaN 保持，且 max_daily 随总数同步）")


def test_real_data_end_to_end():
    """真实数据：跑完 300 个账号，检查 Issue1 的 dz 方向与全部约束。"""
    import pandas as pd
    from config import RAW_CSV_PATH, PROCESSED_DIR

    if not (PROCESSED_DIR / "clean_accounts.parquet").exists():
        print("  [SKIP] 无真实数据，跳过端到端检查")
        return
    df = pd.read_parquet(PROCESSED_DIR / "clean_accounts.parquet")
    raw = pd.read_csv(RAW_CSV_PATH, parse_dates=["date"]).sort_values(["user_id", "date"])
    prod_avg = raw.groupby("product_id")["rating"].mean().to_dict()
    prod_pool = raw["product_id"].drop_duplicates().to_numpy()
    events = {int(u): (g["date"].values.astype("datetime64[D]"),
                       g["rating"].values.astype(float),
                       g["product_id"].values.astype(int))
              for u, g in raw.groupby("user_id")}
    valid = df[(df["account_is_fake"] == 0) & (df["n_reviews"] >= 5)]
    idx = valid.sample(n=min(300, len(valid)), random_state=m.BASE_SEED).index

    dz = {t: {k: [] for k in FEATURES} for t in TEMPLATES}
    for i, ix in enumerate(idx):
        uid = int(df.loc[ix, "user_id"])
        tpl = TEMPLATES[min(i * 3 // max(len(idx), 1), 2)]
        old, new, *_ = m.simulate_account(uid, *events[uid], tpl, prod_avg, prod_pool)
        for k in FEATURES:
            d = (new[k] - old[k]) / 1.0     # 相对量，用于方向判定
            if np.isnan(d):
                d = 0.0
            dz[tpl][k].append(d)

    for tpl in ("A_Time_Burst", "C_Stealth_Mix"):
        assert np.mean(dz[tpl]["burstiness"]) > 0, f"{tpl} burstiness dz 不为正（Issue1 未修好）"
    assert np.mean(dz["A_Time_Burst"]["max_daily_reviews"]) > 0
    assert np.mean(dz["B_Rating_Polarization"]["max_daily_reviews"]) < 0.5
    print("  真实数据端到端：A burst Δ=%+.2f | C burst Δ=%+.2f | B max_daily Δ=%+.2f  ok"
          % (np.mean(dz["A_Time_Burst"]["burstiness"]),
             np.mean(dz["C_Stealth_Mix"]["burstiness"]),
             np.mean(dz["B_Rating_Polarization"]["max_daily_reviews"])))


def main():
    print("[1] Φ 与 step1 一致性")
    test_phi_against_table()
    print("[2] 模板物理不变量")
    test_templates_invariants()
    print("[3] 方向性（Issue1 / Issue2）")
    test_direction()
    print("[4] 可复现性")
    test_replay_determinism()
    print("[5] 边界：单日账号")
    test_single_day_account()
    print("[6] 真实数据端到端")
    test_real_data_end_to_end()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
