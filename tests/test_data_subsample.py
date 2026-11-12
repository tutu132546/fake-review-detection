"""派生视图（YelpZip 规模对齐子样本）的单元测试。

这组测试盯的是三件容易悄悄错掉的事：
1. 抽样量必须真的落在目标附近（否则"规模对齐"是假的）；
2. 抽样必须整商户进行（半个商户进去会把该商户的标签分布截断）；
3. 分层不能扭曲假评比例（否则稳健性检查会被抽样本身污染）。
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import subsample  # noqa: E402


def _toy_frame(n_items: int = 200, seed: int = 0) -> pd.DataFrame:
    """构造一个小型 YelpZip 形状的表：商户大小差异很大，假评比例固定 10%。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        # 商户规模从 1 条到 200 条不等，模拟真实的长尾
        n = int(rng.integers(1, 200))
        for j in range(n):
            rows.append(
                {
                    "user_id": f"u{rng.integers(0, 500)}",
                    "item_id": f"b{i}",
                    "rating": float(rng.integers(1, 6)),
                    "date": pd.Timestamp("2010-01-01"),
                    "label": int(rng.random() < 0.10),
                    "text": "x",
                }
            )
    return pd.DataFrame(rows)


class TestStratifiedSubsample(unittest.TestCase):
    def setUp(self):
        self.df = _toy_frame()
        self.max_item_rows = int(self.df["item_id"].value_counts().max())

    def test_rows_land_near_target(self):
        target = len(self.df) // 2
        sub, audit = subsample.stratified_subsample(self.df, target_rows=target, n_strata=5, seed=0)
        # 每层最多超出一个商户，所以上界是 层数 × 最大商户规模
        self.assertGreaterEqual(len(sub), target)
        self.assertLessEqual(len(sub) - target, 5 * self.max_item_rows)
        self.assertEqual(audit["rows"], len(sub))

    def test_items_are_kept_whole(self):
        sub, _ = subsample.stratified_subsample(self.df, target_rows=len(self.df) // 2, n_strata=5)
        full = self.df["item_id"].value_counts()
        got = sub["item_id"].value_counts()
        # 每个被选中的商户，评论数必须与原表完全一致（不能只抽走一部分）
        for item, n in got.items():
            self.assertEqual(int(n), int(full[item]), f"商户 {item} 被截断了")

    def test_fake_ratio_is_not_distorted(self):
        sub, audit = subsample.stratified_subsample(self.df, target_rows=len(self.df) // 2, n_strata=5)
        gap = abs(sub["label"].mean() - self.df["label"].mean())
        self.assertLess(gap, 0.02, "分层抽样把假评比例抽歪了，稳健性检查会因此失真")
        self.assertAlmostEqual(audit["fake_ratio"], float(sub["label"].mean()), places=6)

    def test_different_seeds_give_different_sets(self):
        a, _ = subsample.stratified_subsample(self.df, target_rows=len(self.df) // 2, seed=0)
        b, _ = subsample.stratified_subsample(self.df, target_rows=len(self.df) // 2, seed=1)
        self.assertNotEqual(set(a["item_id"]), set(b["item_id"]))

    def test_same_seed_is_reproducible(self):
        a, _ = subsample.stratified_subsample(self.df, target_rows=len(self.df) // 2, seed=7)
        b, _ = subsample.stratified_subsample(self.df, target_rows=len(self.df) // 2, seed=7)
        self.assertEqual(sorted(a["item_id"].unique()), sorted(b["item_id"].unique()))

    def test_target_larger_than_data_is_rejected(self):
        with self.assertRaises(ValueError):
            subsample.stratified_subsample(self.df, target_rows=len(self.df) + 1)

    def test_missing_item_column_is_rejected(self):
        with self.assertRaises(KeyError):
            subsample.stratified_subsample(self.df.drop(columns=["item_id"]), target_rows=10)


class TestNamingAndManifest(unittest.TestCase):
    def test_view_name_encodes_scale(self):
        self.assertEqual(subsample.view_name(359052), "yelpzip_scale359k")
        self.assertEqual(subsample.view_name(100000), "yelpzip_scale100k")

    def test_default_target_comes_from_yelpnyc_spec(self):
        # 目标量级必须与 schema 里 yelpnyc 的官方规模同源，避免两处各写一个数
        from data.schema import DATASET_SPECS

        self.assertEqual(subsample.TARGET_ROWS, int(DATASET_SPECS["yelpnyc"].expected_rows))

    def test_build_views_writes_one_file_per_seed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            df = _toy_frame(n_items=40)
            target = len(df) // 2
            manifest = subsample.build_views(df, Path(td), target_rows=target, seeds=(0, 1), n_strata=4)
            prefix = subsample.view_name(target)
            files = sorted(p.name for p in Path(td).glob("*.parquet"))
            self.assertEqual(files, [f"{prefix}_seed0.parquet", f"{prefix}_seed1.parquet"])
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertTrue((Path(td) / "manifest.json").exists())
            # 报告里必须出现"不是 YelpNYC"的免责说明，防止下游误用
            self.assertIn("不能当 YelpNYC 用", subsample.render_report(manifest))


if __name__ == "__main__":
    unittest.main()
