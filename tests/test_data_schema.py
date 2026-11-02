"""统一数据 schema 与解析逻辑的单元测试。

原则：不依赖任何真实数据文件，全部用内存构造的小样本；因此 CI / 队友本机
只要装好依赖就能跑：

    python -m unittest discover -s tests -p "test_data_schema.py" -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 让测试可以从仓库根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.readers import (  # noqa: E402
    _clean_text,
    _parse_dates,
    map_columns,
    sniff_separator,
)
from data.schema import (  # noqa: E402
    CANONICAL_COLUMNS,
    SchemaError,
    coerce_dtypes,
    empty_reviews_frame,
    label_from_yelp,
    summary_stats,
    validate_reviews,
)


def _toy_frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": "toy",
            "review_id": [f"toy-{i:09d}" for i in range(n)],
            "global_user_id": [f"toy:u{i}" for i in range(n)],
            "user_id": [f"u{i}" for i in range(n)],
            "item_id": [f"i{i}" for i in range(n)],
            "rating": np.array([5.0, 1.0, 3.0, np.nan], dtype="float32"),
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "label": np.array([1, 0, 0, 1], dtype="int8"),
            "label_source": ["yelp_filter"] * n,
            "is_injected": np.zeros(n, dtype="int8"),
            "text_available": np.ones(n, dtype="int8"),
            "text": ["a", "b", "c", "d"],
        }
    )


class TestLabelMapping(unittest.TestCase):
    def test_rayana_numeric_encoding(self):
        """Rayana 原始编码：1=推荐(真实)，-1=过滤(虚假)。"""
        out = label_from_yelp(pd.Series([1, -1, 1, -1]))
        self.assertEqual(out.tolist(), [0, 1, 0, 1])

    def test_zero_one_encoding(self):
        """部分再加工版本：1=真实，0=虚假。"""
        out = label_from_yelp(pd.Series([1, 0, 1]))
        self.assertEqual(out.tolist(), [0, 1, 0])

    def test_string_input(self):
        """CSV 全按字符串读入时也要正确翻转。"""
        out = label_from_yelp(pd.Series(["1", "-1", "1"]))
        self.assertEqual(out.tolist(), [0, 1, 0])

    def test_textual_encoding(self):
        out = label_from_yelp(pd.Series(["genuine", "fake", "real"]))
        self.assertEqual(out.tolist(), [0, 1, 0])

    def test_unknown_label_raises(self):
        """遇到未知标签必须报错，绝不能默认当成真实样本。"""
        with self.assertRaises(SchemaError):
            label_from_yelp(pd.Series([1, 2, -1]))

    def test_missing_label_raises(self):
        with self.assertRaises(SchemaError):
            label_from_yelp(pd.Series([1, None]))


class TestColumnMapping(unittest.TestCase):
    def test_yelp_aliases(self):
        mapping = map_columns(["user_id", "prod_id", "rating", "label", "date", "text"])
        self.assertEqual(mapping["prod_id"], "item_id")
        self.assertEqual(mapping["user_id"], "user_id")

    def test_fraudsquad_aliases(self):
        """FraudSquad 用 star/reviewText/reviewTime/user/product 这套列名。"""
        mapping = map_columns(
            ["star", "reviewSummary", "reviewText", "product_child", "product", "user", "reviewTime", "label"]
        )
        self.assertEqual(mapping["star"], "rating")
        self.assertEqual(mapping["reviewText"], "text")
        self.assertEqual(mapping["reviewTime"], "date")
        self.assertEqual(mapping["user"], "user_id")
        self.assertEqual(mapping["product"], "item_id")
        # product 与 product_child 同时存在时，取父商品 product，子商品不参与映射
        self.assertNotIn("product_child", mapping)


class TestDateParsing(unittest.TestCase):
    def test_iso_and_unix(self):
        s = pd.Series(["2020-01-01", "1651503614"])
        out = _parse_dates(s)
        self.assertEqual(out.iloc[0], pd.Timestamp("2020-01-01"))
        self.assertEqual(out.iloc[1], pd.Timestamp("2022-05-02 15:00:14"))

    def test_garbage_becomes_nat(self):
        out = _parse_dates(pd.Series(["not-a-date"]))
        self.assertTrue(pd.isna(out.iloc[0]))


class TestTextCleaning(unittest.TestCase):
    def test_whitespace_and_missing(self):
        out = _clean_text(pd.Series(["  hello   world ", np.nan, "N/A", ""]))
        self.assertEqual(out.iloc[0], "hello world")
        self.assertEqual(out.iloc[1], "")
        self.assertEqual(out.iloc[2], "")
        self.assertEqual(out.iloc[3], "")


class TestSeparatorSniffing(unittest.TestCase):
    def test_comma(self):
        self.assertEqual(sniff_separator("a,b,c\n1,2,3\n"), ",")

    def test_tab(self):
        self.assertEqual(sniff_separator("a\tb\tc\n1\t2\t3\n"), "\t")

    def test_space(self):
        self.assertEqual(sniff_separator("201 0 5.0 1 2011-06-08\n202 0 3.0 1 2011-08-30\n"), " ")

    def test_empty_raises(self):
        with self.assertRaises(SchemaError):
            sniff_separator("   \n  ")


class TestValidation(unittest.TestCase):
    def test_valid_frame_passes_and_sorts(self):
        df = _toy_frame()
        out = validate_reviews(df)
        self.assertTrue(out["date"].is_monotonic_increasing)
        self.assertEqual(list(out.columns), CANONICAL_COLUMNS)

    def test_bad_label_detected(self):
        df = _toy_frame()
        df.loc[0, "label"] = 2
        with self.assertRaises(SchemaError) as ctx:
            validate_reviews(df)
        self.assertIn("label", str(ctx.exception))

    def test_duplicate_review_id_detected(self):
        df = _toy_frame()
        df.loc[1, "review_id"] = df.loc[0, "review_id"]
        with self.assertRaises(SchemaError) as ctx:
            validate_reviews(df)
        self.assertIn("review_id", str(ctx.exception))

    def test_missing_date_detected(self):
        df = _toy_frame()
        df.loc[2, "date"] = pd.NaT
        with self.assertRaises(SchemaError):
            validate_reviews(df)

    def test_rating_out_of_range_detected(self):
        df = _toy_frame()
        df.loc[0, "rating"] = 7.0
        with self.assertRaises(SchemaError):
            validate_reviews(df)

    def test_non_strict_collects_problems(self):
        df = _toy_frame()
        df.loc[0, "label"] = 5
        out = validate_reviews(df, strict=False)
        self.assertTrue(out.attrs.get("schema_problems"))

    def test_missing_column_detected(self):
        df = _toy_frame().drop(columns=["text"])
        with self.assertRaises(SchemaError):
            validate_reviews(df)

    def test_unexpected_column_detected(self):
        """schema 之外的列必须显式声明，避免下游拿到口径不明的字段。"""
        df = _toy_frame()
        df["mystery"] = 1
        with self.assertRaises(SchemaError):
            validate_reviews(df)


class TestDtypesAndStats(unittest.TestCase):
    def test_empty_frame_has_canonical_columns(self):
        self.assertEqual(list(empty_reviews_frame().columns), CANONICAL_COLUMNS)

    def test_coerce_fills_missing_text(self):
        df = _toy_frame()
        df.loc[0, "text"] = None
        out = coerce_dtypes(df)
        self.assertEqual(out.loc[0, "text"], "")
        self.assertEqual(str(out["label"].dtype), "int8")

    def test_summary_stats(self):
        stats = summary_stats(_toy_frame())
        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["users"], 4)
        self.assertEqual(stats["fake_rows"], 2)
        self.assertAlmostEqual(stats["fake_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
