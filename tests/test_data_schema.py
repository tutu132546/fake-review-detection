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
    split_partial_rows,
    strip_partial_rows,
)
from data.schema import (  # noqa: E402
    ACCOUNT_COLUMNS,
    CANONICAL_COLUMNS,
    DATASET_SPECS,
    PUBLIC_DATASETS,
    VIEW_ONLY_DATASETS,
    SchemaError,
    build_account_table,
    coerce_dtypes,
    empty_reviews_frame,
    label_from_yelp,
    summary_stats,
    validate_accounts,
    validate_reviews,
)
from data.pipeline import COMBINE_EXCLUDE  # noqa: E402


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
        """Rayana 原始编码：1=推荐(真实)，-1=过滤(虚假)，auto 可自动识别。"""
        out = label_from_yelp(pd.Series([1, -1, 1, -1]))
        self.assertEqual(out.tolist(), [0, 1, 0, 1])

    def test_zero_one_encoding_is_ambiguous_and_raises(self):
        """0/1 编码方向不可从数据判断，必须显式指定 scheme —— 绝不默认猜。"""
        with self.assertRaises(SchemaError) as ctx:
            label_from_yelp(pd.Series([1, 0, 1]))
        msg = str(ctx.exception)
        self.assertIn("binary_1_fake", msg)
        self.assertIn("binary_0_fake", msg)

    def test_binary_1_fake_scheme(self):
        """显式声明 1=虚假、0=真实。"""
        out = label_from_yelp(pd.Series([1, 0, 1]), scheme="binary_1_fake")
        self.assertEqual(out.tolist(), [1, 0, 1])

    def test_binary_0_fake_scheme(self):
        """显式声明 0=虚假、1=真实（方向相反，必须能区分开）。"""
        out = label_from_yelp(pd.Series([1, 0, 1]), scheme="binary_0_fake")
        self.assertEqual(out.tolist(), [0, 1, 0])

    def test_scheme_domain_violation_raises(self):
        """指定 signed 方案后出现 0，说明数据版本不符，必须报错而不是猜。"""
        with self.assertRaises(SchemaError):
            label_from_yelp(pd.Series([1, 0]), scheme="signed_1_real_minus1_fake")

    def test_unknown_scheme_raises(self):
        with self.assertRaises(SchemaError):
            label_from_yelp(pd.Series([1, -1]), scheme="whatever")

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


class TestYelpZipPartialRowRepair(unittest.TestCase):
    """镜像文件里"被吞进评论文本的导出残缺行"必须能被识别并剥离。"""

    def test_clean_text_untouched(self):
        clean, rows = split_partial_rows("good food, nice place")
        self.assertEqual(clean, "good food, nice place")
        self.assertEqual(rows, [])

    def test_normal_multiline_review_untouched(self):
        """真实带换行的评论（YelpZip 里有 18 条）不能被误切。"""
        text = "first line\nsecond line\nthird line"
        clean, rows = split_partial_rows(text)
        self.assertEqual(clean, text)
        self.assertEqual(rows, [])

    def test_contaminated_text_is_split(self):
        text = "Sabarsky is much prettier\n58123\t1419\t2012-12-09\tgreat food\nend of another review"
        clean, rows = split_partial_rows(text)
        self.assertEqual(clean, "Sabarsky is much prettier")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "58123")
        self.assertEqual(rows[0][1], "1419")
        self.assertEqual(rows[0][2], "2012-12-09")
        self.assertIn("great food", rows[0][3])

    def test_multiple_rows_and_tail(self):
        text = "head\n1\t2\t2012-01-01\trow one\n3\t4\t2012-01-02\trow two\nspilled tail"
        clean, rows = split_partial_rows(text)
        self.assertEqual(clean, "head")
        self.assertEqual(len(rows), 2)
        # 末尾那段属于最后一条残缺行（它的正文被换行截断了）
        self.assertIn("spilled tail", rows[-1][3])

    def test_series_level_audit(self):
        s = pd.Series(["ok", "head\n9\t8\t2012-01-01\tx", "another\n7\t6\t2012-01-02\ty"])
        cleaned, audit = strip_partial_rows(s)
        self.assertEqual(audit["contaminated_records"], 2)
        self.assertEqual(audit["partial_rows"], 2)
        self.assertEqual(cleaned.tolist()[0], "ok")
        self.assertEqual(cleaned.tolist()[1], "head")
        self.assertNotIn("\t", cleaned.str.cat(sep=" "))


class TestAccountTable(unittest.TestCase):
    def _reviews(self) -> pd.DataFrame:
        # u1: 2 条真实 + 1 条假（被接管）；u2: 2 条真实；u3: 1 条注入
        return pd.DataFrame(
            {
                "dataset": ["d"] * 5,
                "user_id": ["u1", "u1", "u1", "u2", "u3"],
                "item_id": ["i1", "i2", "i3", "i1", "i9"],
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-01-05", "2020-01-06", "2020-01-02", "2020-01-09"]
                ),
                "label": np.array([0, 0, 1, 0, 1], dtype="int8"),
                "is_injected": np.array([0, 0, 0, 0, 1], dtype="int8"),
            }
        )

    def test_counts_and_labels(self):
        acc = build_account_table(self._reviews(), label_basis="hijacked_by_llm")
        acc = acc.set_index("account_id")
        self.assertEqual(acc.loc["u1", "n_reviews"], 3)
        self.assertEqual(acc.loc["u1", "n_fake_reviews"], 1)
        self.assertEqual(acc.loc["u1", "n_fake_items"], 1)
        self.assertEqual(acc.loc["u1", "account_label"], 1)
        self.assertEqual(acc.loc["u2", "account_label"], 0)
        self.assertEqual(acc.loc["u2", "n_fake_items"], 0)  # 无假评论也要是 0 而不是 NaN

    def test_injected_basis_uses_n_injected_not_label(self):
        """注入集的正例必须是"被注入过"，不能是"有假评论"（那会混进平台标签）。"""
        acc = build_account_table(self._reviews(), label_basis="injected").set_index("account_id")
        self.assertEqual(acc.loc["u3", "account_label"], 1)
        self.assertEqual(acc.loc["u1", "account_label"], 0)  # u1 有假评论但没被注入

    def test_unknown_basis_raises(self):
        with self.assertRaises(SchemaError):
            build_account_table(self._reviews(), label_basis="nope")

    def test_validate_accounts(self):
        acc = build_account_table(self._reviews(), label_basis="hijacked_by_llm")
        validate_accounts(acc)  # 不抛
        bad = acc.copy()
        bad.loc[0, "n_fake_reviews"] = 999
        with self.assertRaises(SchemaError):
            validate_accounts(bad)

    def test_empty_frame_returns_contract(self):
        acc = build_account_table(empty_reviews_frame(), label_basis="platform_filtered")
        self.assertEqual(list(acc.columns), ACCOUNT_COLUMNS)


class TestDatasetSpecs(unittest.TestCase):
    def test_fraudsquad_is_split_into_three(self):
        for name, model in (
            ("fraudsquad_llama3", "llama3_8b"),
            ("fraudsquad_qwen2", "qwen_72b"),
            ("fraudsquad_qwen_dsr1", "qwen_dsr1"),
        ):
            spec = DATASET_SPECS[name]
            self.assertEqual(spec.single_model, model)
            self.assertEqual(spec.derived_from, "fraudsquad_amazon")

    def test_merged_view_excluded_from_public_and_combined(self):
        self.assertNotIn("fraudsquad_amazon", PUBLIC_DATASETS)
        self.assertIn("fraudsquad_amazon", VIEW_ONLY_DATASETS)
        self.assertIn("fraudsquad_amazon", COMBINE_EXCLUDE)

    def test_no_duplicate_dataset_names_in_public(self):
        self.assertEqual(len(PUBLIC_DATASETS), len(set(PUBLIC_DATASETS)))

    def test_yelpnyc_is_restaurant_not_hotel(self):
        """YelpNYC 是餐厅数据；早期文档误写成酒店，这里锁死。"""
        notes = DATASET_SPECS["yelpnyc"].notes
        self.assertIn("餐厅", notes)
        self.assertIn("不是酒店", notes)


if __name__ == "__main__":
    unittest.main()
