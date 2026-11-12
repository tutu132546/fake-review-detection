"""Yelp 官方「metadata + reviewContent」双文件读取器的单元测试。

重点不是"能不能读出来"，而是**文本贴错时必须报错**：
文本与元数据错位是静默错误——模型照样训练、指标照样好看，
但每条评论挂的是别人的标签。所以这里把三条保护线各自钉死：

1. 行数、键全等 → 走按位置贴（最快路径），且结果正确；
2. 行数相等但键错位 → 自动降级为键连接，仍能对上；
3. 匹配率过低 → 直接抛 SchemaError，绝不生成可能错位的数据。
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import readers  # noqa: E402
from data.schema import SchemaError  # noqa: E402


def _write_pair(tmp: Path, meta_rows, content_rows):
    """按官方格式落盘：metadata 5 列、reviewContent 4 列，均 Tab 分隔无表头。"""
    meta_path = tmp / "metadata"
    content_path = tmp / "reviewContent"
    meta_path.write_text(
        "\n".join("\t".join(map(str, r)) for r in meta_rows) + "\n", encoding="utf-8"
    )
    content_path.write_text(
        "\n".join("\t".join(map(str, r)) for r in content_rows) + "\n", encoding="utf-8"
    )
    return meta_path, content_path


META = [
    ("100", "0", "5.0", "1", "2014-01-01"),
    ("101", "0", "1.0", "-1", "2014-02-02"),
    ("102", "1", "4.0", "1", "2014-03-03"),
]
CONTENT = [
    ("100", "0", "2014-01-01", "Great food, will come back."),
    ("101", "0", "2014-02-02", "Terrible, don't go."),
    ("102", "1", "2014-03-03", 'Average. Said "fine", nothing more.'),
]


class TestYelpPairHappyPath(unittest.TestCase):
    def test_positional_join(self):
        with tempfile.TemporaryDirectory() as td:
            mp, cp = _write_pair(Path(td), META, CONTENT)
            df = readers.read_yelp_pair(mp, cp, "yelpnyc")

        self.assertEqual(len(df), 3)
        self.assertEqual(int(df["label"].sum()), 1)          # 只有 -1 那条是假
        self.assertEqual(df.loc[df["user_id"] == "101", "label"].iloc[0], 1)
        self.assertEqual(df.loc[df["user_id"] == "100", "label"].iloc[0], 0)
        self.assertEqual((df["text_available"] == 1).all(), True)
        self.assertEqual(int(df["is_injected"].sum()), 0)

    def test_text_keeps_quotes_and_commas(self):
        with tempfile.TemporaryDirectory() as td:
            mp, cp = _write_pair(Path(td), META, CONTENT)
            df = readers.read_yelp_pair(mp, cp, "yelpnyc")
        text = df.loc[df["user_id"] == "102", "text"].iloc[0]
        # 通用 CSV 解析器在 QUOTE_MINIMAL 下会把引号当语法，这里必须原样保留语义
        self.assertIn("Average", text)
        self.assertIn("nothing more", text)

    def test_audit_reports_positional_and_zero_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            mp, cp = _write_pair(Path(td), META, CONTENT)
            df = readers.read_yelp_pair(mp, cp, "yelpnyc")
        audit = df.attrs["text_join_audit"]
        self.assertEqual(audit["strategy"], "positional")
        self.assertEqual(audit["key_mismatches"], 0)
        self.assertEqual(audit["matched"], 3)
        self.assertEqual(audit["rows_metadata"], 3)
        self.assertEqual(audit["rows_content"], 3)


class TestYelpPairMisalignment(unittest.TestCase):
    def test_same_length_but_shuffled_keys_falls_back_to_merge(self):
        """行数相等但顺序不同：不许按位置贴，必须按键连上。"""
        shuffled = [CONTENT[2], CONTENT[0], CONTENT[1]]
        with tempfile.TemporaryDirectory() as td:
            mp, cp = _write_pair(Path(td), META, shuffled)
            df = readers.read_yelp_pair(mp, cp, "yelpnyc")

        audit = df.attrs["text_join_audit"]
        self.assertEqual(audit["strategy"], "keyed_merge")
        self.assertGreater(audit["key_mismatches"], 0)
        # 键连接后文本必须跟对的人
        self.assertIn("Great food", df.loc[df["user_id"] == "100", "text"].iloc[0])
        self.assertIn("Terrible", df.loc[df["user_id"] == "101", "text"].iloc[0])

    def test_too_few_matches_raises(self):
        """只有 1/3 能对上：必须抛错，而不是产出一张文本大半为空/错位的表。"""
        broken = [CONTENT[0], ("999", "9", "2020-01-01", "who?"), ("998", "9", "2020-01-02", "who?")]
        with tempfile.TemporaryDirectory() as td:
            mp, cp = _write_pair(Path(td), META, broken)
            with self.assertRaises(SchemaError):
                readers.read_yelp_pair(mp, cp, "yelpnyc")

    def test_malformed_content_line_is_skipped(self):
        """少一列的残缺行跳过，不参与对齐（官方包里也可能出现）。"""
        with_malformed = CONTENT + [("103", "1")]   # 只有两列
        with tempfile.TemporaryDirectory() as td:
            mp, cp = _write_pair(Path(td), META, with_malformed)
            df = readers.read_yelp_pair(mp, cp, "yelpnyc")
        self.assertEqual(len(df), 3)
        self.assertEqual(df.attrs["text_join_audit"]["malformed_content_lines"], 1)


class TestYelpContentParser(unittest.TestCase):
    def test_tab_inside_text_is_preserved(self):
        """文本里出现 Tab 时应留在正文里，而不是把记录切成 5 段后丢弃。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reviewContent"
            p.write_text("1\t2\t2014-01-01\tbefore\tafter\n", encoding="utf-8")
            df = readers._read_yelp_content(p)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["text"].iloc[0], "before\tafter")

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reviewContent"
            p.write_text("\n\n", encoding="utf-8")
            with self.assertRaises(SchemaError):
                readers._read_yelp_content(p)


if __name__ == "__main__":
    unittest.main()
