import inspect
import json
import os
import tempfile
import unittest

from datasketch import MinHash, MinHashLSHEnsemble
from fk_discovery import (
    discover_foreign_keys,
    load_csv_tables,
    render_report,
    result_to_dict,
)


def find_pair(result, child, parent):
    for c in result.candidates:
        if (c.child_table, c.child_column) == child and (c.parent_table, c.parent_column) == parent:
            return c
    return None


class TestValidForeignKey(unittest.TestCase):
    def test_clean_fk_is_valid(self):
        tables = {
            "users": {"user_id": ["u1", "u2", "u3"], "name": ["a", "b", "c"]},
            "orders": {"user_id": ["u1", "u2", "u2", "u3", "u1"]},
        }
        result = discover_foreign_keys(tables, threshold=0.5)
        pair = find_pair(result, ("orders", "user_id"), ("users", "user_id"))
        self.assertIsNotNone(pair)
        self.assertTrue(pair.is_valid_fk)
        self.assertEqual(pair.distinct_containment, 1.0)
        self.assertEqual(pair.row_match_rate, 1.0)
        self.assertEqual(pair.orphan_rows, ())
        self.assertTrue(pair.parent_key_qualified)
        self.assertIn(pair, result.relationships)

    def test_reverse_direction_not_valid_when_parent_has_duplicates(self):
        # users.user_id 的不同值也全包含于 orders.user_id, 但 orders.user_id 有重复值,
        # 不能标成有效外键 —— 靠父键资格区分方向。
        tables = {
            "users": {"user_id": ["u1", "u2", "u3"]},
            "orders": {"user_id": ["u1", "u2", "u2", "u3", "u1"]},
        }
        result = discover_foreign_keys(tables, threshold=0.5)
        pair = find_pair(result, ("users", "user_id"), ("orders", "user_id"))
        self.assertIsNotNone(pair)
        self.assertEqual(pair.distinct_containment, 1.0)
        self.assertFalse(pair.is_valid_fk)
        self.assertNotIn(pair, result.relationships)
        self.assertIn("父列含重复值", pair.invalid_reasons)


class TestOrphansAndRowStats(unittest.TestCase):
    def test_orphans_kept_per_row_and_duplicates_not_lost(self):
        tables = {
            "parent": {"code": ["A"]},
            "child": {"code": ["B", "A", "B", None, "B"]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("child", "code"), ("parent", "code"))
        self.assertIsNotNone(pair)
        # 不同值包含率: {A,B} 中只有 A 命中
        self.assertEqual(pair.distinct_containment, 0.5)
        # 行匹配率按原始行: 非空 4 行中 1 行命中 (空值行不进分母)
        self.assertEqual(pair.rows_total, 5)
        self.assertEqual(pair.null_rows, 1)
        self.assertEqual(pair.matched_rows, 1)
        self.assertEqual(pair.row_match_rate, 0.25)
        # 每条孤儿行都保留: 3 条 B, 带 0 基行号, 重复行不丢
        self.assertEqual(pair.orphan_rows, ((0, "B"), (2, "B"), (4, "B")))
        self.assertFalse(pair.is_valid_fk)
        self.assertIn("未全包含（存在孤儿值）", pair.invalid_reasons)

    def test_source_nulls_are_not_orphans(self):
        tables = {
            "parent": {"code": ["A", "B"]},
            "child": {"code": [None, "A", None, "B", None]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("child", "code"), ("parent", "code"))
        self.assertIsNotNone(pair)
        self.assertEqual(pair.orphan_rows, ())
        self.assertEqual(pair.null_rows, 3)
        self.assertEqual(pair.matched_rows, 2)
        self.assertEqual(pair.row_match_rate, 1.0)

    def test_matched_rows_count_duplicates(self):
        tables = {
            "parent": {"code": ["A"]},
            "child": {"code": ["A", "A", "A"]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("child", "code"), ("parent", "code"))
        self.assertIsNotNone(pair)
        self.assertEqual(pair.rows_total, 3)
        self.assertEqual(pair.matched_rows, 3)
        self.assertEqual(pair.row_match_rate, 1.0)


class TestParentKeyQualification(unittest.TestCase):
    def test_parent_with_duplicates_not_valid_even_full_containment(self):
        tables = {
            "parent": {"code": ["A", "B", "B", "C"]},  # 重复值 B
            "child": {"code": ["A", "C"]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("child", "code"), ("parent", "code"))
        self.assertIsNotNone(pair)
        self.assertEqual(pair.distinct_containment, 1.0)
        self.assertTrue(pair.parent_has_duplicates)
        self.assertFalse(pair.parent_key_qualified)
        self.assertFalse(pair.is_valid_fk)
        self.assertNotIn(pair, result.relationships)

    def test_parent_with_nulls_not_valid_even_full_containment(self):
        tables = {
            "parent": {"code": ["A", "B", None, "C"]},  # 含空值
            "child": {"code": ["A", "C"]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("child", "code"), ("parent", "code"))
        self.assertIsNotNone(pair)
        self.assertEqual(pair.distinct_containment, 1.0)
        self.assertTrue(pair.parent_has_nulls)
        self.assertFalse(pair.parent_key_qualified)
        self.assertFalse(pair.is_valid_fk)
        self.assertIn("父列含空值", pair.invalid_reasons)


class TestEmptyStringAndNaN(unittest.TestCase):
    def test_empty_string_is_valid_value(self):
        tables = {
            "coupons": {"code": ["", "C1", "C2"]},  # 空字符串是合格父键的一员
            "orders": {"coupon": ["", "", None, "C1"]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("orders", "coupon"), ("coupons", "code"))
        self.assertIsNotNone(pair)
        self.assertTrue(pair.is_valid_fk)
        self.assertEqual(pair.child_distinct_size, 2)  # {"" 和 "C1"}
        self.assertEqual(pair.matched_rows, 3)  # 两个 "" 和一个 "C1" 都命中
        self.assertEqual(pair.orphan_rows, ())

    def test_empty_string_can_be_orphan(self):
        tables = {
            "parent": {"code": ["A", "B"]},
            "child": {"code": ["A", ""]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("child", "code"), ("parent", "code"))
        self.assertIsNotNone(pair)
        # 空字符串是有效值, 不在父列值集中就是孤儿, 而不是被当成空值跳过
        self.assertEqual(pair.orphan_rows, ((1, ""),))
        self.assertEqual(pair.null_rows, 0)

    def test_nan_treated_as_null(self):
        tables = {
            "parent": {"v": [1.0, 2.0, 3.0]},
            "child": {"v": [1.0, float("nan"), 2.0]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        pair = find_pair(result, ("child", "v"), ("parent", "v"))
        self.assertIsNotNone(pair)
        self.assertEqual(pair.null_rows, 1)
        self.assertEqual(pair.orphan_rows, ())
        self.assertEqual(pair.row_match_rate, 1.0)
        self.assertTrue(pair.is_valid_fk)


class TestAllNullColumns(unittest.TestCase):
    def test_all_null_source_column_reports_no_evidence(self):
        tables = {
            "parent": {"code": ["A", "B"]},
            "legacy": {"old_id": [None, None, None], "note": ["A", "B", "C"]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        self.assertIn(("legacy", "old_id"), result.no_evidence_columns)
        # 全空列不产生任何候选, 更不会自动认定关系
        for c in result.candidates:
            self.assertNotEqual((c.child_table, c.child_column), ("legacy", "old_id"))
            self.assertNotEqual((c.parent_table, c.parent_column), ("legacy", "old_id"))
        self.assertEqual(result.relationships, [r for r in result.relationships if r.is_valid_fk])

    def test_all_null_column_via_nan(self):
        tables = {
            "t": {"c": [float("nan"), None]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        self.assertIn(("t", "c"), result.no_evidence_columns)
        self.assertEqual(result.candidates, [])


class TestRecallBoundary(unittest.TestCase):
    def test_below_threshold_pair_not_recalled_and_not_evaluated(self):
        # 包含率 2/20 = 0.1, 远低于阈值 0.5; 未被召回的列对不补成精确结果
        tables = {
            "logs": {"tag": [f"t{i:02d}" for i in range(20)]},
            "archive": {"tag": ["t00", "t01"] + [f"a{i:02d}" for i in range(18)]},
        }
        result = discover_foreign_keys(tables, threshold=0.5)
        self.assertIsNone(find_pair(result, ("logs", "tag"), ("archive", "tag")))
        self.assertIsNone(find_pair(result, ("archive", "tag"), ("logs", "tag")))
        self.assertTrue(result.recall_limited)

    def test_self_pair_excluded(self):
        tables = {"t": {"c": ["A", "B", "C"]}}
        result = discover_foreign_keys(tables, threshold=0.1)
        self.assertEqual(result.candidates, [])

    def test_deterministic_with_fixed_seed(self):
        tables = {
            "users": {"user_id": ["u1", "u2", "u3"]},
            "orders": {"user_id": ["u1", "u2", "u9"]},
        }
        r1 = discover_foreign_keys(tables, threshold=0.5)
        r2 = discover_foreign_keys(tables, threshold=0.5)
        self.assertEqual(
            [(c.child_table, c.child_column, c.parent_table, c.parent_column) for c in r1.candidates],
            [(c.child_table, c.child_column, c.parent_table, c.parent_column) for c in r2.candidates],
        )

    def test_empty_input(self):
        result = discover_foreign_keys({}, threshold=0.5)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.relationships, [])
        self.assertEqual(result.no_evidence_columns, [])


class TestCsvLoader(unittest.TestCase):
    def test_null_tokens_and_empty_string(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.csv"), "w", encoding="utf-8") as f:
                f.write("a,b\n\\N,\n1,\\N\n1,\n")  # 含重复行, 不丢
            tables = load_csv_tables(d, null_tokens=["\\N"])
        self.assertEqual(tables["t"]["a"], [None, "1", "1"])  # 重复行保留
        self.assertEqual(tables["t"]["b"], ["", None, ""])  # 空字符串是有效值

    def test_no_null_tokens_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.csv"), "w", encoding="utf-8") as f:
                f.write('a\n""\n')  # 带引号的空字段
            tables = load_csv_tables(d)
        self.assertEqual(tables["t"]["a"], [""])  # 默认空字符串是有效值, 不是空值

    def test_blank_lines_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.csv"), "w", encoding="utf-8") as f:
                f.write("a\n1\n\n2\n")  # 中间的空白行不是数据
            tables = load_csv_tables(d)
        self.assertEqual(tables["t"]["a"], ["1", "2"])

    def test_ragged_rows(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.csv"), "w", encoding="utf-8") as f:
                f.write("a,b\n1\n2,3,4\n")
            tables = load_csv_tables(d)
        self.assertEqual(tables["t"]["a"], ["1", "2"])
        self.assertEqual(tables["t"]["b"], [None, "3"])  # 短行补 None
        self.assertEqual(tables["t"]["__extra_2"], [None, "4"])  # 长行不丢数据

    def test_duplicate_header_names(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.csv"), "w", encoding="utf-8") as f:
                f.write("a,a,\n1,2,3\n")
            tables = load_csv_tables(d)
        self.assertEqual(sorted(tables["t"].keys()), ["a", "a#2", "column_2"])


class TestReportAndJson(unittest.TestCase):
    def test_report_contains_key_sections(self):
        tables = {
            "users": {"user_id": ["u1", "u2"]},
            "orders": {"user_id": ["u1", "u9", None]},
            "legacy": {"old_id": [None, None]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        report = render_report(result)
        self.assertIn("有效外键关系", report)
        self.assertIn("达到阈值的全部候选", report)
        self.assertIn("无证据列", report)
        self.assertIn("legacy.old_id", report)
        self.assertIn("孤儿行 (1 条): 行1='u9'", report)

    def test_json_serializable_and_keeps_all_orphans(self):
        tables = {
            "parent": {"code": ["A"]},
            "child": {"code": ["B"] * 50 + ["A"]},
        }
        result = discover_foreign_keys(tables, threshold=0.1)
        data = result_to_dict(result)
        text = json.dumps(data, ensure_ascii=False, default=str)
        self.assertIn("orphan_rows", text)
        pair = find_pair(result, ("child", "code"), ("parent", "code"))
        self.assertEqual(len(pair.orphan_rows), 50)  # 每条孤儿行都保留
        self.assertEqual(len(data["candidates"][0]["orphan_rows"]), 50)


class TestOriginalLSHEnsembleInterface(unittest.TestCase):
    def test_interface_preserved(self):
        # 工具依赖的是 MinHashLSHEnsemble 原有接口, 这里锁定它没有被改动
        sig = inspect.signature(MinHashLSHEnsemble.__init__)
        for param in ("threshold", "num_perm", "num_part", "m"):
            self.assertIn(param, sig.parameters)
        self.assertTrue(hasattr(MinHashLSHEnsemble, "index"))
        self.assertTrue(hasattr(MinHashLSHEnsemble, "query"))

    def test_original_interface_end_to_end(self):
        # 按文档原用法直接驱动一遍 LSH Ensemble
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        for d in ["cat", "dog"]:
            m1.update(d.encode("utf8"))
        for d in ["cat", "dog", "fish"]:
            m2.update(d.encode("utf8"))
        ensemble = MinHashLSHEnsemble(threshold=0.5, num_perm=128)
        ensemble.index([("m2", m2, 3)])
        self.assertIn("m2", ensemble)
        self.assertEqual(list(ensemble.query(m1, 2)), ["m2"])


if __name__ == "__main__":
    unittest.main()
