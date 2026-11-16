r"""采集表外键发现工具：LSH Ensemble 召回候选父列 + 原始值精确核实。

流程：对每列取"去重非空值集"，用 :class:`datasketch.MinHashLSHEnsemble`
（保持其原有接口，不修改 datasketch 库本身）按包含率阈值召回候选父列；
仅对召回的列对用原始值逐行精确核实，报告不同值包含率与按源原始行计算的
匹配率，并逐条保留孤儿行。

口径约定（使用前先读）：

- 空值仅指 ``None`` 与 float ``NaN``；空字符串 ``""`` 是有效值，照常参与
  集合与行级统计。
- 源列（子列）的空值行不算孤儿，也不计入行匹配率的分母。
- 重复行不丢：行匹配率与孤儿行都按原始行逐条统计，不去重。
- 孤儿行 = 非空且其值不在父列值集中的行，每条都保留（0 基行号 + 原始值）。
- 全空源列只报"无证据"，不参与召回，也绝不自动认定任何关系。
- 有效外键仅当：子列不同值 100% 包含于父列，且父键合格（父列无空值、
  无重复值）。父列有重复值或空值时，即使集合全包含也不标为有效外键。
- LSH Ensemble 是召回层：未被召回的列对不做精确核实，不补成全库精确
  结果，因此候选集合不是全库精确包含率结论。

命令行用法::

    python fk_discovery.py CSV_DIR [--threshold 0.5] [--null-token "\N"] \
        [--json report.json]

每个 ``*.csv`` 文件视为一张采集表（文件名去后缀为表名，首行为列名）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Optional, Sequence

from datasketch import MinHash, MinHashLSHEnsemble

# 列的标识：(表名, 列名)
ColumnKey = tuple  # tuple[str, str]

__all__ = [
    "CandidatePair",
    "ColumnProfile",
    "FKDiscoveryResult",
    "discover_foreign_keys",
    "load_csv_tables",
    "main",
    "render_report",
    "result_to_dict",
    "write_json",
]


# ---------------------------------------------------------------------------
# 空值与值编码
# ---------------------------------------------------------------------------


def _is_null(value: Any) -> bool:
    """空值：``None`` 或 float ``NaN``。空字符串不是空值。"""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _normalize(value: Any) -> Hashable:
    """把不可哈希的值（如 list/dict）转成 repr，保证可以进集合。"""
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _encode(value: Hashable) -> bytes:
    """把值确定性地编码成字节串，带类型标签，避免 1 与 "1" 等跨类型混淆。"""
    if isinstance(value, bool):
        return b"\x00" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return b"\x01" + str(value).encode("utf-8")
    if isinstance(value, float):
        return b"\x02" + repr(value).encode("utf-8")
    if isinstance(value, str):
        return b"\x03" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"\x04" + value
    return b"\x05" + repr(value).encode("utf-8")


def _minhash_of(values, num_perm: int, seed: int) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=seed)
    for v in values:
        m.update(_encode(v))
    return m


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnProfile:
    """单列画像：保留原始行（含重复、含空值）与去重非空值集。"""

    table: str
    column: str
    rows_total: int
    null_rows: int
    values: tuple  # 原始行顺序的值（已做可哈希归一化），重复行不丢
    distinct: frozenset  # 去重非空值集

    @property
    def key(self) -> ColumnKey:
        return (self.table, self.column)

    @property
    def nonnull_rows(self) -> int:
        return self.rows_total - self.null_rows

    @property
    def has_nulls(self) -> bool:
        return self.null_rows > 0

    @property
    def has_duplicates(self) -> bool:
        return len(self.distinct) != self.nonnull_rows

    @property
    def key_qualified(self) -> bool:
        """父键合格：有非空行、无空值、无重复值（可作主键）。"""
        return self.nonnull_rows > 0 and not self.has_nulls and not self.has_duplicates


@dataclass(frozen=True)
class CandidatePair:
    """一对经 LSH Ensemble 召回并用原始值核实过的 子列 -> 父列 候选。"""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    # 不同值（集合）层面
    child_distinct_size: int
    parent_distinct_size: int
    matched_distinct: int
    distinct_containment: float  # |子集 ∩ 父集| / |子集|
    # 原始行层面（重复行计入，空值行不计入分母）
    rows_total: int
    null_rows: int
    matched_rows: int
    row_match_rate: float  # 命中行 / 非空行
    orphan_rows: tuple  # 每条孤儿行都保留：((0 基行号, 原始值), ...)
    # 父键资格
    parent_has_nulls: bool
    parent_has_duplicates: bool
    parent_key_qualified: bool
    # 结论
    is_valid_fk: bool  # 全包含 且 父键合格
    invalid_reasons: tuple  # 不成立的原因（为空表示成立）


@dataclass
class FKDiscoveryResult:
    """发现结果。candidates 是召回并核实过的全部候选；relationships 是其中
    全包含且父键合格的子集。未召回的列对未做精确核实（recall_limited）。
    """

    threshold: float
    num_perm: int
    indexed_columns: int  # 参与召回的列数（去重非空值集非空的列）
    candidates: list  # list[CandidatePair]，达到阈值的全部候选
    relationships: list  # list[CandidatePair]，有效外键关系
    no_evidence_columns: list  # list[ColumnKey]，全空源列：只报无证据
    recall_limited: bool = True  # 始终为 True：未召回列对不做精确核实


# ---------------------------------------------------------------------------
# 核心：画像、召回、核实
# ---------------------------------------------------------------------------


def _profile_column(table: str, column: str, values: Sequence[Any]) -> ColumnProfile:
    normalized = tuple(_normalize(v) for v in values)
    null_rows = sum(1 for v in normalized if _is_null(v))
    distinct = frozenset(v for v in normalized if not _is_null(v))
    return ColumnProfile(
        table=table,
        column=column,
        rows_total=len(normalized),
        null_rows=null_rows,
        values=normalized,
        distinct=distinct,
    )


def _verify(child: ColumnProfile, parent: ColumnProfile) -> CandidatePair:
    """用原始值精确核实一对 子列 -> 父列。"""
    parent_set = parent.distinct
    matched_distinct = len(child.distinct & parent_set)
    containment = matched_distinct / len(child.distinct)
    matched_rows = 0
    orphans = []
    for row_no, value in enumerate(child.values):
        if _is_null(value):
            continue  # 源空值不算孤儿，也不计入分母
        if value in parent_set:
            matched_rows += 1
        else:
            orphans.append((row_no, value))  # 逐条保留，重复行不丢
    nonnull = child.nonnull_rows
    # 子列能进入核实必然有非空行；防御性兜底
    row_match_rate = matched_rows / nonnull if nonnull > 0 else 1.0
    full_containment = matched_distinct == len(child.distinct)
    reasons = []
    if not full_containment:
        reasons.append("未全包含（存在孤儿值）")
    if parent.has_nulls:
        reasons.append("父列含空值")
    if parent.has_duplicates:
        reasons.append("父列含重复值")
    is_valid = full_containment and parent.key_qualified
    return CandidatePair(
        child_table=child.table,
        child_column=child.column,
        parent_table=parent.table,
        parent_column=parent.column,
        child_distinct_size=len(child.distinct),
        parent_distinct_size=len(parent.distinct),
        matched_distinct=matched_distinct,
        distinct_containment=containment,
        rows_total=child.rows_total,
        null_rows=child.null_rows,
        matched_rows=matched_rows,
        row_match_rate=row_match_rate,
        orphan_rows=tuple(orphans),
        parent_has_nulls=parent.has_nulls,
        parent_has_duplicates=parent.has_duplicates,
        parent_key_qualified=parent.key_qualified,
        is_valid_fk=is_valid,
        invalid_reasons=tuple(reasons),
    )


def discover_foreign_keys(
    tables: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    threshold: float = 0.5,
    num_perm: int = 256,
    num_part: int = 16,
    m: int = 8,
    seed: int = 1,
) -> FKDiscoveryResult:
    """从采集表中发现可能的外键关系。

    Args:
        tables: ``{表名: {列名: [原始值, ...]}}``。空值用 ``None``（或 float
            ``NaN``）表示；空字符串是有效值。
        threshold: LSH Ensemble 的包含率召回阈值，达到该阈值的候选才会被
            精确核实并报告。
        num_perm: MinHash 的排列数（越大召回越准）。
        num_part: LSH Ensemble 的分区数。
        m: LSH Ensemble 的内存因子。
        seed: MinHash 随机种子，固定则结果可复现。

    Returns:
        FKDiscoveryResult。未被 LSH Ensemble 召回的列对不做精确核实。

    """
    profiles: dict[ColumnKey, ColumnProfile] = {}
    for table, columns in tables.items():
        for column, values in columns.items():
            if isinstance(values, (str, bytes)):
                raise TypeError(f"列 {table}.{column} 的值应是序列，而不是字符串")
            profiles[(table, column)] = _profile_column(table, column, values)

    # 全空源列：去重非空值集为空，无法索引也无法查询，只报无证据
    no_evidence = sorted(k for k, p in profiles.items() if len(p.distinct) == 0)
    indexable = {k: p for k, p in profiles.items() if len(p.distinct) > 0}

    candidates: list[CandidatePair] = []
    if indexable:
        minhashes = {k: _minhash_of(p.distinct, num_perm, seed) for k, p in indexable.items()}
        # 原接口：index([(key, minhash, size), ...])，所有列都作为候选父列入索引
        ensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part, m=m)
        ensemble.index([(k, minhashes[k], len(p.distinct)) for k, p in indexable.items()])
        for child_key in sorted(indexable):
            child = indexable[child_key]
            # 原接口：query(minhash, size) 召回包含率达到阈值的父列
            for parent_key in ensemble.query(minhashes[child_key], len(child.distinct)):
                if parent_key == child_key:
                    continue  # 自己必然包含自己，跳过
                candidates.append(_verify(child, indexable[parent_key]))

    candidates.sort(
        key=lambda c: (c.child_table, c.child_column, -c.distinct_containment, c.parent_table, c.parent_column)
    )
    relationships = [c for c in candidates if c.is_valid_fk]
    return FKDiscoveryResult(
        threshold=threshold,
        num_perm=num_perm,
        indexed_columns=len(indexable),
        candidates=candidates,
        relationships=relationships,
        no_evidence_columns=no_evidence,
    )


# ---------------------------------------------------------------------------
# CSV 采集表加载
# ---------------------------------------------------------------------------


def _dedupe_header(header: list) -> list:
    """列名去重/补名：空列名补 column_N，重名追加 #2、#3 ..."""
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(header):
        name = name if name else f"column_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}#{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def load_csv_tables(
    directory,
    *,
    null_tokens=(),
    encoding: str = "utf-8-sig",
) -> dict:
    """把目录下每个 ``*.csv`` 读成一张表：``{表名: {列名: [值, ...]}}``。

    空字符串保持为有效值；只有命中 ``null_tokens`` 的字段才视为空值
    ``None``。短行缺失的字段补 ``None``；长行多出的字段放进 ``__extra_N``
    列，不丢数据。重复行原样保留。完全空白的行跳过（与 pandas 一致；
    单列文件里的空字符串请写成带引号的 ``""``，否则与空行无法区分）。
    """
    tokens = frozenset(null_tokens)
    tables: dict[str, dict[str, list]] = {}
    for path in sorted(Path(directory).glob("*.csv")):
        with open(path, newline="", encoding=encoding) as f:
            rows = list(csv.reader(f))
        if not rows:
            tables[path.stem] = {}
            continue
        header = _dedupe_header(rows[0])
        columns: dict[str, list] = {name: [] for name in header}
        n_data = 0
        for raw in rows[1:]:
            if not raw:
                continue  # 空白行不是数据
            if len(raw) > len(header):
                for i in range(len(header), len(raw)):
                    extra = f"__extra_{i}"
                    if extra not in columns:
                        columns[extra] = [None] * n_data
                    header.append(extra)
            if len(raw) < len(header):
                raw = raw + [None] * (len(header) - len(raw))
            for name, value in zip(header, raw):
                columns[name].append(None if value in tokens else value)
            n_data += 1
        tables[path.stem] = columns
    return tables


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _render_pair(c: CandidatePair, max_orphans: Optional[int]) -> list:
    status = "有效" if c.is_valid_fk else "候选"
    lines = [f"  [{status}] {c.child_table}.{c.child_column} -> {c.parent_table}.{c.parent_column}"]
    lines.append(f"    不同值包含率: {c.distinct_containment:.4f} ({c.matched_distinct}/{c.child_distinct_size})")
    nonnull = c.rows_total - c.null_rows
    lines.append(
        f"    行匹配率: {c.row_match_rate:.4f} (非空行 {c.matched_rows}/{nonnull}; "
        f"总行数 {c.rows_total}, 空值行 {c.null_rows})"
    )
    if c.parent_key_qualified:
        lines.append("    父键: 合格 (无空值, 无重复)")
    else:
        problems = []
        if c.parent_has_nulls:
            problems.append("含空值")
        if c.parent_has_duplicates:
            problems.append("含重复值")
        lines.append(f"    父键: 不合格 ({', '.join(problems)})")
    if not c.orphan_rows:
        lines.append("    孤儿行: 无")
    else:
        shown = c.orphan_rows if max_orphans is None else c.orphan_rows[:max_orphans]
        items = ", ".join(f"行{i}={_short(repr(v))}" for i, v in shown)
        suffix = "" if len(shown) == len(c.orphan_rows) else f", ... 共 {len(c.orphan_rows)} 条(完整见 JSON)"
        lines.append(f"    孤儿行 ({len(c.orphan_rows)} 条): {items}{suffix}")
    if c.is_valid_fk:
        lines.append("    结论: 有效外键 (不同值全包含 且 父键合格)")
    else:
        lines.append(f"    结论: 不成立 ({'; '.join(c.invalid_reasons)})")
    return lines


def render_report(result: FKDiscoveryResult, *, max_orphans: Optional[int] = None) -> str:
    """渲染人类可读的中文报告。``max_orphans`` 只限制打印条数，结果数据
    中始终保留全部孤儿行。
    """
    lines = ["=" * 64, "外键发现报告"]
    lines.append(
        f"阈值 threshold={result.threshold}, num_perm={result.num_perm}, "
        f"参与召回列数={result.indexed_columns}, 精确核实列对数={len(result.candidates)}"
    )
    lines.append("=" * 64)

    lines.append("")
    lines.append(f"【有效外键关系】(不同值全包含 且 父键合格) 共 {len(result.relationships)} 条")
    if not result.relationships:
        lines.append("  (无)")
    for c in result.relationships:
        lines.extend(_render_pair(c, max_orphans))

    lines.append("")
    lines.append(f"【达到阈值的全部候选】(LSH Ensemble 召回并经原始值核实) 共 {len(result.candidates)} 条")
    if not result.candidates:
        lines.append("  (无)")
    for c in result.candidates:
        lines.extend(_render_pair(c, max_orphans))

    lines.append("")
    lines.append(f"【无证据列】(全空源列, 不认定任何关系) 共 {len(result.no_evidence_columns)} 列")
    if not result.no_evidence_columns:
        lines.append("  (无)")
    for table, column in result.no_evidence_columns:
        lines.append(f"  {table}.{column}")

    lines.append("")
    lines.append("说明:")
    lines.append("  - 候选由 LSH Ensemble 按各列去重非空值集召回; 未被召回的列对未做")
    lines.append("    精确核实, 本报告不是全库精确包含率结论。")
    lines.append("  - 有效外键需同时满足: 子列不同值 100% 包含于父列, 且父列无空值、无重复值。")
    lines.append("  - 空值=None/NaN; 空字符串是有效值; 源空值行不算孤儿; 行级统计保留重复行。")
    return "\n".join(lines)


def _pair_to_dict(c: CandidatePair) -> dict:
    return {
        "child": {"table": c.child_table, "column": c.child_column},
        "parent": {"table": c.parent_table, "column": c.parent_column},
        "distinct_containment": c.distinct_containment,
        "matched_distinct": c.matched_distinct,
        "child_distinct_size": c.child_distinct_size,
        "parent_distinct_size": c.parent_distinct_size,
        "rows_total": c.rows_total,
        "null_rows": c.null_rows,
        "matched_rows": c.matched_rows,
        "row_match_rate": c.row_match_rate,
        "orphan_rows": [{"row": i, "value": v} for i, v in c.orphan_rows],
        "parent_key_qualified": c.parent_key_qualified,
        "parent_has_nulls": c.parent_has_nulls,
        "parent_has_duplicates": c.parent_has_duplicates,
        "is_valid_fk": c.is_valid_fk,
        "invalid_reasons": list(c.invalid_reasons),
    }


def result_to_dict(result: FKDiscoveryResult) -> dict:
    """导出为可 JSON 序列化的字典（孤儿行逐条保留）。"""
    return {
        "threshold": result.threshold,
        "num_perm": result.num_perm,
        "indexed_columns": result.indexed_columns,
        "recall_limited": result.recall_limited,
        "note": "未被 LSH Ensemble 召回的列对未做精确核实, 候选集合不是全库精确结果",
        "relationships": [_pair_to_dict(c) for c in result.relationships],
        "candidates": [_pair_to_dict(c) for c in result.candidates],
        "no_evidence_columns": [{"table": t, "column": c} for t, c in result.no_evidence_columns],
    }


def write_json(result: FKDiscoveryResult, path) -> None:
    Path(path).write_text(
        json.dumps(result_to_dict(result), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="采集表外键发现: LSH Ensemble 召回候选父列, 原始值精确核实",
    )
    parser.add_argument("csv_dir", help="采集表目录, 每个 *.csv 为一张表")
    parser.add_argument("--threshold", type=float, default=0.5, help="包含率召回阈值 (默认 0.5)")
    parser.add_argument("--num-perm", type=int, default=256, help="MinHash 排列数 (默认 256)")
    parser.add_argument("--num-part", type=int, default=16, help="LSH Ensemble 分区数 (默认 16)")
    parser.add_argument("--m", type=int, default=8, help="LSH Ensemble 内存因子 (默认 8)")
    parser.add_argument(
        "--null-token",
        action="append",
        default=[],
        help="视为空值的字段文本, 可多次指定; 默认无 (空字符串是有效值)",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 编码 (默认 utf-8-sig)")
    parser.add_argument("--json", dest="json_path", default=None, help="把完整结果 (含全部孤儿行) 写到该 JSON 文件")
    parser.add_argument("--max-orphans", type=int, default=None, help="仅限制终端打印的孤儿行条数")
    args = parser.parse_args(argv)

    tables = load_csv_tables(args.csv_dir, null_tokens=args.null_token, encoding=args.encoding)
    result = discover_foreign_keys(
        tables,
        threshold=args.threshold,
        num_perm=args.num_perm,
        num_part=args.num_part,
        m=args.m,
    )
    print(render_report(result, max_orphans=args.max_orphans))
    if args.json_path:
        write_json(result, args.json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
