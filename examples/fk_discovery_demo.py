"""外键发现演示：合成一批带脏数据的采集表，用 fk_discovery 找可能的外键。

涵盖的脏数据情形：
- 干净外键（orders.user_id -> users.user_id）
- 孤儿行（orders.product_sku 里有产品表不存在的 sku）
- 空字符串是有效值（orders.coupon -> coupons.code）
- 父列有重复值 / 空值，即使全包含也不成立（departments / rooms）
- 全空源列只报无证据（legacy.old_id）
- 远低于阈值的列对不被召回，也不补做精确核实（logs.tag vs archive.tag）

运行：python examples/fk_discovery_demo.py
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fk_discovery import discover_foreign_keys, load_csv_tables, render_report, write_json

TABLES = {
    "users": {
        "user_id": ["u1", "u2", "u3", "u4", "u5"],
        "name": ["张三", "李四", "王五", "赵六", "钱七"],
        "city": ["北京", "上海", "北京", None, "广州"],
    },
    "orders": {
        "order_id": [f"o{i}" for i in range(10)],
        # 干净外键 -> users.user_id
        "user_id": ["u1", "u2", "u2", "u3", "u5", "u5", "u1", "u4", "u2", "u3"],
        # 脏: p8/p9 不在 products.sku 里 -> 孤儿行 (行2, 行4, 行6, 重复行不丢)
        "product_sku": ["p1", "p2", "p8", "p1", "p9", "p3", "p8", "p4", "p2", "p1"],
        # 空字符串是有效值, 空值不算孤儿 -> coupons.code
        "coupon": ["", "", None, "C1", "", "C1", None, "", "C1", ""],
    },
    "products": {
        "sku": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "title": ["键盘", "鼠标", "显示器", "耳机", "音箱", "摄像头"],
    },
    "coupons": {
        # 合格父键: 无空值无重复, 且空字符串是其中一个有效值
        "code": ["", "C1", "C2"],
    },
    "departments": {
        # 父列有重复值 D2: 即使 employees.dept 全包含也不成立
        "dept_code": ["D1", "D2", "D2", "D3"],
        "dept_name": ["研发", "市场", "市场", "财务"],
    },
    "employees": {
        "emp_id": [f"e{i}" for i in range(6)],
        "dept": ["D1", "D2", "D2", "D3", "D1", "D2"],
    },
    "rooms": {
        # 父列含空值: 即使 assets.room 全包含也不成立
        "room_no": ["101", "102", None, "103"],
    },
    "assets": {
        "asset_id": [f"a{i}" for i in range(4)],
        "room": ["101", "102", "103", "101"],
    },
    "legacy": {
        # 全空源列: 只报无证据, 不认定任何关系
        "old_id": [None, None, None],
        "note": ["x1", "x2", "x3"],
    },
    "logs": {
        # 与 archive.tag 只重叠 2/20, 低于阈值, 不应出现在结果里
        "tag": [f"t{i:02d}" for i in range(20)],
    },
    "archive": {
        "tag": ["t00", "t01"] + [f"a{i:02d}" for i in range(18)],
    },
}


def main():
    result = discover_foreign_keys(TABLES, threshold=0.5)
    print(render_report(result))

    out_dir = tempfile.mkdtemp(prefix="fk_demo_")
    json_path = os.path.join(out_dir, "fk_report.json")
    write_json(result, json_path)
    print(f"\n完整 JSON 报告 (含全部孤儿行) 已写入: {json_path}")

    # CSV 采集表路径: 把同样的表写成 CSV (空值写成 \N), 走加载器再跑一次。
    # 用 QUOTE_ALL 是为了让单列文件里的空字符串 "" 不与空白行混淆。
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir)
    for table, columns in TABLES.items():
        with open(os.path.join(csv_dir, f"{table}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            names = list(columns)
            writer.writerow(names)
            for row in zip(*(columns[n] for n in names)):
                writer.writerow(["\\N" if v is None else v for v in row])
    csv_result = discover_foreign_keys(load_csv_tables(csv_dir, null_tokens=["\\N"]), threshold=0.5)
    same = [(c.child_table, c.child_column, c.parent_table, c.parent_column) for c in result.relationships] == [
        (c.child_table, c.child_column, c.parent_table, c.parent_column) for c in csv_result.relationships
    ]
    print(f"CSV 路径复跑, 有效关系一致: {same}")
    print(f"命令行用法: python fk_discovery.py {csv_dir} --null-token '\\N'")


if __name__ == "__main__":
    main()
