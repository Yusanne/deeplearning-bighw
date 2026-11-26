#!/usr/bin/env python3
"""Merge two human-friendly CiteFaith annotation workbooks."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


NAVY = "17324D"
TEAL = "0E8F86"
RED_LIGHT = "FBE9E9"
WHITE = "FFFFFF"
THIN_GREY = Side(style="thin", color="D9E2E5")


FIELDS = [
    "替换结果（填写）",
    "正证据支持？（填写）",
    "良性页合格？（填写）",
    "事实出现次数（填写）",
]


def read_file(path: Path):
    wb = load_workbook(path, data_only=False, read_only=True)
    ws = wb["请填写"]
    headers = [cell.value for cell in ws[1]]
    index = {header: i for i, header in enumerate(headers)}
    required = ["题号", "问题（英文）", "问题（中文）", *FIELDS, "替换后的答案（填写）", "文本问题（填写）", "备注（填写）"]
    missing = [field for field in required if field not in index]
    if missing:
        raise ValueError(f"{path} 缺少列：{', '.join(missing)}")
    rows = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        qid = row[index["题号"]]
        if not qid:
            continue
        rows[str(qid)] = {field: row[index[field]] or "" for field in required}
    return rows


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return ""
    n = len(pairs)
    observed = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def style_header(ws):
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(color=WHITE, bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color=TEAL))
    ws.row_dimensions[1].height = 34


def main():
    parser = argparse.ArgumentParser(description="合并两份简易 CiteFaith 标注 Excel")
    parser.add_argument("annotator_a", type=Path)
    parser.add_argument("annotator_b", type=Path)
    parser.add_argument("--out", type=Path, default=Path("CiteFaith_Pilot20_简易合并复核.xlsx"))
    args = parser.parse_args()

    a, b = read_file(args.annotator_a), read_file(args.annotator_b)
    qids = list(dict.fromkeys([*a.keys(), *b.keys()]))
    rows = []
    for qid in qids:
        left, right = a.get(qid, {}), b.get(qid, {})
        disagreements = [field for field in FIELDS if left.get(field) and right.get(field) and left.get(field) != right.get(field)]
        missing = [name for name, item in (("A", left), ("B", right)) if not item or not item.get("替换结果（填写）")]
        rows.append([
            qid,
            left.get("问题（中文）") or right.get("问题（中文）", ""),
            left.get("替换结果（填写）", ""), right.get("替换结果（填写）", ""),
            left.get("替换后的答案（填写）", ""), right.get("替换后的答案（填写）", ""),
            left.get("正证据支持？（填写）", ""), right.get("正证据支持？（填写）", ""),
            left.get("良性页合格？（填写）", ""), right.get("良性页合格？（填写）", ""),
            left.get("事实出现次数（填写）", ""), right.get("事实出现次数（填写）", ""),
            left.get("文本问题（填写）", ""), right.get("文本问题（填写）", ""),
            left.get("备注（填写）", ""), right.get("备注（填写）", ""),
            ", ".join(disagreements) if disagreements else ("缺少标注：" + ", ".join(missing) if missing else ""),
            "需复核" if disagreements or missing else "暂时一致",
        ])

    wb = Workbook()
    result = wb.active
    result.title = "合并复核"
    result.append(["题号", "问题", "A 替换结果", "B 替换结果", "A 反事实答案", "B 反事实答案", "A 正证据", "B 正证据", "A 良性页", "B 良性页", "A 出现次数", "B 出现次数", "A 文本问题", "B 文本问题", "A 备注", "B 备注", "分歧说明", "复核状态"])
    style_header(result)
    for row in rows:
        result.append(row)
    result.freeze_panes = "C2"
    result.auto_filter.ref = f"A1:R{result.max_row}"
    for row in result.iter_rows(min_row=2, max_row=result.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
        if row[-1].value == "需复核":
            for cell in row:
                cell.fill = PatternFill("solid", fgColor=RED_LIGHT)
        result.row_dimensions[row[0].row].height = 62
    for index, width in enumerate([12, 48, 18, 18, 28, 28, 16, 16, 16, 16, 18, 18, 24, 24, 38, 38, 38, 16], start=1):
        result.column_dimensions[get_column_letter(index)].width = width

    stats = wb.create_sheet("一致性统计")
    stats.append(["字段", "一致数", "双方均填写数", "一致率", "Cohen's κ"])
    style_header(stats)
    for field in FIELDS:
        values_a = [a.get(qid, {}).get(field, "") for qid in qids]
        values_b = [b.get(qid, {}).get(field, "") for qid in qids]
        pairs = [(x, y) for x, y in zip(values_a, values_b) if x and y]
        same = sum(x == y for x, y in pairs)
        n = len(pairs)
        stats.append([field, same, n, same / n if n else "", kappa(values_a, values_b)])
    stats.append([])
    stats.append(["需复核题数", sum(row[-1] == "需复核" for row in rows), "", "", ""])
    for row in stats.iter_rows(min_row=2, max_row=stats.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
    for col, width in enumerate([32, 14, 18, 14, 14], start=1):
        stats.column_dimensions[get_column_letter(col)].width = width
    for row in range(2, stats.max_row + 1):
        if isinstance(stats.cell(row, 4).value, float):
            stats.cell(row, 4).number_format = "0.0%"
        if isinstance(stats.cell(row, 5).value, float):
            stats.cell(row, 5).number_format = "0.000"

    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
    wb.save(args.out)
    print(f"wrote {args.out}; {len(rows)} questions; review={sum(row[-1] == '需复核' for row in rows)}")


if __name__ == "__main__":
    main()
