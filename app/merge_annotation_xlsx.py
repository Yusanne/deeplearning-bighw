#!/usr/bin/env python3
"""Merge two independently completed CiteFaith annotation workbooks."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


NAVY = "17324D"
TEAL = "0E8F86"
TEAL_LIGHT = "E6F5F2"
RED_LIGHT = "FBE9E9"
GREY_DARK = "60717B"
WHITE = "FFFFFF"
THIN_GREY = Side(style="thin", color="D9E2E5")


def read_annotations(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    wb = load_workbook(path, data_only=False, read_only=True)
    ws = wb["标注表"]
    headers = [cell.value for cell in ws[1]]
    index = {header: i for i, header in enumerate(headers)}
    required = ["qid", "问题", "证据文件状态", "swap_validity", "正证据支持", "反事实答案", "替换后可回答", "良性页有效", "事实出现次数", "文本问题", "标注信心", "备注"]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"{path} 缺少列：{', '.join(missing)}")
    annotations = {}
    annotator = ""
    for row in ws.iter_rows(min_row=2, values_only=True):
        qid = row[index["qid"]]
        if not qid:
            continue
        annotator = annotator or str(row[index.get("标注人", 0)] or "")
        annotations[str(qid)] = {name: row[index[name]] for name in required}
    return annotator, annotations


def kappa(values_a: list[str], values_b: list[str]) -> float | None:
    pairs = [(a, b) for a, b in zip(values_a, values_b) if a and b]
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in set(counts_a) | set(counts_b))
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def exact_agreement(values_a: list[str], values_b: list[str]) -> tuple[int, int, float | None]:
    pairs = [(a, b) for a, b in zip(values_a, values_b) if a and b]
    if not pairs:
        return 0, 0, None
    same = sum(a == b for a, b in pairs)
    return same, len(pairs), same / len(pairs)


def style_header(ws, row=1):
    for cell in ws[row]:
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(color=WHITE, bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color=TEAL))
    ws.row_dimensions[row].height = 32


def main():
    parser = argparse.ArgumentParser(description="合并两位标注人的 CiteFaith Excel")
    parser.add_argument("annotator_a", type=Path)
    parser.add_argument("annotator_b", type=Path)
    parser.add_argument("--out", type=Path, default=Path("CiteFaith_Pilot20_合并复核.xlsx"))
    args = parser.parse_args()

    name_a, data_a = read_annotations(args.annotator_a)
    name_b, data_b = read_annotations(args.annotator_b)
    qids = list(dict.fromkeys([*data_a.keys(), *data_b.keys()]))

    comparison_fields = ["swap_validity", "正证据支持", "替换后可回答", "良性页有效", "事实出现次数"]
    output_rows = []
    for qid in qids:
        a = data_a.get(qid, {})
        b = data_b.get(qid, {})
        disagreements = [field for field in comparison_fields if a.get(field) and b.get(field) and a.get(field) != b.get(field)]
        missing = [label for label, row in (("A", a), ("B", b)) if not row or not row.get("swap_validity")]
        status = "需复核" if disagreements or missing else "一致/待最终裁定"
        output_rows.append([
            qid,
            a.get("问题") or b.get("问题", ""),
            a.get("证据文件状态") or b.get("证据文件状态", ""),
            a.get("swap_validity", ""), b.get("swap_validity", ""),
            a.get("正证据支持", ""), b.get("正证据支持", ""),
            a.get("反事实答案", ""), b.get("反事实答案", ""),
            a.get("替换后可回答", ""), b.get("替换后可回答", ""),
            a.get("良性页有效", ""), b.get("良性页有效", ""),
            a.get("事实出现次数", ""), b.get("事实出现次数", ""),
            a.get("文本问题", ""), b.get("文本问题", ""),
            a.get("标注信心", ""), b.get("标注信心", ""),
            a.get("备注", ""), b.get("备注", ""),
            ", ".join(disagreements) if disagreements else ("缺少标注：" + ", ".join(missing) if missing else ""),
            status,
        ])

    wb = Workbook()
    ws = wb.active
    ws.title = "合并结果"
    headers = ["qid", "问题", "证据文件状态", "A swap", "B swap", "A 正证据", "B 正证据", "A 反事实答案", "B 反事实答案", "A 可回答", "B 可回答", "A 良性", "B 良性", "A 出现次数", "B 出现次数", "A 文本问题", "B 文本问题", "A 信心", "B 信心", "A 备注", "B 备注", "分歧字段", "复核状态"]
    ws.append(headers)
    style_header(ws)
    for row in output_rows:
        ws.append(row)
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:W{ws.max_row}"
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
        if row[-1].value == "需复核":
            for cell in row:
                cell.fill = PatternFill("solid", fgColor=RED_LIGHT)
        ws.row_dimensions[row[0].row].height = 62
    widths = [13, 44, 16, 14, 14, 13, 13, 24, 24, 13, 13, 13, 13, 14, 14, 23, 23, 11, 11, 38, 38, 35, 18]
    for col, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col)].width = width

    summary = wb.create_sheet("一致性统计")
    summary.append(["CiteFaith Pilot 20 · 双人标注一致性统计", ""])
    summary["A1"].font = Font(size=16, bold=True, color=NAVY)
    summary.append(["标注文件 A", str(args.annotator_a)])
    summary.append(["标注文件 B", str(args.annotator_b)])
    summary.append(["标注人字段", f"A={name_a or '未填写'}；B={name_b or '未填写'}"])
    summary.append([])
    summary.append(["字段", "一致数", "双方均填写数", "一致率", "Cohen's κ", "说明"])
    style_header(summary, 6)
    for field in comparison_fields:
        values_a = [data_a.get(qid, {}).get(field, "") for qid in qids]
        values_b = [data_b.get(qid, {}).get(field, "") for qid in qids]
        same, n, rate = exact_agreement(values_a, values_b)
        summary.append([field, same, n, rate if rate is not None else "", kappa(values_a, values_b) if rate is not None else "", "只对双方均填写的题目计算"])
    summary.append([])
    summary.append(["需复核题数", sum(row[-1] == "需复核" for row in output_rows), "", "", "", "复核结果页中的红色行"])
    summary.freeze_panes = "A7"
    for row in summary.iter_rows(min_row=7, max_row=summary.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
    summary.column_dimensions["A"].width = 24
    summary.column_dimensions["B"].width = 14
    summary.column_dimensions["C"].width = 18
    summary.column_dimensions["D"].width = 14
    summary.column_dimensions["E"].width = 14
    summary.column_dimensions["F"].width = 38
    for row in range(7, summary.max_row + 1):
        if isinstance(summary.cell(row, 4).value, float):
            summary.cell(row, 4).number_format = "0.0%"
        if isinstance(summary.cell(row, 5).value, float):
            summary.cell(row, 5).number_format = "0.000"

    info = wb.create_sheet("复核说明")
    info.append(["合并复核说明"])
    info["A1"].font = Font(size=16, bold=True, color=NAVY)
    notes = [
        "红色行表示两位标注人在核心字段上有分歧，或其中一人没有完成标注。",
        "先讨论 swap_validity、正证据支持、替换后可回答和事实出现次数，再形成最终裁定。",
        "Cohen's κ 只对双方均填写的题目计算，同时应结合原始一致率和分歧案例解释。",
        "不要用自动合并结果替代人工复核；合并表用于定位分歧和生成统计。",
    ]
    for note in notes:
        info.append([note])
    info.column_dimensions["A"].width = 115
    for row in info.iter_rows(min_row=2, max_row=info.max_row):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
        info.row_dimensions[row[0].row].height = 32

    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0

    wb.save(args.out)
    print(f"wrote {args.out}; {len(output_rows)} questions; review={sum(row[-1] == '需复核' for row in output_rows)}")


if __name__ == "__main__":
    main()
