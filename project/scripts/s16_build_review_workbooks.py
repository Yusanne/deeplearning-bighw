#!/usr/bin/env python3
"""Build self-contained Excel review workbooks from the fair assignment CSVs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "project/results/raw"

NAVY = "17324D"
TEAL = "0E8F86"
TEAL_LIGHT = "E6F5F2"
YELLOW = "FFF9D9"
GREY = "EEF2F3"
RED_LIGHT = "FBE9E9"
WHITE = "FFFFFF"
THIN_GREY = Side(style="thin", color="D9E2E5")


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_pages(doc_id: str) -> dict[str, str]:
    path = ROOT / "project/data/pages" / f"{doc_id}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): clean(value) for key, value in payload.items()}


def parse_pages(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def style_header(ws, row: int = 1, color: str = NAVY) -> None:
    for cell in ws[row]:
        cell.fill = PatternFill("solid", fgColor=color)
        cell.font = Font(color=WHITE, bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color=TEAL))
    ws.row_dimensions[row].height = 34


def add_validation(ws, column: str, last_row: int, values: list[str]) -> None:
    validation = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True)
    validation.error = "请使用下拉菜单选择。"
    validation.errorTitle = "选项无效"
    ws.add_data_validation(validation)
    validation.add(f"{column}2:{column}{last_row}")


def add_intro(ws, reviewer: str, action_count: int, unique_count: int, overlap_count: int) -> None:
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 110
    ws.merge_cells("A1:B1")
    ws["A1"] = f"CiteFaith-20 · {reviewer}复核任务"
    ws["A1"].font = Font(size=18, bold=True, color=NAVY)
    ws.merge_cells("A2:B2")
    ws["A2"] = f"本文件包含 {action_count} 次复核：{unique_count} 个独立候选 + {overlap_count} 个共同复核候选。"
    ws["A2"].font = Font(size=12, bold=True, color=TEAL)
    ws.append([])
    ws.append(["先看这里", "说明"])
    style_header(ws, 4, TEAL)
    notes = [
        ("怎么开始", "打开“请填写”表，每行是一道任务；先读问题、参考答案和证据页范围。"),
        ("重点看什么", "打开“看证据”表，重点看原始证据、替换证据和良性证据。删除证据不是本轮重点。"),
        ("替换结果", "改变了答案=替换证据支持不同且明确的新答案；应该拒答=替换证据不足；没改变=事实仍相同；看不清=无法可靠判断。"),
        ("其他字段", "正证据支持？判断原始证据能否支持参考答案；良性页合格？判断加入良性页后答案是否应保持不变；事实出现次数不确定时选不确定。"),
        ("中文翻译", "问题和证据的中文列是 Excel 365 的辅助翻译公式。数字、日期、专有名词请回看英文原文。"),
        ("独立判断", "不要看对方的标注，不要看模型输出；只根据问题、参考答案和证据页判断。"),
        ("回传", "填写黄色列后直接把本 Excel 发回。看不清的题目允许选“看不清”，并在备注说明原因。"),
    ]
    for label, text in notes:
        ws.append([label, text])
    for row in ws.iter_rows(min_row=5, max_row=ws.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
        row[0].font = Font(bold=True, color=NAVY)
        ws.row_dimensions[row[0].row].height = 42
    ws.freeze_panes = "A5"


def build(path: Path, reviewer: str) -> None:
    rows = read_rows(path)
    action_count = len(rows)
    unique_count = sum(1 for row in rows if row["review_mode"] != "共同复核")
    overlap_count = sum(1 for row in rows if row["review_mode"] == "共同复核")

    workbook = Workbook()
    intro = workbook.active
    intro.title = "开始这里"
    add_intro(intro, reviewer, action_count, unique_count, overlap_count)

    form = workbook.create_sheet("请填写")
    headers = [
        "题号", "原 pilot 题号", "文档", "问题（英文）", "问题（中文）", "参考答案",
        "原始证据页", "替换页", "良性页", "复核模式", "替换结果（填写）", "替换后的答案（填写）",
        "正证据支持？（填写）", "良性页合格？（填写）", "事实出现次数（填写）", "文本问题（填写）", "备注（填写）",
    ]
    form.append(headers)
    style_header(form)
    form.freeze_panes = "K2"
    editable = {11, 12, 13, 14, 15, 16, 17}
    for row in rows:
        form.append([
            row["assignment_id"], row["pilot_qid"], row["doc_id"], row["question"], "",
            row["gold_answer"], row["gold_pages"], row["swap_page"], row["benign_pages"], row["review_mode"],
            "", "", "", "", "", "", "",
        ])
    for row in form.iter_rows(min_row=2, max_row=form.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
            if cell.column in editable:
                cell.fill = PatternFill("solid", fgColor=YELLOW)
        row[4].value = f'=IF(D{row[0].row}="","",TRANSLATE(D{row[0].row},"en","zh"))'
        form.row_dimensions[row[0].row].height = 72
    widths = [14, 13, 35, 56, 56, 22, 15, 12, 15, 18, 20, 28, 18, 18, 18, 24, 42]
    for index, width in enumerate(widths, start=1):
        form.column_dimensions[get_column_letter(index)].width = width
    add_validation(form, "K", form.max_row, ["改变了答案", "应该拒答", "没改变", "看不清"])
    add_validation(form, "M", form.max_row, ["是", "否", "不确定"])
    add_validation(form, "N", form.max_row, ["是", "否", "不确定"])
    add_validation(form, "O", form.max_row, ["1 次", "2 次以上", "不确定"])
    form.conditional_formatting.add(
        f"K2:K{form.max_row}",
        CellIsRule(operator="equal", formula=['"改变了答案"'], fill=PatternFill("solid", fgColor=TEAL_LIGHT)),
    )
    form.conditional_formatting.add(
        f"K2:K{form.max_row}",
        CellIsRule(operator="equal", formula=['"看不清"'], fill=PatternFill("solid", fgColor=RED_LIGHT)),
    )
    form.auto_filter.ref = f"A1:Q{form.max_row}"

    evidence = workbook.create_sheet("看证据")
    evidence.append(["题号", "文档", "问题", "证据类型", "页码", "英文证据全文", "中文辅助翻译（Excel 365）"])
    style_header(evidence, color=TEAL)
    evidence.freeze_panes = "A2"
    for row in rows:
        pages = read_pages(row["doc_id"])
        page_groups = [
            ("原始证据", parse_pages(row["gold_pages"])),
            ("替换证据", parse_pages(row["swap_page"])),
            ("良性证据", parse_pages(row["benign_pages"])),
        ]
        seen: set[tuple[str, str]] = set()
        for kind, page_numbers in page_groups:
            for page_number in page_numbers:
                key = (kind, page_number)
                if key in seen:
                    continue
                seen.add(key)
                text = pages.get(page_number, "[找不到该页的抽取文本，请打开 PDF 核对]")
                evidence.append([row["assignment_id"], row["doc_id"], row["question"], kind, page_number, text, ""])
    for row in evidence.iter_rows(min_row=2, max_row=evidence.max_row):
        row[6].value = f'=IF(F{row[0].row}="","",TRANSLATE(F{row[0].row},"en","zh"))'
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
        evidence.row_dimensions[row[0].row].height = 130
    for index, width in enumerate([14, 35, 56, 15, 10, 105, 105], start=1):
        evidence.column_dimensions[get_column_letter(index)].width = width
    evidence.auto_filter.ref = f"A1:G{evidence.max_row}"

    for sheet in workbook.worksheets:
        sheet.sheet_view.showGridLines = False
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
    output = RAW / f"review_task_{'pan' if reviewer == '潘雨萱' else 'zhang'}.xlsx"
    workbook.save(output)
    print(f"wrote {output}: {action_count} review actions, {evidence.max_row - 1} evidence rows")


def main() -> None:
    build(RAW / "review_task_pan.csv", "潘雨萱")
    build(RAW / "review_task_zhang.csv", "张轩晨")


if __name__ == "__main__":
    main()
