#!/usr/bin/env python3
"""Build a shareable Excel workbook for CiteFaith pilot annotation.

The workbook intentionally excludes LLM outputs from the main annotation sheets
to avoid biasing the human judgment of swap validity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "week1/data/selected20.json"
WORLDS_DIR = ROOT / "week1/results/worlds"
OUTPUT = ROOT / "CiteFaith_Pilot20_标注模板.xlsx"

NAVY = "17324D"
TEAL = "0E8F86"
TEAL_LIGHT = "E6F5F2"
AMBER = "BD7218"
AMBER_LIGHT = "FFF3DF"
RED_LIGHT = "FBE9E9"
GREY = "E9EFF1"
GREY_DARK = "60717B"
YELLOW = "FFF9D9"
WHITE = "FFFFFF"
THIN_GREY = Side(style="thin", color="D9E2E5")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(clean(x) for x in value)
    if isinstance(value, bool):
        return "是" if value else "否"
    return ILLEGAL_CHARACTERS_RE.sub("", str(value))


def page_text(rows: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Page {clean(row.get('page'))}]\n{clean(row.get('text'))}" for row in rows
    )


def style_header(ws, row=1, fill=NAVY):
    for cell in ws[row]:
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.font = Font(color=WHITE, bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color=TEAL))
    ws.row_dimensions[row].height = 32


def style_body(ws, start_row, end_row, editable_columns=None):
    editable_columns = set(editable_columns or [])
    for row in ws.iter_rows(min_row=start_row, max_row=end_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
            if cell.column in editable_columns:
                cell.fill = PatternFill("solid", fgColor=YELLOW)
        ws.row_dimensions[row[0].row].height = 70


def add_dropdown(ws, column, start, end, values):
    formula = '"' + ",".join(values) + '"'
    validation = DataValidation(type="list", formula1=formula, allow_blank=True)
    validation.error = "请从下拉选项中选择。"
    validation.errorTitle = "无效选项"
    validation.prompt = "请使用下拉菜单填写，必要时在备注中解释。"
    validation.promptTitle = "标注提示"
    ws.add_data_validation(validation)
    validation.add(f"{column}{start}:{column}{end}")


def add_title_note(ws, title, note, width=110):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=1)
    ws["A1"] = title
    ws["A1"].font = Font(size=17, bold=True, color=NAVY)
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 32
    ws["A2"] = note
    ws["A2"].font = Font(size=10, color=GREY_DARK)
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = width
    ws.row_dimensions[2].height = 36


def main():
    selected = load_json(SELECTED)
    selected_rows = selected.get("questions", selected)
    by_qid = {row.get("qid"): row for row in selected_rows}
    world_rows = {}
    for path in sorted(WORLDS_DIR.glob("*.json")):
        row = load_json(path)
        if row.get("qid"):
            world_rows[row["qid"]] = row

    wb = Workbook()
    ws_info = wb.active
    ws_info.title = "填写说明"
    ws_main = wb.create_sheet("标注表")
    ws_pages = wb.create_sheet("证据页")

    # Instructions sheet.
    add_title_note(
        ws_info,
        "CiteFaith Pilot 20 · 双人独立标注模板",
        "用途：判断替换证据 E− 是否真的改变了题目的事实结论。主标注表不包含模型输出，避免影响人工判断。",
        width=115,
    )
    instructions = [
        ("使用流程", "复制一份 Excel 发给每位标注人；两人分别填写，不要在独立标注阶段查看对方结果。完成后将两份文件交回合并。"),
        ("阅读证据", "在“证据页”工作表中按 qid 和 world 筛选。plus 是原始证据，swap 是候选替换证据，benign 是良性增补，del 是删除金标证据后的世界。"),
        ("flip", "替换页支持一个与原答案不同、且可以明确判定的新答案。"),
        ("unanswerable", "替换页不能支持新答案，正确行为应是拒答或说明证据不足。"),
        ("invalid", "替换页主题不符、没有改变事实、文本不自然，或无法可靠判断。"),
        ("事实出现次数", "只统计原始文档中支持答案的事实出现次数；1 处是 single-support，2+ 处表示可能存在冗余支撑。"),
        ("文本问题", "扫描件、答案格式不一致、图表/复杂版面依赖等情况不要直接判为 swap 失败，应单独标记。"),
        ("规则代理", "标注表里的规则代理只作提示，不是最终标签。最终判断必须基于证据页全文。"),
        ("建议顺序", "先填 swap_validity 和反事实答案，再填正证据、良性页、事实出现次数、文本问题、信心和备注。"),
        ("交回材料", "请保留 qid、所有黄色标注列和备注；文件名建议使用 CiteFaith_Pilot20_标注_姓名.xlsx。"),
        ("异常题", "cfw1_019 当前没有生成四世界文件，已在标注表标记为缺失证据；不要猜测，直接在文本问题和备注中记录。"),
    ]
    ws_info.append([])
    ws_info.append(["项目", "说明"])
    style_header(ws_info, 4, TEAL)
    for item in instructions:
        ws_info.append(list(item))
    for row in ws_info.iter_rows(min_row=5, max_row=4 + len(instructions)):
        row[0].font = Font(bold=True, color=NAVY)
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
    ws_info.column_dimensions["A"].width = 18
    ws_info.column_dimensions["B"].width = 100
    for i in range(5, 5 + len(instructions)):
        ws_info.row_dimensions[i].height = 38
    ws_info.freeze_panes = "A5"

    # Main annotation sheet.
    columns = [
        "序号", "qid", "doc_id", "问题", "参考答案", "答案格式", "正证据页", "swap页",
        "规则代理", "代理原因", "良性页", "证据文件状态", "标注人", "swap_validity",
        "正证据支持", "反事实答案", "替换后可回答", "良性页有效", "事实出现次数",
        "文本问题", "标注信心", "备注", "完成状态",
    ]
    ws_main.append(columns)
    style_header(ws_main, 1, NAVY)
    ws_main.freeze_panes = "D2"
    ws_main.auto_filter.ref = f"A1:W{len(selected_rows) + 1}"

    annotation_columns = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
    for index, selected_row in enumerate(selected_rows, start=1):
        qid = selected_row.get("qid")
        world = world_rows.get(qid, {})
        swap = world.get("swap", {})
        gold_pages = world.get("gold_pages_textview") or selected_row.get("evidence_pages", [])
        benign_pages = world.get("benign_pages", [])
        world_status = "有四世界证据" if world else "缺少 worlds 文件"
        proxy = "可能有效" if swap.get("valid_proxy") else "待人工确认"
        if not world:
            proxy = "无规则代理"
        values = [
            index,
            qid,
            selected_row.get("doc_id", world.get("doc_id", "")),
            selected_row.get("question", world.get("question", "")),
            selected_row.get("answer", world.get("answer", "")),
            selected_row.get("answer_format", world.get("answer_format", "")),
            clean(gold_pages),
            clean(world.get("swap_page", "")),
            proxy,
            swap.get("reason", "") if world else "请先确认是否存在证据文件",
            clean(benign_pages),
            world_status,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            f'=IF(AND(N{index + 1}<>"",O{index + 1}<>"",Q{index + 1}<>"",R{index + 1}<>"",S{index + 1}<>"",U{index + 1}<>""),"已完成","未完成")',
        ]
        ws_main.append(values)

    style_body(ws_main, 2, len(selected_rows) + 1, annotation_columns)
    widths = [7, 13, 34, 44, 20, 13, 11, 10, 13, 28, 13, 16, 12, 16, 13, 24, 14, 13, 14, 23, 11, 38, 11]
    for col, width in enumerate(widths, start=1):
        ws_main.column_dimensions[get_column_letter(col)].width = width
    for row in ws_main.iter_rows(min_row=2, max_row=len(selected_rows) + 1):
        row[3].alignment = Alignment(vertical="top", wrap_text=True)
        row[21].alignment = Alignment(vertical="top", wrap_text=True)
    add_dropdown(ws_main, "M", 2, len(selected_rows) + 1, ["A", "B"])
    add_dropdown(ws_main, "N", 2, len(selected_rows) + 1, ["flip", "unanswerable", "invalid"])
    for column in ("O", "Q", "R"):
        add_dropdown(ws_main, column, 2, len(selected_rows) + 1, ["yes", "no", "uncertain"])
    add_dropdown(ws_main, "S", 2, len(selected_rows) + 1, ["1", "2+", "unknown"])
    add_dropdown(ws_main, "T", 2, len(selected_rows) + 1, ["none", "text_not_extractable", "answer_format_needs_review", "visual_or_layout_dependent"])
    add_dropdown(ws_main, "U", 2, len(selected_rows) + 1, ["high", "medium", "low"])
    ws_main.conditional_formatting.add(f"N2:N{len(selected_rows) + 1}", CellIsRule(operator="equal", formula=['"flip"'], fill=PatternFill("solid", fgColor=TEAL_LIGHT)))
    ws_main.conditional_formatting.add(f"N2:N{len(selected_rows) + 1}", CellIsRule(operator="equal", formula=['"invalid"'], fill=PatternFill("solid", fgColor=RED_LIGHT)))
    ws_main.conditional_formatting.add(f"W2:W{len(selected_rows) + 1}", CellIsRule(operator="equal", formula=['"已完成"'], fill=PatternFill("solid", fgColor=TEAL_LIGHT)))

    # Evidence sheet: one page per row, easy to filter by qid/world.
    page_columns = ["qid", "doc_id", "问题", "world", "世界含义", "page", "chars", "文本"]
    ws_pages.append(page_columns)
    style_header(ws_pages, 1, TEAL)
    ws_pages.freeze_panes = "A2"
    for qid in [row.get("qid") for row in selected_rows]:
        row = world_rows.get(qid)
        if not row:
            continue
        for world, meta in (("plus", "原始世界：支持答案"), ("del", "删除世界：删除金标证据"), ("swap", "替换世界：候选冲突证据"), ("benign", "良性世界：无关增补")):
            for page in row.get("worlds", {}).get(world, []):
                ws_pages.append([
                    qid,
                    row.get("doc_id", ""),
                    row.get("question", ""),
                    world,
                    meta,
                    page.get("page", ""),
                    page.get("chars", len(clean(page.get("text", "")))),
                    clean(page.get("text", "")),
                ])
    style_body(ws_pages, 2, max(2, ws_pages.max_row), set())
    for col, width in enumerate([13, 34, 44, 10, 25, 9, 10, 110], start=1):
        ws_pages.column_dimensions[get_column_letter(col)].width = width
    for row in ws_pages.iter_rows(min_row=2, max_row=ws_pages.max_row):
        row[7].alignment = Alignment(vertical="top", wrap_text=True)
        ws_pages.row_dimensions[row[0].row].height = 150
    ws_pages.auto_filter.ref = f"A1:H{ws_pages.max_row}"

    # Workbook defaults.
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.page_margins.left = 0.25
        ws.page_margins.right = 0.25
        ws.page_margins.top = 0.4
        ws.page_margins.bottom = 0.4

    wb.active = 0
    wb.save(OUTPUT)
    print(f"wrote {OUTPUT} ({len(selected_rows)} questions, {ws_pages.max_row - 1} evidence rows)")


if __name__ == "__main__":
    main()
