#!/usr/bin/env python3
"""Build a human-friendly CiteFaith annotation workbook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "week1/data/selected20.json"
WORLDS_DIR = ROOT / "week1/results/worlds"
OUTPUT = ROOT / "CiteFaith_Pilot20_简易标注表.xlsx"

NAVY = "17324D"
TEAL = "0E8F86"
TEAL_LIGHT = "E6F5F2"
YELLOW = "FFF9D9"
GREY = "EEF2F3"
GREY_DARK = "60717B"
WHITE = "FFFFFF"
RED_LIGHT = "FBE9E9"
THIN_GREY = Side(style="thin", color="D9E2E5")

QUESTION_ZH = {
    "cfw1_001": "除了公共假日之外，根据给出的营业时间，柜台在星期二几点关闭？",
    "cfw1_002": "第 1 页开头提到的日期是什么？请按 YYYY-MM-DD 格式回答。",
    "cfw1_003": "第 14 页中 INF SERCRL LLP 的传真号码是多少？",
    "cfw1_004": "因果关系阶梯中的第二级指的是什么？",
    "cfw1_005": "Kildee 先生从哪一年开始参与儿童营养工作？",
    "cfw1_006": "这份文件的日期是什么？请按 YYYY-MM-DD 格式回答。",
    "cfw1_007": "德克萨斯州大休斯敦地区有多少个地理标记？",
    "cfw1_008": "亚马逊 2017 财年未兑现礼品卡的负债金额是多少？请以十亿美元为单位回答。",
    "cfw1_009": "文件第一页提到的州长叫什么名字？",
    "cfw1_010": "亚马逊 2017 年不动产和设备的折旧费用是多少？请以十亿美元为单位回答。",
    "cfw1_011": "这辆车的 Wi-Fi Connect 使用什么技术？",
    "cfw1_012": "哪篇论文使用了 Fast NMS 处理？",
    "cfw1_013": "“计算机化医疗信息的主要安全问题”图表讨论的是哪些年份？请列出所有年份。",
    "cfw1_014": "从 2016 年到 2017 年，allowance 回报率的变化百分比是多少？结果四舍五入到一位小数。",
    "cfw1_015": "截至 2023 年 1 月 28 日的财年，Best Buy 的商誉是多少？",
    "cfw1_016": "Adobe 在 2015 财年主要通过什么渠道销售产品？",
    "cfw1_017": "Laurent Nahmiash 先生的电子邮箱地址是什么？",
    "cfw1_018": "编辑一条短信需要多少步？",
    "cfw1_019": "PPT 的“coding time”部分介绍了哪个用于训练的数据集？",
    "cfw1_020": "这篇论文的实验总共使用了多少个数据集？请给出整数。",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(clean(item) for item in value)
    text = str(value)
    return "".join(char for char in text if char in "\n\r\t" or ord(char) >= 32)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pages_text(pages: list[dict[str, Any]]) -> str:
    parts = []
    for page in pages:
        parts.append(f"【Page {clean(page.get('page'))}】\n{clean(page.get('text'))}")
    return "\n\n".join(parts)


def add_list_validation(ws, column: str, last_row: int, values: list[str]):
    validation = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True)
    validation.error = "请使用下拉菜单选择。"
    validation.errorTitle = "选项无效"
    ws.add_data_validation(validation)
    validation.add(f"{column}2:{column}{last_row}")


def style_header(ws, row=1, color=NAVY):
    for cell in ws[row]:
        cell.fill = PatternFill("solid", fgColor=color)
        cell.font = Font(color=WHITE, bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color=TEAL))
    ws.row_dimensions[row].height = 34


def main():
    selected = load(SELECTED)
    selected_rows = selected.get("questions", selected)
    worlds = {}
    for path in sorted(WORLDS_DIR.glob("*.json")):
        row = load(path)
        worlds[row["qid"]] = row

    wb = Workbook()
    intro = wb.active
    intro.title = "开始这里"
    form = wb.create_sheet("请填写")
    evidence = wb.create_sheet("看证据")

    # Simple instructions.
    intro.column_dimensions["A"].width = 22
    intro.column_dimensions["B"].width = 105
    intro["A1"] = "CiteFaith Pilot 20 · 简易人工标注表"
    intro["A1"].font = Font(size=18, bold=True, color=NAVY)
    intro["A2"] = "给师兄的版本：只需要填写“请填写”表里的黄色列。"
    intro["A2"].font = Font(size=12, color=TEAL, bold=True)
    intro.merge_cells("A1:B1")
    intro.merge_cells("A2:B2")
    intro.append([])
    intro.append(["怎么做", "说明"])
    style_header(intro, 4, TEAL)
    notes = [
        ("第 1 步", "打开“请填写”表，每一行是一道题。先读问题和参考答案。"),
        ("第 2 步", "打开“看证据”表，按题号找同一道题。重点只看“原始证据”和“替换证据”；“良性证据”用于确认它不会改变答案。“删除证据”可以先不看。"),
        ("第 3 步", "回到“请填写”表，只填写黄色列：替换结果、替换后的答案、正证据是否支持、良性页是否合格、事实出现次数、备注。问题中文翻译只用于理解，不需要修改。"),
        ("替换结果", "选“改变了答案”：替换证据支持一个不同的明确答案；选“应该拒答”：替换证据不足以回答；选“没改变”：替换并没有改变事实；看不清时选“看不清”。"),
        ("正证据支持", "原始证据页能否直接支持参考答案。"),
        ("良性页合格", "加入 benign 页面后，不应改变原问题的事实答案。"),
        ("事实出现次数", "不需要为了这一项通读整份文档。看不出来就选“不确定”；只有明确发现同一事实在多个页面重复时才选“2 次以上”。"),
        ("四个世界是什么", "它们不是四篇文章，而是同一道题的四种证据版本：原始=正常证据；删除=拿掉支持答案的页；替换=换成相似但事实不同的页；良性=加入无关页。"),
        ("需要看全文吗", "不需要。每题先看问题、参考答案、原始证据和替换证据即可；只有遇到明显重复或分歧时，才额外查看其他页面。"),
        ("中文翻译", "“看证据”表最后一列是中文辅助翻译。Microsoft 365 Excel 支持时会自动生成；如果显示为空或函数不支持，请以英文原文为准，或使用 Excel/WPS 自带翻译。"),
        ("注意", "不要看模型输出；这是人工独立判断。规则代理没有放进这个版本，避免影响判断。"),
        ("异常题", "cfw1_019 没有四世界证据文件，请在备注写“缺少证据文件”，不要猜测。"),
        ("交回", "填好后直接把 Excel 发回即可。文件名可改成 CiteFaith_Pilot20_标注_姓名.xlsx。"),
    ]
    for item in notes:
        intro.append(list(item))
    for row in intro.iter_rows(min_row=5, max_row=intro.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
        row[0].font = Font(bold=True, color=NAVY)
        intro.row_dimensions[row[0].row].height = 34
    intro.freeze_panes = "A5"

    # Human-facing form. Keep it narrow and self-explanatory.
    headers = [
        "题号", "问题（英文）", "问题（中文）", "参考答案", "正证据页", "替换页", "良性页", "证据状态",
        "替换结果（填写）", "替换后的答案（填写）", "正证据支持？（填写）",
        "良性页合格？（填写）", "事实出现次数（填写）", "文本问题（填写）", "备注（填写）",
    ]
    form.append(headers)
    style_header(form)
    form.freeze_panes = "H2"
    form.auto_filter.ref = f"A1:O{len(selected_rows) + 1}"
    editable = set(range(9, 16))

    for selected_row in selected_rows:
        qid = selected_row.get("qid")
        row = worlds.get(qid)
        if row:
            gold_pages = row.get("gold_pages_textview", [])
            swap_pages = row.get("swap_page", "")
            benign_pages = row.get("benign_pages", [])
            status = "有证据"
        else:
            gold_pages = selected_row.get("evidence_pages", [])
            swap_pages = ""
            benign_pages = ""
            status = "缺少 worlds 文件"
        form.append([
            qid,
            selected_row.get("question", row.get("question", "") if row else ""),
            QUESTION_ZH.get(qid, ""),
            selected_row.get("answer", row.get("answer", "") if row else ""),
            clean(gold_pages),
            clean(swap_pages),
            clean(benign_pages),
            status,
            "", "", "", "", "", "", "",
        ])

    for row in form.iter_rows(min_row=2, max_row=form.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
            if cell.column in editable:
                cell.fill = PatternFill("solid", fgColor=YELLOW)
        form.row_dimensions[row[0].row].height = 72
    widths = [12, 48, 48, 22, 12, 12, 12, 16, 22, 28, 18, 18, 18, 24, 42]
    for index, width in enumerate(widths, start=1):
        form.column_dimensions[get_column_letter(index)].width = width
    add_list_validation(form, "I", form.max_row, ["改变了答案", "应该拒答", "没改变", "看不清"])
    add_list_validation(form, "K", form.max_row, ["是", "否", "不确定"])
    add_list_validation(form, "L", form.max_row, ["是", "否", "不确定"])
    add_list_validation(form, "M", form.max_row, ["1 次", "2 次以上", "不确定"])
    add_list_validation(form, "N", form.max_row, ["无", "扫描件/无文本层", "答案格式问题", "图表/复杂版面"])
    form.conditional_formatting.add(f"I2:I{form.max_row}", CellIsRule(operator="equal", formula=['"改变了答案"'], fill=PatternFill("solid", fgColor=TEAL_LIGHT)))
    form.conditional_formatting.add(f"I2:I{form.max_row}", CellIsRule(operator="equal", formula=['"看不清"'], fill=PatternFill("solid", fgColor=RED_LIGHT)))

    # Evidence sheet: one page per row; filter by qid and world.
    evidence.append(["题号", "问题", "world", "这组证据是什么", "页码", "英文证据全文", "中文辅助翻译（Excel 365）"])
    style_header(evidence, color=TEAL)
    evidence.freeze_panes = "A2"
    for selected_row in selected_rows:
        qid = selected_row.get("qid")
        row = worlds.get(qid)
        if not row:
            continue
        meaning = {
            "plus": "原始世界：支持参考答案",
            "del": "删除世界：删除金标证据",
            "swap": "替换世界：候选冲突证据",
            "benign": "良性世界：加入无关页面",
        }
        for world, pages in row.get("worlds", {}).items():
            for page in pages:
                evidence.append([
                    qid,
                    row.get("question", ""),
                    {"plus": "原始证据", "del": "删除证据", "swap": "替换证据", "benign": "良性证据"}.get(world, world),
                    meaning.get(world, world),
                    page.get("page", ""),
                    f"【Page {clean(page.get('page'))}】\n{clean(page.get('text'))}",
                    "",
                ])
    for row in evidence.iter_rows(min_row=2, max_row=evidence.max_row):
        row[6].value = f'=IF(F{row[0].row}="","",TRANSLATE(F{row[0].row},"en","zh"))'
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=THIN_GREY)
        evidence.row_dimensions[row[0].row].height = 140
    for index, width in enumerate([12, 48, 14, 27, 10, 105, 105], start=1):
        evidence.column_dimensions[get_column_letter(index)].width = width
    evidence.auto_filter.ref = f"A1:F{evidence.max_row}"

    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0

    wb.save(OUTPUT)
    print(f"wrote {OUTPUT} ({len(selected_rows)} questions, {evidence.max_row - 1} evidence rows)")


if __name__ == "__main__":
    main()
