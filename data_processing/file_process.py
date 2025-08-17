from docx import Document
import re
import json
import os
from docx.oxml.table import CT_Tbl
from docx.table import Table

def parse_docx(file_path):
    doc = Document(file_path)

    # 定义匹配规则
    section_keywords = [
        "基本情况分析", "经营管理分析", "产品及市场分析", "财务分析", "企业担保用途分析", "综合分析"
    ]
    subsection_mapping = {
        "企业基本资质评价": "基本情况分析",
        "企业经营管理能力评价": "经营管理分析",
        "企业产品（服务）及市场评价": "产品及市场分析",
        "企业产品及市场评价": "产品及市场分析",  # 兼容不同的标题
        "企业财务状况评价": "财务分析",
        "担保用途及偿还能力评价": "企业担保用途分析"
    }

    section_pattern = re.compile(r"(?:第[一二三四五六七八九十]+部分[\t\s]*)?(基本情况分析|经营管理分析|产品及市场分析|财务分析|企业担保用途分析|综合分析)")
    subsection_pattern = re.compile(r"^(?:[一二三四五六七八九十]+[、.])?(企业基本资质评价|企业经营管理能力评价|企业产品（服务）及市场评价|企业产品及市场评价|企业财务状况评价|担保用途及偿还能力评价)")

    current_section = None
    current_subsection = None
    is_in_subsection = False  # 判断是否已经进入子分类
    sections_data = {key: {"title": key, "content": "", "summary": ""} for key in section_keywords}

    def extract_text_from_table(table):
        """提取表格中的所有文本"""
        table_text = []
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                table_text.append(row_text)
        return "\n".join(table_text)

    # 遍历 Word 文档结构，确保文本和表格按顺序解析
    body = doc.element.body
    for element in body:
        # 处理文本段落
        if element.tag.endswith("p"):
            text = "".join(node.text for node in element.findall('.//w:t', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}) if node.text).strip()
            if not text:
                continue

            # 识别主分类
            section_match = section_pattern.search(text)
            if section_match:
                current_section = section_match.group(1)
                is_in_subsection = False  # 遇到新部分时结束子分类收集
                continue

            # 处理综合分析的子分类
            if current_section == "综合分析":
                subsection_match = subsection_pattern.match(text)
                if subsection_match:
                    current_subsection = subsection_match.group(1)
                    mapped_section = subsection_mapping.get(current_subsection)
                    if mapped_section:
                        sections_data[mapped_section]["summary"] += text + "\n"
                        is_in_subsection = True  # 进入子分类，继续收集正文内容
                    continue

            # 存储正文内容
            if current_section and current_section in sections_data:
                if is_in_subsection:
                    sections_data[mapped_section]["summary"] += text + "\n"
                else:
                    sections_data[current_section]["content"] += text + "\n"

        # 处理表格
        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)  # 直接转换当前 element 为表格对象
            table_text = extract_text_from_table(table)
            if current_section and current_section in sections_data:
                if is_in_subsection:
                    sections_data[mapped_section]["summary"] += table_text + "\n"
                else:
                    sections_data[current_section]["content"] += table_text + "\n"

    return sections_data

def write_to_jsonl(data, jsonl_path, filename):
    # 移除综合分析部分
    if "综合分析" in data:
        del data["综合分析"]
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps({"filename": filename, "content": list(data.values())}, ensure_ascii=False)
        f.write(json_line + '\n')


# 处理多个文件夹中的所有 DOCX 文件
folder_paths = [
    r'/home/data/test',
]
jsonl_path = 'input_test.jsonl'
open(jsonl_path, 'w', encoding='utf-8').close()  # 清空或创建输出文件

for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            file_path = os.path.join(folder_path, filename)
            parsed_data = parse_docx(file_path)
            write_to_jsonl(parsed_data, jsonl_path, filename)

print(f"处理完成，结果保存至 {jsonl_path}")

    