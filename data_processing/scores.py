import json
import re
import numpy as np
from collections import defaultdict

# 定义正则表达式，匹配评分（格式：X/5 或 X/25）
score_pattern = re.compile(r"（评分：(\d+)/(\d+)）|总评分：(\d+)/25")

# 定义要处理的标题
titles_to_process = [
    "基本情况分析", "经营管理分析", "产品及市场分析", "财务分析", "企业担保用途分析"
]

# 读取 JSONL 文件并统计评分
input_file = ""
scores_dict = defaultdict(list)  # 存储每个标题的评分
overall_scores_list = []  # 用于存储所有评分数据（计算总平均分）

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        for section in data.get("content", []):  # 确保 "content" 字段存在
            title = section.get("title", "")
            if title in titles_to_process:  # 只处理指定标题的部分
                score_result = section.get("score_result", "")

                # 提取评分（仅获取满分为5的评分）
                scores = [int(match.group(1)) for match in score_pattern.finditer(score_result) if match.group(2) == "5"]
                
                # 获取总评分（满分为25）
                total_score = next((int(match.group(3)) for match in score_pattern.finditer(score_result) if match.group(3) is not None), None)

                if len(scores) == 5 and total_score is not None:
                    scores.append(total_score)  # 追加总评分
                    scores_dict[title].append(scores)
                    overall_scores_list.append(scores)  # 存入总评分列表

# 计算每个部分的统计数据
for title, scores_list in scores_dict.items():
    if scores_list:
        scores_array = np.array(scores_list)  # 转换为 NumPy 数组

        avg_scores = np.mean(scores_array, axis=0)
        min_scores = np.min(scores_array, axis=0)
        max_scores = np.max(scores_array, axis=0)

        print(f"=== {title} 评分统计 ===")
        print(f"真实性与准确性 平均分: {avg_scores[0]:.2f} / 5")
        print(f"业务逻辑性    平均分: {avg_scores[1]:.2f} / 5")
        print(f"风险分析全面性 平均分: {avg_scores[2]:.2f} / 5")
        print(f"重点突出与简洁性 平均分: {avg_scores[3]:.2f} / 5")
        print(f"再担保决策支持价值 平均分: {avg_scores[4]:.2f} / 5")
        print(f"总评分 平均分: {avg_scores[5]:.2f} / 25")
        
        print("\n=== 最高分与最低分 ===")
        print(f"最高总评分: {max_scores[5]} / 25")
        print(f"最低总评分: {min_scores[5]} / 25")
        print("\n")
    else:
        print(f"未找到有效评分数据：{title}\n")

# 计算所有类别的综合平均分
if overall_scores_list:
    overall_scores_array = np.array(overall_scores_list)

    overall_avg_scores = np.mean(overall_scores_array, axis=0)
    overall_min_scores = np.min(overall_scores_array, axis=0)
    overall_max_scores = np.max(overall_scores_array, axis=0)

    print(f"=== 🌟 总体评分统计 🌟 ===")
    print(f"真实性与准确性 平均分: {overall_avg_scores[0]:.2f} / 5")
    print(f"业务逻辑性    平均分: {overall_avg_scores[1]:.2f} / 5")
    print(f"风险分析全面性 平均分: {overall_avg_scores[2]:.2f} / 5")
    print(f"重点突出与简洁性 平均分: {overall_avg_scores[3]:.2f} / 5")
    print(f"再担保决策支持价值 平均分: {overall_avg_scores[4]:.2f} / 5")
    print(f"总评分 平均分: {overall_avg_scores[5]:.2f} / 25")

    print("\n=== 🌟 总体最高分与最低分 🌟 ===")
    print(f"最高总评分: {overall_max_scores[5]} / 25")
    print(f"最低总评分: {overall_min_scores[5]} / 25")
    print("\n")
else:
    print("未找到任何有效评分数据！")
