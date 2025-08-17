import json
import jsonlines

# 文件路径
generated_file = ''  # 含生成总结和打分
reference_file = ''  # 含正确总结

# 读取 reference 文件并建立映射：{(filename, title): summary_LLM}
reference_map = {}
with jsonlines.open(reference_file) as ref_reader:
    for ref_data in ref_reader:
        filename = ref_data.get("filename")
        for section in ref_data["content"]:
            title = section.get("title")
            summary = section.get("summary_LLM", "")
            key = (filename, title)
            reference_map[key] = summary

# 用于存储转换后的数据
transformed_data = []

# 读取生成文件并构造训练数据
with jsonlines.open(generated_file) as reader:
    for original_data in reader:
        filename = original_data.get("filename")
        for section in original_data["content"]:
            title = section.get("title")
            summary_generated = section.get("summary_LLM")
            # score = section.get("score_result")
            score = section.get("score_data", {}).get("raw_output")  

            key = (filename, title)
            correct_summary = reference_map.get(key)

            # 如果找到了对应的参考总结才处理
            if summary_generated and correct_summary:
                instruction = "作为资深再担保业务评审员，请针对以下文本（含原始内容、已有总结及专家评分与评价）进行总结优化。需严格参考专家评分与评价逻辑"
                input_text = f"标题: {title}\n内容: {section['content']}\n总结: {summary_generated}"
                output = f"评价：{score}\n 优化总结：{correct_summary}\n "

                transformed_item = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                }

                transformed_data.append(transformed_item)

# 保存到新 JSON 文件
with open('', 'w', encoding='utf-8') as file:
    json.dump(transformed_data, file, ensure_ascii=False, indent=4)









