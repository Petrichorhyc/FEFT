from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import jsonlines
import re # 导入正则表达式模块

# 模型路径
model_name = "" # 请确保路径正确

# 加载 vLLM 运行引擎
# 实际使用时请取消注释下面两行，并删除或注释掉模拟部分:
llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.6, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

SCORING_CRITERIA = """
请对以下总结内容进行评分，每个维度 5 分，总分 25 分： 

1️⃣ **真实性与准确性（5分）**
- 5 分：总结内容完全准确，无遗漏，所有关键数据均包含，表达清晰。
- 4 分：总结基本准确，但有轻微遗漏或数据轻微偏差。
- 3 分：大部分准确，但有一定程度的信息缺失或错误。
- 2 分：存在较多错误，核心内容表达有偏差。
- 1 分：严重偏离原文信息，存在大量错误或遗漏。

2️⃣ **业务逻辑性（5分）**
- 5 分：总结逻辑清晰，层次分明，符合再担保业务的评审逻辑。
- 4 分：逻辑基本清晰，但部分信息组织不够合理。
- 3 分：存在逻辑跳跃，部分信息前后矛盾。
- 2 分：逻辑混乱，难以理解总结内容。
- 1 分：完全没有逻辑，内容组织凌乱。

3️⃣ **风险分析全面性（5分）**
- 5 分：清晰指出企业的信用风险、市场风险、财务风险等，分析到位。
- 4 分：识别出主要风险，但部分细节分析不够深入。
- 3 分：有一定风险分析，但较为片面。
- 2 分：风险分析不充分，仅提到个别风险点。
- 1 分：未进行有效的风险分析。

4️⃣ **重点突出与简洁性（5分）**
- 5 分：总结精准，内容简练，突出核心信息，无冗余。
- 4 分：较为简练，但部分内容仍可精简。
- 3 分：有一定冗余，部分内容过长或不必要。
- 2 分：总结过于冗长，信息提炼不到位。
- 1 分：完全没有抓住重点，信息杂乱无章。

5️⃣ **再担保决策支持价值（5分）**
- 5 分：总结清晰阐述再担保机构决策所需的核心信息，提供有价值的判断依据。
- 4 分：对决策有所帮助，但仍可优化。
- 3 分：有一定帮助，但缺少关键信息。
- 2 分：对决策帮助有限，信息过于浅显。
- 1 分：无决策参考价值。

请严格按照以下格式返回评分：
```
**1. 真实性与准确性（评分：X/5）** 评分理由：……  

**2. 业务逻辑性（评分：X/5）** 评分理由：……  

**3. 风险分析全面性（评分：X/5）** 评分理由：……  

**4. 重点突出与简洁性（评分：X/5）** 评分理由：……  

**5. 再担保决策支持价值（评分：X/5）** 评分理由：……  

**总评分：X/25**
```
"""

def parse_score_result(result_text):
    """
    解析LLM返回的评分文本，提取各维度分数、理由和计算总分。
    """
    parsed_data = {
        "scores": {},
        "reasons": {},
        "calculated_total_score": 0,
        "llm_reported_total_score": None,
        "parsing_error": False,
        "raw_output": result_text
    }
    
    dimensions = [
        "真实性与准确性",
        "业务逻辑性",
        "风险分析全面性",
        "重点突出与简洁性",
        "再担保决策支持价值"
    ]
    
    current_total_score = 0
    all_dimensions_successfully_parsed = True

    for i, dim_name in enumerate(dimensions):
        next_dim_lookahead = rf"\*\*{re.escape(str(i + 2))}\." if i + 1 < len(dimensions) else r"\*\*总评分："
        pattern_str = rf"\*\*{re.escape(str(i + 1))}\.\s*{re.escape(dim_name)}（评分：\s*(\d)\s*/5）\*\*\s*评分理由：(.*?)(?=\n\s*(?:{next_dim_lookahead})|$)"
        
        match = re.search(pattern_str, result_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            try:
                score_str = match.group(1)
                score = int(score_str)
                if not (0 <= score <= 5):
                    raise ValueError(f"分数 '{score_str}' 超出0-5范围")
                reason = match.group(2).strip()
                
                parsed_data["scores"][dim_name] = score
                parsed_data["reasons"][dim_name] = reason
                current_total_score += score
            except ValueError as e:
                parsed_data["scores"][dim_name] = None
                parsed_data["reasons"][dim_name] = f"分数解析或校验错误: '{score_str if 'score_str' in locals() else '未知内容'}' ({e})"
                all_dimensions_successfully_parsed = False
                parsed_data["parsing_error"] = True
        else:
            parsed_data["scores"][dim_name] = None
            parsed_data["reasons"][dim_name] = f"未找到该维度 '{dim_name}' 评分或格式不匹配"
            all_dimensions_successfully_parsed = False
            parsed_data["parsing_error"] = True
            
    if all_dimensions_successfully_parsed:
        parsed_data["calculated_total_score"] = current_total_score
    else:
        parsed_data["calculated_total_score"] = current_total_score

    total_score_match = re.search(r"\*\*总评分：\s*(\d+)\s*/25\*\*", result_text, re.IGNORECASE)
    if total_score_match:
        try:
            parsed_data["llm_reported_total_score"] = int(total_score_match.group(1))
        except ValueError:
            parsed_data["llm_reported_total_score"] = "LLM报告总分解析错误"
    else:
        parsed_data["llm_reported_total_score"] = "未找到LLM报告总分"
            
    return parsed_data


def get_summary_score(original_text, summary_text):
    system_message = "你是一个经验丰富且一丝不苟的再担保业务评审专家。你的任务是严格按照提供的评分标准，对照原始报告和总结内容，进行客观、详细的打分，并给出充分的评分理由。请确保输出完全符合要求的格式。"
    
    user_prompt_detail = (
        "请仔细阅读并理解上述每一项评分标准及其细则。对于每一个评分维度，您都需要：\n"
        "1. 严格依据“原始报告内容”和“总结内容”进行对比分析。\n"
        "2. 严格对照该维度的评分细则（1分到5分的具体描述）来决定分数。\n"
        "3. 在“评分理由”中，清晰、具体地解释您为什么给出该分数，必须引用“总结内容”中的相关表述，并结合“原始报告内容”进行对比佐证，指出亮点或不足。\n\n"
        "请提供详细评分，并严格按照以下格式输出（确保每个维度的评分都以 'X/5' 的形式给出，总评分行也请填写）："
    )

    user_message_content = (
        f"你是一位资深的再担保业务评审员。你的任务是对提供的总结内容进行评分，并给出评分理由。\n\n"
        f"### 原始报告内容\n{original_text}\n\n"
        f"### 总结内容\n{summary_text}\n\n"
        f"### 评分标准\n{SCORING_CRITERIA}\n\n"
        f"{user_prompt_detail}"
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content}
    ]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"错误：应用聊天模板时出错: {e}")
        text = f"{system_message}\nUSER: {user_message_content}\nASSISTANT:"


    sampling_params = SamplingParams(
        max_tokens=4096, 
        temperature=0.3, 
        top_p=0.3, 
        top_k=50,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    try:
        outputs = llm.generate([text], sampling_params)
        if outputs and len(outputs) > 0 and outputs[0].outputs and len(outputs[0].outputs) > 0:
            result = outputs[0].outputs[0].text.strip()
        else:
            print("错误：LLM生成了空的或意外的输出。")
            result = "LLM生成错误：输出为空或格式不正确"
    except Exception as e:
        print(f"错误：LLM生成过程中发生错误: {e}")
        result = f"LLM生成错误: {str(e)}"
    
    return result

# ------------------- 请将以下文件名替换为您实际的输入输出文件名 -------------------
input_file_for_script = "" # 您的输入文件名
output_file_for_script = "" # 您期望的输出文件名
# ---------------------------------------------------------------------------------

print(f"INFO: Starting processing. Input: '{input_file_for_script}', Output: '{output_file_for_script}'")
try:
    with open(input_file_for_script, "r", encoding="utf-8") as f, \
         open(output_file_for_script, "w", encoding="utf-8") as out_f:
        
        for idx, line in enumerate(f):
            line_number = idx + 1
            try:
                data = json.loads(line)
                if "content" not in data or not isinstance(data["content"], list):
                    print(f"警告：第 {line_number} 行数据格式不正确，缺少 'content' 列表，已跳过。")
                    error_data = {"original_line_number": line_number, "error": "Invalid data format, missing 'content' list", "line_content": line.strip()}
                    out_f.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                    continue

                for section_idx, section in enumerate(data["content"]):
                    section_number = section_idx + 1
                    original_text = section.get("content", "").strip()
                    summary_text = section.get("summary_LLM", "").strip()
                    
                    section["score_data"] = {
                        "scores": {}, "reasons": {},
                        "calculated_total_score": 0, "llm_reported_total_score": None,
                        "parsing_error": False, "status_message": "", "raw_llm_output": ""
                    }

                    if not original_text:
                        msg = f"原文为空，未评分"
                        print(f"信息：第 {line_number} 条记录, 第 {section_number} 部分总结: {msg}")
                        section["score_data"]["status_message"] = msg
                        continue
                    if not summary_text:
                        msg = f"总结为空，未评分"
                        print(f"信息：第 {line_number} 条记录, 第 {section_number} 部分总结: {msg}")
                        section["score_data"]["status_message"] = msg
                        continue

                    print(f"正在评分第 {line_number} 条记录, 第 {section_number} 部分总结...")
                    try:
                        llm_raw_output = get_summary_score(original_text, summary_text)
                        section["score_data"]["raw_llm_output"] = llm_raw_output
                        
                        if "LLM生成错误" in llm_raw_output:
                             parsed_score_data = {"parsing_error": True, "status_message": llm_raw_output}
                        else:
                            parsed_score_data = parse_score_result(llm_raw_output)
                        
                        section["score_data"].update(parsed_score_data)

                        if section["score_data"].get("parsing_error"):
                            status_msg = section["score_data"].get("status_message") or "LLM输出解析失败或不完整"
                            section["score_data"]["status_message"] = status_msg
                            print(f"警告：第 {line_number} 条记录, 第 {section_number} 部分总结: {status_msg}")
                        else:
                            section["score_data"]["status_message"] = "评分完成"
                            print(f"评分完成：第 {line_number} 条记录, 第 {section_number} 部分总结。计算总分: {section['score_data']['calculated_total_score']}/25")

                    except Exception as e_score:
                        err_msg = f"评分过程中发生错误: {str(e_score)}"
                        print(f"错误：在评分第 {line_number} 条记录, 第 {section_number} 部分总结时: {err_msg}")
                        section["score_data"]["status_message"] = err_msg
                        section["score_data"]["parsing_error"] = True
                
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                out_f.flush()
                print("-" * 80)
            
            except json.JSONDecodeError:
                err_msg = f"第 {line_number} 行不是有效的JSON格式，已跳过。"
                print(f"错误：{err_msg} 行内容：{line.strip()}")
                error_data = {"original_line_number": line_number, "error": "Invalid JSON format", "line_content": line.strip()}
                out_f.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e_main:
                err_msg = f"处理第 {line_number} 行时发生未知错误: {str(e_main)}。"
                print(f"严重错误：{err_msg} 行内容：{line.strip()}")
                error_data = {"original_line_number": line_number, "error": f"Unknown error: {str(e_main)}", "line_content": line.strip()}
                out_f.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                out_f.flush()

except FileNotFoundError:
    print(f"错误：输入文件 '{input_file_for_script}' 未找到。请确保文件路径正确。")
except Exception as e_global:
    print(f"脚本执行过程中发生严重错误: {e_global}")

print(f"所有评分已完成或尝试完成，结果已保存到 {output_file_for_script}")
