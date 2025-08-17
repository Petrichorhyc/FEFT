from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import jsonlines

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("")

# 设置采样参数
sampling_params = SamplingParams(temperature=0.3, top_p=0.8, repetition_penalty=1.05, max_tokens=20480)

# 加载本地模型
llm = LLM(model="", gpu_memory_utilization=0.7, tensor_parallel_size=4)


def read_prompt_template(file_path):
    """从文本文件读取提示模板"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def generate_summary(text, prompt_template):
    """使用模型生成摘要"""
    full_prompt = f"{prompt_template}{text}"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": full_prompt}
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate([formatted_text], sampling_params)
    return outputs[0].outputs[0].text if outputs else ""


def process_jsonl(input_path, output_path, template_path):
    """处理JSONL文件"""
    # 读取提示模板
    prompt_template = read_prompt_template(template_path)
    file_count = 0  # 初始化文件计数器

    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
        for obj in reader:
            for content_block in obj.get("content", []):
                if content_block.get("content", "").strip():  # 仅处理有内容的块
                    input_text = f"{content_block.get('title', '')}\n{content_block['content']}"
                    content_block["summary_LLM"] = generate_summary(input_text, prompt_template)
            writer.write(obj)
            file_count += 1  # 每处理完一个文件，计数器加1
            print(f"{obj.get('filename', 'Unknown')} 已完成，已处理文件数: {file_count}")

    return file_count


# 使用示例
input_jsonl = ""    # 输入JSONL文件路径
output_jsonl = ""  # 输出JSONL文件路径
template_path = ""      # 提示模板文件路径
total_files_processed = process_jsonl(input_jsonl, output_jsonl, template_path)
print(f"处理完成，共处理 {total_files_processed} 个文件，结果已保存到: {output_jsonl}")