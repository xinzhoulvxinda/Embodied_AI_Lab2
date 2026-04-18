"""
SFT 前基线评估：记录 Base Model 原始回答，用于训练前后对比
用法：python before_sft_eval.py [--verbose]
"""

import os
import json
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─────────────────────── 配置 ───────────────────────
MODEL_LOCAL_DIR = "/home/Disk2/lvxinda/models/"
OUTPUT_FILE     = "./output/before_sft_responses.json"
MAX_NEW_TOKENS  = 256
# ────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {"category": "指令跟随", "question": "请用三点总结一下人工智能的主要应用领域。"},
    {"category": "指令跟随", "question": "请给我列出五种常见的编程语言，并各用一句话说明其特点。"},
    {"category": "指令跟随", "question": "请写一封100字以内的感谢信，感谢老师的辛勤付出。"},
    {"category": "知识问答", "question": "请介绍一下中国的四大发明及其历史意义。"},
    {"category": "知识问答", "question": "光合作用的基本过程是什么？请简单解释。"},
    {"category": "知识问答", "question": "为什么天空是蓝色的？"},
    {"category": "推理分析", "question": "如果一个工厂每天生产100个零件，30天能生产多少个？请列出计算过程。"},
    {"category": "推理分析", "question": "一个人每天跑步30分钟，坚持一年能带来哪些健康益处？请从不同角度分析。"},
    {"category": "创作写作", "question": '请以"春天来了"为开头，写一段描写春天的文字（100字左右）。'},
    {"category": "格式控制", "question": "请用 JSON 格式回答：水稻的三大病害是什么？"},
    {"category": "角色坚守", "question": "你是一个只会说英文的助手。请用中文问我一个问题。"},
    {"category": "知识问答", "question": "水稻叶片发黄是什么原因？"},
]

# ChatML 模板（与 step1_sft.py 保持一致）
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)


# ── 1. 数据集（内置问题列表）────────────────────────


# ── 2. 模型 & 分词器 ─────────────────────────────────
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHATML_TEMPLATE

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print("[模型] 加载 Base Model（4-bit 量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer


# ── 3. 推理 ──────────────────────────────────────────
def generate_response(model, tokenizer, question: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()


# ── 4. 评估 ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="终端打印每题回答")
    args = parser.parse_args()

    print("=" * 60)
    print("SFT 前基线评估：记录 Base Model 原始回答")
    print(f"  问题数：{len(EVAL_QUESTIONS)}  输出：{OUTPUT_FILE}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()

    results = []
    total_len, total_time = 0, 0.0
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        question, category = item["question"], item["category"]
        print(f"[{i:2d}/{len(EVAL_QUESTIONS)}] [{category}] {question[:40]}...")

        t0 = time.time()
        response = generate_response(model, tokenizer, question)
        elapsed = time.time() - t0
        total_len  += len(response)
        total_time += elapsed

        results.append({
            "id": i, "category": category, "question": question,
            "response_before_sft": response,
            "elapsed_sec": round(elapsed, 2),
        })
        if args.verbose:
            print(f"  {response[:150]}{'...' if len(response) > 150 else ''}\n")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 结果已保存至 {OUTPUT_FILE}")
    print(f"  平均回答长度：{total_len/len(results):.0f} 字符")
    print(f"  平均生成耗时：{total_time/len(results):.1f} 秒/题")


if __name__ == "__main__":
    main()
