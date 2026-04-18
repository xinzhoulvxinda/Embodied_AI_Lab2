"""
步骤四：模型评估与对比分析
对比 Base → SFT → GRPO 三个阶段模型的回答质量与奖励分数
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# ─────────────────────── 配置 ───────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_MODEL    = "./outputs/sft_model"
PPO_MODEL    = "./outputs/ppo_model"
REWARD_MODEL = "./outputs/reward_model"
OUTPUT_FILE  = "./outputs/evaluation_results.json"
MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.7
SYSTEM_MSG     = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
# ────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "如何提高大豆的固氮效率？",
    "温室番茄种植中常见的营养缺乏症有哪些？",
    "什么是保护性耕作技术？它对土壤有什么好处？",
    "农业病虫害预测预报的主要方法有哪些？",
    "如何判断果树是否需要修剪？修剪的基本原则是什么？",
]


# ── 1. 数据集（测试问题，无需加载外部数据集）────────────


# ── 2. 模型 & 分词器 ─────────────────────────────────
def load_model_and_tokenizer(model_path: str, base_model: str = None):
    if not os.path.exists(model_path):
        print(f"  [警告] {model_path} 不存在，回退到 {base_model}")
        model_path = base_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ── 3. 推理 ──────────────────────────────────────────
def generate_response(model, tokenizer, question: str) -> str:
    device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=TEMPERATURE, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_reward_score(reward_model, reward_tokenizer, question: str, answer: str, device) -> float:
    text = f"System: {SYSTEM_MSG}\nUser: {question}\nAssistant: {answer}"
    enc = reward_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        return reward_model(**enc).logits[0].item()


# ── 4. 评估 ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("步骤四：模型评估与对比分析")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")

    reward_available = os.path.exists(REWARD_MODEL)
    if reward_available:
        print(f"\n[奖励模型] 加载 {REWARD_MODEL}...")
        reward_tok = AutoTokenizer.from_pretrained(REWARD_MODEL, trust_remote_code=True)
        if reward_tok.pad_token is None:
            reward_tok.pad_token = reward_tok.eos_token
        reward_mdl = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL, num_labels=1, trust_remote_code=True, torch_dtype=torch.float16,
        ).to(device).eval()
    else:
        print("\n[奖励模型] 未找到，跳过自动评分")

    models_to_eval = [
        ("Base",     BASE_MODEL),
        ("SFT",      SFT_MODEL),
        ("PPO-RLHF", PPO_MODEL),
    ]

    all_results = []
    for q_idx, question in enumerate(TEST_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"问题 {q_idx+1}/{len(TEST_QUESTIONS)}: {question}")
        print("=" * 60)
        q_results = {"question": question, "responses": []}

        for model_name, model_path in models_to_eval:
            print(f"\n[{model_name}]")
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, BASE_MODEL)
                response = generate_response(model, tokenizer, question)

                reward_score = None
                if reward_available:
                    reward_score = compute_reward_score(reward_mdl, reward_tok, question, response, device)
                    print(f"  奖励分数: {reward_score:.4f}")
                print(f"  回答: {response[:300]}{'...' if len(response) > 300 else ''}")

                q_results["responses"].append({
                    "model": model_name, "response": response, "reward_score": reward_score,
                })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [错误] {e}")
                q_results["responses"].append({"model": model_name, "response": f"[错误: {e}]", "reward_score": None})

        all_results.append(q_results)

    print(f"\n{'='*60}")
    print("评估统计摘要")
    print("=" * 60)
    if reward_available:
        for model_name, _ in models_to_eval:
            scores = [
                r["reward_score"]
                for qr in all_results
                for r in qr["responses"]
                if r["model"] == model_name and r["reward_score"] is not None
            ]
            if scores:
                print(f"  {model_name:12s}: 平均奖励分数 = {sum(scores)/len(scores):.4f}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 评估结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
