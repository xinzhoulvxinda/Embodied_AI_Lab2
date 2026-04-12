#!/usr/bin/env python3
"""
实验步骤4: 模型评估与对比分析
对比 基础模型 → SFT模型 → PPO-RLHF模型 的输出质量变化

运行方式:
    python step4_evaluate.py

输出:
    - 终端打印各模型对比输出
    - evaluation_results.json 保存详细对比结果
    - 奖励分数统计分析
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# ======================== 配置区域 ========================
BASE_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_MODEL     = "./outputs/sft_model"
PPO_MODEL     = "./outputs/ppo_model"
REWARD_MODEL  = "./outputs/reward_model"
OUTPUT_FILE   = "./outputs/evaluation_results.json"

# 评估用测试问题（未见过的新问题）
TEST_QUESTIONS = [
    "如何提高大豆的固氮效率？",
    "温室番茄种植中常见的营养缺乏症有哪些？",
    "什么是保护性耕作技术？它对土壤有什么好处？",
    "农业病虫害预测预报的主要方法有哪些？",
    "如何判断果树是否需要修剪？修剪的基本原则是什么？",
]

MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.7
# =========================================================

def load_model_and_tokenizer(model_path: str, base_model: str = None):
    """加载模型和分词器，若路径不存在则回退到基础模型"""
    if not os.path.exists(model_path):
        print(f"  警告: {model_path} 不存在，使用基础模型 {base_model}")
        model_path = base_model
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    """生成模型回复"""
    device = next(model.parameters()).device
    system_msg = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 只保留新生成的 token
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def compute_reward_score(reward_model, reward_tokenizer, question: str, answer: str, device) -> float:
    """用奖励模型给回答打分"""
    system_msg = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    text = f"System: {system_msg}\nUser: {question}\nAssistant: {answer}"
    
    enc = reward_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        score = reward_model(**enc).logits[0].item()
    return score

def main():
    print("=" * 60)
    print("  步骤 4: 模型评估与对比分析")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")

    # 加载奖励模型用于自动评分
    reward_available = False
    if os.path.exists(REWARD_MODEL):
        print(f"\n[奖励模型] 加载: {REWARD_MODEL}")
        reward_tok = AutoTokenizer.from_pretrained(REWARD_MODEL, trust_remote_code=True)
        if reward_tok.pad_token is None:
            reward_tok.pad_token = reward_tok.eos_token
        reward_mdl = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL, num_labels=1, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()
        reward_available = True
    else:
        print("\n[奖励模型] 未找到奖励模型，跳过自动评分")

    # 要对比的模型列表
    models_to_eval = [
        ("基础模型 (Base)", BASE_MODEL),
        ("SFT 模型", SFT_MODEL),
        ("PPO-RLHF 模型", PPO_MODEL),
    ]

    all_results = []
    
    for q_idx, question in enumerate(TEST_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"问题 {q_idx+1}: {question}")
        print(f"{'='*60}")
        
        q_results = {"question": question, "responses": []}
        
        for model_name, model_path in models_to_eval:
            print(f"\n[{model_name}]")
            print(f"  加载模型: {model_path if os.path.exists(model_path) else BASE_MODEL}")
            
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, BASE_MODEL)
                response = generate_response(model, tokenizer, question, MAX_NEW_TOKENS)
                
                # 计算奖励分数
                reward_score = None
                if reward_available:
                    reward_score = compute_reward_score(reward_mdl, reward_tok, question, response, device)
                    print(f"  奖励分数: {reward_score:.4f}")
                
                print(f"  回答: {response[:300]}{'...' if len(response) > 300 else ''}")
                
                q_results["responses"].append({
                    "model": model_name,
                    "response": response,
                    "reward_score": reward_score,
                })
                
                # 释放显存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  错误: {e}")
                q_results["responses"].append({
                    "model": model_name, "response": f"[错误: {e}]", "reward_score": None
                })
        
        all_results.append(q_results)
    
    # 统计分析
    print(f"\n{'='*60}")
    print("  评估统计摘要")
    print(f"{'='*60}")
    
    if reward_available:
        for model_name, _ in models_to_eval:
            scores = []
            for qr in all_results:
                for r in qr["responses"]:
                    if r["model"] == model_name and r["reward_score"] is not None:
                        scores.append(r["reward_score"])
            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {model_name:20s}: 平均奖励分数 = {avg:.4f}")
    
    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[完成] 评估结果已保存至: {OUTPUT_FILE}")
    print("\n实验分析要点:")
    print("  1. 对比各阶段模型回答的详细程度和专业性")
    print("  2. 观察奖励分数随训练阶段的变化趋势")
    print("  3. 分析 PPO 模型是否在专业性上优于 SFT 模型")
    print("  4. 思考 KL 散度约束对模型输出多样性的影响")

if __name__ == "__main__":
    main()
