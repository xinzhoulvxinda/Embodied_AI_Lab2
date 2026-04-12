#!/usr/bin/env python3
"""
实验步骤3: PPO 强化学习训练 (Proximal Policy Optimization)
使用奖励模型信号通过 PPO 算法进一步优化语言模型

RLHF 最终阶段：策略模型(Policy)在奖励模型的引导下不断优化输出质量
同时通过 KL 散度约束防止策略偏离参考模型太远（崩坏）

运行方式:
    python step3_ppo_train.py

环境要求:
    pip install transformers trl datasets peft accelerate
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType
import os

# ======================== 配置区域 ========================
SFT_MODEL      = "./outputs/sft_model"        # 策略模型（待优化）
REWARD_MODEL   = "./outputs/reward_model"     # 奖励模型
BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct" # 回退选项
DATA_PATH      = "./data/agriculture_sft.jsonl"  # 用问题部分作为 PPO 查询
OUTPUT_DIR     = "./outputs/ppo_model"
MAX_NEW_TOKENS = 200      # 每次生成的最大 token 数
BATCH_SIZE     = 2        # PPO mini-batch 大小
PPO_EPOCHS     = 2        # PPO 内部更新轮数
TOTAL_STEPS    = 20       # 总训练步数（实验用，正式训练建议100-500步）
KL_COEFF       = 0.1      # KL 散度惩罚系数（防止策略崩坏）
LEARNING_RATE  = 1e-5
# =========================================================

def load_queries(path: str, tokenizer, n_samples: int = 30):
    """
    从 SFT 数据集中提取问题作为 PPO 训练查询
    每步 PPO 训练: 模型生成回答 → 奖励模型打分 → 策略梯度更新
    """
    system_msg = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    queries = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": item["prompt"]},
            ]
            query_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            queries.append({"query": query_text, "prompt": item["prompt"]})
    
    queries = queries[:n_samples]
    print(f"[数据加载] 共准备 {len(queries)} 条 PPO 训练查询")
    return queries

def main():
    print("=" * 60)
    print("  步骤 3: PPO 强化学习训练")
    print("=" * 60)
    print("\nPPO-RLHF 训练流程:")
    print("  1. 策略模型(SFT) 生成回答")
    print("  2. 奖励模型对回答打分")
    print("  3. 计算 KL 惩罚（防止偏离参考策略太远）")
    print("  4. PPO 更新策略模型参数")
    print("  重复上述步骤直到收敛\n")

    policy_path = SFT_MODEL if os.path.exists(SFT_MODEL) else BASE_MODEL
    reward_path = REWARD_MODEL if os.path.exists(REWARD_MODEL) else BASE_MODEL
    
    # 1. 加载分词器
    print(f"[1/5] 加载分词器: {policy_path}")
    tokenizer = AutoTokenizer.from_pretrained(policy_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # 生成时使用 left padding

    # 2. 加载策略模型（带价值头，用于 PPO）
    print(f"\n[2/5] 加载策略模型（带价值头）: {policy_path}")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
    )

    # 3. 加载奖励模型（用于对生成文本打分）
    print(f"\n[3/5] 加载奖励模型: {reward_path}")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path, trust_remote_code=True)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    reward_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_model = reward_model.to(device)

    # 4. 准备训练数据
    print(f"\n[4/5] 准备 PPO 训练查询")
    queries = load_queries(DATA_PATH, tokenizer, n_samples=30)

    # 5. PPO 训练循环
    print(f"\n[5/5] 开始 PPO 训练 (共 {TOTAL_STEPS} 步)")
    
    ppo_config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        kl_penalty="kl",
        init_kl_coef=KL_COEFF,
        target_kl=6.0,              # 自适应 KL 目标值
        adap_kl_ctrl=True,          # 自动调整 KL 系数
        log_with=None,
        model_name=policy_path,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None,             # 使用 PEFT，ref_model 设为 None（自动对比基础权重）
        tokenizer=tokenizer,
        dataset=Dataset.from_list(queries),
        data_collator=lambda x: x,
    )

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    step_logs = []
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= TOTAL_STEPS:
            break
        
        # a. 编码查询
        query_tensors = [
            tokenizer(b["query"], return_tensors="pt").input_ids.squeeze(0).to(device)
            for b in batch
        ]
        
        # b. 策略模型生成回答
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **gen_kwargs
        )
        
        # c. 解码生成文本
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        
        # d. 奖励模型打分
        rewards = []
        for resp_text, query_item in zip(responses, batch):
            full_text = query_item["query"] + resp_text
            enc = reward_tokenizer(
                full_text, return_tensors="pt",
                max_length=512, truncation=True
            ).to(device)
            with torch.no_grad():
                score = reward_model(**enc).logits[0].item()
            rewards.append(torch.tensor(score, dtype=torch.float32))
        
        # e. PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        print(f"  Step {step+1:3d}/{TOTAL_STEPS} | "
              f"Mean Reward: {mean_reward:.3f} | "
              f"KL Div: {stats.get('objective/kl', 0):.3f}")
        step_logs.append({"step": step+1, "mean_reward": mean_reward})

    # 保存最终模型
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ppo_trainer.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存训练日志
    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(step_logs, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] PPO 模型已保存至: {OUTPUT_DIR}")
    print(f"  训练日志: {log_path}")
    print("\n下一步: 运行 step4_evaluate.py 对比各阶段模型效果")

if __name__ == "__main__":
    main()
