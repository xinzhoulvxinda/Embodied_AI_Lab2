#!/usr/bin/env python3
"""
实验步骤2: 奖励模型训练 (Reward Model Training)
使用人类偏好数据（chosen/rejected对）训练奖励模型

奖励模型学习区分"好回答"(chosen)和"差回答"(rejected)，
为后续 PPO 强化学习阶段提供奖励信号。

运行方式:
    python step2_reward_train.py

环境要求:
    pip install transformers trl datasets peft accelerate
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType

# ======================== 配置区域 ========================
BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_MODEL   = "./outputs/sft_model"        # 在SFT模型基础上训练奖励模型
DATA_PATH   = "./data/agriculture_reward.jsonl"
OUTPUT_DIR  = "./outputs/reward_model"
MAX_SEQ_LEN = 512
EPOCHS      = 2
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
LEARNING_RATE = 1e-4
# =========================================================

def load_reward_dataset(path: str, tokenizer) -> Dataset:
    """
    加载偏好数据集并格式化为奖励模型训练格式
    
    奖励模型需要: input_ids_chosen, input_ids_rejected
    分别对应"好回答"和"差回答"的 token 序列
    """
    records = []
    system_prompt = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            # 格式化 chosen（好答案）
            chosen_messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": item["prompt"]},
                {"role": "assistant", "content": item["chosen"]},
            ]
            # 格式化 rejected（差答案）
            rejected_messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": item["prompt"]},
                {"role": "assistant", "content": item["rejected"]},
            ]
            
            chosen_text   = tokenizer.apply_chat_template(chosen_messages,   tokenize=False, add_generation_prompt=False)
            rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
            
            chosen_enc   = tokenizer(chosen_text,   max_length=MAX_SEQ_LEN, truncation=True, padding=False)
            rejected_enc = tokenizer(rejected_text, max_length=MAX_SEQ_LEN, truncation=True, padding=False)
            
            records.append({
                "input_ids_chosen":         chosen_enc["input_ids"],
                "attention_mask_chosen":    chosen_enc["attention_mask"],
                "input_ids_rejected":       rejected_enc["input_ids"],
                "attention_mask_rejected":  rejected_enc["attention_mask"],
            })
    
    print(f"[数据加载] 共加载 {len(records)} 对偏好数据")
    return Dataset.from_list(records)

def main():
    print("=" * 60)
    print("  步骤 2: 奖励模型训练 (Reward Model)")
    print("=" * 60)

    # 1. 加载分词器（优先使用SFT模型）
    model_path = SFT_MODEL if __import__("os").path.exists(SFT_MODEL) else BASE_MODEL
    print(f"\n[1/4] 加载分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载偏好数据集
    print(f"\n[2/4] 加载偏好数据集: {DATA_PATH}")
    dataset = load_reward_dataset(DATA_PATH, tokenizer)
    
    # 划分训练/验证集 (80/20)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  训练集: {len(train_ds)} 条, 验证集: {len(eval_ds)} 条")

    # 3. 加载奖励模型（在语言模型基础上添加分类头）
    print(f"\n[3/4] 初始化奖励模型")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,          # 输出单个标量作为奖励分数
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    # 使用 LoRA 减少显存占用
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    # 4. 训练奖励模型
    print(f"\n[4/4] 开始训练奖励模型")
    print("  训练目标: Bradley-Terry 排序损失，使 score(chosen) > score(rejected)")
    
    training_args = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        logging_steps=5,
        save_steps=50,
        fp16=True,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n[完成] 奖励模型已保存至: {OUTPUT_DIR}")
    print("\n奖励模型训练原理:")
    print("  损失函数: L = -log(σ(r_chosen - r_rejected))")
    print("  优化目标: 使好回答的分数 > 差回答的分数")
    print("\n下一步: 运行 step3_ppo_train.py 进行 PPO 强化学习训练")

if __name__ == "__main__":
    main()
