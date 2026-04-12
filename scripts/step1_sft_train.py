#!/usr/bin/env python3
"""
实验步骤1: 监督微调 (Supervised Fine-Tuning, SFT)
使用农业问答数据对基础模型进行指令微调

运行方式:
    python step1_sft_train.py

环境要求:
    pip install transformers trl datasets peft accelerate bitsandbytes
"""

import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

# ======================== 配置区域 ========================
MODEL_NAME = "qwen/Qwen2.5-1.5B-Instruct"   # ModelScope 模型 ID
DATA_PATH  = "./data/agriculture_sft.jsonl"
OUTPUT_DIR = "./outputs/sft_model"
MAX_SEQ_LEN = 512
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4          # 等效 batch_size = 8
LEARNING_RATE = 2e-4
LORA_R = 8              # LoRA rank，越大效果可能越好但显存需求增加
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# =========================================================

def load_dataset_from_jsonl(path: str) -> Dataset:
    """加载 JSONL 格式的农业问答数据集"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[数据加载] 共加载 {len(records)} 条训练样本")
    return Dataset.from_list(records)

def format_chat_prompt(example: dict, tokenizer) -> dict:
    """
    将问答对格式化为模型对话格式
    使用 chat_template 将 prompt/response 转换为训练文本
    """
    messages = [
        {"role": "system",  "content": "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"},
        {"role": "user",    "content": example["prompt"]},
        {"role": "assistant","content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

def main():
    print("=" * 60)
    print("  步骤 1: 监督微调 (SFT)")
    print("=" * 60)

    # 1. 加载分词器
    print(f"\n[1/4] 加载分词器: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据集并格式化
    print(f"\n[2/4] 加载并预处理数据集: {DATA_PATH}")
    raw_dataset = load_dataset_from_jsonl(DATA_PATH)
    dataset = raw_dataset.map(
        lambda ex: format_chat_prompt(ex, tokenizer),
        remove_columns=raw_dataset.column_names
    )
    print(f"  示例文本 (前200字):\n  {dataset[0]['text'][:200]}...")

    # 3. 配置 LoRA（低秩适配，大幅降低显存需求）
    print(f"\n[3/4] 配置 LoRA 参数高效微调")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 注意力层
        bias="none",
    )
    print(f"  LoRA rank={LORA_R}, alpha={LORA_ALPHA}, target: q/k/v/o_proj")

    # 4. 训练配置
    print(f"\n[4/4] 开始训练")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=True,                    # 使用半精度训练
        dataloader_num_workers=0,
        report_to="none",             # 实验环境关闭wandb
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    print(f"\n  模型参数统计:")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    print(f"  可训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  总参数量:   {total:,}")

    trainer.train()

    # 保存模型
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[完成] SFT 模型已保存至: {OUTPUT_DIR}")
    print("下一步: 运行 step2_reward_train.py 训练奖励模型")

if __name__ == "__main__":
    main()
