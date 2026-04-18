#!/usr/bin/env python3
"""
实验步骤2: 奖励模型训练 (Reward Model Training)
使用人类偏好数据（chosen/rejected 对）训练奖励模型

奖励模型学习区分"好回答"(chosen)和"差回答"(rejected)，
为后续强化学习阶段提供奖励信号。

【学生任务说明】
  本文件包含 3 处 TODO，请按照注释中的"设计要求"独立补全。
  完成后运行：python step2_reward.py

环境要求:
    pip install transformers trl datasets peft accelerate
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType

# ======================== 配置区域（无需修改）========================
BASE_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_MODEL     = "./outputs/sft_model"       # 优先在 SFT 模型基础上训练
DATA_PATH     = "./data/agriculture_reward.jsonl"
OUTPUT_DIR    = "./outputs/reward_model"
MAX_SEQ_LEN   = 512
EPOCHS        = 2
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
LEARNING_RATE = 1e-4
# ===================================================================


def load_reward_dataset(path: str, tokenizer) -> Dataset:
    """
    加载偏好数据集并格式化为奖励模型训练格式。

    奖励模型的训练需要成对的 chosen/rejected 编码结果：
      - input_ids_chosen      / attention_mask_chosen
      - input_ids_rejected    / attention_mask_rejected

    # ===================================================================
    # TODO[2-1]: 实现偏好数据集的加载与格式化
    # ===================================================================
    # 设计要求:
    #   对 JSONL 文件中的每一条记录（含 "prompt"、"chosen"、"rejected" 字段）：
    #   1. 分别为 chosen 和 rejected 构造包含 system/user/assistant 的消息列表
    #      - system 内容: "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    #      - user   内容: item["prompt"]
    #      - assistant 内容: 分别为 item["chosen"] 和 item["rejected"]
    #   2. 使用 tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=False)
    #      将消息列表转换为字符串
    #   3. 使用 tokenizer(text, max_length=MAX_SEQ_LEN, truncation=True, padding=False)
    #      分别对 chosen 和 rejected 编码，得到 input_ids 与 attention_mask
    #   4. 将以下字段追加到 records 列表：
    #      - "input_ids_chosen"       : chosen 的 input_ids
    #      - "attention_mask_chosen"  : chosen 的 attention_mask
    #      - "input_ids_rejected"     : rejected 的 input_ids
    #      - "attention_mask_rejected": rejected 的 attention_mask
    #   5. 返回 Dataset.from_list(records)
    #
    # 提示:
    #   - RewardTrainer 要求数据集字段名必须严格匹配上述四个名称
    #   - padding=False 表示不做填充（由 DataCollator 统一处理）
    # ===================================================================
    """
    system_msg = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # TODO[2-1]: 构造 chosen/rejected 消息列表，编码，追加到 records
            chosen_messages   = None  # 请补全
            rejected_messages = None  # 请补全

            chosen_text   = None  # 请调用 apply_chat_template
            rejected_text = None  # 请调用 apply_chat_template

            chosen_enc   = None  # 请调用 tokenizer 编码
            rejected_enc = None  # 请调用 tokenizer 编码

            records.append({
                "input_ids_chosen":          chosen_enc["input_ids"],
                "attention_mask_chosen":     chosen_enc["attention_mask"],
                "input_ids_rejected":        rejected_enc["input_ids"],
                "attention_mask_rejected":   rejected_enc["attention_mask"],
            })

    print(f"[数据加载] 共加载 {len(records)} 条偏好对")
    return Dataset.from_list(records)


def main():
    print("=" * 60)
    print("  步骤 2: 奖励模型训练 (Reward Model Training)")
    print("=" * 60)

    import os
    # 优先使用 SFT 模型，否则回退到基础模型
    model_path = SFT_MODEL if os.path.exists(SFT_MODEL) else BASE_MODEL
    print(f"\n[1/4] 加载分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[2/4] 加载并预处理偏好数据集: {DATA_PATH}")
    dataset = load_reward_dataset(DATA_PATH, tokenizer)

    print(f"\n[3/4] 配置模型与 LoRA")

    # ===================================================================
    # TODO[2-2]: 填写 LoRA task_type
    # ===================================================================
    # 设计要求:
    #   奖励模型基于 AutoModelForSequenceClassification（序列分类任务），
    #   对应的 PEFT TaskType 是哪个枚举值？
    #   候选: TaskType.CAUSAL_LM / TaskType.SEQ_CLS / TaskType.SEQ_2_SEQ_LM
    #
    # 提示:
    #   - CAUSAL_LM 对应语言模型生成（step1 SFT 使用）
    #   - SEQ_CLS 对应序列分类（打分/分类任务）
    #   思考: 为什么奖励模型使用序列分类而不是因果语言模型？
    # ===================================================================

    lora_config = LoraConfig(
        task_type=None,          # TODO[2-2]: 填写正确的 TaskType
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"\n[4/4] 开始训练奖励模型")

    # ===================================================================
    # TODO[2-3]: 补全 RewardConfig 关键参数
    # ===================================================================
    # 设计要求:
    #   A. evaluation_strategy : 评估策略，建议 "epoch"（每轮结束后评估）
    #   B. max_length          : 奖励模型处理的最大序列长度，使用配置区域的变量
    #
    # 提示:
    #   - evaluation_strategy="epoch" 表示每个 epoch 结束时在验证集上评估一次
    #   - max_length 决定了奖励模型能处理的最大输入长度，与 SFT 保持一致
    # ===================================================================

    reward_config = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        evaluation_strategy=None,    # TODO-A: 填写评估策略
        max_length=None,             # TODO-B: 填写最大序列长度
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  总参数量:   {total:,}")

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[完成] 奖励模型已保存至: {OUTPUT_DIR}")
    print("下一步: 运行 step3_rlhf.py 进行强化学习训练")


if __name__ == "__main__":
    main()
