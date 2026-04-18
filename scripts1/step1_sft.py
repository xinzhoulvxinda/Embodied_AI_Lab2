#!/usr/bin/env python3
"""
实验步骤1: 监督微调 (Supervised Fine-Tuning, SFT)
使用农业问答数据对基础模型进行指令微调

【学生任务说明】
  本文件包含 3 处 TODO，请按照注释中的"设计要求"独立补全。
  完成后运行：python step1_sft.py

环境要求:
    pip install transformers trl datasets peft accelerate bitsandbytes
"""

import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

# ======================== 配置区域（无需修改）========================
MODEL_NAME    = "qwen/Qwen2.5-1.5B-Instruct"   # ModelScope 模型 ID
DATA_PATH     = "./data/agriculture_sft.jsonl"
OUTPUT_DIR    = "./outputs/sft_model"
MAX_SEQ_LEN   = 512
EPOCHS        = 3
BATCH_SIZE    = 2
GRAD_ACCUM    = 4          # 等效 batch_size = 8
LEARNING_RATE = 2e-4
LORA_R        = 8          # LoRA rank，越大效果可能越好但显存需求增加
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
# ===================================================================


def load_dataset_from_jsonl(path: str) -> Dataset:
    """加载 JSONL 格式的农业问答数据集（已实现，无需修改）"""
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
    将问答对格式化为模型标准对话格式，用于 SFT 训练。

    # ===================================================================
    # TODO[1-1]: 实现对话格式化函数
    # ===================================================================
    # 设计要求:
    #   1. 构造包含三个角色的消息列表 messages（list of dict）:
    #        - role="system"   : 内容为农业顾问角色描述（见下方提示）
    #        - role="user"     : 内容来自 example["prompt"]
    #        - role="assistant": 内容来自 example["response"]
    #   2. 调用 tokenizer.apply_chat_template() 将消息列表转换为训练字符串
    #        参数: tokenize=False, add_generation_prompt=False
    #   3. 返回字典 {"text": <转换后的字符串>}
    #
    # 提示:
    #   - system 内容: "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    #   - apply_chat_template 会自动添加模型专用特殊标记（如 <|im_start|>）
    #   - 格式化后的文本示例:
    #       <|im_start|>system\n你是...<|im_end|>\n
    #       <|im_start|>user\n问题...<|im_end|>\n
    #       <|im_start|>assistant\n回答...<|im_end|>
    # ===================================================================

    messages = None   # TODO: 请补全消息列表
    text = None       # TODO: 请调用 apply_chat_template 生成训练文本
    return {"text": text}


def main():
    print("=" * 60)
    print("  步骤 1: 监督微调 (SFT)")
    print("=" * 60)

    # ---------- 1. 加载分词器 ----------
    print(f"\n[1/4] 加载分词器: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- 2. 加载数据集并格式化 ----------
    print(f"\n[2/4] 加载并预处理数据集: {DATA_PATH}")
    raw_dataset = load_dataset_from_jsonl(DATA_PATH)
    dataset = raw_dataset.map(
        lambda ex: format_chat_prompt(ex, tokenizer),
        remove_columns=raw_dataset.column_names
    )
    print(f"  示例文本 (前200字):\n  {dataset[0]['text'][:200]}...")

    # ---------- 3. 配置 LoRA ----------
    print(f"\n[3/4] 配置 LoRA 参数高效微调")

    # ===================================================================
    # TODO[1-2]: 填写 LoRA 目标模块列表
    # ===================================================================
    # 设计要求:
    #   选择 Qwen2.5 注意力层中需要注入 LoRA 适配器的模块名称列表。
    #   候选模块: "q_proj", "k_proj", "v_proj", "o_proj",
    #             "gate_proj", "up_proj", "down_proj"
    #
    # 提示:
    #   - 至少选择 q_proj 和 v_proj（查询与值投影，影响注意力质量）
    #   - 加入 k_proj、o_proj 可提升表达能力，但会增加可训练参数量
    #   - 加入 gate/up/down_proj 覆盖 FFN 层，效果更全面
    #   - 权衡: 模块越多 → 可训练参数↑ → 显存需求↑ → 潜在效果↑
    #   思考: 为什么 LoRA 比全量微调更节省显存？
    # ===================================================================

    target_modules = None  # TODO: 填写模块名称列表，例如 ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
    )
    print(f"  LoRA rank={LORA_R}, alpha={LORA_ALPHA}, 目标模块: {target_modules}")

    # ---------- 4. 训练配置与启动 ----------
    print(f"\n[4/4] 开始训练")

    # ===================================================================
    # TODO[1-3]: 补全 SFTConfig 关键训练参数
    # ===================================================================
    # 设计要求:
    #   填写下方标注 TODO 的三个参数：
    #   A. fp16         : 是否启用半精度（FP16）训练以节省显存 (True/False)
    #   B. max_seq_length: 训练时截断/填充的最大序列长度，使用配置区域的变量
    #   C. dataset_text_field: 数据集中存放训练文本的字段名（对应 format_chat_prompt 返回的 key）
    #
    # 提示:
    #   - FP16 在 NVIDIA GPU 上可将显存减少约 50%，训练速度提升约 1.5~2x
    #   - max_seq_length 超过此长度的样本会被截断，太短则丢失信息
    #   - dataset_text_field 必须与 format_chat_prompt 返回的字典 key 完全一致
    # ===================================================================

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
        fp16=None,                    # TODO-A: 填写 True 或 False
        dataloader_num_workers=0,
        report_to="none",
        max_seq_length=None,          # TODO-B: 填写最大序列长度
        dataset_text_field=None,      # TODO-C: 填写文本字段名
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
    print("下一步: 运行 step2_reward.py 训练奖励模型")


if __name__ == "__main__":
    main()
