"""
步骤一：监督微调 (SFT)
数据集：AI-ModelScope/alpaca-gpt4-data-zh（52K 条中文指令数据）
模型  ：Qwen/Qwen2.5-3B（Base Model）
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import inspect
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ─────────────────────── 配置 ───────────────────────
MODEL_LOCAL_DIR  = "/home/Disk2/lvxinda/models"
DATASET_ID       = "AI-ModelScope/alpaca-gpt4-data-zh"
DATA_SAMPLE_SIZE = 3000
OUTPUT_DIR       = "./output/sft_model"
MAX_LENGTH       = 512
NUM_EPOCHS       = 3
BATCH_SIZE       = 2
LR               = 2e-4
LORA_R           = 16
CUDA_DEVICE      = "cuda:0"   # CUDA_VISIBLE_DEVICES=1 后程序内唯一可见卡为 cuda:0
# ────────────────────────────────────────────────────


# ── 1. 数据集 ────────────────────────────────────────
def load_dataset(tokenizer) -> Dataset:
    from modelscope import MsDataset
    print(f"[数据] 加载 {DATASET_ID}（前 {DATA_SAMPLE_SIZE} 条）...")
    ms_dataset = MsDataset.load(DATASET_ID, split="train")

    records = []
    for i, item in enumerate(ms_dataset):
        if DATA_SAMPLE_SIZE is not None and i >= DATA_SAMPLE_SIZE:
            break
        instruction = (item.get("instruction") or "").strip()
        inp         = (item.get("input") or "").strip()
        output      = (item.get("output") or "").strip()
        if not instruction or not output:
            continue
        user_msg = f"{instruction}\n{inp}" if inp else instruction
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}, {"role": "assistant", "content": output}],
            tokenize=False, add_generation_prompt=False,
        )
        records.append({"text": text})

    dataset = Dataset.from_list(records)
    print(f"[数据] 加载完成：{len(dataset)} 条，示例：\n{dataset[0]['text'][:200]}\n{'─'*40}")
    return dataset


# ── 2. 模型 & 分词器 ─────────────────────────────────
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit QLoRA 必须用 fp16，bf16+4bit 会导致梯度 NaN
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        quantization_config=bnb_config,
        device_map=CUDA_DEVICE,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ── 3. 训练 ──────────────────────────────────────────
def train(model, tokenizer, dataset):
    # 兼容不同 trl 版本：有的版本把 dataset_text_field/max_seq_length 放在 SFTTrainer，
    # 有的版本放在 SFTConfig；tokenizer 参数名也可能是 tokenizer 或 processing_class。
    sft_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        dataloader_num_workers=2,
        packing=False,
    )
    sft_fields = getattr(SFTConfig, "__dataclass_fields__", {})
    if "dataset_text_field" in sft_fields:
        sft_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_fields:
        sft_kwargs["max_seq_length"] = MAX_LENGTH
    elif "max_length" in sft_fields:
        sft_kwargs["max_length"] = MAX_LENGTH

    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in sig.parameters and "dataset_text_field" not in sft_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sig.parameters and "max_seq_length" not in sft_kwargs:
        trainer_kwargs["max_seq_length"] = MAX_LENGTH
    elif "max_length" in sig.parameters and "max_length" not in sft_kwargs:
        trainer_kwargs["max_length"] = MAX_LENGTH

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    return trainer


# ── 4. 保存 & 评估 ────────────────────────────────────
def save_and_evaluate(model, tokenizer):
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    nan_keys = [k for k, v in lora_state.items() if torch.isnan(v).any()]
    if nan_keys:
        print(f"\n[警告] LoRA 权重存在 NaN（{len(nan_keys)}/{len(lora_state)} 层），训练无效，不保存")
        print("  NaN 层示例：", nan_keys[:3])
        return

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[完成] 模型已保存至 {OUTPUT_DIR}")
    print("  下一步：运行 step2_reward_model.py")


# ── 主流程 ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("步骤一：监督微调 (SFT)")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset(tokenizer)
    train(model, tokenizer, dataset)
    save_and_evaluate(model, tokenizer)


if __name__ == "__main__":
    main()
