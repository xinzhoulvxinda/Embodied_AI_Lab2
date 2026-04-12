"""
实验步骤一：监督微调 (SFT - Supervised Fine-Tuning)
使用农业问答数据集对 Qwen2.5-1.5B-Instruct 进行微调

环境要求：
  pip install transformers datasets peft accelerate bitsandbytes trl torch
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ─────────────────────── 配置区域 ───────────────────────
MODEL_NAME   = "qwen/Qwen2.5-1.5B-Instruct"  # ModelScope 模型 ID
DATA_PATH    = "./data/agriculture_sft.jsonl"
OUTPUT_DIR   = "./output/sft_model"
MAX_LENGTH   = 512
NUM_EPOCHS   = 3
BATCH_SIZE   = 4
LR           = 2e-4
# ──────────────────────────────────────────────────────

def download_model_if_needed(model_name, local_dir="./models/base"):
    """
    优先使用本地缓存；若不存在则从魔塔社区（ModelScope）自动下载。
    需要提前安装：pip install modelscope
    """
    import os
    if os.path.isdir(local_dir) and any(
        f.endswith((".safetensors", ".bin", ".pt")) for f in os.listdir(local_dir)
    ):
        print(f"[模型] 发现本地缓存：{local_dir}，跳过下载")
        return local_dir

    print(f"[模型] 本地缓存不存在，从魔塔社区下载：{model_name}")
    try:
        from modelscope import snapshot_download
        saved_path = snapshot_download(model_name, cache_dir="./models")
        print(f"[模型] 下载完成，保存至：{saved_path}")
        return saved_path
    except ImportError:
        raise RuntimeError(
            "未找到 modelscope 库，请先执行：pip install modelscope"
        )


def load_data(path):
    """加载并格式化数据集"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    def format_sample(item):
        # 使用 ChatML 格式拼接指令和输出
        instruction = item["instruction"]
        inp  = item.get("input", "")
        out  = item["output"]
        if inp:
            user_msg = f"{instruction}\n\n补充信息：{inp}"
        else:
            user_msg = instruction
        return {"text": f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"}

    return Dataset.from_list([format_sample(d) for d in data])


def main():
    print("=" * 60)
    print("步骤一：监督微调 (SFT)")
    print("=" * 60)

    # ── 1. 加载分词器 ──
    # ── 0. 确保模型已下载 ──
    MODEL_PATH = download_model_if_needed(MODEL_NAME)

    print("\n[1/4] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 量化配置（4-bit，显存友好）──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 3. 加载模型 ──
    print("[2/4] 加载基础模型（4-bit量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── 4. LoRA 配置 ──
    print("[3/4] 注入 LoRA 适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,               # LoRA 秩（秩越大，参数越多，显存越高）
        lora_alpha=32,      # 缩放因子
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 查看可训练参数占比

    # ── 5. 加载数据 ──
    dataset = load_data(DATA_PATH)
    print(f"\n数据集大小：{len(dataset)} 条")
    print(f"示例数据：\n{dataset[0]['text'][:200]}...\n")

    # ── 6. 训练配置 ──
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )

    # ── 7. 启动训练 ──
    print("[4/4] 开始训练...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        packing=False,
    )
    trainer.train()

    # ── 8. 保存 ──
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ SFT 训练完成！模型已保存至 {OUTPUT_DIR}")

    # ── 9. 简单推理测试 ──
    print("\n─── 推理测试 ───")
    test_input = "水稻叶片发黄是什么原因？"
    inputs = tokenizer(
        f"<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"问题：{test_input}")
    print(f"回答：{response}")


if __name__ == "__main__":
    main()
