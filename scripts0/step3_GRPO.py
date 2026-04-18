"""
步骤三：GRPO 强化学习微调 (Group Relative Policy Optimization)
数据集：内置提示词列表（无需外部数据集）
模型  ：Base + SFT LoRA Adapter（来自步骤一）+ 奖励模型（来自步骤二）
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from step2_reward_model import RewardModel

# ─────────────────────── 配置 ───────────────────────
BASE_MODEL_PATH   = "/home/Disk2/lvxinda/models"
SFT_MODEL_PATH    = "./output/sft_model"
REWARD_MODEL_PATH = "./output/reward_model"
OUTPUT_DIR        = "./output/grpo_model"
MAX_NEW_TOKENS    = 200
NUM_TRAIN_EPOCHS  = 1
BATCH_SIZE        = 4
NUM_GENERATIONS   = 4   # 每个 prompt 生成的候选数（GRPO 组大小 G）
LR                = 1.41e-5
KL_COEF           = 0.1
MAX_STEPS         = 20
# ────────────────────────────────────────────────────

TRAIN_PROMPTS = [
    "如何看待网络暴力现象？",
    "人工智能会取代人类工作吗？",
    "如何保持良好的心理健康？",
    "气候变化对我们的生活有什么影响？",
    "怎样才能成为一个诚实守信的人？",
    "如何正确处理与他人的矛盾和冲突？",
    "网络谣言有什么危害？该如何辨别？",
    "为什么要保护个人隐私？",
    "如何培养良好的学习习惯？",
    "社交媒体对人际关系有什么影响？",
    "如何理性看待失败和挫折？",
    "环境保护为什么重要？我们能做什么？",
    "如何在工作中保持高效率？",
    "什么是责任感？为什么重要？",
    "如何看待传统文化与现代化的关系？",
    "消费主义对社会有什么影响？",
]


# ── 1. 数据集 ────────────────────────────────────────
def load_dataset(tokenizer, num_repeats=5) -> Dataset:
    print(f"[数据] 构建训练数据集（{len(TRAIN_PROMPTS)} 条提示词 × {num_repeats} 次重复）...")
    records = []
    for _ in range(num_repeats):
        for p in TRAIN_PROMPTS:
            formatted = (
                f"<|im_start|>user\n{p}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            records.append({"prompt": formatted, "raw_prompt": p})
    random.shuffle(records)
    dataset = Dataset.from_list(records)
    print(f"[数据] 数据集大小：{len(dataset)} 条")
    return dataset


# ── 2. 模型 & 分词器 ─────────────────────────────────
def load_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # GRPO 生成时需要左填充

    print("[模型] 加载 Base Model + SFT LoRA Adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    actor_model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH, is_trainable=True)
    total     = sum(p.numel() for p in actor_model.parameters()) / 1e6
    trainable = sum(p.numel() for p in actor_model.parameters() if p.requires_grad) / 1e6
    print(f"  总参数：{total:.1f}M，可训练：{trainable:.1f}M")

    print("[模型] 加载奖励模型...")
    reward_model = RewardModel(BASE_MODEL_PATH).to(device)
    reward_model.load_state_dict(
        torch.load(f"{REWARD_MODEL_PATH}/reward_model.pt", map_location=device)
    )
    reward_model.eval()

    return actor_model, reward_model, tokenizer


# ── 3. 训练 ──────────────────────────────────────────
def build_reward_fn(reward_model, tokenizer, device):
    def reward_fn(prompts, completions, **kwargs):
        scores = []
        for prompt, completion in zip(prompts, completions):
            text = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{completion}<|im_end|>"
            )
            enc = tokenizer(
                text, max_length=512, truncation=True,
                padding="max_length", return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                score = reward_model(enc["input_ids"], enc["attention_mask"])
            scores.append(score.squeeze().float().item())
        return scores
    return reward_fn


def train(actor_model, reward_model, tokenizer, dataset, device):
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        beta=KL_COEF,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=0.7,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        logging_steps=1,        # 每步打印 loss / reward / kl 等指标
        save_steps=MAX_STEPS,
        remove_unused_columns=False,
        report_to="none",
    )
    reward_fn = build_reward_fn(reward_model, tokenizer, device)
    trainer = GRPOTrainer(
        model=actor_model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )
    print(f"\n[训练] 开始 GRPO 训练（batch={BATCH_SIZE}, G={NUM_GENERATIONS}, max_steps={MAX_STEPS}）...")
    trainer.train()
    return trainer


# ── 4. 保存 & 评估 ────────────────────────────────────
def save_and_evaluate(trainer, actor_model, reward_model, tokenizer, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[完成] GRPO 模型已保存至 {OUTPUT_DIR}")
    print("  下一步：运行 step4_evaluate.py")

    test_prompt = "水稻叶片发黄的原因和处理方法是什么？"
    formatted = (
        f"<|im_start|>user\n{test_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    actor_model.eval()
    with torch.no_grad():
        out = actor_model.generate(
            **inputs, max_new_tokens=200,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    text = (
        f"<|im_start|>user\n{test_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    enc = tokenizer(text, max_length=512, truncation=True,
                    padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        grpo_score = reward_model(enc["input_ids"], enc["attention_mask"]).item()

    print(f"\n[评估] 问题：{test_prompt}")
    print(f"  GRPO 回答（奖励分数={grpo_score:.4f}）：\n  {response}")


# ── 主流程 ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("步骤三：GRPO 强化学习微调")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    actor_model, reward_model, tokenizer = load_model_and_tokenizer(device)
    dataset = load_dataset(tokenizer)
    trainer = train(actor_model, reward_model, tokenizer, dataset, device)
    save_and_evaluate(trainer, actor_model, reward_model, tokenizer, device)


if __name__ == "__main__":
    main()
