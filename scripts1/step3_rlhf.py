#!/usr/bin/env python3
"""
实验步骤3: 基于人类反馈的强化学习训练 (RLHF)
使用奖励模型信号通过强化学习算法进一步优化语言模型

RLHF 最终阶段：策略模型(Policy)在奖励模型的引导下不断优化输出质量，
同时通过 KL 散度约束防止策略偏离参考模型太远（防止崩坏）。

【学生任务说明】
  本文件包含 3 处 TODO，其中 TODO[3-3] 是核心难点（RL 训练主循环）。

  【算法选择】你可以选择以下任意一种算法完成 TODO[3-3]：
    A. PPO  (Proximal Policy Optimization) ← 默认框架，已搭好骨架
    B. DPO  (Direct Preference Optimization) ← 更简洁，无需 RL 循环，可参考 TRL 文档
    C. GRPO (Group Relative Policy Optimization) ← DeepSeek-R1 采用的方法

  完成后运行：python step3_rlhf_student.py

环境要求:
    pip install transformers trl datasets peft accelerate
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType
import os

# ======================== 配置区域（无需修改）========================
SFT_MODEL      = "./outputs/sft_model"
REWARD_MODEL   = "./outputs/reward_model"
BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH      = "./data/agriculture_sft.jsonl"
OUTPUT_DIR     = "./outputs/ppo_model"
MAX_NEW_TOKENS = 200      # 每次生成的最大 token 数
BATCH_SIZE     = 2        # PPO mini-batch 大小
PPO_EPOCHS     = 2        # PPO 内部更新轮数
TOTAL_STEPS    = 20       # 总训练步数（实验用，正式训练建议 100~500 步）
KL_COEFF       = 0.1      # KL 散度惩罚系数
LEARNING_RATE  = 1e-5
# ===================================================================


def load_queries(path: str, tokenizer, n_samples: int = 30):
    """
    从 SFT 数据集中提取问题，作为 PPO/RL 训练的 query 输入。
    每步训练: 模型生成回答 → 奖励模型打分 → 策略梯度更新

    # ===================================================================
    # TODO[3-1]: 实现 query 加载函数
    # ===================================================================
    # 设计要求:
    #   读取 DATA_PATH 中的每条 JSONL 记录（含 "prompt" 字段）：
    #   1. 构造 system + user 消息列表（不含 assistant，因为要让模型生成）
    #      - system 内容: "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    #      - user   内容: item["prompt"]
    #   2. 调用 tokenizer.apply_chat_template(
    #          messages, tokenize=False, add_generation_prompt=True)
    #      注意: add_generation_prompt=True 会在末尾添加 <|im_start|>assistant\n，
    #            引导模型开始生成回答
    #   3. 将 {"query": query_text, "prompt": item["prompt"]} 追加到 queries
    #   4. 取前 n_samples 条，返回 queries 列表
    #
    # 提示:
    #   - 与 SFT 格式化的区别: 这里不包含 assistant 消息，且 add_generation_prompt=True
    #   - query 是模型的输入，response 是模型的输出，两者共同送入奖励模型打分
    # ===================================================================
    """
    system_msg = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    queries = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # TODO: 构造消息列表并生成 query_text
            messages   = None  # 请补全
            query_text = None  # 请调用 apply_chat_template
            queries.append({"query": query_text, "prompt": item["prompt"]})

    queries = queries[:n_samples]
    print(f"[数据加载] 共准备 {len(queries)} 条 RL 训练查询")
    return queries


def main():
    print("=" * 60)
    print("  步骤 3: 基于人类反馈的强化学习训练 (RLHF)")
    print("=" * 60)
    print("\nPPO-RLHF 训练流程:")
    print("  Query → [策略模型生成] → Response")
    print("  Response → [奖励模型打分] → Reward")
    print("  Reward - KL惩罚 → [PPO策略梯度] → 更新策略模型")
    print("  重复上述步骤直到奖励分数稳定上升\n")

    policy_path = SFT_MODEL if os.path.exists(SFT_MODEL) else BASE_MODEL
    reward_path = REWARD_MODEL if os.path.exists(REWARD_MODEL) else BASE_MODEL

    # ---------- 1. 加载分词器 ----------
    print(f"[1/5] 加载分词器: {policy_path}")
    tokenizer = AutoTokenizer.from_pretrained(policy_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # 生成时使用 left padding

    # ---------- 2. 加载策略模型（带价值头，用于 PPO）----------
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

    # ---------- 3. 加载奖励模型 ----------
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

    # ---------- 4. 准备训练数据 ----------
    print(f"\n[4/5] 准备 RL 训练查询")
    queries = load_queries(DATA_PATH, tokenizer, n_samples=30)

    # ---------- 5. PPO 训练 ----------
    print(f"\n[5/5] 开始 PPO 训练 (共 {TOTAL_STEPS} 步)")

    # ===================================================================
    # TODO[3-2]: 补全 PPOConfig 关键强化学习参数
    # ===================================================================
    # 设计要求:
    #   A. kl_penalty    : KL 散度的计算方式，填写字符串 "kl"
    #   B. init_kl_coef  : KL 惩罚系数初始值，使用配置区域的 KL_COEFF 变量
    #   C. target_kl     : 自适应 KL 目标值（float），建议 6.0
    #                      当实际 KL > target_kl 时，自动增大惩罚系数
    #   D. adap_kl_ctrl  : 是否启用自适应 KL 控制 (True/False)
    #
    # 提示:
    #   - KL 散度衡量策略模型与参考模型（SFT）的偏离程度
    #   - init_kl_coef 过大 → 模型更新保守，奖励提升慢
    #   - init_kl_coef 过小 → 模型可能崩坏（输出乱码或重复文本）
    #   - adap_kl_ctrl=True 可动态调整系数，是更稳健的选择
    #   思考: 为什么 RLHF 需要 KL 约束？去掉后会发生什么？
    # ===================================================================

    ppo_config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        kl_penalty=None,         # TODO-A: 填写 KL 计算方式
        init_kl_coef=None,       # TODO-B: 填写初始 KL 系数
        target_kl=None,          # TODO-C: 填写目标 KL 值
        adap_kl_ctrl=None,       # TODO-D: 是否自适应调整 KL 系数
        log_with=None,
        model_name=policy_path,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None,          # 使用 PEFT，ref_model 设为 None（自动对比基础权重）
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

    # ===================================================================
    # TODO[3-3]: 实现 PPO 训练主循环（核心难点）
    # ===================================================================
    # 设计要求（每个 step 依次执行以下步骤）:
    #
    #   a. 编码查询（已给出，无需修改）
    #      query_tensors: 将 batch 中每条 query 文本编码为 1D input_ids tensor
    #
    #   b. 策略模型生成回答
    #      调用: ppo_trainer.generate(query_tensors, return_prompt=False, **gen_kwargs)
    #      返回: response_tensors（list of tensor，仅含新生成 token）
    #
    #   c. 解码生成文本
    #      对 response_tensors 中每个 tensor 调用 tokenizer.decode(..., skip_special_tokens=True)
    #      返回: responses（list of str）
    #
    #   d. 奖励模型打分
    #      对每条 (query + response) 拼接文本：
    #        full_text = query_item["query"] + resp_text
    #      使用 reward_tokenizer 编码后送入 reward_model，
    #      取 reward_model(**enc).logits[0].item() 作为分数，
    #      封装为 torch.tensor(score, dtype=torch.float32)
    #      返回: rewards（list of tensor，长度等于 batch_size）
    #
    #   e. PPO 策略梯度更新
    #      调用: ppo_trainer.step(query_tensors, response_tensors, rewards)
    #      返回: stats（包含 KL 散度等训练统计信息的字典）
    #
    # 提示:
    #   - rewards 必须是 Python list，元素为 0-dim 或 1-dim torch.tensor
    #   - 奖励模型编码时需将 enc 移至 device（.to(device)）
    #   - 使用 torch.no_grad() 包裹奖励模型的前向传播
    #   - stats.get('objective/kl', 0) 可获取当前步的 KL 散度值
    # ===================================================================

    step_logs = []
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= TOTAL_STEPS:
            break

        # a. 编码查询（已给出，无需修改）
        query_tensors = [
            tokenizer(b["query"], return_tensors="pt").input_ids.squeeze(0).to(device)
            for b in batch
        ]

        # TODO: 步骤 b - 策略模型生成回答
        response_tensors = None   # 请补全

        # TODO: 步骤 c - 解码生成文本
        responses = None          # 请补全

        # TODO: 步骤 d - 奖励模型打分
        rewards = None            # 请补全

        # TODO: 步骤 e - PPO 策略梯度更新
        stats = None              # 请补全

        # 日志记录（已给出，无需修改）
        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        print(f"  Step {step+1:3d}/{TOTAL_STEPS} | "
              f"Mean Reward: {mean_reward:.3f} | "
              f"KL Div: {stats.get('objective/kl', 0):.3f}")
        step_logs.append({"step": step + 1, "mean_reward": mean_reward})

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
    print("\n下一步: 运行 step4_evaluate_student.py 对比各阶段模型效果")


# ===========================================================================
# 【可选扩展】使用 DPO 替代 PPO
# ===========================================================================
# DPO (Direct Preference Optimization) 是 PPO 的简化替代方案：
#   - 无需单独的奖励模型，直接在偏好对上优化策略
#   - 无需 RL 训练循环，直接用监督学习方式优化
#   - 公式: L_DPO = -log σ(β · (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
#
# TRL 实现（可参考官方文档）:
#   from trl import DPOTrainer, DPOConfig
#   dpo_config = DPOConfig(beta=0.1, ...)
#   trainer = DPOTrainer(model, ref_model, args=dpo_config,
#                        train_dataset=preference_dataset, tokenizer=tokenizer)
#   trainer.train()
#
# 若选择 DPO，需修改:
#   1. 数据格式: {"prompt": ..., "chosen": ..., "rejected": ...}（即 step2 的数据）
#   2. 删除奖励模型相关代码
#   3. 将上方 PPO 训练循环替换为 DPOTrainer.train()
# ===========================================================================


if __name__ == "__main__":
    main()
