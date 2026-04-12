"""
实验步骤三：PPO 强化学习微调 (Proximal Policy Optimization)
利用奖励模型的反馈信号，通过 PPO 算法进一步优化语言模型

核心组件：
  - Actor (策略网络)：需要优化的语言模型（来自步骤一）
  - Critic (价值网络)：估计状态价值
  - Reward Model：来自步骤二，提供奖励信号
  - Reference Model：原始 SFT 模型，用于 KL 散度约束
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from step2_reward_model import RewardModel  # 复用步骤二的类

# ─────────────────────── 配置区域 ───────────────────────
SFT_MODEL_PATH    = "./output/sft_model"
REWARD_MODEL_PATH = "./output/reward_model"
OUTPUT_DIR        = "./output/ppo_model"
MAX_NEW_TOKENS    = 200
PPO_EPOCHS        = 4        # 每批数据的 PPO 更新次数
BATCH_SIZE        = 4
LR                = 1.41e-5
KL_COEF           = 0.2      # KL 散度惩罚系数（防止模型偏离太远）
# ──────────────────────────────────────────────────────

# 测试用提示词（生成回答后由奖励模型打分）
TEST_PROMPTS = [
    "水稻出现叶尖枯黄该怎么处理？",
    "玉米螟的综合防治方法有哪些？",
    "如何判断土壤缺氮并进行补救？",
    "大豆根腐病怎么预防？",
    "苹果树腐烂病的处理步骤是什么？",
    "草莓白粉病如何进行防治？",
    "奶牛发情有哪些判断指标？",
    "黄瓜霜霉病怎么识别和防治？",
]


def compute_reward(reward_model, tokenizer, prompt, response, device):
    """用奖励模型对生成的回答打分"""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    enc = tokenizer(text, max_length=512, truncation=True,
                    padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        score = reward_model(enc["input_ids"], enc["attention_mask"])
    return score.squeeze().float()


def main():
    print("=" * 60)
    print("步骤三：PPO 强化学习微调")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")

    # ── 1. 加载分词器 ──
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 加载 Actor（带价值头的策略模型）──
    print("\n[1/3] 加载 Actor 模型（SFT 微调结果）...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        SFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── 3. 加载奖励模型 ──
    print("[2/3] 加载奖励模型...")
    reward_model = RewardModel(SFT_MODEL_PATH).to(device)
    reward_model.load_state_dict(torch.load(f"{REWARD_MODEL_PATH}/reward_model.pt", map_location=device))
    reward_model.eval()

    # ── 4. PPO 配置 ──
    print("[3/3] 初始化 PPO 训练器...")
    ppo_config = PPOConfig(
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        kl_penalty="kl",
        init_kl_coef=KL_COEF,
        target_kl=6.0,           # 自适应 KL 目标值
        horizon=10000,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,           # PPO 裁剪范围
        vf_coef=0.1,             # 价值函数损失权重
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    # ── 5. PPO 训练循环 ──
    print("\n开始 PPO 训练...")
    print("训练流程：生成回答 → 奖励模型打分 → PPO 更新策略\n")

    num_steps = 20   # 总训练步数（实验用，可适当增大）
    for step in range(num_steps):
        # 随机采样 batch 大小的提示词
        import random
        batch_prompts = random.choices(TEST_PROMPTS, k=BATCH_SIZE)

        # 5.1 Tokenize 提示词
        query_tensors = []
        for prompt in batch_prompts:
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            enc = tokenizer(formatted, return_tensors="pt").input_ids.squeeze(0)
            query_tensors.append(enc.to(device))

        # 5.2 生成回答（Actor 采样）
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

        # 5.3 解码回答文本
        responses = [
            tokenizer.decode(r.squeeze()[len(q):], skip_special_tokens=True)
            for q, r in zip(query_tensors, response_tensors)
        ]

        # 5.4 奖励模型打分
        rewards = [
            compute_reward(reward_model, tokenizer, p, r, device)
            for p, r in zip(batch_prompts, responses)
        ]

        # 5.5 PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 5.6 日志输出
        mean_reward = torch.stack(rewards).mean().item()
        kl = stats.get("objective/kl", 0)
        pg_loss = stats.get("ppo/loss/policy", 0)
        print(f"Step {step+1:3d}/{num_steps} | "
              f"平均奖励: {mean_reward:.4f} | "
              f"KL散度: {kl:.4f} | "
              f"策略损失: {pg_loss:.4f}")
        if step % 5 == 0:
            print(f"  └─ 示例生成：{responses[0][:80]}...")

    # ── 6. 保存最终模型 ──
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ppo_trainer.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ PPO 训练完成！模型已保存至 {OUTPUT_DIR}")

    # ── 7. 对比实验：SFT vs PPO ──
    print("\n─── SFT vs PPO 效果对比 ───")
    test_prompt = "水稻叶片发黄的原因和处理方法是什么？"
    formatted = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    ppo_response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    ppo_score = compute_reward(reward_model, tokenizer, test_prompt, ppo_response, device).item()

    print(f"问题：{test_prompt}")
    print(f"\nPPO 模型回答（奖励分数: {ppo_score:.4f}）：\n{ppo_response}")
    print("\n提示：分数越高表示奖励模型认为回答质量越好。")


if __name__ == "__main__":
    main()
