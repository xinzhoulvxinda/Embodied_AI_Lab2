"""
实验评估脚本：对比 SFT 和 PPO 模型的效果
生成对比报告，计算各项指标
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from step2_reward_model import RewardModel

SFT_MODEL_PATH    = "./output/sft_model"
PPO_MODEL_PATH    = "./output/ppo_model"
REWARD_MODEL_PATH = "./output/reward_model"

TEST_QUESTIONS = [
    "水稻叶片发黄是什么原因？怎么处理？",
    "玉米螟有哪些防治方法？",
    "小麦赤霉病的最佳防治时期是什么时候？",
    "如何判断土壤缺钾？",
    "温室黄瓜得了霜霉病怎么办？",
]


def generate_response(model, tokenizer, prompt, device, max_new_tokens=150):
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def compute_reward(reward_model, tokenizer, prompt, response, device):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    enc = tokenizer(text, max_length=512, truncation=True,
                    padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        return reward_model(enc["input_ids"], enc["attention_mask"]).item()


def response_length(response):
    return len(response)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("加载模型中...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    ppo_model = AutoModelForCausalLM.from_pretrained(
        PPO_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    reward_model = RewardModel(SFT_MODEL_PATH).to(device)
    reward_model.load_state_dict(torch.load(f"{REWARD_MODEL_PATH}/reward_model.pt", map_location=device))
    reward_model.eval()

    results = []
    print("\n" + "=" * 80)
    print("SFT vs PPO 模型对比评估")
    print("=" * 80)

    for q in TEST_QUESTIONS:
        sft_resp = generate_response(sft_model, tokenizer, q, device)
        ppo_resp = generate_response(ppo_model, tokenizer, q, device)
        sft_score = compute_reward(reward_model, tokenizer, q, sft_resp, device)
        ppo_score = compute_reward(reward_model, tokenizer, q, ppo_resp, device)

        results.append({
            "question": q,
            "sft_response": sft_resp,
            "ppo_response": ppo_resp,
            "sft_reward": sft_score,
            "ppo_reward": ppo_score,
            "sft_length": response_length(sft_resp),
            "ppo_length": response_length(ppo_resp),
        })

        print(f"\n问题：{q}")
        print(f"SFT 奖励分数：{sft_score:.4f}  |  PPO 奖励分数：{ppo_score:.4f}  "
              f"| {'PPO ↑' if ppo_score > sft_score else 'SFT ↑'}")

    # 汇总统计
    sft_rewards = [r["sft_reward"] for r in results]
    ppo_rewards = [r["ppo_reward"] for r in results]
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"SFT  平均奖励：{np.mean(sft_rewards):.4f} ± {np.std(sft_rewards):.4f}")
    print(f"PPO  平均奖励：{np.mean(ppo_rewards):.4f} ± {np.std(ppo_rewards):.4f}")
    wins = sum(1 for s, p in zip(sft_rewards, ppo_rewards) if p > s)
    print(f"PPO 胜率（奖励分数更高）：{wins}/{len(results)} = {wins/len(results):.0%}")

    # 保存报告
    with open("./output/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n✅ 详细评估报告已保存至 ./output/evaluation_report.json")


if __name__ == "__main__":
    main()
