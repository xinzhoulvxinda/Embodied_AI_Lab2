#!/usr/bin/env python3
"""
实验步骤4: 模型评估与对比分析
对比 基础模型 → SFT模型 → PPO-RLHF模型 三个阶段的输出质量变化

【学生任务说明】
  本文件包含 3 处 TODO，请按照注释中的"设计要求"独立补全。
  完成后运行：python step4_evaluate_student.py

输出:
    - 终端打印各阶段模型对比输出及奖励分数
    - evaluation_results.json 保存完整对比结果
    - 统计各模型平均奖励分数，验证 RLHF 管道效果
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# ======================== 配置区域（无需修改）========================
BASE_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_MODEL     = "./outputs/sft_model"
PPO_MODEL     = "./outputs/ppo_model"
REWARD_MODEL  = "./outputs/reward_model"
OUTPUT_FILE   = "./outputs/evaluation_results.json"

# 评估用测试问题（模型训练时未见过）
TEST_QUESTIONS = [
    "如何提高大豆的固氮效率？",
    "温室番茄种植中常见的营养缺乏症有哪些？",
    "什么是保护性耕作技术？它对土壤有什么好处？",
    "农业病虫害预测预报的主要方法有哪些？",
    "如何判断果树是否需要修剪？修剪的基本原则是什么？",
]

MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.7
# ===================================================================


def load_model_and_tokenizer(model_path: str, base_model: str = None):
    """加载模型和分词器，若路径不存在则回退到基础模型（已实现，无需修改）"""
    if not os.path.exists(model_path):
        print(f"  警告: {model_path} 不存在，使用基础模型 {base_model}")
        model_path = base_model

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    """
    使用指定模型生成对问题的回答。

    # ===================================================================
    # TODO[4-1]: 实现模型推理生成函数
    # ===================================================================
    # 设计要求:
    #   1. 获取模型所在 device: device = next(model.parameters()).device
    #   2. 构造 system + user 消息列表:
    #        system: "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    #        user:   question 参数
    #   3. 调用 tokenizer.apply_chat_template(
    #          messages, tokenize=False, add_generation_prompt=True)
    #      生成带生成提示的文本
    #   4. 对文本编码: tokenizer(text, return_tensors="pt").to(device)
    #   5. 在 torch.no_grad() 下调用 model.generate():
    #        参数: max_new_tokens=max_new_tokens, do_sample=True,
    #              temperature=TEMPERATURE, top_p=0.9,
    #              pad_token_id=tokenizer.pad_token_id,
    #              eos_token_id=tokenizer.eos_token_id
    #   6. 只保留新生成的 token（跳过 prompt 部分）:
    #        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    #   7. 解码并返回: tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    #
    # 提示:
    #   - add_generation_prompt=True 在末尾添加 <|im_start|>assistant\n
    #   - outputs[0] 包含完整序列（prompt + 生成内容），需截取新生成部分
    #   - .strip() 去除首尾空白字符
    # ===================================================================
    """
    # TODO[4-1] 实现开始
    device = next(model.parameters()).device

    messages = [
        {"role": "system", "content": "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"},
        {"role": "user",   "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # TODO[4-1] 实现结束


def compute_reward_score(reward_model, reward_tokenizer, question: str, answer: str, device) -> float:
    """
    使用奖励模型对 (问题, 回答) 对打分，返回奖励分数（float）。

    # ===================================================================
    # TODO[4-2]: 实现奖励打分函数
    # ===================================================================
    # 设计要求:
    #   1. 拼接对话文本（与奖励模型训练时的格式保持一致）:
    #        system_msg = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    #        text = f"System: {system_msg}\nUser: {question}\nAssistant: {answer}"
    #   2. 使用 reward_tokenizer 编码:
    #        enc = reward_tokenizer(text, return_tensors="pt",
    #                               max_length=512, truncation=True)
    #   3. 将 enc 移至 device: enc = enc.to(device)
    #   4. 在 torch.no_grad() 下调用奖励模型:
    #        score = reward_model(**enc).logits[0].item()
    #   5. 返回 score（float）
    #
    # 提示:
    #   - reward_model 是 AutoModelForSequenceClassification(num_labels=1)
    #   - logits 形状: [batch_size, 1]，logits[0].item() 取第一个样本的标量分数
    #   - 分数越高 → 回答越符合人类偏好
    # ===================================================================
    """
    # TODO[4-2] 实现开始
    system_msg = "你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。"
    text = f"System: {system_msg}\nUser: {question}\nAssistant: {answer}"

    enc = reward_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    enc = enc.to(device)

    with torch.no_grad():
        score = reward_model(**enc).logits[0].item()

    return score
    # TODO[4-2] 实现结束


def main():
    print("=" * 60)
    print("  步骤 4: 模型评估与对比分析")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")

    # 加载奖励模型（用于自动评分）
    reward_available = False
    if os.path.exists(REWARD_MODEL):
        print(f"\n[奖励模型] 加载: {REWARD_MODEL}")
        reward_tok = AutoTokenizer.from_pretrained(REWARD_MODEL, trust_remote_code=True)
        if reward_tok.pad_token is None:
            reward_tok.pad_token = reward_tok.eos_token
        reward_mdl = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL, num_labels=1, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()
        reward_available = True
    else:
        print("\n[奖励模型] 未找到奖励模型，跳过自动评分")

    # 要对比的三个阶段模型
    models_to_eval = [
        ("基础模型 (Base)", BASE_MODEL),
        ("SFT 模型",        SFT_MODEL),
        ("PPO-RLHF 模型",   PPO_MODEL),
    ]

    all_results = []

    for q_idx, question in enumerate(TEST_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"问题 {q_idx+1}: {question}")
        print(f"{'='*60}")

        q_results = {"question": question, "responses": []}

        for model_name, model_path in models_to_eval:
            print(f"\n[{model_name}]")
            print(f"  加载模型: {model_path if os.path.exists(model_path) else BASE_MODEL}")

            try:
                model, tokenizer = load_model_and_tokenizer(model_path, BASE_MODEL)
                response = generate_response(model, tokenizer, question, MAX_NEW_TOKENS)

                reward_score = None
                if reward_available:
                    reward_score = compute_reward_score(
                        reward_mdl, reward_tok, question, response, device
                    )
                    print(f"  奖励分数: {reward_score:.4f}")

                print(f"  回答: {response[:300]}{'...' if len(response) > 300 else ''}")

                q_results["responses"].append({
                    "model": model_name,
                    "response": response,
                    "reward_score": reward_score,
                })

                # 释放显存
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  错误: {e}")
                q_results["responses"].append({
                    "model": model_name, "response": f"[错误: {e}]", "reward_score": None
                })

        all_results.append(q_results)

    # ===================================================================
    # TODO[4-3]: 实现各模型平均奖励分数的统计与打印
    # ===================================================================
    # 设计要求:
    #   遍历 models_to_eval 中的每个 (model_name, _)：
    #   1. 从 all_results 中收集该模型的所有非 None 奖励分数
    #      （遍历 all_results → 遍历每个 qr["responses"] → 匹配 r["model"] == model_name）
    #   2. 若 scores 非空，计算平均值: avg = sum(scores) / len(scores)
    #   3. 打印格式: "  {model_name:20s}: 平均奖励分数 = {avg:.4f}"
    #
    # 提示:
    #   - 通过观察三阶段的平均分数，验证 RLHF 管道是否有效提升了回答质量
    #   - 预期结果（理想情况）: Base < SFT < PPO-RLHF
    #   - 若结果不符合预期，思考: 数据量不足？训练轮数不够？KL 系数设置是否合理？
    # ===================================================================

    print(f"\n{'='*60}")
    print("  评估统计摘要")
    print(f"{'='*60}")

    if reward_available:
        # TODO[4-3] 实现开始
        for model_name, _ in models_to_eval:
            scores = [
                r["reward_score"]
                for qr in all_results
                for r in qr["responses"]
                if r["model"] == model_name and r["reward_score"] is not None
            ]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {model_name:20s}: 平均奖励分数 = {avg:.4f}")
            else:
                print(f"  {model_name:20s}: 无可用奖励分数")
        # TODO[4-3] 实现结束

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 评估结果已保存至: {OUTPUT_FILE}")
    print("\n实验分析要点（请在实验报告中回答）:")
    print("  1. 各阶段模型回答在专业性、详细程度上有何差异？")
    print("  2. 奖励分数随训练阶段是否呈上升趋势？趋势是否符合预期？")
    print("  3. PPO 的 KL 散度约束对模型输出多样性有何影响？")
    print("  4. 如果你选择了 DPO/GRPO，与 PPO 相比有何优劣？")


if __name__ == "__main__":
    main()
