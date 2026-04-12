"""
实验步骤二：奖励模型训练 (Reward Model Training)
使用人类偏好对数据训练奖励模型

原理：给定同一问题的"好回答"(chosen)和"差回答"(rejected)，
训练模型使得 score(chosen) > score(rejected)
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─────────────────────── 配置区域 ───────────────────────
SFT_MODEL_PATH = "./output/sft_model"   # 步骤一的输出
DATA_PATH      = "./data/agriculture_reward.jsonl"
OUTPUT_DIR     = "./output/reward_model"
MAX_LENGTH     = 256
NUM_EPOCHS     = 5
BATCH_SIZE     = 2
LR             = 1e-5
# ──────────────────────────────────────────────────────


class RewardModel(nn.Module):
    """在语言模型基础上添加打分头"""

    def __init__(self, base_model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        hidden_size = self.model.config.hidden_size
        # 线性打分头：将隐藏状态映射为标量分数
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 取最后一个 token 的隐藏状态作为序列表示
        last_hidden = outputs.last_hidden_state[:, -1, :]
        score = self.score_head(last_hidden.float())
        return score.squeeze(-1)


class PreferenceDataset(Dataset):
    """偏好对数据集"""

    def __init__(self, path, tokenizer, max_len):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def tokenize(self, prompt, response):
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        enc = self.tokenizer(
            text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("prompt", item.get("instruction", ""))
        chosen_ids,  chosen_mask  = self.tokenize(prompt, item["chosen"])
        rejected_ids, rejected_mask = self.tokenize(prompt, item["rejected"])
        return chosen_ids, chosen_mask, rejected_ids, rejected_mask


def preference_loss(chosen_scores, rejected_scores):
    """Bradley-Terry 偏好损失：使 chosen 分数高于 rejected"""
    return -torch.log(torch.sigmoid(chosen_scores - rejected_scores)).mean()


def main():
    print("=" * 60)
    print("步骤二：奖励模型训练 (Reward Model)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")

    # ── 1. 加载分词器 ──
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 构建数据集 ──
    dataset = PreferenceDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"偏好对数量：{len(dataset)} 条")

    # ── 3. 初始化奖励模型 ──
    print("\n初始化奖励模型...")
    model = RewardModel(SFT_MODEL_PATH).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量：{total_params/1e6:.1f}M，可训练：{trainable_params/1e6:.1f}M")

    # ── 4. 优化器 ──
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))

    # ── 5. 训练循环 ──
    print("\n开始训练奖励模型...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0

        for step, (c_ids, c_mask, r_ids, r_mask) in enumerate(dataloader):
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            r_ids, r_mask = r_ids.to(device), r_mask.to(device)

            chosen_scores   = model(c_ids, c_mask)
            rejected_scores = model(r_ids, r_mask)

            loss = preference_loss(chosen_scores, rejected_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            # 准确率：chosen 分数 > rejected 分数的比例
            correct += (chosen_scores > rejected_scores).sum().item()

        avg_loss = total_loss / len(dataloader)
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | 偏好准确率: {acc:.2%}")

    # ── 6. 保存 ──
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/reward_model.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ 奖励模型训练完成！已保存至 {OUTPUT_DIR}")

    # ── 7. 推理演示 ──
    print("\n─── 奖励分数对比演示 ───")
    model.eval()
    test_cases = [
        {
            "prompt": "水稻叶片枯黄是什么原因？",
            "good": "水稻叶尖枯黄常见原因有缺钾症、纹枯病、干旱胁迫或高温热害，建议根据发病部位综合判断后对症处理。",
            "bad": "水稻叶子黄了多施肥就行，买点化肥撒上去应该会好的。"
        }
    ]
    for case in test_cases:
        def get_score(response):
            ids, mask = PreferenceDataset.tokenize.__func__(
                PreferenceDataset(DATA_PATH, tokenizer, MAX_LENGTH),
                case["prompt"], response
            )
            with torch.no_grad():
                return model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)).item()

        good_score = get_score(case["good"])
        bad_score  = get_score(case["bad"])
        print(f"问题：{case['prompt']}")
        print(f"好回答分数：{good_score:.4f}")
        print(f"差回答分数：{bad_score:.4f}")
        print(f"结论：{'✅ 模型正确区分偏好' if good_score > bad_score else '❌ 模型未能区分偏好，需要更多训练'}\n")


if __name__ == "__main__":
    main()
