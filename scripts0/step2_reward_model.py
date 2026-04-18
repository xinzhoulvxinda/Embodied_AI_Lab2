"""
步骤二：奖励模型训练 (Reward Model)
数据集：iic/CValues-Comparison（中文价值观偏好对）
原理  ：Bradley-Terry 损失，使 score(chosen) > score(rejected)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─────────────────────── 配置 ───────────────────────
BASE_MODEL_PATH = "/home/Disk2/lvxinda/models"
OUTPUT_DIR      = "./output/reward_model"
MAX_LENGTH      = 256
NUM_EPOCHS      = 3
BATCH_SIZE      = 4
LR              = 1e-5
DATA_LIMIT      = 2000
LOG_STEPS       = 50
# ────────────────────────────────────────────────────


# ── 1. 数据集 ────────────────────────────────────────
def load_dataset(tokenizer):
    from modelscope.msdatasets import MsDataset
    print(f"[数据] 加载 iic/CValues-Comparison（前 {DATA_LIMIT} 条）...")
    ds = MsDataset.load("iic/CValues-Comparison", split="train")
    records = []
    for item in ds:
        records.append({
            "prompt":   item["prompt"],
            "chosen":   item["pos_resp"],
            "rejected": item["neg_resp"],
        })
        if len(records) >= DATA_LIMIT:
            break
    print(f"[数据] 加载完成：{len(records)} 条偏好对")
    return PreferenceDataset(records, tokenizer, MAX_LENGTH)


class PreferenceDataset(Dataset):
    def __init__(self, records, tokenizer, max_len):
        self.data      = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def tokenize(self, prompt, response):
        text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        enc = self.tokenizer(
            text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        c_ids, c_mask = self.tokenize(item["prompt"], item["chosen"])
        r_ids, r_mask = self.tokenize(item["prompt"], item["rejected"])
        return c_ids, c_mask, r_ids, r_mask


# ── 2. 模型 ──────────────────────────────────────────
class RewardModel(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            base_model_path, trust_remote_code=True,
            dtype=torch.bfloat16, device_map=None,
        )
        hidden_size = self.model.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state[:, -1, :]
        return self.score_head(last_hidden).squeeze(-1)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = RewardModel(BASE_MODEL_PATH)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数：{total/1e6:.1f}M，可训练：{trainable/1e6:.1f}M")
    return model, tokenizer


# ── 3. 训练 ──────────────────────────────────────────
def preference_loss(chosen_scores, rejected_scores):
    return -torch.log(torch.sigmoid(chosen_scores - rejected_scores)).mean()


def train(model, dataset, device):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler  = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))

    print("\n[训练] 开始训练奖励模型...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct = 0.0, 0
        total_steps = len(dataloader)

        for step, (c_ids, c_mask, r_ids, r_mask) in enumerate(dataloader):
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            r_ids, r_mask = r_ids.to(device), r_mask.to(device)

            chosen_scores   = model(c_ids, c_mask)
            rejected_scores = model(r_ids, r_mask)
            loss = preference_loss(chosen_scores, rejected_scores)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct    += (chosen_scores > rejected_scores).sum().item()

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f"  [Step {step+1}] loss={loss.item():.4f}  ← NaN/Inf，终止训练")
                return

            if (step + 1) % LOG_STEPS == 0 or (step + 1) == total_steps:
                avg_loss = total_loss / (step + 1)
                acc      = correct / ((step + 1) * BATCH_SIZE)
                lr_now   = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Step {step+1}/{total_steps} | "
                    f"loss={avg_loss:.4f} | "
                    f"grad_norm={grad_norm:.4f} | "
                    f"acc={acc:.2%} | "
                    f"lr={lr_now:.2e}"
                )

        epoch_loss = total_loss / len(dataloader)
        epoch_acc  = correct / len(dataset)
        print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f} | 偏好准确率={epoch_acc:.2%}")


# ── 4. 保存 & 评估 ────────────────────────────────────
def save_and_evaluate(model, tokenizer, dataset, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/reward_model.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[完成] 奖励模型已保存至 {OUTPUT_DIR}")
    print("  下一步：运行 step3_GRPO.py")

    model.eval()
    test_prompt = "如何看待网络暴力？"
    good = "网络暴力是一种严重的社会问题，会对受害者造成心理伤害。我们应该倡导文明上网，对他人保持尊重，遇到网络暴力行为应及时举报。"
    bad  = "网络暴力很正常，喷人又不犯法，不爽就骂，反正对方也不知道你是谁。"

    def score(response):
        ids, mask = dataset.tokenize(test_prompt, response)
        with torch.no_grad():
            return model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)).item()

    good_score = score(good)
    bad_score  = score(bad)
    print(f"\n[评估] 问题：{test_prompt}")
    print(f"  好回答分数：{good_score:.4f}")
    print(f"  差回答分数：{bad_score:.4f}")
    print(f"  结论：{'正确区分偏好' if good_score > bad_score else '未能区分偏好，建议增加训练轮数'}")


# ── 主流程 ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("步骤二：奖励模型训练 (Reward Model)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    model, tokenizer = load_model_and_tokenizer()
    model = model.to(device)
    dataset = load_dataset(tokenizer)
    train(model, dataset, device)
    save_and_evaluate(model, tokenizer, dataset, device)


if __name__ == "__main__":
    main()
