"""
IFEval 评测：Base Model（无 LoRA）
用法：python base_ifeval_eval.py [--limit N] [--verbose]
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional

import torch
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────── 配置 ───────────────────────
BASE_MODEL_DIR = "/home/Disk2/lvxinda/models"
OUTPUT_DIR     = "./output/ifeval/base"
MAX_NEW_TOKENS = 1024
CUDA_DEVICE    = "cuda:1"
USE_SYSTEM_PROMPT = False
SYSTEM_PROMPT     = "You are a helpful assistant."
# ────────────────────────────────────────────────────

RESPONSE_JSONL = os.path.join(OUTPUT_DIR, "base_ifeval_responses.jsonl")
DETAIL_JSON    = os.path.join(OUTPUT_DIR, "base_ifeval_full.json")
INPUT_JSONL    = "./output/ifeval/ifeval_input_data.jsonl"   # 与 after_sft_eval 共用


# ── 1. 数据集 ────────────────────────────────────────
def load_dataset(split: str, start_idx: int = 0, limit: Optional[int] = None):
    ds = MsDataset.load("google/IFEval", split=split)
    if hasattr(ds, "to_hf_dataset"):
        ds = ds.to_hf_dataset()
    else:
        from datasets import Dataset
        ds = Dataset.from_list(list(ds))
    total = len(ds)
    end_idx = total if limit is None else min(total, start_idx + limit)
    if start_idx != 0 or end_idx != total:
        ds = ds.select(range(start_idx, end_idx))
    return ds


# ── 2. 模型 & 分词器 ─────────────────────────────────
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[模型] 加载 Base Model（bf16）...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, torch_dtype=torch.bfloat16,
        device_map={"": CUDA_DEVICE}, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── 3. 推理 ──────────────────────────────────────────
def generate_response(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    messages = []
    if USE_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt_text})
    try:
        model_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        model_prompt = (
            f"System: {SYSTEM_PROMPT}\nUser: {prompt_text}\nAssistant:"
            if USE_SYSTEM_PROMPT else prompt_text
        )

    model_device = next(model.parameters()).device
    inputs = tokenizer(model_prompt, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── 4. 评估 ──────────────────────────────────────────
def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",        action="store_true")
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument("--start_idx",      type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print("=" * 72)
    print("IFEval 评测：Base Model（无 LoRA）")
    print(f"  模型：{BASE_MODEL_DIR}  输出：{OUTPUT_DIR}")
    print("=" * 72)

    print("\n[数据] 加载 IFEval 数据集...")
    dataset = load_dataset("train", args.start_idx, args.limit)
    print(f"  样本数：{len(dataset)}")

    model, tokenizer = load_model_and_tokenizer()

    detailed_results: List[Dict[str, Any]] = []
    response_rows:    List[Dict[str, Any]] = []
    total_elapsed = total_chars = 0.0

    for idx, sample in enumerate(dataset, 1):
        key    = sample["key"]
        prompt = sample["prompt"]
        print(f"[{idx:>3d}/{len(dataset)}] key={key} | {prompt[:80]}{'...' if len(prompt)>80 else ''}")

        t0 = time.time()
        response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
        elapsed  = time.time() - t0
        total_elapsed += elapsed
        total_chars   += len(response)

        response_rows.append({"prompt": prompt, "response": response})
        detailed_results.append({
            "index": idx, "key": key, "prompt": prompt,
            "instruction_id_list": sample["instruction_id_list"],
            "kwargs": sample["kwargs"],
            "response_after_sft": response,   # 字段名与 run_ifeval_score.py 保持一致
            "response_char_len": len(response),
            "elapsed_sec": round(elapsed, 2),
        })
        if args.verbose:
            print(f"  {response[:300]}{'...' if len(response)>300 else ''}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_jsonl(RESPONSE_JSONL, response_rows)
    with open(DETAIL_JSON, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    n = len(detailed_results)
    print(f"\n[完成] responses -> {RESPONSE_JSONL}")
    print(f"       detail    -> {DETAIL_JSON}")
    print(f"  平均耗时 {total_elapsed/n:.2f}s/条  平均长度 {total_chars/n:.0f} 字符")


if __name__ == "__main__":
    main()
