"""
SFT 后评估：在 IFEval 上测试 LoRA 模型的指令跟随能力
用法：python after_sft_eval.py [--limit N] [--verbose] [--run_official_eval --ifeval_repo_dir /path]
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional

import torch
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────── 配置 ───────────────────────
BASE_MODEL_DIR = "/home/Disk2/lvxinda/models"
SFT_MODEL_DIR  = "./output/sft_model"
OUTPUT_DIR     = "./output/ifeval"
MAX_NEW_TOKENS = 1024
CUDA_DEVICE    = "cuda:1"
USE_SYSTEM_PROMPT = False
SYSTEM_PROMPT     = "You are a helpful assistant."
# ────────────────────────────────────────────────────

INPUT_JSONL_FILE    = os.path.join(OUTPUT_DIR, "ifeval_input_data.jsonl")
RESPONSE_JSONL_FILE = os.path.join(OUTPUT_DIR, "after_sft_ifeval_responses.jsonl")
DETAIL_JSON_FILE    = os.path.join(OUTPUT_DIR, "after_sft_ifeval_full.json")
OFFICIAL_EVAL_DIR   = os.path.join(OUTPUT_DIR, "official_eval")


# ── 1. 数据集 ────────────────────────────────────────
def load_dataset(split: str, start_idx: int = 0, limit: Optional[int] = None):
    ds = MsDataset.load("google/IFEval", split=split)
    if hasattr(ds, "to_hf_dataset"):
        ds = ds.to_hf_dataset()
    else:
        from datasets import Dataset
        ds = Dataset.from_list(list(ds))
    total = len(ds)
    if start_idx < 0 or start_idx >= total:
        raise ValueError(f"start_idx={start_idx} 越界，数据总量为 {total}")
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
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, torch_dtype=torch.bfloat16,
        device_map={"": CUDA_DEVICE}, trust_remote_code=True,
    )
    print("[模型] 加载 SFT LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, SFT_MODEL_DIR)
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


def try_run_official_ifeval(ifeval_repo_dir, input_jsonl, response_jsonl, output_dir):
    if not ifeval_repo_dir:
        raise ValueError("启用 --run_official_eval 时，必须提供 --ifeval_repo_dir")
    repo_root = os.path.abspath(ifeval_repo_dir)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"IFEval 仓库目录不存在：{repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from instruction_following_eval import evaluation_lib
    except Exception as e:
        raise ImportError("无法导入 instruction_following_eval.evaluation_lib") from e

    os.makedirs(output_dir, exist_ok=True)
    inputs = evaluation_lib.read_prompt_list(input_jsonl)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(response_jsonl)
    for func, name in [
        (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
        (evaluation_lib.test_instruction_following_loose,  "eval_results_loose"),
    ]:
        outputs = [func(inp, prompt_to_response) for inp in inputs]
        out_path = os.path.join(output_dir, name + ".jsonl")
        evaluation_lib.write_outputs(out_path, outputs)
        print(f"\n[官方评测] {out_path}")
        evaluation_lib.print_report(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",          action="store_true")
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--start_idx",        type=int, default=0)
    parser.add_argument("--max_new_tokens",   type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--run_official_eval", action="store_true")
    parser.add_argument("--ifeval_repo_dir",  type=str, default="")
    args = parser.parse_args()

    print("=" * 72)
    print("IFEval 评测：SFT LoRA 模型")
    print(f"  Base: {BASE_MODEL_DIR}  SFT: {SFT_MODEL_DIR}")
    print("=" * 72)

    print("\n[数据] 加载 IFEval 数据集...")
    dataset = load_dataset("train", args.start_idx, args.limit)
    print(f"  样本数：{len(dataset)}")

    model, tokenizer = load_model_and_tokenizer()

    detailed_results, input_rows, response_rows = [], [], []
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

        input_rows.append({
            "key": key, "prompt": prompt,
            "instruction_id_list": sample["instruction_id_list"],
            "kwargs": sample["kwargs"],
        })
        response_rows.append({"prompt": prompt, "response": response})
        detailed_results.append({
            "index": idx, "key": key, "prompt": prompt,
            "instruction_id_list": sample["instruction_id_list"],
            "kwargs": sample["kwargs"],
            "response_after_sft": response,
            "response_char_len": len(response),
            "elapsed_sec": round(elapsed, 2),
        })
        if args.verbose:
            print(f"  {response[:300]}{'...' if len(response)>300 else ''}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_jsonl(INPUT_JSONL_FILE, input_rows)
    write_jsonl(RESPONSE_JSONL_FILE, response_rows)
    with open(DETAIL_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    n = len(detailed_results)
    print(f"\n[完成] input_data -> {INPUT_JSONL_FILE}")
    print(f"       responses  -> {RESPONSE_JSONL_FILE}")
    print(f"       detail     -> {DETAIL_JSON_FILE}")
    print(f"  平均耗时 {total_elapsed/n:.2f}s/条  平均长度 {total_chars/n:.0f} 字符")

    if args.run_official_eval:
        try:
            try_run_official_ifeval(
                args.ifeval_repo_dir, INPUT_JSONL_FILE, RESPONSE_JSONL_FILE, OFFICIAL_EVAL_DIR,
            )
        except Exception as e:
            print(f"\n[警告] 官方评测失败：{e}")


if __name__ == "__main__":
    main()
