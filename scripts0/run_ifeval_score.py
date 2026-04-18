"""
run_ifeval_score.py

直接从 after_sft_ifeval_full.json 读取 prompt / response / kwargs，
调用官方 evaluator 输出 strict / loose 评分。

用法：
  python run_ifeval_score.py
  python run_ifeval_score.py --full_json ./output/ifeval/after_sft_ifeval_full.json
"""

import sys
import json
import argparse
import dataclasses
from typing import Dict, Optional, Union

IFEVAL_REPO = "/home/Disk2/lvxinda/google-research"
FULL_JSON   = "./output/ifeval/after_sft_ifeval_full.json"

sys.path.insert(0, IFEVAL_REPO)
from instruction_following_eval import instructions_registry


# ── 复用官方 dataclass，不依赖 evaluation_lib.read_prompt_list ──────────
@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list
    prompt: str
    kwargs: list


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list


def _build_kwargs(raw: dict) -> dict:
    """过滤掉 full.json 里所有值为 None 的 kwargs，避免传入不接受该参数的 checker。"""
    return {k: v for k, v in raw.items() if v is not None}


def _eval_one_strict(inp: InputExample, response: str) -> OutputExample:
    is_following_list = []
    for idx, iid in enumerate(inp.instruction_id_list):
        cls = instructions_registry.INSTRUCTION_DICT[iid]
        inst = cls(iid)
        inst.build_description(**_build_kwargs(inp.kwargs[idx]))
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=inp.prompt)
        is_following_list.append(bool(response.strip() and inst.check_following(response)))
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def _eval_one_loose(inp: InputExample, response: str) -> OutputExample:
    r = response.split("\n")
    variants = [response]
    variants.append("\n".join(r[1:]).strip())
    variants.append("\n".join(r[:-1]).strip())
    variants.append("\n".join(r[1:-1]).strip())
    variants += [v.replace("*", "") for v in variants[:]]

    is_following_list = []
    for idx, iid in enumerate(inp.instruction_id_list):
        cls = instructions_registry.INSTRUCTION_DICT[iid]
        inst = cls(iid)
        inst.build_description(**_build_kwargs(inp.kwargs[idx]))
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=inp.prompt)
        followed = any(v.strip() and inst.check_following(v) for v in variants)
        is_following_list.append(followed)
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def print_report(outputs, label: str):
    import collections
    prompt_total = prompt_correct = instr_total = instr_correct = 0
    tier_total: Dict[str, int] = collections.defaultdict(int)
    tier_correct: Dict[str, int] = collections.defaultdict(int)

    for ex in outputs:
        prompt_total += 1
        if all(ex.follow_instruction_list):
            prompt_correct += 1
        instr_total += len(ex.instruction_id_list)
        instr_correct += sum(ex.follow_instruction_list)
        for iid, ok in zip(ex.instruction_id_list, ex.follow_instruction_list):
            tier_total[iid] += 1
            if ok:
                tier_correct[iid] += 1

    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  prompt-level    : {prompt_correct}/{prompt_total} = {prompt_correct/prompt_total:.4f}")
    print(f"  instruction-level: {instr_correct}/{instr_total} = {instr_correct/instr_total:.4f}")
    print()
    for iid in sorted(tier_total):
        acc = tier_correct[iid] / tier_total[iid]
        print(f"  {iid:<55} {acc:.4f}  ({tier_correct[iid]}/{tier_total[iid]})")
    print('='*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_json", default=FULL_JSON)
    args = parser.parse_args()

    with open(args.full_json, encoding="utf-8") as f:
        records = json.load(f)

    print(f"加载 {len(records)} 条记录：{args.full_json}")

    strict_outputs, loose_outputs = [], []
    for rec in records:
        inp = InputExample(
            key=rec["key"],
            instruction_id_list=rec["instruction_id_list"],
            prompt=rec["prompt"],
            kwargs=rec["kwargs"],
        )
        response = rec["response_after_sft"]
        strict_outputs.append(_eval_one_strict(inp, response))
        loose_outputs.append(_eval_one_loose(inp, response))

    print_report(strict_outputs, "Strict")
    print_report(loose_outputs,  "Loose")


if __name__ == "__main__":
    main()
