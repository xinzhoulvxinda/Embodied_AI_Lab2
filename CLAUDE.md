# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Chinese-language RLHF (Reinforcement Learning from Human Feedback) teaching lab** — a step-by-step hands-on experiment for students to implement the full RLHF pipeline fine-tuning LLMs on an agricultural Q&A domain.

- **`scripts0/`** — Complete reference implementations (teacher's solutions) using Qwen3.5-9B-Base + 4-bit quantization
- **`scripts1/`** — Student exercise templates with `TODO` stubs, using Qwen2.5-1.5B-Instruct (lighter, for student GPUs)
- **`data/`** — Two small JSONL datasets (agriculture domain, Chinese)
- **`docs/`** — Lab manual (`RLHF实验指导书.docx`, Chinese)

No build system, test framework, linter, or CI/CD. Run scripts directly: `python <script>.py`.

**Important**: All scripts must be run from **within** `scripts0/` or `scripts1/` — relative paths (`./data/`, `./output/`, `./outputs/`) are relative to those directories.

---

## Running the Experiments

```bash
# Reference implementations (scripts0)
cd scripts0
python before_sft_eval.py          # Optional: baseline before SFT
python step1_sft.py                # Step 1: SFT with LoRA
python step2_reward_model.py       # Step 2: Train reward model
python step3_ppo.py                # Step 3: PPO RL
python step4_evaluate.py           # Step 4: Evaluate & compare
python evaluate.py                 # SFT vs PPO comparison report

# Student exercises (scripts1) — fill in TODO stubs first
cd scripts1
python step1_sft.py                # TODO[1-1,1-2,1-3]
python step2_reward.py             # TODO[2-1,2-2,2-3]
python step3_rlhf.py               # TODO[3-1,3-2,3-3]
python step4_evalute.py            # TODO[4-1,4-2,4-3]  (typo in filename is intentional)
```

**Output directories differ**: scripts0 → `./output/`; scripts1 → `./outputs/` (don't mix them).

---

## Environment Setup

```bash
bash scripts0/setup_env.sh
# Or manually:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.0 datasets peft accelerate bitsandbytes trl modelscope
pip install numpy matplotlib jupyter ipywidgets
```

- Python 3.13.x, NVIDIA GPU with CUDA required
- scripts0 expects Qwen3.5-9B-Base at `/home/Disk2/lvxinda/models/Qwen3.5-9B-Base` (downloaded via ModelScope `snapshot_download`)
- scripts1 loads `Qwen/Qwen2.5-1.5B-Instruct` directly from HuggingFace/ModelScope cache
- If CUDA OOM during SFT, reduce `BATCH_SIZE` from 4 to 2

---

## RLHF Pipeline Architecture

### Step 0 (optional): Baseline
`before_sft_eval.py` — records base model responses before any training for comparison. Output: `./output/before_sft_responses.json`

### Step 1: Supervised Fine-Tuning (SFT)
- **Data**: `agriculture_sft.jsonl` — `{instruction, input, output}` triples (15 samples); scripts0 additionally fetches `AI-ModelScope/alpaca-gpt4-data-zh` (52K samples)
- **Technique**: QLoRA — 4-bit NF4 quantization + LoRA adapters
- **Output**: `./output[s]/sft_model/`

### Step 2: Reward Model Training
- **Data**: `agriculture_reward.jsonl` — `{prompt, chosen, rejected}` preference pairs (12 samples)
- **scripts0**: Custom `RewardModel` class (base LM + 2-layer scoring head); Bradley-Terry loss: `L = -log σ(r_chosen − r_rejected)`
- **scripts1**: `AutoModelForSequenceClassification(num_labels=1)` via TRL `RewardTrainer`
- **Output**: `./output[s]/reward_model/`

### Step 3: PPO Reinforcement Learning
- **Algorithm**: TRL `PPOTrainer` with Actor (policy), Critic (value head), Reward Model, Reference Model
- **KL penalty**: `init_kl_coef=0.1`, `target_kl=6.0`, adaptive — prevents policy from drifting far from SFT reference
- **scripts1 alternative**: Commented-out DPO instructions using `trl.DPOTrainer`
- **Note**: `scripts0/step3_ppo.py` does `from step2_reward_model import RewardModel` — must run from within `scripts0/`
- **Output**: `./output[s]/ppo_model/`

### Step 4: Evaluation
- Loads Base → SFT → PPO-RLHF models, generates responses to 5 test questions, scores with reward model
- Expected reward trend: Base < SFT < PPO-RLHF
- Output: `./output[s]/evaluation_results.json`

---

## Key Implementation Details

### Chat Templating
- **scripts0** (Base model, no built-in template): Manually constructs ChatML strings:
  ```
  <|im_start|>user
  {question}<|im_end|>
  <|im_start|>assistant
  {response}<|im_end|>
  ```
- **scripts1** (Instruct model): Uses `tokenizer.apply_chat_template()` with system/user/assistant dicts

### System Prompt (scripts1, consistent across all steps)
```
你是一位专业的农业技术顾问，请用专业、准确、详细的语言回答农业相关问题。
```

### QLoRA Configuration
```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
```
- scripts0 LoRA: `r=16`, `lora_alpha=32`, targets all 7 projection layers (`q/k/v/o_proj`, `gate/up/down_proj`)
- scripts1 LoRA: `r=8`, `lora_alpha=16`, minimum `q_proj` + `v_proj`; FFN layers optional

### Inference
- Evaluation uses greedy decoding (`do_sample=False`) for reproducibility
- PPO training uses sampling (`do_sample=True, temperature=0.7`)
- In scripts1 PPO: `ref_model=None` — TRL auto-uses frozen base weights as reference when using PEFT

### Reward Scoring Format (scripts1)
```
System: {system_msg}
User: {question}
Assistant: {answer}
```

---

## Student TODO Reference

### `scripts1/step1_sft.py`
| TODO | Implementation |
|---|---|
| `[1-1]` `format_chat_prompt()` | Build `messages` list (system/user/assistant); call `tokenizer.apply_chat_template(tokenize=False, add_generation_prompt=False)`; return `{"text": ...}` |
| `[1-2]` `target_modules` | e.g. `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| `[1-3]` `SFTConfig` params | `fp16=True`, `max_seq_length=MAX_SEQ_LEN`, `dataset_text_field="text"` |

### `scripts1/step2_reward.py`
| TODO | Implementation |
|---|---|
| `[2-1]` `load_reward_dataset()` | Build chosen/rejected message lists; apply_chat_template; tokenize; return `Dataset.from_list(records)` with `input_ids_chosen/rejected` + `attention_mask_chosen/rejected` |
| `[2-2]` `LoraConfig task_type` | `TaskType.SEQ_CLS` |
| `[2-3]` `RewardConfig` params | `evaluation_strategy="epoch"`, `max_length=MAX_SEQ_LEN` |

### `scripts1/step3_rlhf.py`
| TODO | Implementation |
|---|---|
| `[3-1]` `load_queries()` | system+user messages (no assistant); `apply_chat_template(add_generation_prompt=True)`; return `[{"query": ..., "prompt": ...}]` |
| `[3-2]` `PPOConfig` params | `kl_penalty="kl"`, `init_kl_coef=KL_COEFF`, `target_kl=6.0`, `adap_kl_ctrl=True` |
| `[3-3]` PPO training loop | `ppo_trainer.generate(..., return_prompt=False)`; decode responses; score with `reward_model(**enc).logits[0].item()`; `ppo_trainer.step(query_tensors, response_tensors, rewards)` |

### `scripts1/step4_evalute.py`
| TODO | Implementation |
|---|---|
| `[4-1]` `generate_response()` | system+user messages; `apply_chat_template(add_generation_prompt=True)`; `model.generate`; decode only new tokens |
| `[4-2]` `compute_reward_score()` | Format as reward scoring format above; tokenize; `reward_model(**enc).logits[0].item()` |
| `[4-3]` Statistics | Collect reward scores per model; compute and print averages |
