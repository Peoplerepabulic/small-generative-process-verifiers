"""SFT baseline: the rejection-sampling proxy (ThinkPRM-style).

Trains the verifier to *imitate* gold verifications, using only
agreement-filtered data:
- PRM800K: chosen completions only (all steps rated positive) -> the model only
  ever sees "correct" judgments, exactly like rejection sampling which discards
  disagreeing chains.
- Math-Shepherd: only solutions whose every step is labeled "+".

Target completion is the gold verification in the exact format from rewards.py.
Compare against GRPO (train_grpo.py), which trains on ALL chains including
negatives, with an explicit calibration reward.

T4-friendly: LoRA (r=16) + gradient checkpointing (same rationale as
train_discprm.py -- full-FT AdamW states OOM on a 14.5GB T4). Adapters are
merged into the base model before saving, so evaluate_hf.py sees a plain
full model.
"""
import argparse
import json
import random
import yaml

from datasets import Dataset
from rewards import build_verifier_prompt


def gold_completion(gold):
    lines = [f"Step {i + 1}: {'correct' if g == 1 else 'incorrect'}"
             for i, g in enumerate(gold)]
    first_err = next((str(i + 1) for i, g in enumerate(gold) if g == 0), "none")
    final = "correct" if all(g == 1 for g in gold) else "incorrect"
    # SFT has no learned confidence; use the empirical label as the target
    # (temperature scaling on top of this is baseline #3 in the proposal).
    conf = "0.95" if final == "correct" else "0.05"
    lines += [f"First error step: {first_err}", f"Final: {final}", f"Confidence: {conf}"]
    return "\n".join(lines)


def build_examples(paths, agreement_only=True, max_examples=None, seed=0):
    examples = []
    for path in paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                step_texts, gold = [], []
                for s in d["steps"]:
                    cands = s["candidates"]
                    ci = s["chosen_idx"] if s["chosen_idx"] is not None else 0
                    step_texts.append(cands[ci]["text"])
                    gold.append(cands[ci]["gold"])
                if agreement_only and not all(g == 1 for g in gold):
                    continue  # rejection sampling: discard disagreeing chains
                prompt = build_verifier_prompt(d["problem"], step_texts)
                examples.append({"prompt": prompt, "completion": gold_completion(gold)})
    if max_examples is not None and len(examples) > max_examples:
        examples = random.Random(seed).sample(examples, max_examples)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_1p5b.yaml")
    ap.add_argument("--max_examples", type=int, default=None,
                    help="pilot cap on SFT examples (overrides config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer

    max_examples = args.max_examples
    if max_examples is None:
        max_examples = cfg["training"].get("max_examples")
    examples = build_examples(cfg["data"]["train_paths"],
                              max_examples=max_examples,
                              seed=cfg.get("seed", 0))
    ds = Dataset.from_list(examples)
    print(f"{len(ds)} SFT examples (agreement-filtered)")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    tcfg = cfg["training"]
    # T4 has fp16 tensor cores but no bf16 tensor cores: fp16 is 2-3x faster
    # than bf16 on T4. Use bf16 on Ampere and newer.
    precision = tcfg.get("precision", "bf16")
    sfcfg = SFTConfig(
        output_dir=cfg["output_dir"],
        model_name_or_path=cfg["model"],
        max_length=tcfg["max_length"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=tcfg["num_train_epochs"],
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        gradient_checkpointing=True,
        logging_steps=tcfg.get("logging_steps", 10),
        save_steps=tcfg.get("save_steps", 200),
        report_to=tcfg.get("report_to", "none"),
        seed=cfg.get("seed", 0),
    )

    def formatting_fn(ex):
        return [p + "\n" + c for p, c in zip(ex["prompt"], ex["completion"])]

    trainer = SFTTrainer(model=cfg["model"], args=sfcfg, train_dataset=ds,
                         formatting_func=formatting_fn, peft_config=lora_cfg)
    trainer.train()
    # Merge adapters so outputs/<name> is a plain full model for evaluate_hf.py.
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(cfg["output_dir"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
