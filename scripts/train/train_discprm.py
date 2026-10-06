"""Discriminative PRM baseline (DiscPRM).

Per-step binary classifier: input = problem + solution prefix up to step i,
label = gold label of step i. Standard cross-entropy, same training data as the
generative models (all step labels, not rejection-sampled) so the comparison
isolates the *paradigm* (generative vs discriminative), not the data.

T4-friendly: LoRA (r=16) + gradient checkpointing. Full fine-tuning with AdamW
needs ~12GB for fp32 optimizer states alone on 1.5B params, which OOMs on a
14.5GB T4 (no batch-size tweak fixes that). LoRA keeps the comparison
apples-to-apples across the DiscPRM/SFT/GRPO baselines; adapters are merged
into the base model before saving, so downstream code sees a plain full model.
"""
import argparse
import json
import random
import yaml

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def build_examples(paths, tokenizer, max_length=1024, max_examples=None,
                   seed=0):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                for s in d["steps"]:
                    for c in s["candidates"]:
                        rows.append({"problem": d["problem"], "step": c["text"],
                                     "label": c["gold"]})
    if max_examples is not None and len(rows) > max_examples:
        rows = random.Random(seed).sample(rows, max_examples)
    ds = Dataset.from_list(rows)

    def tok(ex):
        t = tokenizer(ex["problem"] + "\n" + ex["step"],
                      truncation=True, max_length=max_length)
        t["labels"] = ex["label"]
        return t

    return ds.map(tok, batched=False, remove_columns=["problem", "step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/discprm_1p5b.yaml")
    ap.add_argument("--max_examples", type=int, default=None,
                    help="pilot cap on step examples (overrides config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=2, trust_remote_code=True,
        torch_dtype=torch.bfloat16)
    # Qwen2 checkpoints define no pad token: the tokenizer fallback above is
    # not enough -- the model's config must also carry pad_token_id, otherwise
    # forward() raises "Cannot handle batch sizes > 1 if no padding token is
    # defined." on batched input.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    # LoRA + gradient checkpointing (see module docstring for why).
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],  # classification head trains fully
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    max_examples = args.max_examples
    if max_examples is None:
        max_examples = cfg["training"].get("max_examples")
    ds = build_examples(cfg["data"]["train_paths"], tok,
                        max_examples=max_examples, seed=cfg.get("seed", 0))
    print(f"{len(ds)} step-classification examples")

    tcfg = cfg["training"]
    # T4 has fp16 tensor cores but no bf16 tensor cores: fp16 is 2-3x faster
    # than bf16 on T4. Use bf16 on Ampere and newer.
    precision = tcfg.get("precision", "bf16")
    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=tcfg["num_train_epochs"],
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        logging_steps=10, save_steps=200,
        report_to=tcfg.get("report_to", "none"),
        seed=cfg.get("seed", 0),
    )
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=DataCollatorWithPadding(tokenizer=tok)).train()
    # Merge adapters so outputs/<name> is a plain full model.
    model = model.merge_and_unload()
    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
