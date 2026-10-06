"""Discriminative PRM baseline (DiscPRM).

Per-step binary classifier: input = problem + solution prefix up to step i,
label = gold label of step i. Standard cross-entropy, same training data as the
generative models (all step labels, not rejection-sampled) so the comparison
isolates the *paradigm* (generative vs discriminative), not the data.

Cheap: fits on a Kaggle T4.
"""
import argparse
import json
import yaml

import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def build_examples(paths, tokenizer, max_length=1024):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                for s in d["steps"]:
                    for c in s["candidates"]:
                        rows.append({"problem": d["problem"], "step": c["text"],
                                     "label": c["gold"]})
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

    ds = build_examples(cfg["data"]["train_paths"], tok)
    print(f"{len(ds)} step-classification examples")

    tcfg = cfg["training"]
    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=tcfg["num_train_epochs"],
        bf16=True,
        logging_steps=10, save_steps=200,
        report_to=tcfg.get("report_to", "none"),
        seed=cfg.get("seed", 0),
    )
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=DataCollatorWithPadding(tokenizer=tok)).train()
    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
