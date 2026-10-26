"""Evaluate a trained DiscPRM classifier: accuracy + calibration.

The DiscPRM is a per-step binary classifier (AutoModelForSequenceClassification,
input = problem + "\\n" + step text, label = gold step label). This script
reports the same metric set as the generative evaluators so the baselines are
directly comparable:
  - per-step judgment accuracy (+ step-level ECE/Brier)
  - solution-level accuracy (solution correct iff all steps predicted correct)
  - first-error-step localization accuracy (on solutions containing an error)
  - solution-level ECE (10 bins) and Brier, with solution confidence = min
    over steps of P(step correct)

Usage:
  python scripts/evaluation/evaluate_discprm.py --model <path-to-merged-model> \
      --data data/prm800k_unified.jsonl --split phase1_test --max_n 200
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)


def load_test(path, split, max_n=None):
    """(problem, [(step_text, gold), ...]) per problem, chosen candidates."""
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["split"] != split:
                continue
            steps = []
            for s in d["steps"]:
                cands = s["candidates"]
                ci = s["chosen_idx"] if s["chosen_idx"] is not None else 0
                steps.append((cands[ci]["text"], cands[ci]["gold"]))
            items.append((d["problem"], steps))
            if max_n and len(items) >= max_n:
                break
    return items


def ece_score(confs, labels, n_bins=10):
    confs = np.asarray(confs)
    labels = np.asarray(labels)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum():
            ece += (m.sum() / len(confs)) * abs(confs[m].mean() - labels[m].mean())
    return ece


class StepDataset(Dataset):
    def __init__(self, items, tokenizer, max_length=1024):
        self.rows = []
        for pi, (prob, steps) in enumerate(items):
            for si, (text, gold) in enumerate(steps):
                self.rows.append((pi, si, prob, text, gold))
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pi, si, prob, text, gold = self.rows[i]
        t = self.tok(prob + "\n" + text, truncation=True,
                     max_length=self.max_length)
        t["label"] = gold
        t["prob_idx"] = pi
        t["step_idx"] = si
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="phase1_test")
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to(device).eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    items = load_test(args.data, args.split, args.max_n)
    ds = StepDataset(items, tok)
    collator = DataCollatorWithPadding(tokenizer=tok)
    # keep prob_idx/step_idx/label out of the model inputs
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)

    # per (problem, step) -> (p_correct, pred)
    probs = {}
    with torch.no_grad():
        for batch in loader:
            # transformers>=4.4x DataCollatorWithPadding renames "label"->"labels"
            labels = batch.pop("labels") if "labels" in batch else batch.pop("label")
            pis = batch.pop("prob_idx")
            sis = batch.pop("step_idx")
            batch = {k: v.to(device) for k, v in batch.items()
                     if torch.is_tensor(v)}
            logits = model(**batch).logits.float()
            p1 = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            pred = (p1 >= 0.5).astype(int)
            for pi, si, pp, pr, g in zip(pis, sis, p1, pred,
                                        labels.numpy()):
                probs[(int(pi), int(si))] = (float(pp), int(pr), int(g))

    step_ok = step_tot = 0
    step_confs, step_labels = [], []
    sol_ok = sol_tot = 0
    sol_confs, sol_labels = [], []
    loc_ok = loc_tot = 0
    for pi, (prob, steps) in enumerate(items):
        preds, p1s, golds = [], [], []
        for si, (text, gold) in enumerate(steps):
            pp, pr, g = probs[(pi, si)]
            preds.append(pr)
            p1s.append(pp)
            golds.append(g)
            step_tot += 1
            step_ok += (pr == g)
            step_confs.append(pp)
            step_labels.append(g)
        gold_final = 1 if all(g == 1 for g in golds) else 0
        pred_final = 1 if all(p == 1 for p in preds) else 0
        sol_tot += 1
        sol_ok += (pred_final == gold_final)
        # solution confidence = weakest step (a solution is only as
        # trustworthy as its least-confident step)
        sol_confs.append(min(p1s) if pred_final == 1
                         else 1.0 - max(p1s))
        sol_labels.append(gold_final)
        gold_err = next((i + 1 for i, g in enumerate(golds) if g == 0), None)
        if gold_err is not None:
            loc_tot += 1
            pred_err = next((i + 1 for i, p in enumerate(preds) if p == 0),
                            None)
            loc_ok += (pred_err == gold_err)

    step_confs = np.array(step_confs)
    step_labels = np.array(step_labels)
    sol_confs = np.array(sol_confs)
    sol_labels = np.array(sol_labels)
    print(f"problems: {len(items)}, steps: {step_tot}")
    print(f"step accuracy:      {step_ok / max(step_tot, 1):.3f}")
    print(f"step ECE (10 bins): {ece_score(step_confs, step_labels):.4f}")
    print(f"step Brier:         {np.mean((step_confs - step_labels) ** 2):.4f}")
    print(f"solution accuracy:  {sol_ok / max(sol_tot, 1):.3f}")
    print(f"solution ECE:       {ece_score(sol_confs, sol_labels):.4f}")
    print(f"solution Brier:     {np.mean((sol_confs - sol_labels) ** 2):.4f}")
    print(f"localization acc:   {loc_ok / max(loc_tot, 1):.3f}  (n={loc_tot})")


if __name__ == "__main__":
    main()
