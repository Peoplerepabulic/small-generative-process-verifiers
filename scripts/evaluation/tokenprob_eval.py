"""Revision: token-probability scoring (ThinkPRM-style P(yes)/(P(yes)+P(no))).

For each test trace, walk the verifier's own judgment sequence step by step.
At step i the context is the standard prompt plus the model's own previous
judgment lines plus "Step i:"; read the next-token logits at the verdict
position and compute p_incorrect = softmax over {" incorrect", " correct"}.
The appended line uses the two-way argmax (the standard scoring protocol).

Dumps per problem {id, gold_steps, p_inc} and prints a summary: step AUROC
of p_inc, step balanced accuracy at 0.5, and noisy-OR solution scoring
P(correct) = prod(1 - p_inc): AUROC, accuracy at 0.5, ECE.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from rewards import build_verifier_prompt

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(confs, labels, n_bins=10):
    confs = np.asarray(confs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    out = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (confs > lo) & (confs <= hi) if b > 0 else (confs >= lo) & (confs <= hi)
        if m.sum() > 0:
            out += m.mean() * abs(confs[m].mean() - labels[m].mean())
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data)]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    id_corr = tok.encode(" correct", add_special_tokens=False)[0]
    id_inc = tok.encode(" incorrect", add_special_tokens=False)[0]
    print(f"verdict token ids: correct={id_corr} incorrect={id_inc}")

    all_rows = []
    for ri, d in enumerate(records):
        texts = [s["candidates"][s["chosen_idx"] if s["chosen_idx"] is not None
                                else 0]["text"] for s in d["steps"]]
        gold = [s["candidates"][s["chosen_idx"] if s["chosen_idx"] is not None
                               else 0]["gold"] for s in d["steps"]]
        context = build_verifier_prompt(d["problem"], texts)
        p_incs = []
        for i in range(len(texts)):
            prefix = context + f"Step {i + 1}:"
            enc = tok(prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits[0, -1]
            pair = torch.softmax(
                torch.stack([logits[id_inc], logits[id_corr]]), dim=0)
            p_inc = float(pair[0])
            p_incs.append(p_inc)
            verdict = "incorrect" if p_inc > 0.5 else "correct"
            context += f"Step {i + 1}: {verdict}\n"
        all_rows.append({"id": d["id"], "gold_steps": gold, "p_inc": p_incs})
        if (ri + 1) % 25 == 0:
            print(f"{ri + 1}/{len(records)} problems scored")

    with open(args.out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    step_scores, step_labels = [], []
    sol_scores, sol_labels = [], []
    for r in all_rows:
        for p, g in zip(r["p_inc"], r["gold_steps"]):
            step_scores.append(p)
            step_labels.append(1 - g)  # positive class = step incorrect
        sol_scores.append(float(np.prod([1 - p for p in r["p_inc"]])))
        sol_labels.append(int(all(g == 1 for g in r["gold_steps"])))
    step_pred = [1 if p > 0.5 else 0 for p in step_scores]
    tp = sum(1 for p, l in zip(step_pred, step_labels) if p == 1 and l == 1)
    fn = sum(1 for p, l in zip(step_pred, step_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(step_pred, step_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(step_pred, step_labels) if p == 1 and l == 0)
    summary = {
        "step_auroc_p_inc": auroc(step_scores, step_labels),
        "step_balanced_acc": float(0.5 * (tp / max(tp + fn, 1)
                                          + tn / max(tn + fp, 1))),
        "step_neg_recall": float(tp / max(tp + fn, 1)),
        "noisyor_sol_auroc": auroc([-s for s in sol_scores],
                                   [1 - l for l in sol_labels]),
        "noisyor_sol_acc": float(np.mean(
            [(s >= 0.5) == bool(l) for s, l in zip(sol_scores, sol_labels)])),
        "noisyor_sol_ece": ece(sol_scores, sol_labels),
    }
    print("TOKENPROB_SUMMARY " + json.dumps(summary))
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
