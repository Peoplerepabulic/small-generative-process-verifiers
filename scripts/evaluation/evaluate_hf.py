"""Evaluate a generative verifier with plain transformers (no vLLM).

Same metrics as evaluate.py (step/solution/localization accuracy, ECE, Brier)
but uses AutoModelForCausalLM.generate, so it runs anywhere -- e.g. Kaggle T4
where vLLM is unavailable. Slower; use --max_n 200 for a quick read.

Usage:
  python scripts/evaluation/evaluate_hf.py --model outputs/sft-1p5b \
      --data data/prm800k_unified.jsonl --split phase1_test --max_n 200
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from rewards import build_verifier_prompt, parse_verification

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_test(path, split, max_n=None):
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["split"] != split:
                continue
            step_texts, gold = [], []
            for s in d["steps"]:
                cands = s["candidates"]
                ci = s["chosen_idx"] if s["chosen_idx"] is not None else 0
                step_texts.append(cands[ci]["text"])
                gold.append(cands[ci]["gold"])
            items.append((d["problem"], step_texts, gold))
            if max_n and len(items) >= max_n:
                break
    return items


def ece_score(confs, labels, n_bins=10):
    confs = np.asarray(confs); labels = np.asarray(labels)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum():
            ece += m.mean() * abs(confs[m].mean() - labels[m].mean())
    return ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="phase1_test")
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()

    items = load_test(args.data, args.split, args.max_n)
    step_ok = step_tot = sol_ok = sol_tot = loc_ok = loc_tot = parsed = 0
    confs, labels = [], []
    for prob, texts, gold in items:
        prompt = build_verifier_prompt(prob, texts)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id or tok.pad_token_id)
        text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        p = parse_verification(text)
        if p is None:
            continue
        parsed += 1
        for i, g in enumerate(gold, start=1):
            step_tot += 1
            step_ok += (p["judgments"].get(i) == g)
        gold_final = 1 if all(g == 1 for g in gold) else 0
        sol_tot += 1
        sol_ok += (p["final"] == gold_final)
        confs.append(p["confidence"]); labels.append(gold_final)
        gold_err = next((i + 1 for i, g in enumerate(gold) if g == 0), None)
        if gold_err is not None:
            loc_tot += 1
            loc_ok += (p["first_error"] == gold_err)

    confs = np.array(confs); labels = np.array(labels)
    print(f"parsed: {parsed}/{len(items)}")
    print(f"step accuracy:      {step_ok / max(step_tot, 1):.3f}")
    print(f"solution accuracy:  {sol_ok / max(sol_tot, 1):.3f}")
    print(f"localization acc:   {loc_ok / max(loc_tot, 1):.3f}  (n={loc_tot})")
    print(f"ECE (10 bins):      {ece_score(confs, labels):.4f}")
    print(f"Brier score:        {np.mean((confs - labels) ** 2):.4f}")


if __name__ == "__main__":
    main()
