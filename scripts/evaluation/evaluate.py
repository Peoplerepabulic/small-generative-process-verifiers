"""Evaluate a generative verifier: accuracy + calibration.

Reads the unified JSONL test split, runs the verifier on each (problem, steps),
parses the output and reports:
  - per-step judgment accuracy
  - solution-level (Final verdict) accuracy
  - first-error-step localization accuracy (on solutions that contain an error)
  - ECE (10 bins) and Brier score on Confidence vs gold Final label

Usage:
  python scripts/evaluation/evaluate.py --model <hf-id-or-path> \
      --data data/prm800k_unified.jsonl --split phase1_test --max_n 200

ProcessBench wiring is a week-2 task: add a loader that maps ProcessBench items
to (problem, steps, gold) and reuse the metrics below.
"""
import argparse
import json

import numpy as np
from rewards import build_verifier_prompt, parse_verification


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
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.0, max_tokens=2048)

    items = load_test(args.data, args.split, args.max_n)
    prompts = [build_verifier_prompt(p, s) for p, s, _ in items]
    outs = llm.generate(prompts, sp)
    texts = [o.outputs[0].text for o in outs]

    step_ok = step_tot = 0
    sol_ok = sol_tot = 0
    loc_ok = loc_tot = 0
    confs, labels = [], []
    parsed = 0
    for text, (_, _, gold) in zip(texts, items):
        p = parse_verification(text)
        if p is None:
            continue
        parsed += 1
        for i, g in enumerate(gold, start=1):
            step_tot += 1
            if p["judgments"].get(i) == g:
                step_ok += 1
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
