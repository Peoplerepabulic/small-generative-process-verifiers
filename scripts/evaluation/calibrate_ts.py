"""Temperature scaling (post-hoc calibration) for the generative verifier.

The verifier emits a scalar "Confidence" for its Final verdict. Protocol
( honest, no fit/eval overlap ): repeated random split-half on the test set --
fit the single temperature T on one half by minimizing NLL of
sigmoid(logit(p) / T) against gold_final, evaluate ECE/Brier with the scaled
confidences on the held-out half, average over many splits.

Also dumps per-problem (confidence, gold_final, model_final) to --dump for
the paper's records. Greedy decoding, same as evaluate_hf.py.

Usage:
  python scripts/evaluation/calibrate_ts.py --model /root/grpo-1p5b-v3-merged \
      --data data/prm800k_phase1_test.jsonl --split phase1_test \
      --dump /workspace/ts_dump_grpo_v3.jsonl
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
    confs = np.asarray(confs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum():
            ece += m.mean() * abs(confs[m].mean() - labels[m].mean())
    return float(ece)


def brier(confs, labels):
    return float(np.mean((np.asarray(confs) - np.asarray(labels)) ** 2))


def fit_temperature(confs, labels):
    """Grid-search T minimizing NLL of sigmoid(logit(p)/T) vs labels."""
    p = np.clip(np.asarray(confs, dtype=float), 1e-3, 1 - 1e-3)
    y = np.asarray(labels, dtype=float)
    z = np.log(p / (1 - p))
    grid = np.concatenate([np.linspace(0.05, 1.0, 39),
                           np.linspace(1.05, 5.0, 80),
                           np.linspace(5.25, 10.0, 20)])
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        q = 1.0 / (1.0 + np.exp(-z / t))
        q = np.clip(q, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def apply_temperature(confs, t):
    p = np.clip(np.asarray(confs, dtype=float), 1e-3, 1 - 1e-3)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-z / t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="phase1_test")
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    items = load_test(args.data, args.split, args.max_n)
    rows = []
    for prob, texts, gold in items:
        prompt = build_verifier_prompt(prob, texts)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id or tok.pad_token_id)
        text = tok.decode(out[0][inp.input_ids.shape[1]:],
                          skip_special_tokens=True)
        p = parse_verification(text)
        if p is None:
            continue
        gold_final = 1 if all(g == 1 for g in gold) else 0
        rows.append({"confidence": p["confidence"], "gold_final": gold_final,
                     "model_final": p["final"]})
    print(f"parsed: {len(rows)}/{len(items)}")

    if args.dump:
        with open(args.dump, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"dumped -> {args.dump}")

    confs = np.array([r["confidence"] for r in rows])
    labels = np.array([r["gold_final"] for r in rows], dtype=float)
    uniq, counts = np.unique(confs, return_counts=True)
    print("confidence distribution:",
          {float(u): int(c) for u, c in zip(uniq, counts)})
    print(f"RAW   ECE: {ece_score(confs, labels):.4f}   "
          f"Brier: {brier(confs, labels):.4f}")

    rng = np.random.RandomState(args.seed)
    n = len(rows)
    ts, eces, briers = [], [], []
    for _ in range(args.splits):
        idx = rng.permutation(n)
        fit_i, eval_i = idx[: n // 2], idx[n // 2:]
        t = fit_temperature(confs[fit_i], labels[fit_i])
        cal = apply_temperature(confs[eval_i], t)
        ts.append(t)
        eces.append(ece_score(cal, labels[eval_i]))
        briers.append(brier(cal, labels[eval_i]))
    ts, eces, briers = map(np.array, (ts, eces, briers))
    print(f"TS (split-half x{args.splits}): T = {ts.mean():.2f} +- {ts.std():.2f}")
    print(f"TS held-out ECE:   {eces.mean():.4f} +- {eces.std():.4f}")
    print(f"TS held-out Brier: {briers.mean():.4f} +- {briers.std():.4f}")
    print("TS_SUMMARY " + json.dumps({
        "n": n, "raw_ece": ece_score(confs, labels),
        "raw_brier": brier(confs, labels),
        "T_mean": float(ts.mean()), "T_std": float(ts.std()),
        "ts_ece_mean": float(eces.mean()), "ts_ece_std": float(eces.std()),
        "ts_brier_mean": float(briers.mean()),
        "ts_brier_std": float(briers.std())}))


if __name__ == "__main__":
    main()
