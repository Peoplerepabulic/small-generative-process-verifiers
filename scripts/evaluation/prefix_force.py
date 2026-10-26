"""Revision Part B: prefix forcing — the clean causal test of interference.

ThinkPRM's conjecture is about the model's own judgment history. So instead
of conditioning on what the model happened to say at step k (post-treatment,
endogenous), we INTERVENE on it: take the model's own unforced joint
generation, keep its lines before step k, teacher-force line k to
"Step k: correct" or "Step k: incorrect", and let it generate the rest.
Every counterfactual pair receives both treatments (n=84 pairs per model).

Metrics per forcing, on post-k steps: accuracy; drag = false-negative rate
on gold-correct post-k steps. Paired differences (incorrect - correct) with
bootstrap 95% CIs over problems.
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from rewards import build_verifier_prompt, parse_verification
from partB_interference import build_counterfactual_pairs

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_spec = importlib.util.spec_from_file_location(
    "partB_run", os.path.join(os.path.dirname(__file__), "partB_run.py"))
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)
batched_generate = _pb.batched_generate


def boot_ci(vals, n_boot=1000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(vals.mean()),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))],
            "n": int(len(vals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = build_counterfactual_pairs([args.data], max_n=100)
    print(f"{len(pairs)} counterfactual pairs")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    # unforced joint generations (orig variant), batched
    prompts = [build_verifier_prompt(p["problem"], p["variants"]["orig"][0])
               for p in pairs]
    unforced = batched_generate(model, tok, prompts, 1024, 8)

    # forced continuations
    jobs, cont_prompts = [], []
    for pi, (p, gen_text) in enumerate(zip(pairs, unforced)):
        texts, gold = p["variants"]["orig"]
        k = p["k"]
        lines = [l for l in gen_text.split("\n")
                 if l.strip().startswith("Step ")]
        head = lines[:k]  # model's own lines for steps 1..k (0-indexed < k)
        base = prompts[pi] + "\n".join(head) + ("\n" if head else "")
        for forced in ("correct", "incorrect"):
            jobs.append((pi, forced))
            cont_prompts.append(base + f"Step {k + 1}: {forced}\n")
    cont_texts = batched_generate(model, tok, cont_prompts, 1024, 8)
    forced_parsed = {}
    for (pi, forced), ct in zip(jobs, cont_texts):
        p = pairs[pi]
        texts, gold = p["variants"]["orig"]
        k = p["k"]
        lines = [l for l in unforced[pi].split("\n")
                 if l.strip().startswith("Step ")]
        head = lines[:k]
        full = ("\n".join(head) + ("\n" if head else "")
                + f"Step {k + 1}: {forced}\n" + ct)
        forced_parsed[(pi, forced)] = parse_verification(full)

    records = []
    for pi, p in enumerate(pairs):
        texts, gold = p["variants"]["orig"]
        k = p["k"]
        post = list(range(k + 1, len(texts)))
        if not post:
            continue
        row = {"id": p["id"], "k": k, "n_post": len(post)}
        ok = True
        for forced in ("correct", "incorrect"):
            pr = forced_parsed.get((pi, forced))
            if pr is None:
                ok = False
                break
            acc = float(np.mean([pr["judgments"].get(i + 1) == gold[i]
                                 for i in post]))
            gc = [i for i in post if gold[i] == 1]
            drag = (float(np.mean([pr["judgments"].get(i + 1) == 0
                                   for i in gc])) if gc else None)
            row[f"acc_{forced}"] = acc
            row[f"drag_{forced}"] = drag
        if ok:
            records.append(row)

    diffs_acc = [r["acc_incorrect"] - r["acc_correct"] for r in records]
    diffs_drag = [r["drag_incorrect"] - r["drag_correct"] for r in records
                  if r["drag_incorrect"] is not None
                  and r["drag_correct"] is not None]
    summary = {
        "n_pairs": len(records),
        "acc_forced_correct": boot_ci([r["acc_correct"] for r in records]),
        "acc_forced_incorrect": boot_ci(
            [r["acc_incorrect"] for r in records]),
        "acc_diff_incorrect_minus_correct": boot_ci(diffs_acc),
        "drag_forced_correct": boot_ci(
            [r["drag_correct"] for r in records
             if r["drag_correct"] is not None]),
        "drag_forced_incorrect": boot_ci(
            [r["drag_incorrect"] for r in records
             if r["drag_incorrect"] is not None]),
        "drag_diff": boot_ci(diffs_drag),
    }
    print("PREFIX_FORCE_SUMMARY " + json.dumps(summary))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
