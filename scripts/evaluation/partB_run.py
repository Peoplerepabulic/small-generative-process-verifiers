"""Part B runner: step-label interference, HF batched generation (no vLLM).

Reuses the counterfactual construction from scripts/train/partB_interference.py
but fixes two scaffold issues: vLLM (unreliable on our pods) is replaced by
batched transformers generation, and the independent-mode outputs are indexed
by the STEP index (the scaffold indexed by enumeration over post-k steps,
which mismatches whenever k > 0).

For each counterfactual pair (same solution; step k swapped to an
opposite-gold candidate from PRM800K's pool), in both variants (orig/flipped):
  JOINT:       one generation judges all steps (standard verifier prompt).
  INDEPENDENT: one call per post-k step; step i is judged from problem +
               steps 1..i only (no later steps, no own earlier judgments).
  gap = acc_independent(post-k) - acc_joint(post-k)   (per record)

Reports: overall gap + bootstrap 95% CI; difference-in-differences
(gap_flipped - gap_orig per pair); and the direct "drag" statistic — in joint
mode, the false-negative rate on post-k GOLD-CORRECT steps, split by whether
the model judged step k itself incorrect.

Usage:
  python scripts/evaluation/partB_run.py --model /root/grpo-1p5b-v3-merged \
      --data data/prm800k_phase1_test.jsonl --max_pairs 100 \
      --out /workspace/partB_results_grpo_v3.json
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from rewards import build_verifier_prompt, parse_verification
from partB_interference import build_counterfactual_pairs, independent_prompts

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def batched_generate(model, tok, prompts, max_new_tokens, batch_size):
    outs = []
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s:s + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        for j in range(len(batch)):
            outs.append(tok.decode(gen[j][plen:], skip_special_tokens=True))
    return outs


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
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max_pairs", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--joint_batch", type=int, default=8)
    ap.add_argument("--ind_batch", type=int, default=16)
    args = ap.parse_args()

    pairs = build_counterfactual_pairs(args.data, max_n=args.max_pairs)
    print(f"{len(pairs)} counterfactual pairs built")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    # ---- JOINT generations (all pair-variants in batches) ----
    jobs = []  # (pair_idx, variant)
    prompts = []
    for pi, p in enumerate(pairs):
        for variant, (texts, gold) in p["variants"].items():
            jobs.append((pi, variant))
            prompts.append(build_verifier_prompt(p["problem"], texts))
    print(f"joint generations: {len(prompts)}")
    joint_texts = batched_generate(model, tok, prompts, 1024, args.joint_batch)
    joint_parsed = {}
    for (pi, variant), text in zip(jobs, joint_texts):
        joint_parsed[(pi, variant)] = parse_verification(text)

    # ---- INDEPENDENT generations (post-k steps only) ----
    ind_jobs = []  # (pair_idx, variant, step_i 0-based)
    ind_prompts = []
    for pi, p in enumerate(pairs):
        for variant, (texts, gold) in p["variants"].items():
            all_ind = independent_prompts(p["problem"], texts)
            for i in range(p["k"] + 1, len(texts)):
                ind_jobs.append((pi, variant, i))
                ind_prompts.append(all_ind[i])
    print(f"independent generations: {len(ind_prompts)}")
    ind_texts = batched_generate(model, tok, ind_prompts, 128, args.ind_batch)
    ind_judgment = {}
    for (pi, variant, i), text in zip(ind_jobs, ind_texts):
        pr = parse_verification(text)
        ind_judgment[(pi, variant, i)] = (
            None if pr is None else pr["judgments"].get(i + 1))

    # ---- Metrics ----
    records = []
    skipped = 0
    for pi, p in enumerate(pairs):
        for variant, (texts, gold) in p["variants"].items():
            post = list(range(p["k"] + 1, len(texts)))
            pj = joint_parsed.get((pi, variant))
            if pj is None or not post:
                skipped += 1
                continue
            acc_joint = sum(pj["judgments"].get(i + 1) == gold[i]
                            for i in post) / len(post)
            ind_vals = [ind_judgment.get((pi, variant, i)) for i in post]
            ind_ok = [v == gold[i] for v, i in zip(ind_vals, post)
                      if v is not None]
            if not ind_ok:
                skipped += 1
                continue
            acc_ind = sum(ind_ok) / len(ind_ok)
            # drag statistic inputs: post-k gold-correct steps in joint mode
            gc_post = [i for i in post if gold[i] == 1]
            drag = (sum(1 for i in gc_post
                        if pj["judgments"].get(i + 1) == 0) / len(gc_post)
                    if gc_post else None)
            records.append({
                "id": p["id"], "k": p["k"], "variant": variant,
                "gold_k": gold[p["k"]],
                "joint_judged_k_incorrect": pj["judgments"].get(p["k"] + 1) == 0,
                "acc_joint_post": acc_joint, "acc_ind_post": acc_ind,
                "gap": acc_ind - acc_joint, "n_post": len(post),
                "drag_rate_joint": drag})

    gaps = [r["gap"] for r in records]
    summary = {"records": len(records), "skipped": skipped,
               "interference_gap": boot_ci(gaps),
               "frac_positive_gap": float(np.mean([g > 0 for g in gaps]))
               if gaps else None}

    by_pair = {}
    for r in records:
        by_pair.setdefault(r["id"], {})[r["variant"]] = r["gap"]
    dids = [v["flipped"] - v["orig"] for v in by_pair.values()
            if "orig" in v and "flipped" in v]
    summary["diff_in_diff_gap"] = boot_ci(dids)

    for cond, name in [(True, "joint_k_incorrect"), (False, "joint_k_correct")]:
        sub = [r for r in records if r["joint_judged_k_incorrect"] == cond]
        drags = [r["drag_rate_joint"] for r in sub
                 if r["drag_rate_joint"] is not None]
        summary[name] = {
            "n_records": len(sub),
            "gap": boot_ci([r["gap"] for r in sub]),
            "drag_rate_on_gold_correct_post": boot_ci(drags)}

    print("PARTB_SUMMARY " + json.dumps(summary))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
