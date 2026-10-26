"""V4 walk validation gate: compare the cached walk's p_inc against the
proven tokenprob dump (tokenprob_eval.py, full-string walk) on the first
10 chosen problems. Prints V4_WALK_CHECK with mean |dp| — GO if < 0.05.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "train"))
from rewards import build_verifier_prompt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--ref", default="/workspace/tokenprob_grpo_v3.jsonl")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data)][:args.n]
    ref = {}
    for l in open(args.ref):
        r = json.loads(l)
        ref[r["id"]] = r["p_inc"]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)
    id_corr = tok.encode(" correct", add_special_tokens=False)[0]
    id_inc = tok.encode(" incorrect", add_special_tokens=False)[0]

    def feed(past, text, special=False):
        enc = tok(text, add_special_tokens=special,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(input_ids=enc["input_ids"], past_key_values=past,
                      use_cache=True)
        return o.past_key_values, o.logits[0, -1]

    diffs, ss, sl = [], [], []
    for d in records:
        texts = [s["candidates"][s["chosen_idx"] if s["chosen_idx"] is not None
                                else 0]["text"] for s in d["steps"]]
        gold = [s["candidates"][s["chosen_idx"] if s["chosen_idx"] is not None
                               else 0]["gold"] for s in d["steps"]]
        past, _ = feed(None, build_verifier_prompt(d["problem"], texts),
                       special=True)
        mine = []
        for i in range(len(texts)):
            past, logits = feed(past, f"Step {i + 1}:")
            pair = torch.softmax(torch.stack(
                [logits[id_inc], logits[id_corr]]).float(), dim=0)
            p_inc = float(pair[0])
            mine.append(p_inc)
            verdict = "incorrect" if p_inc > 0.5 else "correct"
            past, _ = feed(past, f" {verdict}\n")
        theirs = ref[d["id"]]
        diffs += [abs(a - b) for a, b in zip(mine, theirs)]
        ss += mine
        sl += [1 - g for g in gold]
    ss = np.asarray(ss)
    sl = np.asarray(sl)
    order = np.argsort(ss, kind="mergesort")
    ranks = np.empty(len(ss))
    ranks[order] = np.arange(1, len(ss) + 1)
    n1 = int(sl.sum())
    n0 = len(sl) - n1
    auroc = float((ranks[sl == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) \
        if n1 and n0 else None
    print("V4_WALK_CHECK " + json.dumps({
        "n_problems": len(records), "n_steps": len(diffs),
        "mean_abs_dp": float(np.mean(diffs)),
        "max_abs_dp": float(np.max(diffs)),
        "subset_step_auroc_mine": auroc}), flush=True)


if __name__ == "__main__":
    main()
