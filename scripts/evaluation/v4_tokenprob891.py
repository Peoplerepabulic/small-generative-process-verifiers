"""V4: verdict-token p_inc for all 891 Part C traces (GRPO v3), cached walk.

Mirrors tokenprob_eval.py's protocol (two-way softmax at each "Step i:"
position, argmax verdict appended), but feeds the context incrementally
with past_key_values. Writes one row per trace incrementally:
{pid, ti, is_chosen, gold_steps, gold_final, p_inc}.
Sanity: chosen-trace subset must roughly reproduce tokenprob_eval's
GRPO v3 numbers (noisy-OR acc 0.835, AUROC 0.946, ECE 0.114).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "train"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards import build_verifier_prompt  # noqa: E402
from partC_scores import build_traces  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def auroc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
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
    confs = np.asarray(confs, float)
    labels = np.asarray(labels, float)
    tot = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (confs > lo) & (confs <= hi) if b else (confs >= lo) & (confs <= hi)
        if m.sum():
            tot += m.mean() * abs(confs[m].mean() - labels[m].mean())
    return float(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--out", default="/workspace/tokenprob891_grpo_v3.jsonl")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data)]
    problems = build_traces(records)
    n_traces = sum(len(p["traces"]) for p in problems)
    print(f"{len(problems)} problems, {n_traces} traces", flush=True)

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
        # NOTE: pass input_ids ONLY. Passing the tokenizer's attention_mask
        # (segment-length) together with past_key_values silently corrupts
        # conditioning on the cached context (transformers 5.x): the walk
        # then ignores the appended judgment lines (sanity AUROC ~0.55,
        # delta-p exactly 0). v28 fix.
        enc = tok(text, add_special_tokens=special,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(input_ids=enc["input_ids"], past_key_values=past,
                      use_cache=True)
        return o.past_key_values, o.logits[0, -1]

    t0 = time.time()
    done = 0
    with open(args.out, "w") as f:
        for p in problems:
            for ti, t in enumerate(p["traces"]):
                prompt = build_verifier_prompt(p["problem"], t["texts"])
                past, _ = feed(None, prompt, special=True)
                p_incs = []
                for i in range(len(t["texts"])):
                    past, logits = feed(past, f"Step {i + 1}:")
                    pair = torch.softmax(torch.stack(
                        [logits[id_inc], logits[id_corr]]).float(), dim=0)
                    p_inc = float(pair[0])
                    p_incs.append(p_inc)
                    verdict = "incorrect" if p_inc > 0.5 else "correct"
                    past, _ = feed(past, f" {verdict}\n")
                f.write(json.dumps({
                    "pid": p["id"], "ti": ti,
                    "is_chosen": bool(t["is_chosen"]),
                    "gold_steps": t["gold"],
                    "gold_final": int(t["gold_final"]),
                    "p_inc": p_incs}) + "\n")
                f.flush()
                done += 1
            if done % 100 < len(p["traces"]):
                print(f"{done}/{n_traces} traces, "
                      f"{time.time() - t0:.0f}s", flush=True)
    print(f"walk done in {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(args.out)]
    for scope, sel in (("all891", rows),
                       ("chosen103", [r for r in rows if r["is_chosen"]])):
        ss, sl, ps, pl = [], [], [], []
        for r in sel:
            pcorr = float(np.prod([1 - x for x in r["p_inc"]]))
            ss.append(pcorr)
            sl.append(r["gold_final"])
            for pv, g in zip(r["p_inc"], r["gold_steps"]):
                ps.append(pv)
                pl.append(1 - g)
        acc = float(np.mean([(s >= 0.5) == bool(l)
                             for s, l in zip(ss, sl)]))
        print("V4_TP891_" + scope.upper() + " " + json.dumps({
            "n": len(sel), "noisyor_acc": acc,
            "noisyor_auroc": auroc([-s for s in ss], [1 - l for l in sl]),
            "noisyor_ece": ece(ss, sl),
            "noisyor_brier": float(np.mean([(s - l) ** 2
                                            for s, l in zip(ss, sl)])),
            "step_auroc": auroc(ps, pl)}), flush=True)


if __name__ == "__main__":
    main()
