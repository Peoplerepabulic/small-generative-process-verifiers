"""Recompute ThinkPRM official summaries from the saved dumps (CPU only).

Reads /workspace/thinkprm_official_dump_v2.jsonl (baseline, reparsed) and
/workspace/thinkprm_swap_dump.jsonl (flipped runs). Replicates the exact
metric code of thinkprm_official_eval.py / thinkprm_official_swap.py.
Prints THINKPRM_BASE_SUMMARY and THINKPRM_SWAP_SUMMARY (JSON).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from partB_interference import build_counterfactual_pairs  # noqa: E402
from thinkprm_official_eval import auroc, ece  # noqa: E402


def boot_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(vals.mean()),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))], "n": int(len(vals))}


def main():
    dump = [json.loads(l) for l in
            open("/workspace/thinkprm_official_dump_v2.jsonl")]
    tp = fp = tn = fn = n_parsed = 0
    sol_ok = der_ok = loc_ok = loc_n = 0
    for d in dump:
        gold_err = next((i + 1 for i, g in enumerate(d["gold_steps"])
                         if g == 0), None)
        if gold_err is not None:
            loc_n += 1
            loc_ok += int(d["first_error_pred"] == gold_err)
        sol_ok += int(d["final_pred"] == d["gold_final"])
        der = 1 if d["first_error_pred"] is None else 0
        der_ok += int(der == d["gold_final"])
        for g, p in zip(d["gold_steps"], d["parsed"]):
            if p is None:
                continue
            n_parsed += 1
            if p == 0 and g == 0: tp += 1
            elif p == 0: fp += 1
            elif g == 0: fn += 1
            else: tn += 1
    n_steps = sum(len(d["gold_steps"]) for d in dump)
    neg_recall = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    yn = [(d["yesno_score"], d["gold_final"]) for d in dump
          if d.get("yesno_score") is not None]
    base_summary = {
        "model": "launch/ThinkPRM-1.5B", "prompt": "official_file",
        "n": len(dump), "step_coverage": n_parsed / max(n_steps, 1),
        "step_acc": (tp + tn) / max(n_parsed, 1),
        "neg_recall": neg_recall,
        "balanced_acc": 0.5 * (neg_recall + spec)
        if neg_recall is not None and spec is not None else None,
        "sol_acc_parsed_final": sol_ok / len(dump),
        "sol_acc_derived": der_ok / len(dump),
        "loc": loc_ok / loc_n if loc_n else None, "loc_n": loc_n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "yesno_auroc": auroc([s for s, _ in yn], [l for _, l in yn]),
        "yesno_acc05": sum(int((s >= 0.5) == bool(l)) for s, l in yn)
        / len(yn) if yn else None,
        "yesno_ece": ece([s for s, _ in yn], [l for _, l in yn]) if yn
        else None,
        "yesno_brier": sum((s - l) ** 2 for s, l in yn) / len(yn) if yn
        else None,
    }
    print("THINKPRM_BASE_SUMMARY " + json.dumps(base_summary))

    base = {d["id"]: d for d in dump}
    pairs = build_counterfactual_pairs(
        ["data/prm800k_phase1_test.jsonl"], max_n=200)
    pairs = {p["id"]: p for p in pairs if p["id"] in base}
    swap = [json.loads(l) for l in
            open("/workspace/thinkprm_swap_dump.jsonl")]
    acc_o, acc_f, drag_o, drag_f, flips = [], [], [], [], []
    for d in swap:
        p = pairs.get(d["id"])
        if p is None:
            continue
        k = d["k"]
        _, gold_o = p["variants"]["orig"]
        parsed_o = base[d["id"]]["parsed"]
        parsed_f = d["parsed_flipped"]
        po, pf = parsed_o[k + 1:], parsed_f[k + 1:]
        gold_post = gold_o[k + 1:]
        idx = [i for i in range(len(gold_post))
               if po[i] is not None and pf[i] is not None]
        if not idx or not gold_post:
            continue
        acc_o.append(np.mean([po[i] == gold_post[i] for i in idx]))
        acc_f.append(np.mean([pf[i] == gold_post[i] for i in idx]))
        gc = [i for i in idx if gold_post[i] == 1]
        if gc:
            drag_o.append(np.mean([po[i] == 0 for i in gc]))
            drag_f.append(np.mean([pf[i] == 0 for i in gc]))
        flips.append(np.mean([po[i] != pf[i] for i in idx]))
    if acc_o:
        summary = {
            "n_pairs_used": len(acc_o), "n_pairs_total": len(swap),
            "postk_acc_orig": boot_ci(acc_o),
            "postk_acc_flipped": boot_ci(acc_f),
            "acc_diff_flipped_minus_orig": boot_ci(
                list(np.array(acc_f) - np.array(acc_o))),
            "drag_orig": boot_ci(drag_o), "drag_flipped": boot_ci(drag_f),
            "drag_diff": boot_ci(list(np.array(drag_f)
                                      - np.array(drag_o))),
            "verdict_flip_rate": boot_ci(flips),
        }
    else:
        summary = {"n_pairs_used": 0, "n_pairs_total": len(swap),
                   "error": "no usable parsed pairs"}
    print("THINKPRM_SWAP_SUMMARY " + json.dumps(summary))

    # diagnostics: why so few usable pairs? coverage + truncation census
    cov_f = cov_fp = tot_f = tot_fp = 0
    trunc = 0
    for d in swap:
        pf = d["parsed_flipped"]
        cov_f += sum(v is not None for v in pf)
        tot_f += len(pf)
        post = pf[d["k"] + 1:]
        cov_fp += sum(v is not None for v in post)
        tot_fp += len(post)
        trunc += int(len(d["gen_text"]) >= 20000)
    cov_o = sum(v is not None for d in dump for v in d["parsed"])
    print("THINKPRM_SWAP_DIAG " + json.dumps({
        "flipped_step_coverage": cov_f / max(tot_f, 1),
        "flipped_postk_coverage": cov_fp / max(tot_fp, 1),
        "flipped_truncated_proxy": trunc, "n_flipped": len(swap),
        "baseline_step_coverage": cov_o / max(n_steps, 1)}))


if __name__ == "__main__":
    main()
