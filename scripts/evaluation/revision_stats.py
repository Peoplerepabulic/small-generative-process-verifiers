"""Revision statistics (CPU): everything computable from the dump files.

Reads:
  --dumps DIR with {model}.jsonl from dump_eval.py (generative models)
  --disc FILE from disc_dump.py
  --partc FILE partC_scores.jsonl (for cascade CV + overlap + DiscPRM TS)
Prints REV_STATS_* JSON lines.
"""
import argparse
import json
import math
import os

import numpy as np


def boot_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(vals.mean()),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))], "n": int(len(vals))}


def ece(confs, labels, n_bins=10):
    confs = np.asarray(confs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    out = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (confs > lo) & (confs <= hi) if b else (confs >= lo) & (confs <= hi)
        if m.sum():
            out += m.mean() * abs(confs[m].mean() - labels[m].mean())
    return float(out)


def ts_fit_eval(confs, labels, seed=0):
    """Split-half x200: fit T on one half (NLL grid+refine), eval other."""
    confs = np.asarray(confs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    p = np.clip(confs, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    rng = np.random.default_rng(seed)
    raw_es, ts_es, Ts = [], [], []
    idx = np.arange(len(confs))
    for _ in range(200):
        rng.shuffle(idx)
        a, b = idx[:len(idx) // 2], idx[len(idx) // 2:]
        def nll(T, ii):
            q = 1 / (1 + np.exp(-logit[ii] / T))
            q = np.clip(q, 1e-9, 1 - 1e-9)
            return -np.mean(labels[ii] * np.log(q)
                            + (1 - labels[ii]) * np.log(1 - q))
        grid = np.concatenate([[0.3, 0.5, 0.75], np.arange(1.0, 15.01, 0.25)])
        T = float(grid[np.argmin([nll(t, a) for t in grid])])
        Ts.append(T)
        qb = 1 / (1 + np.exp(-logit[b] / T))
        raw_es.append(ece(confs[b], labels[b]))
        ts_es.append(ece(qb, labels[b]))
    return {"T": float(np.mean(Ts)), "T_sd": float(np.std(Ts)),
            "ece_raw_split": float(np.mean(raw_es)),
            "ece_ts": float(np.mean(ts_es)),
            "ece_ts_sd": float(np.std(ts_es))}


def step_confusion(golds, preds):
    """Positive class = step incorrect. preds: 1=correct verdict, 0=incorrect."""
    tp = fp = tn = fn = 0
    for g, p in zip(golds, preds):
        if p is None:
            continue
        pred_inc = 1 if p == 0 else 0
        gold_inc = 1 if g == 0 else 0
        if pred_inc and gold_inc: tp += 1
        elif pred_inc and not gold_inc: fp += 1
        elif not pred_inc and gold_inc: fn += 1
        else: tn += 1
    n = tp + fp + tn + fn
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "acc": (tp + tn) / n if n else None,
            "neg_recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "balanced_acc": 0.5 * (tp / (tp + fn) + tn / (tn + fp))
            if tp + fn and tn + fp else None,
            "neg_f1": (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None}


def mcnemar_exact(correct_a, correct_b):
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return {"b": b, "c": c, "p": min(p, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--disc", default=None)
    ap.add_argument("--partc", default=None)
    args = ap.parse_args()

    models = {}
    for fn in sorted(os.listdir(args.dumps)):
        if fn.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(os.path.join(args.dumps, fn))]
            models[fn[:-6]] = rows

    out = {}
    sol_correct = {}
    for name, rows in models.items():
        golds, preds = [], []
        c = d = 0
        finals = []
        for r in rows:
            if r["judgments"] is not None:
                golds.extend(r["gold_steps"])
                preds.extend(r["judgments"])
                flagged = any(j == 0 for j in r["judgments"]
                              if j is not None)
                if r["gold_final"] == 1 and flagged:
                    c += 1
                if r["gold_final"] == 0 and flagged:
                    d += 1
            finals.append(1 if r["gen_final"] == r["gold_final"] else 0)
        sol_correct[name] = finals
        entry = {"step": step_confusion(golds, preds),
                 "derived_c": c, "derived_d": d,
                 "gen_final_acc": boot_ci(finals)}
        conf_rows = [r for r in rows if r["conf"] is not None]
        if conf_rows:
            entry["ts"] = ts_fit_eval([r["conf"] for r in conf_rows],
                                      [r["gold_final"] for r in conf_rows])
            entry["ece_raw_full"] = ece([r["conf"] for r in conf_rows],
                                        [r["gold_final"] for r in conf_rows])
        out[name] = entry
    print("REV_STATS_MODELS " + json.dumps(out))

    pairs = [("sft_v1", "sft_v2"), ("sft_v2", "grpo_v3"),
             ("sft_v1", "grpo_v3")]
    mc = {}
    for a, b in pairs:
        if a in sol_correct and b in sol_correct:
            mc[f"{a}_vs_{b}"] = mcnemar_exact(sol_correct[a], sol_correct[b])
    print("REV_STATS_MCNEMAR " + json.dumps(mc))

    if args.disc:
        rows = [json.loads(l) for l in open(args.disc)]
        golds, preds = [], []
        c = d = 0
        sol_scores, sol_labels, finals = [], [], []
        for r in rows:
            preds_p = [1 if p >= 0.5 else 0 for p in r["step_probs"]]
            golds.extend(r["gold_steps"])
            preds.extend(preds_p)
            flagged = any(p < 0.5 for p in r["step_probs"])
            gf = int(all(g == 1 for g in r["gold_steps"]))
            if gf == 1 and flagged: c += 1
            if gf == 0 and flagged: d += 1
            sol_scores.append(min(r["step_probs"]))
            sol_labels.append(gf)
            finals.append(1 if (min(r["step_probs"]) >= 0.5) == bool(gf)
                          else 0)
        disc = {"step": step_confusion(golds, preds), "derived_c": c,
                "derived_d": d, "gen_final_acc": boot_ci(finals),
                "ts_min_score": ts_fit_eval(sol_scores, sol_labels),
                "ece_raw_full": ece(sol_scores, sol_labels)}
        print("REV_STATS_DISC " + json.dumps(disc))
        if "grpo_v3" in sol_correct:
            print("REV_STATS_MCNEMAR_DISC_V3 " + json.dumps(
                mcnemar_exact(finals, sol_correct["grpo_v3"])))

    if args.partc:
        problems = [json.loads(l) for l in open(args.partc)]
        deltas = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

        def cascade_acc(traces, delta):
            ok = 0
            for t in traces:
                dv = 1 if t["disc_min"] >= 0.5 else 0
                esc = abs(t["disc_min"] - 0.5) < delta
                fin = t["gen_final"] if esc else dv
                ok += int(fin == t["gold_final"])
            return ok / len(traces)

        for scope, traces in [
                ("all", [t for p in problems for t in p["traces"]]),
                ("chosen", [t for p in problems for t in p["traces"]
                            if t["is_chosen"]])]:
            rng = np.random.default_rng(0)
            held = {d: [] for d in deltas}
            stars = []
            idx = np.arange(len(traces))
            for _ in range(200):
                rng.shuffle(idx)
                a = [traces[i] for i in idx[:len(idx) // 2]]
                b = [traces[i] for i in idx[len(idx) // 2:]]
                accs_a = {d: cascade_acc(a, d) for d in deltas}
                d_star = max(deltas, key=lambda d: accs_a[d])
                stars.append(d_star)
                for d in deltas:
                    held[d].append(cascade_acc(b, d))
            cv = {"scope": scope, "n": len(traces),
                  "star_mode": float(np.bincount(
                      [deltas.index(s) for s in stars],
                      minlength=len(deltas)).argmax()),
                  "star_delta_value": deltas[int(np.bincount(
                      [deltas.index(s) for s in stars],
                      minlength=len(deltas)).argmax())],
                  "heldout_acc_by_delta": {
                      str(d): [float(np.mean(held[d])), float(np.std(held[d]))]
                      for d in deltas}}
            print(f"REV_STATS_PARTC_CV_{scope.upper()} " + json.dumps(cv))

        for scope, traces in [
                ("all", [t for p in problems for t in p["traces"]]),
                ("chosen", [t for p in problems for t in p["traces"]
                            if t["is_chosen"]])]:
            tab = {"both_correct": 0, "disc_only": 0, "gen_only": 0,
                   "both_wrong": 0}
            for t in traces:
                dv = 1 if t["disc_min"] >= 0.5 else 0
                dc = int(dv == t["gold_final"])
                gc = int(t["gen_final"] == t["gold_final"])
                if dc and gc: tab["both_correct"] += 1
                elif dc: tab["disc_only"] += 1
                elif gc: tab["gen_only"] += 1
                else: tab["both_wrong"] += 1
            print(f"REV_STATS_OVERLAP_{scope.upper()} " + json.dumps(tab))


if __name__ == "__main__":
    main()
