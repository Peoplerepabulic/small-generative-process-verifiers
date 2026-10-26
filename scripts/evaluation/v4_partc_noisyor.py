"""V4: Part C verification cascade with a noisy-OR second stage (CPU).

Merges partC_scores.jsonl (DiscPRM disc_min + text gen verdict per trace)
with tokenprob891_grpo_v3.jsonl (p_inc per trace, aligned by problem id +
trace index from the deterministic build_traces). Computes, for stage-2
in {text, noisyor}: endpoints, full-sample delta curves, and split-half
x200 cross-validated delta selection, on all-891 and chosen-103 scopes.
"""
import json
import os
import sys

import numpy as np

WS = "/workspace"
GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]


def cascade_acc(dsc, dverd, gverd, golds, delta):
    ok = esc = 0
    for s, dv, gv, gl in zip(dsc, dverd, gverd, golds):
        if abs(s - 0.5) < delta:
            esc += 1
            ok += int(gv == gl)
        else:
            ok += int(dv == gl)
    return ok / len(golds), esc / len(golds)


def cv_delta(dsc, dverd, gverd, golds, seed=0, reps=200):
    dsc = np.asarray(dsc)
    dverd = np.asarray(dverd)
    gverd = np.asarray(gverd)
    golds = np.asarray(golds)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(golds))
    chosen, held, held_fixed = [], [], []
    for _ in range(reps):
        rng.shuffle(idx)
        a, b = idx[:len(idx) // 2], idx[len(idx) // 2:]
        accs = [cascade_acc(dsc[a], dverd[a], gverd[a], golds[a], d)[0]
                for d in GRID]
        d_star = GRID[int(np.argmax(accs))]
        chosen.append(d_star)
        held.append(cascade_acc(dsc[b], dverd[b], gverd[b], golds[b],
                                d_star)[0])
        held_fixed.append(cascade_acc(dsc[b], dverd[b], gverd[b], golds[b],
                                      0.10)[0])
    vals, counts = np.unique(chosen, return_counts=True)
    return {"delta_star_mode": float(vals[int(np.argmax(counts))]),
            "held_acc_at_star": [float(np.mean(held)), float(np.std(held))],
            "held_acc_at_0.10": [float(np.mean(held_fixed)),
                                 float(np.std(held_fixed))]}


def main():
    pc = [json.loads(l) for l in open(os.path.join(WS, "partC_scores.jsonl"))]
    tp = {}
    for l in open(os.path.join(WS, "tokenprob891_grpo_v3.jsonl")):
        r = json.loads(l)
        tp[(r["pid"], r["ti"])] = r
    scopes = {"all891": [], "chosen103": []}
    missing = 0
    for r in pc:
        for ti, t in enumerate(r["traces"]):
            k = (r["id"], ti)
            if k not in tp:
                missing += 1
                continue
            pcorr = float(np.prod([1 - x for x in tp[k]["p_inc"]]))
            row = {"disc": t["disc_min"],
                   "dverd": int(t["disc_min"] >= 0.5),
                   "gtext": int(t["gen_final"]),
                   "gnoisy": int(pcorr >= 0.5),
                   "gold": int(t["gold_final"])}
            scopes["all891"].append(row)
            if t["is_chosen"]:
                scopes["chosen103"].append(row)
    print(f"aligned {len(scopes['all891'])} traces, missing {missing}",
          flush=True)
    out = {}
    for scope, rows in scopes.items():
        dsc = [r["disc"] for r in rows]
        dverd = [r["dverd"] for r in rows]
        golds = [r["gold"] for r in rows]
        entry = {"n": len(rows),
                 "disc_acc": float(np.mean([dv == gl for dv, gl
                                            in zip(dverd, golds)]))}
        for stage, key in (("text", "gtext"), ("noisyor", "gnoisy")):
            gv = [r[key] for r in rows]
            curve = {}
            for d in (0.05, 0.10, 0.15, 0.20):
                a, e = cascade_acc(dsc, dverd, gv, golds, d)
                curve[str(d)] = {"acc": a, "esc": e}
            entry[stage] = {
                "gen_acc": float(np.mean([g == gl for g, gl
                                          in zip(gv, golds)])),
                "curve": curve,
                "cv": cv_delta(dsc, dverd, gv, golds)}
        out[scope] = entry
    print("V4_NOISYOR_CASCADE " + json.dumps(out), flush=True)
    with open(os.path.join(WS, "v4_noisyor_cascade.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
