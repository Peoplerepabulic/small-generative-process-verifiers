"""V6 (CPU): Part C clustered statistics on the 891-trace merge.
(1) V6_XYZ: accuracies on the delta=0.10 escalated subset vs the rest.
(2) V6_GROUPED_CV: split-half x200 CV with splits GROUPED BY PROBLEM
    (all traces of a problem stay in one half), both second stages.
(3) V6_CLUSTER_BOOT: cluster bootstrap (resample 103 problems with
    replacement, 2000 reps) percentile CIs for cascade@0.10 (noisy-OR
    stage), readout-alone, DiscPRM-alone accuracies and paired diffs.
Same merge as v4_partc_noisyor.py.
"""
import json
import os

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


def main():
    pc = [json.loads(l) for l in open(os.path.join(WS, "partC_scores.jsonl"))]
    tp = {}
    for l in open(os.path.join(WS, "tokenprob891_grpo_v3.jsonl")):
        r = json.loads(l)
        tp[(r["pid"], r["ti"])] = r
    rows = []  # (pid, disc_score, disc_verd, text_verd, noisy_verd, gold)
    missing = 0
    for r in pc:
        for ti, t in enumerate(r["traces"]):
            k = (r["id"], ti)
            if k not in tp:
                missing += 1
                continue
            pcorr = float(np.prod([1 - x for x in tp[k]["p_inc"]]))
            rows.append((r["id"], float(t["disc_min"]),
                         int(t["disc_min"] >= 0.5), int(t["gen_final"]),
                         int(pcorr >= 0.5), int(t["gold_final"])))
    print(f"aligned {len(rows)} traces, missing {missing}", flush=True)
    dsc = np.array([r[1] for r in rows])
    dverd = np.array([r[2] for r in rows])
    golds = np.array([r[5] for r in rows])
    pids = np.array([r[0] for r in rows])

    # ---- (1) escalated vs rest at delta = 0.10 (noisy-OR stage) ----
    gnoisy = np.array([r[4] for r in rows])
    esc_mask = np.abs(dsc - 0.5) < 0.10
    xyz = {
        "n_esc": int(esc_mask.sum()), "n_rest": int((~esc_mask).sum()),
        "x_disc_on_esc": float(np.mean(dverd[esc_mask] == golds[esc_mask])),
        "y_noisy_on_esc": float(np.mean(gnoisy[esc_mask] == golds[esc_mask])),
        "z_disc_on_rest": float(np.mean(dverd[~esc_mask] == golds[~esc_mask])),
        "noisy_on_rest": float(np.mean(gnoisy[~esc_mask] == golds[~esc_mask])),
    }
    print("V6_XYZ " + json.dumps(xyz), flush=True)

    # ---- (2) grouped split-half CV ----
    uniq = list(dict.fromkeys(r[0] for r in rows))
    by_pid = {p: [i for i, r in enumerate(rows) if r[0] == p]
              for p in uniq}

    def grouped_cv(gverd_all, seed=0, reps=200):
        rng = np.random.default_rng(seed)
        gv = np.asarray(gverd_all)
        chosen, held, held_fixed = [], [], []
        for _ in range(reps):
            order = uniq[:]
            rng.shuffle(order)
            half = len(order) // 2
            ia = [i for p in order[:half] for i in by_pid[p]]
            ib = [i for p in order[half:] for i in by_pid[p]]
            accs = [cascade_acc(dsc[ia], dverd[ia], gv[ia], golds[ia], d)[0]
                    for d in GRID]
            d_star = GRID[int(np.argmax(accs))]
            chosen.append(d_star)
            held.append(cascade_acc(dsc[ib], dverd[ib], gv[ib],
                                    golds[ib], d_star)[0])
            held_fixed.append(cascade_acc(dsc[ib], dverd[ib], gv[ib],
                                          golds[ib], 0.10)[0])
        vals, counts = np.unique(chosen, return_counts=True)
        return {"delta_star_mode": float(vals[int(np.argmax(counts))]),
                "held_acc_at_star": [float(np.mean(held)),
                                     float(np.std(held))],
                "held_acc_at_0.10": [float(np.mean(held_fixed)),
                                     float(np.std(held_fixed))]}

    gtext = np.array([r[3] for r in rows])
    print("V6_GROUPED_CV " + json.dumps({
        "text": grouped_cv(gtext), "noisyor": grouped_cv(gnoisy)}),
        flush=True)

    # ---- (3) cluster bootstrap ----
    casc_ok = np.where(esc_mask, gnoisy == golds, dverd == golds)
    per_pid = {}
    for i, r in enumerate(rows):
        per_pid.setdefault(r[0], []).append(i)
    plist = list(per_pid)
    n_arr = np.array([len(per_pid[p]) for p in plist], float)
    casc_arr = np.array([float(casc_ok[per_pid[p]].sum()) for p in plist])
    read_arr = np.array([float((gnoisy[per_pid[p]] == golds[per_pid[p]]).sum())
                         for p in plist])
    disc_arr = np.array([float((dverd[per_pid[p]] == golds[per_pid[p]]).sum())
                         for p in plist])
    rng = np.random.default_rng(0)
    B = 2000
    stats = {k: [] for k in
             ("casc", "readout", "disc", "casc_minus_readout",
              "casc_minus_disc")}
    for _ in range(B):
        w = rng.multinomial(len(plist), np.full(len(plist), 1 / len(plist)))
        tot = float((w * n_arr).sum())
        ca = float((w * casc_arr).sum()) / tot
        ra = float((w * read_arr).sum()) / tot
        da = float((w * disc_arr).sum()) / tot
        stats["casc"].append(ca)
        stats["readout"].append(ra)
        stats["disc"].append(da)
        stats["casc_minus_readout"].append(ca - ra)
        stats["casc_minus_disc"].append(ca - da)
    boot = {k: {"mean": float(np.mean(v)),
                "ci95": [float(np.percentile(v, 2.5)),
                         float(np.percentile(v, 97.5))]}
            for k, v in stats.items()}
    print("V6_CLUSTER_BOOT " + json.dumps(boot), flush=True)


if __name__ == "__main__":
    main()
