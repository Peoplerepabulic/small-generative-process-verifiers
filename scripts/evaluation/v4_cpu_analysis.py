"""V4 CPU analysis (reviewer round 2) — everything computable from dumps.

Sections (each prints one V4_* JSON line, failures isolated):
  S0 schemas of input files
  S1 official-dump census: boxed variants, Step-0 headers, header counts,
     explicit-final rate, yesno missing, truncation flag
  S2 coverage breakdown of v2-uncovered steps + parser v3 reparse + metrics
  S3 same-subset (v2-covered 401 steps) metrics for our models
  S4 stats pack: recall CP CI, McNemar exacts, Wilson CIs, DiscPRM Brier,
     ThinkPRM temperature scaling (split-half x200)
  S5 chosen-103 alignment: overlap + cascade, partC stack vs revision stack
"""
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thinkprm_official_eval as ev  # noqa: E402

WS = "/workspace"
OFF = os.path.join(WS, "thinkprm_official_dump_v2.jsonl")
REV = os.path.join(WS, "rev_dumps")
DISC = os.path.join(WS, "rev_dumps_disc.jsonl")
PARTC = os.path.join(WS, "partC_scores.jsonl")
TP = {m: os.path.join(WS, f"tokenprob_{m}.jsonl")
      for m in ("sft_v2", "grpo_v3")}


def load(p):
    return [json.loads(l) for l in open(p)]


def out(tag, obj):
    print(tag + " " + json.dumps(obj), flush=True)


def binom_two_sided(k, n):
    """Exact two-sided binomial p-value under p=0.5."""
    if n == 0:
        return None
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    pk = probs[k]
    return float(sum(p for p in probs if p <= pk * (1 + 1e-9)))


def mcnemar(b, c):
    return {"b": b, "c": c, "p_exact": binom_two_sided(min(b, c), b + c)}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


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


def ts_fit_eval(confs, labels, seed=0, reps=200):
    confs = np.asarray(confs, float)
    labels = np.asarray(labels, float)
    p = np.clip(confs, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    rng = np.random.default_rng(seed)
    raw_es, ts_es, Ts = [], [], []
    idx = np.arange(len(confs))
    for _ in range(reps):
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
    return {"T_mean": float(np.mean(Ts)), "ece_raw_held": float(np.mean(raw_es)),
            "ece_ts_held": float(np.mean(ts_es))}


# ---------------- parser v3 ----------------
BOXED = re.compile(r"\\{1,2}boxed\s*\{\s*(incorrect|correct)\s*\}", re.I)


def parse_v3(text, n_steps):
    headers = [(int(m.group(1)), m.start(), m.end())
               for m in ev._HEADER_RE.finditer(text)]
    raw_idx = [h[0] for h in headers]
    zero_based = bool(raw_idx) and 0 in raw_idx and \
        n_steps not in raw_idx and max(raw_idx) == n_steps - 1
    off = 1 if zero_based else 0
    verdicts = {}
    if headers:
        for j, (idx, s, e) in enumerate(headers):
            mi = idx + off
            if not (1 <= mi <= n_steps) or mi in verdicts:
                continue
            seg_end = headers[j + 1][1] if j + 1 < len(headers) else len(text)
            seg = text[e:seg_end]
            boxes = BOXED.findall(seg)
            if boxes:
                verdicts[mi] = 1 if boxes[-1].lower() == "correct" else 0
                continue
            v = ev._segment_verdict(seg)
            if v is not None:
                verdicts[mi] = v
    parsed = [verdicts.get(i + 1) for i in range(n_steps)]
    for pat in ev.STEP_PATTERNS:
        for m in pat.finditer(text):
            idx = int(m.group(1)) + off
            if 1 <= idx <= n_steps and parsed[idx - 1] is None:
                parsed[idx - 1] = 1 if m.group(2).lower() == "correct" else 0
    boxes_all = BOXED.findall(text)
    if len(boxes_all) == n_steps:
        for i, b in enumerate(boxes_all):
            if parsed[i] is None:
                parsed[i] = 1 if b.lower() == "correct" else 0
    first_err = next((i + 1 for i, v in enumerate(parsed) if v == 0), None)
    final = None
    tail = text[int(len(text) * 0.5):]
    for pat in ev.FINAL_PATTERNS:
        ms = list(pat.finditer(tail))
        if ms:
            final = 1 if ms[-1].group(1).lower() == "correct" else 0
            break
    explicit_final = final is not None
    if final is None:
        final = 1 if first_err is None else 0
    return parsed, first_err, final, {"zero_based": zero_based,
                                      "n_boxed": len(boxes_all),
                                      "explicit_final": explicit_final}


def step_metrics(pairs):
    """pairs: list of (gold, pred) with pred in {0,1} (no Nones)."""
    tp = sum(1 for g, p in pairs if p == 0 and g == 0)
    fp = sum(1 for g, p in pairs if p == 0 and g == 1)
    tn = sum(1 for g, p in pairs if p == 1 and g == 1)
    fn = sum(1 for g, p in pairs if p == 1 and g == 0)
    n = len(pairs)
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "acc": (tp + tn) / n if n else None,
            "neg_recall": rec,
            "balanced_acc": 0.5 * (rec + spec)
            if rec is not None and spec is not None else None}


def main():
    dump = load(OFF)
    # ---------- S0 schemas ----------
    try:
        sch = {}
        for name, path in [("rev_sft_v2", os.path.join(REV, "sft_v2.jsonl")),
                           ("disc", DISC), ("partc", PARTC),
                           ("tokenprob_grpo_v3", TP["grpo_v3"])]:
            try:
                rows = load(path)
                sch[name] = {"n": len(rows), "keys": sorted(rows[0].keys())}
                if name == "partc":
                    sch[name]["trace_keys"] = sorted(rows[0]["traces"][0].keys())
            except Exception as e:  # noqa: BLE001
                sch[name] = {"error": str(e)}
        out("V4_SCHEMA", sch)
    except Exception as e:  # noqa: BLE001
        out("V4_SCHEMA_ERROR", {"error": str(e)})

    # ---------- S1 census ----------
    try:
        n_boxed_any = n_boxed2 = n_step0 = n_trunc = n_yn_none = 0
        n_explicit_final = 0
        hdr_gap = []
        for d in dump:
            t = d["gen_text"]
            if BOXED.search(t):
                n_boxed_any += 1
            if re.search(r"\\\\boxed", t):
                n_boxed2 += 1
            idxs = [int(m.group(1)) for m in ev._HEADER_RE.finditer(t)]
            if 0 in idxs:
                n_step0 += 1
            hdr_gap.append(len(d["gold_steps"]) - len(set(
                i for i in idxs if 1 <= i <= len(d["gold_steps"]))))
            n_trunc += int(bool(d.get("truncated")))
            n_yn_none += int(d.get("yesno_score") is None)
            _, _, _, meta = parse_v3(t, len(d["gold_steps"]))
            n_explicit_final += int(meta["explicit_final"])
        out("V4_CENSUS", {
            "n_problems": len(dump),
            "gens_with_boxed": n_boxed_any,
            "gens_with_double_backslash_boxed": n_boxed2,
            "gens_with_step0_header": n_step0,
            "gens_truncated_proxy": n_trunc,
            "yesno_missing": n_yn_none,
            "explicit_final_rate": n_explicit_final / len(dump)})
    except Exception as e:  # noqa: BLE001
        out("V4_CENSUS_ERROR", {"error": str(e)})

    # ---------- S2 coverage breakdown + parser v3 ----------
    try:
        cats = {"gap_middle": 0, "after_first_flag": 0, "tail_stopped": 0}
        sub = {"gap_with_text_evidence": 0, "tail_with_text_evidence": 0,
               "in_truncated_gen": 0}
        n_unc = 0
        for d in dump:
            parsed, gold, t = d["parsed"], d["gold_steps"], d["gen_text"]
            cov_idx = [i for i, v in enumerate(parsed) if v is not None]
            first_flag = next((i for i, v in enumerate(parsed) if v == 0),
                              None)
            for i, v in enumerate(parsed):
                if v is not None:
                    continue
                n_unc += 1
                if d.get("truncated"):
                    sub["in_truncated_gen"] += 1
                evidence = bool(re.search(rf"step\s*{i + 1}\b", t, re.I)) or \
                    bool(BOXED.search(t))
                if cov_idx and max(cov_idx) > i:
                    cats["gap_middle"] += 1
                    sub["gap_with_text_evidence"] += int(evidence)
                elif first_flag is not None and i > first_flag:
                    cats["after_first_flag"] += 1
                else:
                    cats["tail_stopped"] += 1
                    sub["tail_with_text_evidence"] += int(evidence)
        # parser v3 metrics
        pairs3, sol_ok3 = [], 0
        n_cov3 = 0
        for d in dump:
            p3, fe3, fin3, _ = parse_v3(d["gen_text"], len(d["gold_steps"]))
            sol_ok3 += int(fin3 == d["gold_final"])
            for g, p in zip(d["gold_steps"], p3):
                if p is not None:
                    n_cov3 += 1
                    pairs3.append((g, p))
        m3 = step_metrics(pairs3)
        n_steps_tot = sum(len(d["gold_steps"]) for d in dump)
        # v2 subset trivial baseline
        pairs2 = [(g, p) for d in dump
                  for g, p in zip(d["gold_steps"], d["parsed"])
                  if p is not None]
        triv2 = sum(1 for g, _ in pairs2 if g == 1) / len(pairs2)
        m3.update({"coverage": n_cov3 / n_steps_tot,
                   "sol_acc_parsed_final_v3": sol_ok3 / len(dump),
                   "v2_coverage": len(pairs2) / n_steps_tot,
                   "v2_subset_trivial_acc": triv2,
                   "v2_subset_gold_neg_rate":
                   sum(1 for g, _ in pairs2 if g == 0) / len(pairs2),
                   "full_gold_neg_rate":
                   sum(1 for d in dump for g in d["gold_steps"] if g == 0)
                   / n_steps_tot})
        out("V4_COVERAGE", {"n_uncovered_v2": n_unc, "categories": cats,
                            "subflags": sub, "parser_v3": m3})
    except Exception as e:  # noqa: BLE001
        out("V4_COVERAGE_ERROR", {"error": str(e)})

    # ---------- S3 same-subset ----------
    try:
        covered = {}
        for d in dump:
            covered[d["id"]] = {i for i, v in enumerate(d["parsed"])
                                if v is not None}
        res = {}
        for model in ("sft_v1", "sft_v2", "grpo_v3"):
            rows = load(os.path.join(REV, model + ".jsonl"))
            prs = []
            for r in rows:
                if r["judgments"] is None:
                    continue
                for i in covered.get(r["id"], set()):
                    if r["judgments"][i] is not None:
                        prs.append((r["gold_steps"][i], r["judgments"][i]))
            res[model] = step_metrics(prs)
        rows = load(DISC)
        prs = []
        for r in rows:
            for i in covered.get(r["id"], set()):
                pred = 0 if r["step_probs"][i] < 0.5 else 1
                prs.append((r["gold_steps"][i], pred))
        res["discprm"] = step_metrics(prs)
        out("V4_SAME_SUBSET", res)
    except Exception as e:  # noqa: BLE001
        out("V4_SAME_SUBSET_ERROR", {"error": str(e)})

    # ---------- S4 stats pack ----------
    try:
        stats = {}
        # recall 0/21 Clopper-Pearson (k=0 closed form)
        stats["thinkprm_recall_cp95"] = [0.0, 1 - 0.025 ** (1 / 21)]
        # noisy-OR v2 vs v3 from tokenprob dumps
        tp_rows = {}
        for m, p in TP.items():
            tp_rows[m] = {r["id"]: r for r in load(p)}
        accs, b = {}, 0
        b = c = 0
        for pid, r3 in tp_rows["grpo_v3"].items():
            r2 = tp_rows["sft_v2"].get(pid)
            if r2 is None:
                continue
            gold = int(all(g == 1 for g in r3["gold_steps"]))
            ok3 = int((np.prod([1 - x for x in r3["p_inc"]]) >= 0.5) == gold)
            ok2 = int((np.prod([1 - x for x in r2["p_inc"]]) >= 0.5) == gold)
            accs.setdefault("grpo_v3", []).append(ok3)
            accs.setdefault("sft_v2", []).append(ok2)
            if ok2 and not ok3:
                b += 1
            elif ok3 and not ok2:
                c += 1
        stats["noisyor_acc"] = {
            m: {"acc": float(np.mean(v)), "n": len(v),
                "wilson95": wilson(int(sum(v)), len(v))}
            for m, v in accs.items()}
        stats["noisyor_v2_vs_v3_mcnemar"] = mcnemar(b, c)
        # DiscPRM solution Brier (score = min step prob, P(correct))
        drows = load(DISC)
        sc = [min(r["step_probs"]) for r in drows]
        lb = [int(all(g == 1 for g in r["gold_steps"])) for r in drows]
        stats["discprm_sol_brier"] = float(np.mean(
            [(s - l) ** 2 for s, l in zip(sc, lb)]))
        stats["discprm_sol_ece"] = ece(sc, lb)
        # ThinkPRM TS on yes/no scores
        yn = [(d["yesno_score"], d["gold_final"]) for d in dump
              if d.get("yesno_score") is not None]
        stats["thinkprm_yesno_n"] = len(yn)
        stats["thinkprm_ts"] = ts_fit_eval([s for s, _ in yn],
                                           [l for _, l in yn])
        stats["thinkprm_yesno_brier"] = float(np.mean(
            [(s - l) ** 2 for s, l in yn]))
        out("V4_STATS", stats)
    except Exception as e:  # noqa: BLE001
        out("V4_STATS_ERROR", {"error": str(e)})

    # ---------- S5 alignment ----------
    try:
        pc = load(PARTC)
        b = c = 0
        ov = {"both": 0, "disc_only": 0, "gen_only": 0, "neither": 0}
        for r in pc:
            t = next(t for t in r["traces"] if t["is_chosen"])
            dok = int((t["disc_min"] >= 0.5) == bool(t["gold_final"]))
            gok = int(int(t["gen_final"]) == int(t["gold_final"]))
            if dok and gok:
                ov["both"] += 1
            elif dok:
                ov["disc_only"] += 1
                b += 1
            elif gok:
                ov["gen_only"] += 1
                c += 1
            else:
                ov["neither"] += 1
        partc_align = {"overlap": ov, "mcnemar_disc_vs_gen": mcnemar(b, c),
                       "gen_acc": (ov["both"] + ov["gen_only"]) / sum(ov.values())}
        # revision stack
        disc = {r["id"]: r for r in load(DISC)}
        gen = {r["id"]: r for r in load(os.path.join(REV, "grpo_v3.jsonl"))}
        b = c = 0
        ov2 = {"both": 0, "disc_only": 0, "gen_only": 0, "neither": 0}
        dsc, dverdict, gverdict, golds = [], [], [], []
        for pid, gr in gen.items():
            dr = disc[pid]
            s = min(dr["step_probs"])
            dok = int((s >= 0.5) == bool(gr["gold_final"]))
            gok = int(int(gr["gen_final"]) == int(gr["gold_final"]))
            dsc.append(s)
            dverdict.append(int(s >= 0.5))
            gverdict.append(int(gr["gen_final"]))
            golds.append(int(gr["gold_final"]))
            if dok and gok:
                ov2["both"] += 1
            elif dok:
                ov2["disc_only"] += 1
                b += 1
            elif gok:
                ov2["gen_only"] += 1
                c += 1
            else:
                ov2["neither"] += 1
        rev_align = {"overlap": ov2, "mcnemar_disc_vs_gen": mcnemar(b, c),
                     "gen_acc": (ov2["both"] + ov2["gen_only"]) / sum(ov2.values())}
        # rev-stack chosen cascade at delta grid (full sample)
        casc = {}
        for delta in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
            ok = esc = 0
            for s, dv, gv, gl in zip(dsc, dverdict, gverdict, golds):
                if abs(s - 0.5) < delta:
                    esc += 1
                    ok += int(gv == gl)
                else:
                    ok += int(dv == gl)
            casc[str(delta)] = {"acc": ok / len(golds), "esc": esc / len(golds)}
        rev_align["cascade_full_sample"] = casc
        out("V4_ALIGN", {"partc_stack": partc_align, "rev_stack": rev_align})
    except Exception as e:  # noqa: BLE001
        out("V4_ALIGN_ERROR", {"error": str(e)})


if __name__ == "__main__":
    main()
