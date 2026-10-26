"""V5 extras (CPU): (1) DiscPRM x {text, noisy-OR} 2x2 tables on the
Part C merge (all-891 + chosen-103); (2) official-ThinkPRM parser
conflict count: steps where parser v2 (stored in the dump) and parser
v3 (boxed-priority) both produce a verdict and disagree.
Prints V5_TWOBYTWO and V5_CONFLICTS as JSON lines.
"""
import json
import os
import re
import sys

sys.path.insert(0, "/workspace/thinkprm-v28/thinkprm/scripts/evaluation")
import thinkprm_official_eval as ev  # noqa: E402
import numpy as np  # noqa: E402

WS = "/workspace"
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
    return parsed


def two_by_two(rows, key):
    both = d_only = g_only = neither = 0
    for r in rows:
        d_ok = r["dverd"] == r["gold"]
        g_ok = r[key] == r["gold"]
        if d_ok and g_ok:
            both += 1
        elif d_ok:
            d_only += 1
        elif g_ok:
            g_only += 1
        else:
            neither += 1
    return {"both": both, "disc_only": d_only, "gen_only": g_only,
            "neither": neither}


def main():
    # ---- 2x2 tables (same merge as v4_partc_noisyor.py) ----
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
            row = {"dverd": int(t["disc_min"] >= 0.5),
                   "gtext": int(t["gen_final"]),
                   "gnoisy": int(pcorr >= 0.5),
                   "gold": int(t["gold_final"])}
            scopes["all891"].append(row)
            if t["is_chosen"]:
                scopes["chosen103"].append(row)
    out = {"missing": missing}
    for scope, rows in scopes.items():
        out[scope] = {
            "n": len(rows),
            "disc_x_text": two_by_two(rows, "gtext"),
            "disc_x_noisyor": two_by_two(rows, "gnoisy"),
            "disc_acc": float(np.mean([r["dverd"] == r["gold"]
                                       for r in rows])),
            "text_acc": float(np.mean([r["gtext"] == r["gold"]
                                       for r in rows])),
            "noisyor_acc": float(np.mean([r["gnoisy"] == r["gold"]
                                          for r in rows]))}
    print("V5_TWOBYTWO " + json.dumps(out), flush=True)

    # ---- parser conflicts ----
    dump = [json.loads(l) for l in
            open(os.path.join(WS, "thinkprm_official_dump_v2.jsonl"))]
    joint = dis = dis_gold0 = dis_gold1 = v2only = v3only = 0
    for d in dump:
        v2 = d["parsed"]
        gold = d["gold_steps"]
        v3 = parse_v3(d["gen_text"], len(gold))
        for a, b, g in zip(v2, v3, gold):
            if a is not None and b is not None:
                joint += 1
                if a != b:
                    dis += 1
                    if g == 0:
                        dis_gold0 += 1
                    else:
                        dis_gold1 += 1
            elif a is not None:
                v2only += 1
            elif b is not None:
                v3only += 1
    print("V5_CONFLICTS " + json.dumps({
        "joint_covered": joint, "disagree": dis,
        "disagree_gold_incorrect": dis_gold0,
        "disagree_gold_correct": dis_gold1,
        "v2_only": v2only, "v3_only": v3only}), flush=True)


if __name__ == "__main__":
    main()
