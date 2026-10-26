"""Part C sweep: two cascade analyses from cached dual-verifier scores.

Input: partC_scores.py output (per problem: chosen + multipath traces, each
with DiscPRM and generative-verifier scores and gold_final).

ANALYSIS 1 (primary) — verification cascade over individual traces.
  DiscPRM verdict: 1 iff disc_min >= 0.5. Escalate to the generative
  verifier when DiscPRM is unsure: |disc_min - 0.5| < delta. Final verdict
  = generative Final when escalated, else the DiscPRM verdict.
  Cost per trace (paper formula 2*T*P, P = 1.5e9): DiscPRM = one forward
  per step (T=1 each); generative = gen_tokens when escalated.
  Sweep delta -> accuracy vs mean FLOPs Pareto; report the cheapest point
  matching generative-only accuracy, with bootstrap CIs.

ANALYSIS 2 (secondary) — Best-of-N selection cascade (the scaffold design),
  on problems whose candidate traces mix fully-correct and incorrect:
  accept DiscPRM top-1 unless margin < tau or softmax entropy > eta;
  on escalation verify the top-3 with the generative verifier and pick by
  (gen_score, gen_final, gen_conf).
"""
import argparse
import json
import math

import numpy as np

P = 1.5e9


def entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs)


def gen_rank(tr):
    return (tr["gen_score"], tr["gen_final"], tr["gen_conf"])


def verification_cascade(traces, deltas):
    n = len(traces)
    disc_verd = np.array([1 if t["disc_min"] >= 0.5 else 0 for t in traces])
    gen_verd = np.array([t["gen_final"] for t in traces])
    gold = np.array([t["gold_final"] for t in traces])
    discmin = np.array([t["disc_min"] for t in traces])
    disc_flops = np.array([2 * P * t["n_steps"] for t in traces])
    gen_flops = np.array([2 * P * t["gen_tokens"] for t in traces])
    refs = {"n_traces": n,
            "disc_only_acc": float((disc_verd == gold).mean()),
            "gen_only_acc": float((gen_verd == gold).mean())}
    points = []
    for d in deltas:
        esc = np.abs(discmin - 0.5) < d
        final = np.where(esc, gen_verd, disc_verd)
        fl = disc_flops + np.where(esc, gen_flops, 0.0)
        points.append({"delta": d, "accuracy": float((final == gold).mean()),
                         "flops_per_trace": float(fl.mean()),
                         "gen_call_rate": float(esc.mean())})
    front, best = [], -1.0
    for p in sorted(points, key=lambda x: x["flops_per_trace"]):
        if p["accuracy"] > best:
            front.append(p)
            best = p["accuracy"]
    feas = [p for p in front if p["accuracy"] >= refs["gen_only_acc"] - 1e-9]
    best_c = min(feas, key=lambda p: p["flops_per_trace"]) if feas else None
    out = {"references": refs, "pareto": front, "best_cascade": best_c}
    if best_c is not None:
        rng = np.random.default_rng(0)
        d = best_c["delta"]
        esc = np.abs(discmin - 0.5) < d
        final = np.where(esc, gen_verd, disc_verd)
        boots_c, boots_g = [], []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            boots_c.append((final[idx] == gold[idx]).mean())
            boots_g.append((gen_verd[idx] == gold[idx]).mean())
        out["best_cascade_ci95"] = [float(np.percentile(boots_c, 2.5)),
                                    float(np.percentile(boots_c, 97.5))]
        out["gen_only_ci95"] = [float(np.percentile(boots_g, 2.5)),
                                float(np.percentile(boots_g, 97.5))]
    return out


def disc_order(traces):
    """Deterministic DiscPRM ranking: (disc_min, disc_mean) descending.

    disc_min ties are rampant within a problem (traces share the weakest
    step), and a naive argsort()[::-1] breaks ties toward the LAST trace —
    which, given the error-target sampling order, is systematically the
    most-corrupted candidate (it scored 0.533, below random, in v1 of this
    sweep). The tuple key + stable sort makes the ranking principled and
    reproducible."""
    return sorted(range(len(traces)),
                  key=lambda i: (traces[i]["disc_min"],
                                 traces[i]["disc_mean"]),
                  reverse=True)


def selection_cascade(problems):
    sel = [p for p in problems
           if len(p["traces"]) >= 4
           and 0 < sum(t["gold_final"] for t in p["traces"]) < len(p["traces"])]
    n = len(sel)
    if n == 0:
        return {"n_problems": 0}
    rand_acc = float(np.mean([np.mean([t["gold_final"] for t in p["traces"]])
                              for p in sel]))
    disc_acc = float(np.mean([
        p["traces"][disc_order(p["traces"])[0]]["gold_final"] for p in sel]))
    gen_acc = float(np.mean([
        max(p["traces"], key=gen_rank)["gold_final"] for p in sel]))
    gen_flops = float(np.mean([
        sum(2 * P * t["n_steps"] + 2 * P * t["gen_tokens"]
            for t in p["traces"]) / 1 for p in sel]))
    refs = {"n_problems": n, "random_acc": rand_acc, "disc_only_acc": disc_acc,
            "gen_only_acc": gen_acc, "gen_only_flops_per_problem": gen_flops}
    points = []
    for tau in [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 1.01]:
        for eta in [0.3, 0.6, 1.0, 1.4, 1.8, 99.0]:
            correct, flops, calls, n_esc = 0, 0.0, 0, 0
            for p in sel:
                trs = p["traces"]
                ds = np.array([t["disc_min"] for t in trs])
                order = disc_order(trs)
                margin = ds[order[0]] - ds[order[1]]
                probs = np.exp(ds) / np.exp(ds).sum()
                base = sum(2 * P * t["n_steps"] for t in trs)
                if margin < tau or entropy(probs) > eta:
                    n_esc += 1
                    topk = list(order[:3])
                    pick = max(topk, key=lambda i: gen_rank(trs[i]))
                    flops += base + sum(2 * P * trs[i]["gen_tokens"]
                                        for i in topk)
                    calls += len(topk)
                else:
                    pick = int(order[0])
                    flops += base
                correct += trs[pick]["gold_final"]
            points.append({"tau": tau, "eta": eta, "accuracy": correct / n,
                           "flops_per_problem": flops / n,
                           "gen_calls_per_problem": calls / n,
                           "escalation_rate": n_esc / n})
    front, best = [], -1.0
    for p in sorted(points, key=lambda x: x["flops_per_problem"]):
        if p["accuracy"] > best:
            front.append(p)
            best = p["accuracy"]
    feas = [p for p in front if p["accuracy"] >= gen_acc - 1e-9]
    best_c = min(feas, key=lambda p: p["flops_per_problem"]) if feas else None
    return {"references": refs, "pareto": front, "best_cascade": best_c,
            "flops_saving_vs_gen_only": (
                gen_flops / best_c["flops_per_problem"] if best_c else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="/workspace/partC_scores.jsonl")
    ap.add_argument("--out", default="/workspace/partC_pareto.json")
    args = ap.parse_args()

    problems = [json.loads(l) for l in open(args.scores)]
    all_traces = [t for p in problems for t in p["traces"]]
    chosen = [t for p in problems for t in p["traces"] if t["is_chosen"]]

    deltas = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    ver_all = verification_cascade(all_traces, deltas)
    ver_chosen = verification_cascade(chosen, deltas)
    sel = selection_cascade(problems)

    summary = {"verification_all_traces": ver_all,
               "verification_chosen_only": ver_chosen,
               "selection": sel}
    print("PARTC_VERIFICATION_ALL " + json.dumps(ver_all))
    print("PARTC_VERIFICATION_CHOSEN " + json.dumps(ver_chosen))
    print("PARTC_SELECTION " + json.dumps(sel))
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
