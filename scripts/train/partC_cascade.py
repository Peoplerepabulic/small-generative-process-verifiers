"""Part C: adaptive verification cascade + accuracy-vs-FLOPs Pareto frontier.

Pipeline:
  Stage 1 (cheap): DiscPRM (or short-CoT verifier) scores N candidates.
  Routing:  if top-1 margin < tau OR score entropy > eta -> escalate top-k to
            the expensive generative verifier (ThinkPRM-style); else accept.
  Dynamic K: disagreement across sampled CoTs -> larger K / extra thinking round.

  FLOPs accounting reuses the paper's formula (§F.8): FLOPs ≈ 2*T*P
  (T = generated tokens, P = params). DiscPRM ≈ one forward pass (T=1).

This script works OFFLINE on cached score files, so the τ/η sweep is instant:
  1. First produce scores:
       python scripts/evaluation/evaluate.py --model <discprm> ... # extend to dump scores
     (or any script that writes {"id":..., "disc_scores":[...], "gen_scores":[...],
      "gen_tokens":[...], "gold":...} per problem)
  2. Then sweep:
       python scripts/train/partC_cascade.py --scores scores.jsonl --out outputs/pareto.json

Outputs: Pareto frontier points (total FLOPs, accuracy) + the τ/η operating point
that matches full-ThinkPRM accuracy at minimal FLOPs.
"""
import argparse
import json
import math

import numpy as np


def flops_approx(n_tokens: float, n_params: float) -> float:
    return 2.0 * n_tokens * n_params  # paper §F.8


def entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs)


def cascade_decision(disc_scores, tau, eta):
    """Return (accept_idx, escalate: bool)."""
    order = np.argsort(disc_scores)[::-1]
    top1, top2 = disc_scores[order[0]], disc_scores[order[1]] if len(order) > 1 else 0.0
    margin = top1 - top2
    probs = np.exp(disc_scores) / np.exp(disc_scores).sum()
    if margin < tau or entropy(probs) > eta:
        return order[0], True
    return order[0], False


def evaluate_frontier(problems, taus, etas, p_disc, p_gen, topk=3):
    """problems: list of dicts with disc_scores, gen_scores, gen_tokens, gold_idx."""
    points = []
    for tau in taus:
        for eta in etas:
            correct, flops = 0, 0.0
            n_escalated = 0
            for pr in problems:
                ds = np.array(pr["disc_scores"])
                accept, esc = cascade_decision(ds, tau, eta)
                flops += flops_approx(1, p_disc) * len(ds)  # stage-1: 1 fwd pass each
                if esc:
                    n_escalated += 1
                    order = np.argsort(ds)[::-1][:topk]
                    gs = np.array(pr["gen_scores"])
                    toks = np.array(pr["gen_tokens"])
                    accept = order[int(np.argmax(gs[order]))]
                    flops += flops_approx(float(toks[order].mean()), p_gen) * topk
                correct += (accept == pr["gold_idx"])
            points.append({"tau": tau, "eta": eta,
                           "accuracy": correct / len(problems),
                           "flops": flops,
                           "escalation_rate": n_escalated / len(problems)})
    return points


def pareto(points):
    pts = sorted(points, key=lambda p: p["flops"])
    front, best = [], -1.0
    for p in pts:
        if p["accuracy"] > best:
            front.append(p); best = p["accuracy"]
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="jsonl with disc/gen scores per problem")
    ap.add_argument("--p_disc", type=float, default=1.5e9)
    ap.add_argument("--p_gen", type=float, default=1.5e9)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default="outputs/pareto.json")
    args = ap.parse_args()

    problems = [json.loads(l) for l in open(args.scores)]
    taus = [0.01, 0.05, 0.1, 0.2, 0.5]
    etas = [0.5, 1.0, 1.5, 2.0]

    points = evaluate_frontier(problems, taus, etas, args.p_disc, args.p_gen, args.topk)
    front = pareto(points)

    # reference: full-generative on everything
    full_acc = float(np.mean([int(np.argmax(p["gen_scores"])) == p["gold_idx"]
                                for p in problems]))
    full_flops = sum(flops_approx(float(np.mean(p["gen_tokens"])), args.p_gen) * len(p["gen_scores"])
                     for p in problems)

    # cheapest cascade point matching full-generative accuracy
    feas = [p for p in front if p["accuracy"] >= full_acc - 1e-9]
    best = min(feas, key=lambda p: p["flops"]) if feas else None

    print(f"full-generative: acc={full_acc:.3f} flops={full_flops:.3e}")
    if best:
        print(f"best cascade:  acc={best['accuracy']:.3f} flops={best['flops']:.3e} "
              f"(tau={best['tau']}, eta={best['eta']}, esc_rate={best['escalation_rate']:.2f}) "
              f"-> {full_flops / best['flops']:.1f}x cheaper")
    json.dump({"pareto": front, "full": {"accuracy": full_acc, "flops": full_flops},
               "best_cascade": best}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
