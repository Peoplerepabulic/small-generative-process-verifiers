"""Part B: quantify step label interference with counterfactuals.

The paper (§5.8) names the phenomenon but only describes it qualitatively:
"if the PRM judges a particular step as incorrect, it becomes more likely to
label subsequent steps as incorrect even if they are not."

Design (counterfactual, within-solution):
  For a solution with gold labels and first error at step k (or a fully-correct
  solution where we *inject* an error at step k using a -1 candidate from
  PRM800K's candidate pool):
    - JOINT mode:      one verification generation judges all steps (standard).
    - INDEPENDENT mode: one verification call per step; step i is judged from
                       the problem + steps 1..i only (no later context, and the
                       model never sees its own earlier judgments).
  interference_gap = acc_independent(post-k steps) - acc_joint(post-k steps)
  Positive gap => the early "incorrect" judgment dragged down later judgments.

  Counterfactual pairing: the SAME solution is evaluated (a) as-is and (b) with
  step k flipped (inject or repair). The difference-in-differences between (a)
  and (b) isolates the causal effect of the step-k judgment.

Outputs: interference gap with bootstrap 95% CI + a plot-ready JSON.
GPU needed only for the inference step; dataset construction runs on CPU.
"""
import argparse
import json
import random

from rewards import build_verifier_prompt


def build_counterfactual_pairs(unified_paths, seed=0, max_n=None):
    """Yield (problem, steps, gold, k, variant) where variant in {'orig','flipped'}.

    Uses PRM800K candidate pools: at a chosen step k, swap the chosen candidate
    with an opposite-gold candidate from the same step's pool.
    """
    rng = random.Random(seed)
    pairs = []
    for path in unified_paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["source"] != "prm800k":
                    continue
                steps = d["steps"]
                # candidate step indices that have BOTH gold values in the pool
                flippable = [i for i, s in enumerate(steps)
                             if {c["gold"] for c in s["candidates"]} == {0, 1}]
                if not flippable:
                    continue
                k = rng.choice(flippable)
                base_texts = [s["candidates"][s["chosen_idx"] or 0]["text"] for s in steps]
                base_gold = [s["candidates"][s["chosen_idx"] or 0]["gold"] for s in steps]
                # flipped variant: swap step k to the opposite-gold candidate
                alt = [c for c in steps[k]["candidates"]
                       if c["gold"] != base_gold[k]][0]
                flip_texts = base_texts.copy(); flip_texts[k] = alt["text"]
                flip_gold = base_gold.copy(); flip_gold[k] = alt["gold"]
                pairs.append({"problem": d["problem"], "id": d["id"], "k": k,
                              "variants": {"orig": (base_texts, base_gold),
                                           "flipped": (flip_texts, flip_gold)}})
                if max_n and len(pairs) >= max_n:
                    return pairs
    return pairs


def independent_prompts(problem, steps):
    """One prompt per step: judge step i given problem + steps 1..i."""
    prompts = []
    for i in range(len(steps)):
        ctx = "\n".join(f"Step {j + 1}: {s}" for j, s in enumerate(steps[:i]))
        prompts.append(
            "Verify ONLY the last step below. Earlier steps are given as context.\n\n"
            f"Problem: {problem}\n\nContext:\n{ctx}\n\n"
            f"Step to verify:\nStep {i + 1}: {steps[i]}\n\n"
            "Respond in exactly this format:\n"
            f"Step {i + 1}: <correct|incorrect>\n"
            "Final: <correct|incorrect>\n"
            "Confidence: <0.00-1.00>"
        )
    return prompts


def run_interference(model_path, pairs, seed=0):
    """Run joint vs independent verification. Returns per-pair gap records.

    Requires GPU (vLLM). The gap for pair p:
        gap_p = mean_acc_independent(steps k+1..) - mean_acc_joint(steps k+1..)
    averaged over both variants (orig/flipped) and difference-in-differences
    reported separately.
    """
    from vllm import LLM, SamplingParams
    from rewards import parse_verification
    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.0, max_tokens=2048)

    records = []
    for p in pairs:
        for variant, (texts, gold) in p["variants"].items():
            # JOINT
            out = llm.generate([build_verifier_prompt(p["problem"], texts)], sp)[0]
            pj = parse_verification(out.outputs[0].text)
            # INDEPENDENT
            outs = llm.generate(independent_prompts(p["problem"], texts), sp)
            post = list(range(p["k"] + 1, len(texts)))  # 0-indexed steps after k
            if pj is None or not post:
                continue
            acc_joint = sum(pj["judgments"].get(i + 1) == gold[i] for i in post) / len(post)
            acc_ind, n = 0, 0
            for idx, i in enumerate(post):
                pi = parse_verification(outs[idx].outputs[0].text)
                if pi is None:
                    continue
                acc_ind += (pi["judgments"].get(i + 1) == gold[i]); n += 1
            if n == 0:
                continue
            records.append({"id": p["id"], "k": p["k"], "variant": variant,
                            "gap": acc_ind / n - acc_joint, "n_post": len(post)})
    return records


def summarize(records, n_boot=1000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    gaps = np.array([r["gap"] for r in records])
    if len(gaps) == 0:
        return {"n": 0}
    boots = [rng.choice(gaps, size=len(gaps), replace=True).mean() for _ in range(n_boot)]
    return {
        "n": len(gaps),
        "interference_gap_mean": float(gaps.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "frac_positive": float((gaps > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="unified jsonl paths")
    ap.add_argument("--model", default=None, help="verifier checkpoint; omit for dataset-only mode")
    ap.add_argument("--max_n", type=int, default=500)
    ap.add_argument("--out", default="outputs/partB_interference.json")
    args = ap.parse_args()

    pairs = build_counterfactual_pairs(args.data, max_n=args.max_n)
    print(f"{len(pairs)} counterfactual pairs built")
    json.dump([{"id": p["id"], "k": p["k"]} for p in pairs],
              open(args.out.replace(".json", "_pairs.json"), "w"), indent=1)

    if args.model:
        records = run_interference(args.model, pairs)
        summary = summarize(records)
        print("interference summary:", json.dumps(summary, indent=2))
        json.dump({"summary": summary, "records": records}, open(args.out, "w"), indent=1)
    else:
        print("dataset-only mode: run again with --model <ckpt> to compute gaps")


if __name__ == "__main__":
    main()
