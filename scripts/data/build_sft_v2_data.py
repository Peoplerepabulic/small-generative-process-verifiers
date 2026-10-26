"""Build the SFT v2 training files (consistency-enforced, pattern-diverse).

Diagnosis behind v2 (2026-10-21): SFT v1's targets were already derived from
gold step labels, yet the trained model degenerates to always emitting
"First error step: none / Final: correct" on PRM800K test. The training
mixture was 99.6% Math-Shepherd (avg 3.9 steps, 54% negative steps, errors
dense and early), while PRM800K test traces are long (~18 steps) with SPARSE
errors. Judgment patterns like "17 correct + 1 incorrect in the middle ->
Final: incorrect" essentially never occur in v1 training, so the summary
lines fall back to the all-correct template.

Fix: exploit the unified format's per-step candidate lists (phase1_train:
49,494 completions, ~40% negative) to construct MANY real traces per problem
with stratified target error counts e in {0,1,2,3,5,8}. Every picked
candidate's label is a real human rating of the exact text shown in the
prompt, and the target completion (built by train_sft.gold_completion)
derives First-error / Final / Confidence from the picked labels -- so
consistency holds by construction across the full pattern space.

Outputs (unified format, one candidate per step, chosen_idx=0):
  data/prm800k_multipath.jsonl    chosen path + 12 stratified paths/problem
  data/mathshepherd_sample18k.jsonl  fixed-seed 18k subsample of MS train
"""
import json
import random

SEED = 20261021
ERROR_PLAN = [(0, 4), (1, 2), (2, 2), (3, 2), (5, 1), (8, 1)]  # (target errors, reps)
MS_SAMPLE = 18000


def pick_path(steps, target_errors, rng):
    """Return list of candidate indices, aiming for target_errors negatives.

    Falls back to any candidate when the wanted label is unavailable at a
    step; the caller recomputes judgments from the actual picks, so the
    derived summary is always consistent with the shown texts.
    """
    n = len(steps)
    order = list(range(n))
    rng.shuffle(order)
    err_pos = set(order[:target_errors])
    picks = []
    for i, s in enumerate(steps):
        golds = [c["gold"] for c in s["candidates"]]
        want = 0 if i in err_pos else 1
        choices = [j for j, g in enumerate(golds) if g == want]
        if not choices:
            choices = list(range(len(golds)))
        picks.append(rng.choice(choices))
    return picks


def trace_record(base, picks, rid):
    steps = []
    for s, j in zip(base["steps"], picks):
        c = s["candidates"][j]
        steps.append({"candidates": [{"text": c["text"], "gold": c["gold"]}],
                      "chosen_idx": 0})
    return {"id": rid, "source": "prm800k-multipath",
            "split": "phase1_train_mp", "problem": base["problem"],
            "ground_truth_answer": base.get("ground_truth_answer"),
            "finish_reason": base.get("finish_reason"), "steps": steps}


def main():
    rng = random.Random(SEED)
    recs = []
    with open("data/prm800k_unified.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") == "phase1_train":
                recs.append(d)
    print(f"phase1_train problems: {len(recs)}")

    out = []
    for d in recs:
        # The original chosen path (coherent real trace) always included.
        chosen = [s["chosen_idx"] if s["chosen_idx"] is not None else 0
                  for s in d["steps"]]
        seen = {tuple(chosen)}
        out.append(trace_record(d, chosen, f"{d['id']}-chosen"))
        for e, reps in ERROR_PLAN:
            for _ in range(reps):
                picks = pick_path(d["steps"], e, rng)
                key = tuple(picks)
                if key in seen:
                    continue
                seen.add(key)
                out.append(trace_record(d, picks, f"{d['id']}-e{e}-{len(seen)}"))
    rng.shuffle(out)
    with open("data/prm800k_multipath.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_full = sum(1 for r in out
                 if all(s["candidates"][0]["gold"] == 1 for s in r["steps"]))
    neg = sum(1 for r in out for s in r["steps"]
              if s["candidates"][0]["gold"] == 0)
    tot = sum(len(r["steps"]) for r in out)
    print(f"multipath traces: {len(out)}  fully-correct: {n_full / len(out):.3f}  "
          f"negative-step rate: {neg / tot:.3f}")

    ms = []
    with open("data/mathshepherd_unified.jsonl") as f:
        for line in f:
            ms.append(line)
    rng.shuffle(ms)
    with open("data/mathshepherd_sample18k.jsonl", "w") as f:
        for line in ms[:MS_SAMPLE]:
            f.write(line)
    print(f"mathshepherd sample: {min(MS_SAMPLE, len(ms))}")


if __name__ == "__main__":
    main()
