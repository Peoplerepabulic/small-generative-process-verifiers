"""Convert Math-Shepherd (peiyi9979/Math-Shepherd on HF) to the unified format.

Verified schema (2026-10-05, live via datasets-server):
- fields: input, label, task
- input:  "<problem> Step 1: <text> ки\\nStep 2: <text> ки\\n..."  (ки = step delimiter)
- label:  "<problem> Step 1: <text> +\\nStep 2: <text> -\\n..."  (+ correct / - incorrect)

Unified output record:
    {id, source, split, problem, ground_truth_answer: null,
     finish_reason: null, steps: [{candidates: [{text, gold}], chosen_idx: null}]}
Math-Shepherd has one completion per step (no chosen/alternative split), so each
step has a single candidate. Steps that fail to parse are counted and skipped;
a record is kept only if >=1 step parses.
"""
import argparse
import json
import os
import re
from collections import Counter

STEP_RE = re.compile(r"Step\s+(\d+):\s*(.*?)\s*([+-])(?=\n|$)")


def parse_label(label: str):
    """Return (problem, [(step_text, gold), ...]). Raises on mismatch."""
    steps = [(int(m.group(1)), m.group(2).strip(), 1 if m.group(3) == "+" else 0)
             for m in STEP_RE.finditer(label)]
    if not steps:
        raise ValueError("no steps parsed")
    nums = [s[0] for s in steps]
    if nums != list(range(1, len(nums) + 1)):
        raise ValueError(f"non-sequential step numbers: {nums[:10]}")
    m0 = STEP_RE.search(label)
    problem = label[: m0.start()].strip()
    return problem, [(t, g) for _, t, g in steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mathshepherd_unified.jsonl")
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--streaming", action="store_true",
                    help="stream instead of downloading the full file "
                         "(recommended on Kaggle: only --max_records rows are fetched)")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("peiyi9979/Math-Shepherd", split="train",
                      streaming=args.streaming or args.max_records is not None)

    stats = Counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w") as out:
        for i, row in enumerate(ds):
            if args.max_records and i >= args.max_records:
                break
            try:
                problem, steps = parse_label(row["label"])
            except ValueError:
                stats["parse_failed"] += 1
                continue
            rec = {
                "id": f"mathshepherd-train-{i}",
                "source": "math-shepherd",
                "split": "train",
                "problem": problem,
                "ground_truth_answer": None,
                "finish_reason": None,
                "steps": [{"candidates": [{"text": t, "gold": g}],
                           "chosen_idx": 0} for t, g in steps],
            }
            for _, g in steps:
                stats[f"gold_{g}"] += 1
            stats["records"] += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("stats:", dict(stats))
    print(f"wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
