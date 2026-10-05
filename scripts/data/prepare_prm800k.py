"""Download PRM800K and convert to the unified chain-preserving format.

Verified schema (2026-10-05, live download):
- Source files (Git LFS! raw.githubusercontent.com serves only the LFS *pointer*;
  use the media.githubusercontent.com URL below):
    https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/{phase1_train,phase1_test,phase2_train,phase2_test}.jsonl
- One JSON object per line = one question with ONE labeled step-by-step trace:
    question: {problem, ground_truth_answer}
    label: {steps: [{completions: [{text, rating, flagged}, ...] | null,
                     human_completion, chosen_completion}],
            total_time, finish_reason}
- rating: 1 = positive, 0 = neutral, -1 = negative, null = unlabeled.

Unified output record:
    {id, source, split, problem, ground_truth_answer, finish_reason,
     steps: [{candidates: [{text, gold}], chosen_idx}]}
where gold: 1 = correct (rating 1 or 0), 0 = incorrect (rating -1).
Candidates with rating null are dropped. chosen_idx may be null.

IMPORTANT: unlike the original ThinkPRM-style rejection sampling (which trains only
on the chosen completions), we keep EVERY candidate completion. Measured on
phase1_train (2026-10-05): chosen-only completions are 100% positive (12,961/12,961),
while all completions are ~40% negative (19,656/49,494). The negatives are exactly
the data rejection sampling throws away -- and exactly what GRPO trains on.
"""
import argparse
import json
import os
import urllib.request
from collections import Counter

BASE = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data"
SPLITS = ["phase1_train", "phase1_test", "phase2_train", "phase2_test"]

RATING_TO_GOLD = {1: 1, 0: 1, -1: 0}  # neutral counts as correct (standard practice)


def download(split: str, raw_dir: str) -> str:
    os.makedirs(raw_dir, exist_ok=True)
    url = f"{BASE}/{split}.jsonl"
    path = os.path.join(raw_dir, f"{split}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"[skip] {path} already downloaded")
        return path
    print(f"[dl] {url}")
    urllib.request.urlretrieve(url, path)
    # guard against accidentally fetching the LFS pointer instead of the data
    with open(path) as f:
        head = f.read(60)
    assert not head.startswith("version https://git-lfs"), (
        f"{path} looks like a Git LFS pointer -- use the media.githubusercontent.com URL")
    return path


def convert(path: str, split: str):
    stats = Counter()
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            q = d["question"]
            steps = []
            for s in d["label"]["steps"]:
                comps = s.get("completions") or []
                cands = []
                chosen_idx = s.get("chosen_completion")
                kept = 0
                for j, c in enumerate(comps):
                    r = c.get("rating")
                    if r not in RATING_TO_GOLD:
                        stats["dropped_null_rating"] += 1
                        if j == chosen_idx:
                            chosen_idx = None  # chosen candidate was dropped
                        continue
                    cands.append({"text": c["text"], "gold": RATING_TO_GOLD[r]})
                    stats[f"gold_{RATING_TO_GOLD[r]}"] += 1
                    if j == chosen_idx:
                        chosen_idx = kept
                    kept += 1
                if not cands:
                    stats["empty_steps"] += 1
                    continue
                steps.append({"candidates": cands, "chosen_idx": chosen_idx})
            if not steps:
                stats["empty_records"] += 1
                continue
            stats["records"] += 1
            yield {
                "id": f"prm800k-{split}-{i}",
                "source": "prm800k",
                "split": split,
                "problem": q["problem"],
                "ground_truth_answer": q.get("ground_truth_answer"),
                "finish_reason": d["label"].get("finish_reason"),
                "steps": steps,
            }
    print(f"[{split}] stats:", dict(stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prm800k_unified.jsonl")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w") as out:
        for split in args.splits:
            path = download(split, args.raw_dir)
            for rec in convert(path, split):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
