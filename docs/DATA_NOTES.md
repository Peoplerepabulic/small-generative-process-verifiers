# Data notes — verified live 2026-10-05

## PRM800K (openai/prm800k)

- Files: `prm800k/data/{phase1_train,phase1_test,phase2_train,phase2_test}.jsonl`
- **Git LFS trap**: `raw.githubusercontent.com/...` serves only the 132-byte LFS *pointer*.
  Real data URL: `https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/<file>.jsonl`
  (`prepare_prm800k.py` asserts on this).
- Record = one question + ONE labeled step trace:
  `question: {problem, ground_truth_answer}`,
  `label: {steps: [{completions: [{text, rating, flagged}] | null, human_completion, chosen_completion}], total_time, finish_reason}`
- `rating`: 1 = positive, 0 = neutral, -1 = negative, null = unlabeled.
- phase1_train measured: 949 questions → 931 usable records, 49,494 step-completions.
  Distribution (all completions): **1: 18,782 / 0: 10,234 / -1: 19,656 / null: 822**.
  Neutral (0) is mapped to correct, matching DuaShepherd's convention.
- **Chosen-only completions are 100% positive** (12,961/12,961). This is why a
  rejection-sampling SFT baseline never sees a negative example, and why keeping
  all completions (≈40% negatives) is the core data advantage of Part A.
- ⚠️ **Correction to the proposal**: the "92/8 label imbalance" figure does NOT hold
  for PRM800K raw data (measured ~60/40 correct/incorrect when keeping all
  completions). It presumably describes ThinkPRM's *synthetic* chains. The GRPO
  config therefore uses `incorrect_weight: 2.0`, not 5–10×. Re-measure if the data
  mix changes (e.g. after adding synthetic QwQ chains).

## Math-Shepherd (peiyi9979/Math-Shepherd, HuggingFace)

- `load_dataset("peiyi9979/Math-Shepherd", split="train")`; fields: `input`, `label`, `task`.
- `input`: `<problem> Step 1: <text> ки\nStep 2: <text> ки\n...` ("ки" = step delimiter artifact).
- `label`: same but each step ends with ` +` (correct) or ` -` (incorrect).
- One completion per step (no candidate pool, no chosen split) → each step becomes a
  single-candidate record in the unified format.
- Unlike PRM800K, solutions contain naturally mixed +/- labels (Monte-Carlo estimated),
  so no error injection is needed for these records in GRPO.
- **Validated 2026-10-05 on 2,283 real rows**: 99.9% parse success (2,281/2,283),
  avg 3.73 steps/record, **70.8% of steps labeled incorrect**. The 2 failures are
  genuine data noise (a step missing its +/- terminator; unnumbered-step variant) —
  the parser's sequential-number check correctly rejects them and the script counts skips.

## Unified format (scripts/data/*.py output)

```json
{"id": "...", "source": "prm800k | math-shepherd", "split": "...",
 "problem": "...", "ground_truth_answer": "...|null", "finish_reason": "...|null",
 "steps": [{"candidates": [{"text": "...", "gold": 1|0}], "chosen_idx": 0|null}]}
```
`gold`: 1 = correct, 0 = incorrect.
