# What Breaks in Small Generative Process Verifiers

Code accompanying the paper **"What Breaks in Small Generative Process Verifiers:
Verdict Aggregation, Calibration Limits, and Discriminative–Generative Cascades"**
(Qiwei Li, Washington University in St. Louis; arXiv preprint, 2026).

We study a 1.5B generative process verifier (DeepSeek-R1-Distill-Qwen-1.5B,
fixed step-judgment format, no free-form critique) on PRM800K, and find that
its failures lie mostly where accuracy numbers do not point:

- **Step accuracy is a base-rate mirage.** A trivial all-"correct" judge scores
  0.904 on the test steps; the SFT verifier scores 0.897 while flagging 15 of
  1,857 steps, one correctly. GRPO cannot fix a policy that never varies its
  verdict (zero within-group advantage); training on multipath-consistent
  traces does (solution accuracy 0.417 → 0.583).
- **The verifier knows more than it says.** Verdict-token probabilities reach
  step AUROC 0.81, and noisy-OR aggregation over them gives 0.883 solution
  accuracy at ECE 0.092 with no post-hoc fitting, while the emitted verbalized
  confidence is two-valued by target design and stays miscalibrated.
- **Step-label interference is real but small.** Teacher-forcing an "incorrect"
  judgment at step *k* leaves later hard verdicts unchanged, and raises later
  steps' P(incorrect) by about one probability point on the continuous
  token-probability readout.
- **Cascades trade cost for accuracy.** Escalating only the traces a
  discriminative PRM is unsure about to the token-probability readout reaches
  0.820 accuracy on 891 traces while calling the generative verifier on 20.5%
  of them — significantly above the discriminative PRM alone (0.744) and level
  within noise with the readout on every trace (0.773).

The official ThinkPRM-1.5B checkpoint is evaluated as an external reference:
its prose verdicts parse for only 21.6% of steps (a non-random subset), and
two reasonable parsers disagree on 21 of those steps outright.

## Repo layout

```
├── configs/                 # YAML configs (DiscPRM, SFT, GRPO, pilot)
├── docs/                    # DATA_NOTES.md (verified data facts),
│                            # THINKPRM_REPRO_SPEC.md, WEEK1-2.md
├── notebooks/               # week1_kickoff.ipynb
├── scripts/
│   ├── data/                # download + unify PRM800K / Math-Shepherd
│   ├── train/               # rewards.py; DiscPRM / SFT / GRPO trainers;
│   │                        # partB_interference.py; partC_cascade.py
│   └── evaluation/          # step/solution eval; token-probability readout;
│                            # official-baseline dump + parsers; prefix forcing;
│                            # Part C scores/sweep; revision statistics
├── requirements.txt
└── LICENSE
```

## Pipeline

```bash
pip install -r requirements.txt

# 1. download + unify data (keeps ALL rated candidate chains, not only chosen)
python scripts/data/prepare_prm800k.py --out data/prm800k_unified.jsonl
python scripts/data/prepare_mathshepherd.py --out data/mathshepherd_unified.jsonl

# 2. train the discriminative baseline and the generative verifiers
python scripts/train/train_discprm.py --config configs/discprm_1p5b.yaml
python scripts/train/train_sft.py     --config configs/sft_1p5b.yaml
python scripts/train/train_grpo.py    --config configs/grpo_1p5b.yaml

# 3. evaluate (step/solution accuracy, localization, ECE / Brier)
python scripts/evaluation/evaluate.py --model <checkpoint>
```

Part B (counterfactual step swaps, prefix forcing) lives in
`scripts/train/partB_interference.py`; the Part C cascade sweep
(verification + Best-of-N selection, accuracy-vs-FLOPs Pareto) is
`scripts/train/partC_cascade.py`, which works offline on cached score files.

Data notes that matter (PRM800K Git LFS trap, label distributions, the
unified all-candidates format) are documented in `docs/DATA_NOTES.md`.

## Citation

If you use this code, please cite the paper (arXiv link to be added on release):

> Qiwei Li. *What Breaks in Small Generative Process Verifiers: Verdict
> Aggregation, Calibration Limits, and Discriminative–Generative Cascades.*
> arXiv preprint, 2026.
