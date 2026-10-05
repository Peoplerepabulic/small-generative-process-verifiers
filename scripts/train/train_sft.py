"""SFT baseline: the rejection-sampling proxy (ThinkPRM-style).

Trains the verifier to *imitate* gold verifications, using only
agreement-filtered data:
- PRM800K: chosen completions only (all steps rated positive) -> the model only
  ever sees "correct" judgments, exactly like rejection sampling which discards
  disagreeing chains.
- Math-Shepherd: only solutions whose every step is labeled "+".

Target completion is the gold verification in the exact format from rewards.py.
Compare against GRPO (train_grpo.py), which trains on ALL chains including
negatives, with an explicit calibration reward.
"""
import argparse
import json
import yaml

from datasets import Dataset
from rewards import build_verifier_prompt


def gold_completion(gold):
    lines = [f"Step {i + 1}: {'correct' if g == 1 else 'incorrect'}"
             for i, g in enumerate(gold)]
    first_err = next((str(i + 1) for i, g in enumerate(gold) if g == 0), "none")
    final = "correct" if all(g == 1 for g in gold) else "incorrect"
    # SFT has no learned confidence; use the empirical label as the target
    # (temperature scaling on top of this is baseline #3 in the proposal).
    conf = "0.95" if final == "correct" else "0.05"
    lines += [f"First error step: {first_err}", f"Final: {final}", f"Confidence: {conf}"]
    return "\n".join(lines)


def build_examples(paths, agreement_only=True):
    examples = []
    for path in paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                step_texts, gold = [], []
                for s in d["steps"]:
                    cands = s["candidates"]
                    ci = s["chosen_idx"] if s["chosen_idx"] is not None else 0
                    step_texts.append(cands[ci]["text"])
                    gold.append(cands[ci]["gold"])
                if agreement_only and not all(g == 1 for g in gold):
                    continue  # rejection sampling: discard disagreeing chains
                prompt = build_verifier_prompt(d["problem"], step_texts)
                examples.append({"prompt": prompt, "completion": gold_completion(gold)})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_1p5b.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from trl import SFTConfig, SFTTrainer

    examples = build_examples(cfg["data"]["train_paths"])
    ds = Dataset.from_list(examples)
    print(f"{len(ds)} SFT examples (agreement-filtered)")

    tcfg = cfg["training"]
    sfcfg = SFTConfig(
        output_dir=cfg["output_dir"],
        model_name_or_path=cfg["model"],
        max_length=tcfg["max_length"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=tcfg["num_train_epochs"],
        bf16=True,
        logging_steps=tcfg.get("logging_steps", 10),
        save_steps=tcfg.get("save_steps", 200),
        report_to=tcfg.get("report_to", "none"),
        seed=cfg.get("seed", 0),
    )

    def formatting_fn(ex):
        return [p + "\n" + c for p, c in zip(ex["prompt"], ex["completion"])]

    trainer = SFTTrainer(model=cfg["model"], args=sfcfg, train_dataset=ds,
                         formatting_func=formatting_fn)
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
