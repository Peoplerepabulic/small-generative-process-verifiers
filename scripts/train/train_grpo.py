"""GRPO training of the generative verifier (Part A core).

Data strategy (see docs/DATA_NOTES.md for why):
- Math-Shepherd records: complete solutions with mixed +/- step labels -> used as-is.
- PRM800K records: the chosen path is ~always all-correct, so with probability
  --inject_error_prob we substitute one step with a -1-rated candidate from that
  step's own candidate pool, creating a solution with a known first error.
  (Approximation: downstream steps keep their original labels.)

Needs ~48GB VRAM (A6000) for max_completion_length=2048 + vLLM rollouts.
Pilot on Kaggle T4: reduce --max_completion_length to 512 and --num_generations to 4.
"""
import argparse
import json
import random
import yaml

from datasets import Dataset
from rewards import build_verifier_prompt, build_reward_fns


def build_examples(paths, inject_error_prob=0.5, seed=0):
    rng = random.Random(seed)
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
                # error injection for all-correct PRM800K paths
                if (d["source"] == "prm800k" and all(g == 1 for g in gold)
                        and rng.random() < inject_error_prob):
                    idxs = [i for i, s in enumerate(d["steps"])
                            if any(c["gold"] == 0 for c in s["candidates"])]
                    if idxs:
                        i = rng.choice(idxs)
                        neg = [c for c in d["steps"][i]["candidates"] if c["gold"] == 0]
                        pick = rng.choice(neg)
                        step_texts[i] = pick["text"]
                        gold[i] = 0
                gold_final = 1 if all(g == 1 for g in gold) else 0
                first_err = next((i + 1 for i, g in enumerate(gold) if g == 0), None)
                examples.append({
                    "prompt": build_verifier_prompt(d["problem"], step_texts),
                    "gold_steps": gold,
                    "gold_first_error": first_err if first_err is not None else -1,
                    "gold_final": gold_final,
                })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_1p5b.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from trl import GRPOConfig, GRPOTrainer

    examples = build_examples(cfg["data"]["train_paths"],
                              cfg["data"].get("inject_error_prob", 0.5),
                              cfg.get("seed", 0))
    ds = Dataset.from_list(examples)
    print(f"{len(ds)} GRPO examples")

    reward_fn = build_reward_fns(cfg)
    print("active rewards:", reward_fn.names)

    tcfg = cfg["training"]
    gcfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        model_name_or_path=cfg["model"],
        num_generations=tcfg["num_generations"],
        max_completion_length=tcfg["max_completion_length"],
        max_prompt_length=tcfg["max_prompt_length"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=tcfg["num_train_epochs"],
        bf16=True,
        use_vllm=tcfg.get("use_vllm", True),
        vllm_gpu_memory_utilization=tcfg.get("vllm_gpu_memory_utilization", 0.4),
        logging_steps=tcfg.get("logging_steps", 10),
        save_steps=tcfg.get("save_steps", 200),
        report_to=tcfg.get("report_to", "none"),
        seed=cfg.get("seed", 0),
    )
    trainer = GRPOTrainer(model=cfg["model"], args=gcfg,
                          train_dataset=ds, reward_funcs=reward_fn)
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
