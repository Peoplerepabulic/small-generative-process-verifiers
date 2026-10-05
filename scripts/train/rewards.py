"""Reward functions for GRPO training of the generative verifier (Part A).

The verifier must generate, for a (problem, solution-steps) prompt:
    Step 1: correct
    Step 2: incorrect
    ...
    First error step: 2        (or: none)
    Final: incorrect
    Confidence: 0.83           (probability the Final verdict is right, 0-1)

Three verifiable rewards (all computed against gold step labels):
  r_agree    per-step judgment agreement; steps whose gold label is "incorrect"
             are up-weighted (incorrect_weight) to counter label imbalance.
  r_localize bonus for correctly identifying the FIRST error step.
  r_cal      negative Brier score on (Confidence vs gold Final label):
             directly optimizes calibration.

Each function below follows the TRL reward API:
    fn(prompts, completions, **kwargs) -> list[float]
where kwargs carries per-example gold labels (gold_steps, gold_first_error,
gold_final) supplied as extra dataset columns.
"""
import re

STEP_LINE = re.compile(r"^\s*Step\s+(\d+)\s*:\s*(correct|incorrect)\s*$", re.I | re.M)
FIRST_ERR = re.compile(r"^\s*First error step\s*:\s*(\d+|none)\s*$", re.I | re.M)
FINAL = re.compile(r"^\s*Final\s*:\s*(correct|incorrect)\s*$", re.I | re.M)
CONF = re.compile(r"^\s*Confidence\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*$", re.I | re.M)


def build_verifier_prompt(problem: str, steps: list[str]) -> str:
    numbered = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    return (
        "Verify the following solution step by step. For each step, judge whether it is correct.\n\n"
        f"Problem: {problem}\n\n"
        f"Solution:\n{numbered}\n\n"
        "Respond in exactly this format:\n"
        "Step 1: <correct|incorrect>\n"
        "...\n"
        "First error step: <step number|none>\n"
        "Final: <correct|incorrect>\n"
        "Confidence: <0.00-1.00, your probability that the Final verdict is right>"
    )


def parse_verification(text: str):
    """Parse a generated verification. Returns dict or None if unparseable."""
    judgments = {}
    for m in STEP_LINE.finditer(text):
        judgments[int(m.group(1))] = 1 if m.group(2).lower() == "correct" else 0
    m = FIRST_ERR.search(text)
    first_err = None if m is None else (None if m.group(1).lower() == "none" else int(m.group(1)))
    m = FINAL.search(text)
    final = None if m is None else (1 if m.group(1).lower() == "correct" else 0)
    m = CONF.search(text)
    conf = None if m is None else float(m.group(1))
    if not judgments or final is None or conf is None:
        return None
    return {"judgments": judgments, "first_error": first_err, "final": final, "confidence": conf}


def _gold_first_error(gold_steps: list[int]):
    for i, g in enumerate(gold_steps):
        if g == 0:
            return i + 1  # 1-indexed
    return None


def make_r_agree(incorrect_weight: float = 2.0):
    def r_agree(prompts, completions, gold_steps, **kwargs):
        out = []
        for text, gold in zip(completions, gold_steps):
            p = parse_verification(text)
            if p is None:
                out.append(0.0)
                continue
            total, score = 0.0, 0.0
            for i, g in enumerate(gold, start=1):
                w = incorrect_weight if g == 0 else 1.0
                total += w
                if i in p["judgments"]:
                    score += w if p["judgments"][i] == g else -w
            out.append(score / total if total else 0.0)
        return out
    return r_agree


def make_r_localize():
    def r_localize(prompts, completions, gold_steps, **kwargs):
        out = []
        for text, gold in zip(completions, gold_steps):
            p = parse_verification(text)
            gold_err = _gold_first_error(gold)
            if p is None or gold_err is None:
                out.append(0.0)  # undefined when solution is fully correct
                continue
            out.append(1.0 if p["first_error"] == gold_err else 0.0)
        return out
    return r_localize


def make_r_cal():
    def r_cal(prompts, completions, gold_steps, **kwargs):
        out = []
        for text, gold in zip(completions, gold_steps):
            p = parse_verification(text)
            if p is None:
                out.append(0.0)
                continue
            gold_final = 1 if all(g == 1 for g in gold) else 0
            out.append(-((p["confidence"] - gold_final) ** 2))  # -Brier, in [-1, 0]
        return out
    return r_cal


def make_format_bonus():
    """Small shaping reward so the model first learns the output format."""
    def r_format(prompts, completions, **kwargs):
        return [0.2 if parse_verification(t) is not None else 0.0 for t in completions]
    return r_format


def build_reward_fns(cfg):
    """Combine the three rewards with configurable weights."""
    r = cfg["rewards"]
    fns = [
        ("r_agree", make_r_agree(r["incorrect_weight"]), r["w_agree"]),
        ("r_localize", make_r_localize(), r["w_localize"]),
        ("r_cal", make_r_cal(), r["w_cal"]),
        ("r_format", make_format_bonus(), r.get("w_format", 0.0)),
    ]
    active = [(n, f, w) for n, f, w in fns if w != 0]

    def combined(prompts, completions, **kwargs):
        totals = [0.0] * len(completions)
        for name, fn, w in active:
            for i, v in enumerate(fn(prompts, completions, **kwargs)):
                totals[i] += w * v
        return totals

    combined.names = [n for n, _, _ in active]
    return combined
