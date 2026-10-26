"""Part C scoring: test traces scored by BOTH verifiers (DiscPRM + GRPO v3).

For every phase1_test problem we score:
  - the chosen (human-rated) trace, and
  - up to 10 multipath traces sampled through the PRM800K candidate pools
    (stratified target-error counts, deduped against the chosen path).
This supports two analyses (partC_sweep.py):
  1. Verification cascade over all ~330 held-out traces (primary).
  2. Best-of-N selection cascade on the problems whose candidate set mixes
     fully-correct and incorrect traces (secondary, matches the scaffold).

Stage 1 (cheap): DiscPRM sequence classifier, input exactly as trained
(problem + "\\n" + step text), P(correct) per step; trace scores = min
(weakest link) and mean over steps.
Stage 2 (expensive): GRPO v3 generative verifier, one joint generation per
trace; gen_score = fraction of steps judged correct, plus the model's Final
verdict, Confidence, generated-token count (FLOPs), and parse flag.

Output jsonl, one line per problem:
  {id, traces: [{is_chosen, gold_final, disc_min, disc_mean, gen_score,
                 gen_final, gen_conf, gen_tokens, gen_parsed}]}
"""
import argparse
import importlib.util
import json
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from rewards import build_verifier_prompt, parse_verification

import numpy as np
import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

_spec = importlib.util.spec_from_file_location(
    "build_sft_v2_data",
    os.path.join(os.path.dirname(__file__), "..", "data",
                 "build_sft_v2_data.py"))
_bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bm)
pick_path = _bm.pick_path

ERROR_TARGETS = [0, 0, 0, 0, 1, 1, 2, 2, 3, 5, 8]


def build_traces(records, n_paths=10, seed=20261022):
    rng = random.Random(seed)
    problems = []
    for d in records:
        chosen = [s["chosen_idx"] if s["chosen_idx"] is not None else 0
                  for s in d["steps"]]
        seen = {tuple(chosen)}
        paths = [(chosen, True)]
        for e in ERROR_TARGETS:
            if len(paths) > n_paths:
                break
            picks = pick_path(d["steps"], e, rng)
            if tuple(picks) in seen:
                continue
            seen.add(tuple(picks))
            paths.append((picks, False))
        traces = []
        for picks, is_chosen in paths:
            texts = [d["steps"][i]["candidates"][j]["text"]
                     for i, j in enumerate(picks)]
            gold = [d["steps"][i]["candidates"][j]["gold"]
                    for i, j in enumerate(picks)]
            traces.append({"texts": texts, "gold": gold,
                           "is_chosen": is_chosen, "n_steps": len(texts),
                           "gold_final": 1 if all(g == 1 for g in gold) else 0})
        problems.append({"id": d["id"], "problem": d["problem"],
                         "traces": traces})
    return problems


@torch.no_grad()
def disc_scores(model, tok, problems, batch_size=32, device="cuda"):
    jobs, texts = [], []
    for pi, p in enumerate(problems):
        for ti, t in enumerate(p["traces"]):
            for si, s in enumerate(t["texts"]):
                jobs.append((pi, ti, si))
                texts.append(p["problem"] + "\n" + s)
    probs = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        logits = model(**enc).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    for (pi, ti, si), pr in zip(jobs, probs):
        problems[pi]["traces"][ti].setdefault("step_probs", []).append(pr)
    for p in problems:
        for t in p["traces"]:
            t["disc_min"] = float(min(t["step_probs"]))
            t["disc_mean"] = float(np.mean(t["step_probs"]))
            del t["step_probs"]


@torch.no_grad()
def gen_scores(model, tok, problems, batch_size=8, max_new_tokens=1024):
    jobs, prompts = [], []
    for pi, p in enumerate(problems):
        for ti, t in enumerate(p["traces"]):
            jobs.append((pi, ti))
            prompts.append(build_verifier_prompt(p["problem"], t["texts"]))
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s:s + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        for j, (pi, ti) in enumerate(jobs[s:s + batch_size]):
            ntok = int((gen[j][plen:] != tok.pad_token_id).sum())
            text = tok.decode(gen[j][plen:], skip_special_tokens=True)
            pr = parse_verification(text)
            t = problems[pi]["traces"][ti]
            if pr is None:
                t.update(gen_score=0.0, gen_final=0, gen_conf=0.0,
                         gen_tokens=ntok, gen_parsed=0)
            else:
                gold = t["gold"]
                frac = float(np.mean([pr["judgments"].get(i + 1) == g
                                      for i, g in enumerate(gold)]))
                t.update(gen_score=frac, gen_final=pr["final"],
                         gen_conf=pr["confidence"], gen_tokens=ntok,
                         gen_parsed=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc_model", required=True)
    ap.add_argument("--gen_model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--out", default="/workspace/partC_scores.jsonl")
    ap.add_argument("--n_paths", type=int, default=10)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data)]
    problems = build_traces(records, n_paths=args.n_paths)
    n_traces = sum(len(p["traces"]) for p in problems)
    n_steps = sum(len(t["texts"]) for p in problems for t in p["traces"])
    print(f"{len(problems)} problems, {n_traces} traces, {n_steps} steps")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtok = AutoTokenizer.from_pretrained(args.disc_model,
                                         trust_remote_code=True)
    if dtok.pad_token is None:
        dtok.pad_token = dtok.eos_token
    dmodel = AutoModelForSequenceClassification.from_pretrained(
        args.disc_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)
    disc_scores(dmodel, dtok, problems, device=device)
    del dmodel
    torch.cuda.empty_cache()
    print("disc scoring done")

    gtok = AutoTokenizer.from_pretrained(args.gen_model,
                                         trust_remote_code=True)
    if gtok.pad_token is None:
        gtok.pad_token = gtok.eos_token
    gtok.padding_side = "left"
    gmodel = AutoModelForCausalLM.from_pretrained(
        args.gen_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)
    gen_scores(gmodel, gtok, problems)
    print("gen scoring done")

    with open(args.out, "w") as f:
        for p in problems:
            f.write(json.dumps({
                "id": p["id"],
                "traces": [{k: t[k] for k in
                            ("is_chosen", "gold_final", "n_steps", "disc_min",
                             "disc_mean", "gen_score", "gen_final",
                             "gen_conf", "gen_tokens", "gen_parsed")}
                           for t in p["traces"]]}) + "\n")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
