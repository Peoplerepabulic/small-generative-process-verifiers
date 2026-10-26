"""Revision dump-eval: per-problem judgments for generative verifiers.

Greedy joint generation on phase1_test; dumps everything needed for the
revision statistics (step confusion matrices, solution-level c/d, McNemar,
bootstrap CIs, temperature scaling) so they can be computed offline:
  {id, gold_steps, judgments (aligned list, null if unparsed), gen_final,
   gold_final, first_pred, first_gold, conf}
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from rewards import build_verifier_prompt, parse_verification

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_spec = importlib.util.spec_from_file_location(
    "partB_run", os.path.join(os.path.dirname(__file__), "partB_run.py"))
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)
batched_generate = _pb.batched_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data)]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    prompts, metas = [], []
    for d in records:
        texts = [s["candidates"][s["chosen_idx"] if s["chosen_idx"] is not None
                                else 0]["text"] for s in d["steps"]]
        gold = [s["candidates"][s["chosen_idx"] if s["chosen_idx"] is not None
                               else 0]["gold"] for s in d["steps"]]
        prompts.append(build_verifier_prompt(d["problem"], texts))
        metas.append((d["id"], gold))
    texts_out = batched_generate(model, tok, prompts, 1024, 8)

    n_parsed = 0
    with open(args.out, "w") as f:
        for (rid, gold), text in zip(metas, texts_out):
            pr = parse_verification(text)
            if pr is None:
                row = {"id": rid, "gold_steps": gold, "judgments": None,
                       "gen_final": None, "gold_final": int(all(
                           g == 1 for g in gold)),
                       "first_pred": None,
                       "first_gold": next((i + 1 for i, g in enumerate(gold)
                                           if g == 0), None),
                       "conf": None}
            else:
                n_parsed += 1
                judg = [pr["judgments"].get(i + 1) for i in range(len(gold))]
                row = {"id": rid, "gold_steps": gold, "judgments": judg,
                       "gen_final": pr["final"],
                       "gold_final": int(all(g == 1 for g in gold)),
                       "first_pred": pr["first_error"],
                       "first_gold": next((i + 1 for i, g in enumerate(gold)
                                           if g == 0), None),
                       "conf": pr["confidence"]}
            f.write(json.dumps(row) + "\n")
    print(f"dumped {len(metas)} problems ({n_parsed} parsed) -> {args.out}")


if __name__ == "__main__":
    main()
