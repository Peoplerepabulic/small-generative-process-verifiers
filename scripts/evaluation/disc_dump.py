"""Revision dump for DiscPRM: per-step probabilities on chosen test traces."""
import argparse
import json

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data)]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    jobs, texts = [], []
    for ri, d in enumerate(records):
        for si, s in enumerate(d["steps"]):
            ci = s["chosen_idx"] if s["chosen_idx"] is not None else 0
            jobs.append((ri, si))
            texts.append(d["problem"] + "\n" + s["candidates"][ci]["text"])
    probs = []
    with torch.no_grad():
        for s in range(0, len(texts), args.batch_size):
            enc = tok(texts[s:s + args.batch_size], return_tensors="pt",
                      padding=True, truncation=True, max_length=1024).to(device)
            logits = model(**enc).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    per = {}
    for (ri, si), p in zip(jobs, probs):
        per.setdefault(ri, []).append(p)
    with open(args.out, "w") as f:
        for ri, d in enumerate(records):
            gold = [s["candidates"][s["chosen_idx"] if s["chosen_idx"]
                                    is not None else 0]["gold"]
                    for s in d["steps"]]
            f.write(json.dumps({"id": d["id"], "gold_steps": gold,
                                "step_probs": per[ri]}) + "\n")
    print(f"dumped {len(records)} problems -> {args.out}")


if __name__ == "__main__":
    main()
