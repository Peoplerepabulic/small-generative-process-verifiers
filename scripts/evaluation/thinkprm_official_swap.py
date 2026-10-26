"""ThinkPRM-1.5B on counterfactual step swaps (Part B on the real format).

For each counterfactual pair (step k swapped to an opposite-gold
candidate), generate the official model's verification of the FLIPPED
trace and compare its post-k parsed verdicts against the ORIGINAL run's
verdicts from the baseline dump (same problem id). Post-k gold labels
are carried over from the original trace, as in Part B of the paper.

Prints THINKPRM_SWAP_SUMMARY (JSON).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, HERE)

from partB_interference import build_counterfactual_pairs  # noqa: E402
from thinkprm_official_eval import (BUILTIN_PROMPT,  # noqa: E402
                                    parse_generation)


def boot_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(vals.mean()),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))], "n": int(len(vals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--base_dump",
                    default="/workspace/thinkprm_official_dump.jsonl")
    ap.add_argument("--prompt_template", default=None)
    ap.add_argument("--out",
                    default="/workspace/thinkprm_swap_dump.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if args.prompt_template and os.path.exists(args.prompt_template):
        template = open(args.prompt_template).read()
        prompt_src = "official_file"
    else:
        template = BUILTIN_PROMPT
        prompt_src = "builtin"

    base = {}
    for line in open(args.base_dump):
        d = json.loads(line)
        base[d["id"]] = d

    pairs = build_counterfactual_pairs([args.data], max_n=200)
    pairs = [p for p in pairs if p["id"] in base]
    print(f"{len(pairs)} pairs with baseline runs; prompt={prompt_src}",
          flush=True)

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    prompts = []
    for p in pairs:
        texts, _ = p["variants"]["flipped"]
        user = template.format(problem=p["problem"],
                               solution="\n\n".join(texts))
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True))
    gens = []
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        kw = {"max_new_tokens": args.max_new_tokens,
              "pad_token_id": tok.pad_token_id}
        if args.temperature > 0:
            kw.update(do_sample=True, temperature=args.temperature,
                      top_p=0.95)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        for o in out:
            gens.append(tok.decode(o[enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        print(f"gen {min(i + args.batch, len(prompts))}/{len(prompts)}",
              flush=True)

    acc_o, acc_f, drag_o, drag_f, flips = [], [], [], [], []
    dump = []
    for p, g in zip(pairs, gens):
        k = p["k"]
        texts_f, _ = p["variants"]["flipped"]
        _, gold_o = p["variants"]["orig"]
        parsed_f, fe_f, fin_f = parse_generation(g, len(texts_f))
        parsed_o = base[p["id"]]["parsed"]
        po = parsed_o[k + 1:]
        pf = parsed_f[k + 1:]
        gold_post = gold_o[k + 1:]
        idx = [i for i in range(len(gold_post))
               if po[i] is not None and pf[i] is not None]
        dump.append({"id": p["id"], "k": k, "parsed_flipped": parsed_f,
                     "first_error_pred": fe_f, "final_pred": fin_f,
                     "gen_text": g})
        if not idx or not gold_post:
            continue
        acc_o.append(np.mean([po[i] == gold_post[i] for i in idx]))
        acc_f.append(np.mean([pf[i] == gold_post[i] for i in idx]))
        gc = [i for i in idx if gold_post[i] == 1]
        if gc:
            drag_o.append(np.mean([po[i] == 0 for i in gc]))
            drag_f.append(np.mean([pf[i] == 0 for i in gc]))
        flips.append(np.mean([po[i] != pf[i] for i in idx]))

    with open(args.out, "w") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")

    if acc_o:
        diff = np.array(acc_f) - np.array(acc_o)
        ddrag = (np.array(drag_f) - np.array(drag_o)) if drag_o else []
        summary = {
            "n_pairs_used": len(acc_o), "n_pairs_total": len(pairs),
            "postk_acc_orig": boot_ci(acc_o),
            "postk_acc_flipped": boot_ci(acc_f),
            "acc_diff_flipped_minus_orig": boot_ci(list(diff)),
            "drag_orig": boot_ci(drag_o), "drag_flipped": boot_ci(drag_f),
            "drag_diff": boot_ci(list(ddrag)) if len(ddrag) else None,
            "verdict_flip_rate": boot_ci(flips),
        }
    else:
        summary = {"n_pairs_used": 0, "n_pairs_total": len(pairs),
                   "error": "no usable parsed pairs"}
    print("THINKPRM_SWAP_SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
