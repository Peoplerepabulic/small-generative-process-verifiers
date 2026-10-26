"""V4: continuous prefix-forcing — Delta p_i on post-k steps.

For each counterfactual pair (orig trace, flip step k) and each arm
(force line k = correct / incorrect), walk the post-k steps reading
p_inc at every verdict position (argmax path per arm, mirroring
tokenprob_eval). Reports per-pair mean Delta p (incorrect arm minus
correct arm) over post-k steps, and over gold-correct post-k steps only
(the continuous analogue of drag), with bootstrap CIs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "train"))
from rewards import build_verifier_prompt  # noqa: E402
from partB_interference import build_counterfactual_pairs  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def boot_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return {"mean": float(vals.mean()),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))], "n": int(len(vals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--pf_json", default=None,
                    help="prefix_force_*.json to restrict to its pair ids")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = build_counterfactual_pairs([args.data], max_n=100)
    pf_ids = None
    if args.pf_json and os.path.exists(args.pf_json):
        pf = json.load(open(args.pf_json))
        pf_ids = {r["id"] for r in pf["records"]}
    print(f"{len(pairs)} pairs; pf subset: "
          f"{len(pf_ids) if pf_ids else 'n/a'}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)
    id_corr = tok.encode(" correct", add_special_tokens=False)[0]
    id_inc = tok.encode(" incorrect", add_special_tokens=False)[0]

    def feed(past, text, special=False):
        # See v4_tokenprob891.py: input_ids only — an attention_mask passed
        # alongside past_key_values corrupts cached conditioning (v28 fix).
        enc = tok(text, add_special_tokens=special,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(input_ids=enc["input_ids"], past_key_values=past,
                      use_cache=True)
        return o.past_key_values, o.logits[0, -1]

    def walk(prompt, n, k=None, forced=None, base_path=None):
        """Return (p_incs list, path list). If forced is set, line k
        (0-indexed) uses `forced`; steps < k use base_path verdicts."""
        past, _ = feed(None, prompt, special=True)
        p_incs, path = [], []
        for i in range(n):
            past, logits = feed(past, f"Step {i + 1}:")
            pair = torch.softmax(torch.stack(
                [logits[id_inc], logits[id_corr]]).float(), dim=0)
            p_inc = float(pair[0])
            p_incs.append(p_inc)
            if forced is not None and i == k:
                verdict = forced
            elif forced is not None and i < k:
                verdict = base_path[i]
            else:
                verdict = "incorrect" if p_inc > 0.5 else "correct"
            path.append(verdict)
            past, _ = feed(past, f" {verdict}\n")
        return p_incs, path

    records = []
    for p in pairs:
        texts, gold = p["variants"]["orig"]
        k = p["k"]
        n = len(texts)
        if k + 1 >= n:
            continue
        prompt = build_verifier_prompt(p["problem"], texts)
        _, base_path = walk(prompt, n)
        p_corr, _ = walk(prompt, n, k=k, forced="correct",
                         base_path=base_path)
        p_inc_arm, _ = walk(prompt, n, k=k, forced="incorrect",
                            base_path=base_path)
        post = list(range(k + 1, n))
        dp = [p_inc_arm[i] - p_corr[i] for i in post]
        gc = [i for i in post if gold[i] == 1]
        records.append({
            "id": p["id"], "k": k, "n_post": len(post),
            "dp_mean": float(np.mean(dp)),
            "dp_goldcorrect": float(np.mean([p_inc_arm[i] - p_corr[i]
                                             for i in gc])) if gc else None,
            "p_post_corr": float(np.mean([p_corr[i] for i in post])),
            "p_post_inc": float(np.mean([p_inc_arm[i] for i in post]))})
        if len(records) % 20 == 0:
            print(f"{len(records)} pairs done", flush=True)

    def agg(recs):
        return {
            "n": len(recs),
            "dp_all": boot_ci([r["dp_mean"] for r in recs]),
            "dp_goldcorrect": boot_ci(
                [r["dp_goldcorrect"] for r in recs
                 if r["dp_goldcorrect"] is not None]),
            "p_post_corr_mean": float(np.mean([r["p_post_corr"]
                                               for r in recs])),
            "p_post_inc_mean": float(np.mean([r["p_post_inc"]
                                              for r in recs]))}
    res = {"all_walk_pairs": agg(records)}
    if pf_ids:
        res["prefix_force_71_subset"] = agg(
            [r for r in records if r["id"] in pf_ids])
    print("V4_DELTAP_" + args.tag.upper() + " " + json.dumps(res),
          flush=True)
    out_path = args.out or f"/workspace/v4_delta_p_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": res, "records": records}, f, indent=1)
    print(f"wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
