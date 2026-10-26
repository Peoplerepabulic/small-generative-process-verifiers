"""Official ThinkPRM-1.5B baseline on PRM800K phase-1 test chosen traces.

Uses the OFFICIAL prompt template if available: pass --prompt_template
pointing to a text file with {problem} and {solution} placeholders
(extracted verbatim from the official repo's eval code). Falls back to a
built-in plain instruction prompt otherwise (reported in the summary).

Per problem: chat-template generation (long CoT), tolerant parsing of
per-step verdicts from the generated text, first-error extraction, and
the paper's own yes/no scoring (force "Is the solution correct?" after
the CoT; score = P(Yes)/(P(Yes)+P(No)) at the answer position).

Prints THINKPRM_BASE_SUMMARY (JSON). Saves a dump jsonl + 3 raw samples.
"""
import argparse
import json
import math
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

BUILTIN_PROMPT = (
    "Verify the following mathematical solution step by step. Reason "
    "carefully about each step, state for each step whether it is correct "
    "or incorrect, and conclude whether the solution as a whole is "
    "correct.\n\nProblem: {problem}\n\nSolution:\n{solution}"
)

STEP_PATTERNS = [
    re.compile(r"step\s*(\d+)\s*(?:is|:|—|-)?\s*(incorrect|correct|wrong)",
               re.I),
    re.compile(r"\*\*step\s*(\d+)\*\*\s*(?:is|:)?\s*(incorrect|correct|wrong)",
               re.I),
]
FINAL_PATTERNS = [
    re.compile(r"(?:final\s+(?:verdict|answer)|overall|conclusion)[^.\n]{0,40}?"
               r"\b(incorrect|correct)\b", re.I),
    re.compile(r"(?:the\s+)?solution\s+(?:is|seems|appears)\s+"
               r"(?:to\s+be\s+)?(incorrect|correct)", re.I),
    re.compile(r"(?:the\s+)?answer\s+is\s+(incorrect|correct)", re.I),
]

# --- prose-format parsing (official ThinkPRM writes free-text critiques) ---
_HEADER_RE = re.compile(r"step\s*(\d+)\s*(?=[:.)\]]|\s+is\b|\s+-\s)", re.I)
_NEG_RE = re.compile(
    r"is\s+incorrect|is\s+wrong|is\s+not\s+correct|not\s+correct|"
    r"\bincorrect\b|\bwrong\b|made\s+an?\s+error|contains\s+an?\s+error|"
    r"there\s+is\s+an?\s+error|this\s+is\s+a\s+mistake|\bmistake\b", re.I)
_POS_RE = re.compile(
    r"is\s+correct|\bcorrect\b|no\s+error|no\s+mistake|is\s+valid|"
    r"is\s+accurate|correctly\s+computed|correctly\s+states", re.I)


def _segment_verdict(seg):
    negs = [m.start() for m in _NEG_RE.finditer(seg)]
    poss = [m.start() for m in _POS_RE.finditer(seg)]
    if not negs and not poss:
        return None
    last_neg = max(negs) if negs else -1
    last_pos = max(poss) if poss else -1
    return 0 if last_neg > last_pos else 1


def parse_prose(text, n_steps):
    headers = [(int(m.group(1)), m.start(), m.end())
               for m in _HEADER_RE.finditer(text)]
    headers = [h for h in headers if 1 <= h[0] <= n_steps]
    verdicts = {}
    if headers:
        bounds = [h[2] for h in headers] + [len(text)]
        for j, (idx, _, _) in enumerate(headers):
            seg = text[bounds[j]:bounds[j + 1]] if j + 1 < len(headers) \
                else text[bounds[j]:]
            v = _segment_verdict(seg)
            if v is not None and idx not in verdicts:
                verdicts[idx] = v
    parsed = [verdicts.get(i + 1) for i in range(n_steps)]
    # fall back to the label-style patterns for anything still missing
    for pat in STEP_PATTERNS:
        for m in pat.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= n_steps and parsed[idx - 1] is None:
                parsed[idx - 1] = 1 if m.group(2).lower() == "correct" else 0
    first_err = next((i + 1 for i, v in enumerate(parsed) if v == 0), None)
    final = None
    tail = text[int(len(text) * 0.5):]
    for pat in FINAL_PATTERNS:
        ms = list(pat.finditer(tail))
        if ms:
            final = 1 if ms[-1].group(1).lower() == "correct" else 0
            break
    if final is None:
        final = 1 if first_err is None else 0
    return parsed, first_err, final


def parse_generation(text, n_steps):
    return parse_prose(text, n_steps)


def _parse_generation_labels(text, n_steps):
    verdicts = {}
    for pat in STEP_PATTERNS:
        for m in pat.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= n_steps and idx not in verdicts:
                v = m.group(2).lower()
                verdicts[idx] = 1 if v == "correct" else 0
    parsed = [verdicts.get(i + 1) for i in range(n_steps)]
    first_err = next((i + 1 for i, v in enumerate(parsed) if v == 0), None)
    final = None
    tail = text[int(len(text) * 0.5):]
    for pat in FINAL_PATTERNS:
        ms = list(pat.finditer(tail))
        if ms:
            final = 1 if ms[-1].group(1).lower() == "correct" else 0
            break
    if final is None:
        final = 1 if first_err is None else 0
    return parsed, first_err, final


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        rank_sum += avg * sum(l for _, l in pairs[i:j + 1])
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def ece(confs, labels, n_bins=10):
    out = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = [i for i, c in enumerate(confs)
             if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if m:
            out += len(m) / len(confs) * abs(
                sum(confs[i] for i in m) / len(m)
                - sum(labels[i] for i in m) / len(m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prm800k_phase1_test.jsonl")
    ap.add_argument("--prompt_template", default=None)
    ap.add_argument("--out", default="/workspace/thinkprm_official_dump.jsonl")
    ap.add_argument("--samples_out",
                    default="/workspace/thinkprm_official_samples.txt")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reparse", default=None,
                    help="recompute parses from an existing dump, no GPU")
    args = ap.parse_args()

    if args.prompt_template and os.path.exists(args.prompt_template):
        template = open(args.prompt_template).read()
        prompt_src = "official_file"
    else:
        template = BUILTIN_PROMPT
        prompt_src = "builtin"

    rows = []
    for line in open(args.data):
        d = json.loads(line)
        if d["source"] != "prm800k":
            continue
        texts, gold = [], []
        for s in d["steps"]:
            c = s["candidates"][s["chosen_idx"] or 0]
            texts.append(c["text"])
            gold.append(c["gold"])
        rows.append({"id": d["id"], "problem": d["problem"],
                     "texts": texts, "gold": gold})
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} problems; prompt={prompt_src}", flush=True)

    if args.reparse:
        dump = []
        for line in open(args.reparse):
            d = json.loads(line)
            parsed, first_err, final = parse_generation(
                d["gen_text"], len(d["gold_steps"]))
            d["parsed"], d["first_error_pred"], d["final_pred"] = (
                parsed, first_err, final)
            dump.append(d)
        with open(args.out, "w") as f:
            for d in dump:
                f.write(json.dumps(d) + "\n")
        _print_summary(dump, args, prompt_src)
        return

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).eval().to(device)

    prompts = []
    for r in rows:
        solution = "\n\n".join(r["texts"])
        user = template.format(problem=r["problem"], solution=solution)
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
        for j, o in enumerate(out):
            gens.append(tok.decode(o[enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        print(f"gen {min(i + args.batch, len(prompts))}/{len(prompts)}",
              flush=True)

    # yes/no solution scores, following the paper's protocol
    yes_ids = [tok.encode(v, add_special_tokens=False)[-1]
               for v in ("Yes", " Yes")]
    no_ids = [tok.encode(v, add_special_tokens=False)[-1]
              for v in ("No", " No")]
    yn_scores = []
    for r, g, p in zip(rows, gens, prompts):
        convo = (p + g + "\n\nIs the solution correct? "
                 "Answer with Yes or No.")
        enc = tok(convo, return_tensors="pt").to(device)
        if enc["input_ids"].shape[1] > 12000:
            yn_scores.append(None)
            continue
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        py = sum(probs[t].item() for t in set(yes_ids))
        pn = sum(probs[t].item() for t in set(no_ids))
        yn_scores.append(py / (py + pn) if py + pn > 0 else None)

    dump, samples = [], []
    for r, g, yn in zip(rows, gens, yn_scores):
        parsed, first_err, final = parse_generation(g, len(r["gold"]))
        dump.append({"id": r["id"], "gold_steps": r["gold"],
                     "gold_final": int(all(x == 1 for x in r["gold"])),
                     "gen_text": g, "parsed": parsed,
                     "first_error_pred": first_err, "final_pred": final,
                     "yesno_score": yn,
                     "truncated": len(g) >= 20000})
        if len(samples) < 3:
            samples.append(f"===== {r['id']} =====\n{g}\n")
    with open(args.out, "w") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")
    with open(args.samples_out, "w") as f:
        f.write("\n".join(samples))

    _print_summary(dump, args, prompt_src)


def _print_summary(dump, args, prompt_src):
    tp = fp = tn = fn = n_parsed_steps = 0
    sol_ok = der_ok = loc_ok = loc_n = 0
    for d in dump:
        gold_err = next((i + 1 for i, g in enumerate(d["gold_steps"])
                         if g == 0), None)
        if gold_err is not None:
            loc_n += 1
            loc_ok += int(d["first_error_pred"] == gold_err)
        sol_ok += int(d["final_pred"] == d["gold_final"])
        der = 1 if d["first_error_pred"] is None else 0
        der_ok += int(der == d["gold_final"])
        for g, p in zip(d["gold_steps"], d["parsed"]):
            if p is None:
                continue
            n_parsed_steps += 1
            if p == 0 and g == 0: tp += 1
            elif p == 0: fp += 1
            elif g == 0: fn += 1
            else: tn += 1
    n_steps = sum(len(d["gold_steps"]) for d in dump)
    step_acc = (tp + tn) / max(n_parsed_steps, 1)
    neg_recall = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    yn_pairs = [(d["yesno_score"], d["gold_final"]) for d in dump
                if d["yesno_score"] is not None]
    summary = {
        "model": args.model, "prompt": prompt_src, "n": len(dump),
        "step_coverage": n_parsed_steps / max(n_steps, 1),
        "step_acc": step_acc, "neg_recall": neg_recall,
        "balanced_acc": 0.5 * (neg_recall + spec)
        if neg_recall is not None and spec is not None else None,
        "sol_acc_parsed_final": sol_ok / len(dump),
        "sol_acc_derived": der_ok / len(dump),
        "loc": loc_ok / loc_n if loc_n else None, "loc_n": loc_n,
        "yesno_auroc": auroc([s for s, _ in yn_pairs],
                             [l for _, l in yn_pairs]) if yn_pairs else None,
        "yesno_acc05": sum(int((s >= 0.5) == bool(l)) for s, l in yn_pairs)
        / len(yn_pairs) if yn_pairs else None,
        "yesno_ece": ece([s for s, _ in yn_pairs],
                         [l for _, l in yn_pairs]) if yn_pairs else None,
        "yesno_brier": sum((s - l) ** 2 for s, l in yn_pairs)
        / len(yn_pairs) if yn_pairs else None,
        "n_truncated": sum(int(d["truncated"]) for d in dump),
    }
    print("THINKPRM_BASE_SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
