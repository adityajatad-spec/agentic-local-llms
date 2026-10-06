"""Failure taxonomy and accuracy-by-difficulty from saved per-example traces.

Each wrong answer is assigned exactly one category (first matching rule wins):
  truncated          no answer and a generation hit its token cap
  non_termination    agent loop used all steps without a final answer
  no_answer_other    no answer for any other reason
  verifier_override  the verifier rejected a correct candidate and replaced it with a wrong one
  escalation_harm    the gate escalated, the pipeline was wrong, and plain CoT was right on this problem
  tool_error         an answer was produced after at least one failed tool call (error or timeout)
  reasoning_error    an answer was produced and is wrong (no mechanism above applies)

Run from the project root: python analysis/scripts/failure_taxonomy.py [run ...]
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
MAX_STEPS = 8
CAPS = (4096, 1024, 16384)  # cot / react-step / thinking token caps on MATH
CATS = ["truncated", "non_termination", "no_answer_other", "verifier_override", "escalation_harm",
        "tool_error", "reasoning_error"]
AGENTIC = {"react", "always_on", "always_on_intrinsic", "always_on_none", "adaptive"}


def load(run):
    by = defaultdict(dict)
    meta = json.load(open(os.path.join(ROOT, "results", run, "metrics.json")))["meta"]
    for line in open(os.path.join(ROOT, "results", run, "per_example.jsonl")):
        r = json.loads(line)
        by[r["method"]][r["id"]] = r
    return meta, by


def capped(r):
    return any(t.get("n_out", 0) in CAPS for t in r["trace"])


def classify(m, r, cot_row):
    if r["pred"] is None:
        if capped(r):
            return "truncated"
        if m in AGENTIC and r["tool_calls"] >= MAX_STEPS - 1:
            return "non_termination"
        return "no_answer_other"
    if r.get("flagged") and r.get("cand_correct") == 1:
        return "verifier_override"
    if m == "adaptive" and r.get("escalated") and cot_row is not None and cot_row["correct"]:
        return "escalation_harm"
    if r["tool_calls"] > r["tool_ok"]:
        return "tool_error"
    return "reasoning_error"


def levels():
    """MATH-500 difficulty level per problem id (from the HF dataset, cached locally)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="/tmp/hfc")
    return {f"math-{r['unique_id']}": int(r["level"]) for r in ds}


def main(runs):
    lvl = levels()
    out = {}
    for spec in runs:
        run, _, ref = spec.partition("@")  # rebuilt runs have no CoT of their own; use the reference run's CoT
        meta, by = load(run)
        cot = by.get("cot", {}) if not ref else load(ref)[1]["cot"]
        res = {"model": meta["model"], "methods": {}}
        for m, rows in by.items():
            wrong = [(i, r) for i, r in rows.items() if not r["correct"]]
            cats = Counter(classify(m, r, cot.get(i)) for i, r in wrong)
            acc_lvl = {}
            for L in range(1, 6):
                ids = [i for i in rows if lvl.get(i) == L]
                acc_lvl[L] = {"n": len(ids), "acc": sum(rows[i]["correct"] for i in ids) / len(ids) if ids else None}
            res["methods"][m] = {"n_wrong": len(wrong), "categories": {c: cats.get(c, 0) for c in CATS},
                                 "acc_by_level": acc_lvl}
        out[run] = res
        print(f"\n=== {run} ({meta['model']})")
        print(f"{'method':20s} wrong " + " ".join(f"{c[:11]:>11s}" for c in CATS))
        for m, v in res["methods"].items():
            print(f"{m:20s} {v['n_wrong']:5d} " + " ".join(f"{v['categories'][c]:11d}" for c in CATS))
        print("accuracy by MATH level (n per level: " + ", ".join(
            f"L{L}={next(iter(res['methods'].values()))['acc_by_level'][L]['n']}" for L in range(1, 6)) + ")")
        for m, v in res["methods"].items():
            print(f"  {m:20s} " + " ".join(f"L{L} {100 * v['acc_by_level'][L]['acc']:5.1f}" for L in range(1, 6)))
    json.dump(out, open(os.path.join(ROOT, "analysis", "failures.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or ["run-007", "run-008"])  # e.g. rebuilt-qwen@run-007
