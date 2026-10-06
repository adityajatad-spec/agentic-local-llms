"""Count empty tool observations (code that printed nothing) in text-format ReAct traces with the print-only tool.

Run from the project root: python analysis/scripts/empty_observations.py
Note: traces keep the last 300 characters of each prompt, so counts are a lower bound.
"""
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
out = {}
for run, m in (("run-007", "react"), ("run-007", "always_on"), ("run-017", "react")):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", run, "per_example.jsonl"))]
    R = [r for r in rows if r["method"] == m]
    obs = empty = 0
    with_empty, wrong_with_empty = set(), 0
    for r in R:
        had = False
        for t in r["trace"]:
            for o in re.findall(r"Observation: (.*?)\n(?:Thought:|$)", t["p"], re.S):
                obs += 1
                if o.strip() == "":
                    empty += 1
                    had = True
        if had:
            with_empty.add(r["id"])
            wrong_with_empty += not r["correct"]
    acc_overall = sum(r["correct"] for r in R) / len(R)
    out[f"{run}:{m}"] = {"observations": obs, "empty": empty, "empty_rate": empty / max(1, obs),
                         "problems_with_empty": len(with_empty),
                         "error_rate_with_empty": wrong_with_empty / max(1, len(with_empty)),
                         "error_rate_overall": 1 - acc_overall}
json.dump(out, open(os.path.join(ROOT, "analysis", "empty_observations.json"), "w"), indent=2)
for k, v in out.items():
    print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
