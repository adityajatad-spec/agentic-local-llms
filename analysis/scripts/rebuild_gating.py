"""Rebuild the gate-based methods from independent self-consistency samples (seed-fix, 2026-10).

The original adaptive-controller and gate->thinking runs drew their three gate samples with a shared per-request
seed, which correlated them. Here the gate is recomputed from the first three samples of a self-consistency@5 run
with independent seeds. Everything after the gate is taken from existing clean runs of the escalation target:
  adaptive       escalated -> always-on pipeline row (greedy decoding, so identical to what the controller produces)
  gate_think     escalated -> thinking-mode row (sampled; an independent draw of the same procedure)
Rebuilt cost per problem = tokens of the gate's first k samples + the escalation target's tokens for escalated
problems only, both taken from the source runs. The gate threshold is the one calibrated in each source run
(one temperature-0.7 sample per held-out problem, unaffected by the seed bug).

Run from the project root:
  python analysis/scripts/rebuild_gating.py <sc5_run> <task> <out_name> <method>=<source_run>:<source_method> ...
Example:
  python analysis/scripts/rebuild_gating.py run-023 math rebuilt-qwen adaptive=run-007:always_on gate_think=run-018:think
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import agent  # noqa: E402
from train import bootstrap_ci, summarize  # noqa: E402


def load(run):
    by = defaultdict(dict)
    for line in open(os.path.join(ROOT, "results", run, "per_example.jsonl")):
        r = json.loads(line)
        by[r["method"]][r["id"]] = r
    return by


def threshold(source_run, method):
    """Gate threshold calibrated in the run whose original gated method is being rebuilt."""
    for r in load(source_run)[method].values():
        thr = (r.get("info") or {}).get("thr")
        if thr is not None:
            return thr
    raise ValueError(f"no threshold in {source_run}:{method}")


def gate(sc_row, k, thr, use_conf=True):
    """Accept iff the first k samples all have an answer, agree, and (optionally) mean log-prob exceeds thr."""
    info = sc_row["info"]
    ans, lps = info["answers"][:k], info["lps"][:k]
    tokens = sum(info["n_in"][:k]) + sum(info["n_out"][:k])
    if len(ans) == k and all(a is not None for a in ans) and len({agent.key(a) for a in ans}) == 1:
        if not use_conf or sum(lps) / k > thr:
            return ans[0], tokens
    return None, tokens


def rebuild(sc_run, out_name, method, src_run, src_method, thr_run, thr_method, k=3, use_conf=True):
    sc = load(sc_run)["sc5"]
    src = load(src_run)[src_method]
    thr = threshold(thr_run, thr_method)
    rows = []
    for i, s in sc.items():
        e = src[i]
        acc, gate_tok = gate(s, k, thr, use_conf)
        base = {"id": i, "method": method, "gold": s["gold"], "info": {"thr": thr, "gate_answers": s["info"]["answers"][:k],
                                                                       "gate_source": sc_run, "escalation_source": f"{src_run}:{src_method}"}}
        if acc is not None:
            rows.append({**base, "pred": acc, "correct": float(agent.same(acc, s["gold"])), "escalated": False,
                         "n_in": sum(s["info"]["n_in"][:k]), "n_out": sum(s["info"]["n_out"][:k]), "calls": k,
                         "tool_calls": 0, "tool_ok": 0, "flagged": None, "cand_correct": float(agent.same(acc, s["gold"])),
                         "latency_s": 0.0, "trace": s["trace"][:k]})
        else:
            rows.append({**base, "pred": e["pred"], "correct": e["correct"], "escalated": True,
                         "n_in": sum(s["info"]["n_in"][:k]) + e["n_in"], "n_out": sum(s["info"]["n_out"][:k]) + e["n_out"],
                         "calls": k + e["calls"], "tool_calls": e["tool_calls"], "tool_ok": e["tool_ok"],
                         "flagged": e.get("flagged"), "cand_correct": e.get("cand_correct"), "latency_s": 0.0,
                         "trace": s["trace"][:k] + e["trace"]})
    return rows


def main():
    sc_run, task, out_name = sys.argv[1:4]
    agent.set_task(task)
    sc_rows = load(sc_run)["sc5"]
    distinct = sum(1 for r in sc_rows.values() if len({t["o"] for t in r["trace"][:5]}) == 5)
    out = {}
    for spec in sys.argv[4:]:
        method, src = spec.split("=")
        src_run, src_method = src.split(":")
        # thresholds come from the original gated run: adaptive runs for adaptive, the gate->thinking run for gate_think
        thr_run, thr_method = {"adaptive": (src_run, "adaptive"), "gate_think": (src_run, "gate_think")}[method]
        rows = rebuild(sc_run, out_name, method, src_run, src_method, thr_run, thr_method)
        out[method] = {"summary": summarize(rows), "rows": rows}
        if method == "gate_think":  # cheaper gate variants, from the same independent samples
            for k, use_conf, name in ((2, False, "gate2_think"), (3, False, "gate3noconf_think")):
                r2 = rebuild(sc_run, out_name, name, src_run, src_method, thr_run, thr_method, k=k, use_conf=use_conf)
                out[name] = {"summary": summarize(r2), "rows": r2}
    d = os.path.join(ROOT, "results", out_name)
    os.makedirs(d, exist_ok=True)
    meta = {"model": json.load(open(os.path.join(ROOT, "results", sc_run, "metrics.json")))["meta"]["model"],
            "rebuilt_from": sc_run, "task": task, "specs": sys.argv[4:], "distinct5_samples": distinct,
            "n": len(sc_rows), "note": "gate recomputed from independent SC@5 samples; escalations from clean source runs"}
    json.dump({"meta": meta, "methods": {m: v["summary"] for m, v in out.items()}}, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    with open(os.path.join(d, "per_example.jsonl"), "w") as f:
        for v in out.values():
            for r in v["rows"]:
                f.write(json.dumps(r) + "\n")
    print(f"{out_name}: problems with 5 distinct samples in {sc_run}: {distinct}/{len(sc_rows)}")
    for m, v in out.items():
        s = v["summary"]
        print(f"  {m:18s} acc {s['accuracy']:.3f} tok {s['tokens_per_problem']:.0f} escalated {s['escalation_rate']:.2f}")


if __name__ == "__main__":
    main()
