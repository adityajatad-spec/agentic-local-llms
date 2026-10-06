"""Compute all tables and statistics for the paper from results/ (no hand-entered numbers).

Every method is compared with the CoT baseline from the *same run* (same backend, GPU and batch conditions)
using an exact McNemar test on per-problem correctness; p-values are Holm-corrected within each table.

Run from the project root: python analysis/scripts/compute_stats.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

from scipy.stats import binomtest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import agent  # noqa: E402  (answer matching for the offline gate simulation)

OUT_T = os.path.join(ROOT, "analysis", "tables")
os.makedirs(OUT_T, exist_ok=True)
_cache = {}


def load(run):
    if run not in _cache:
        by = defaultdict(dict)
        for line in open(os.path.join(ROOT, "results", run, "per_example.jsonl")):
            r = json.loads(line)
            by[r["method"]][r["id"]] = r
        m = json.load(open(os.path.join(ROOT, "results", run, "metrics.json")))
        _cache[run] = (m["methods"], by, m["meta"])
    return _cache[run]


def mcnemar(a, b):
    """Exact McNemar on paired correctness. Returns (b_wins, a_wins, p)."""
    ids = sorted(set(a) & set(b))
    a_only = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    b_only = sum(1 for i in ids if b[i]["correct"] and not a[i]["correct"])
    n = a_only + b_only
    return b_only, a_only, (binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def compare_table(spec, fname, caption_cols=True):
    """spec: list of (label, run, method). Returns rows with accuracy, CI, tokens, delta and Holm p vs same-run CoT."""
    rows, ps = [], []
    for item in spec:
        label, run, m = item[:3]
        ref = item[3] if len(item) > 3 else run           # run whose CoT the method is tested against
        run_label = item[4] if len(item) > 4 else run.split("-")[1]
        s, by, meta = load(run)
        cs, cby, _ = load(ref)
        cot_s, cot_rows = cs["cot"], cby["cot"]
        r = {"label": label, "run": run, "ref_run": ref, "run_label": run_label, "method": m, "model": meta.get("model"),
             "accuracy": s[m]["accuracy"], "ci95": s[m]["acc_ci95"], "tokens": s[m]["tokens_per_problem"],
             "cot_same_run": cot_s["accuracy"], "x_cot": s[m]["tokens_per_problem"] / cot_s["tokens_per_problem"]}
        if m != "cot":
            w, l, p = mcnemar(cot_rows, by[m])
            r.update(delta=s[m]["accuracy"] - cot_s["accuracy"], wins=w, losses=l, p_raw=p)
            ps.append(p)
        rows.append(r)
    adj = iter(holm(ps))
    for r in rows:
        if "p_raw" in r:
            r["p_holm"] = next(adj)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Method & Run & Accuracy (95\% CI) & Tokens & $\times$CoT$^\ast$ & $\Delta$ vs CoT$^\ast$ & $p$ (Holm) \\", r"\midrule"]
    for r in rows:
        acc, lo, hi = (100 * r["accuracy"], 100 * r["ci95"][0], 100 * r["ci95"][1])
        d = f"{100 * r['delta']:+.1f}" if "delta" in r else "--"
        p = fmt_p(r["p_holm"]) if "p_holm" in r else "--"
        lines.append(f"{r['label']} & {r['run_label']} & {acc:.1f} [{lo:.1f}, {hi:.1f}] & {r['tokens']:.0f} & "
                     f"{r['x_cot']:.2f} & {d} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT_T, fname), "w").write("\n".join(lines) + "\n")
    return rows


stats = {}

# ---------------------------------------------------------------- main tables: MATH-500, both models
QWEN = [("Direct answer", "run-007", "direct"), ("Chain-of-thought (CoT)", "run-007", "cot"),
        ("Self-Refine", "run-007", "self_refine"), ("Reflection memory", "run-007", "reflexion"),
        ("ReAct, print-only tool", "run-007", "react"), ("ReAct, REPL tool", "run-014", "react_repl"),
        ("ReAct, native calls", "run-014", "react_native"), ("Always-on pipeline", "run-007", "always_on"),
        ("Always-on, native calls", "run-014", "always_on_native"),
        ("Adaptive controller$^\\ddagger$", "rebuilt-qwen", "adaptive", "run-007", "023+007"),
        ("Self-consistency@5", "run-023", "sc5"), ("Thinking mode", "run-018", "think"),
        ("Gate $\\rightarrow$ thinking$^\\ddagger$", "rebuilt-qwen", "gate_think", "run-018", "023+018")]
LLAMA = [("Direct answer", "run-017", "direct"), ("Chain-of-thought (CoT)", "run-017", "cot"),
         ("Self-Refine", "run-017", "self_refine"), ("Reflection memory", "run-017", "reflexion"),
         ("ReAct, print-only tool", "run-017", "react"), ("ReAct, REPL tool", "run-015", "react_repl"),
         ("ReAct, native calls", "run-015", "react_native"), ("Always-on pipeline", "run-017", "always_on"),
         ("Always-on, native calls", "run-015", "always_on_native"),
         ("Adaptive controller$^\\ddagger$", "rebuilt-llama", "adaptive", "run-017", "024+017"),
         ("Self-consistency@5", "run-024", "sc5")]
stats["main_qwen"] = compare_table(QWEN, "main_qwen.tex")
stats["main_llama"] = compare_table(LLAMA, "main_llama.tex")
stats["cot_reproducibility"] = {
    model: {run: load(run)[0]["cot"]["accuracy"] for run in runs}
    for model, runs in (("Qwen3-8B", ["run-007", "run-014", "run-018", "run-023", "run-026", "run-027"]),
                        ("Llama-3.1-8B", ["run-017", "run-015", "run-024"]))}

def cot_note(runs):
    parts = []
    for run in runs:
        c = load(run)[0]["cot"]
        parts.append(f"run~{run.split('-')[1]} {100 * c['accuracy']:.1f}\\%, {c['tokens_per_problem']:.0f} tokens")
    return "Same-run CoT: " + "; ".join(parts) + "."


with open(os.path.join(OUT_T, "macros.tex"), "w") as f:
    f.write("% generated by compute_stats.py\n")
    f.write("\\newcommand{\\cotnoteQwen}{" + cot_note(["run-007", "run-014", "run-018", "run-023"]) + "}\n")
    f.write("\\newcommand{\\cotnoteLlama}{" + cot_note(["run-017", "run-015", "run-024"]) + "}\n")
    f.write("\\newcommand{\\cotnoteGsmh}{" + cot_note(["run-016", "run-025"]) + "}\n")

# ---------------------------------------------------------------- GSM-Hard (Qwen)
GSMH = [("Direct answer", "run-016", "direct"), ("Chain-of-thought (CoT)", "run-016", "cot"),
        ("ReAct, print-only tool", "run-016", "react"), ("ReAct, REPL tool", "run-016", "react_repl"),
        ("ReAct, native calls", "run-016", "react_native"), ("Always-on pipeline", "run-016", "always_on"),
        ("Adaptive controller$^\\ddagger$", "rebuilt-gsmhard", "adaptive", "run-016", "025+016"),
        ("Self-consistency@5", "run-025", "sc5")]
stats["gsmhard_qwen"] = compare_table(GSMH, "gsmhard_qwen.tex")

# ---------------------------------------------------------------- ReAct decomposition (tool semantics vs format)
dec = {}
for model, spec in (("Qwen3-8B", [("print-only", "run-007", "react"), ("REPL", "run-014", "react_repl"),
                                  ("native", "run-014", "react_native")]),
                    ("Llama-3.1-8B", [("print-only", "run-017", "react"), ("REPL", "run-015", "react_repl"),
                                      ("native", "run-015", "react_native")])):
    dec[model] = []
    for variant, run, m in spec:
        s, by, _ = load(run)
        R = list(by[m].values())
        dec[model].append({
            "variant": variant, "run": run, "accuracy": s[m]["accuracy"], "tokens": s[m]["tokens_per_problem"],
            "cot_same_run": s["cot"]["accuracy"], "tool_validity": s[m]["tool_validity"],
            "tool_calls_per_problem": sum(r["tool_calls"] for r in R) / len(R),
            "no_answer": sum(r["pred"] is None for r in R) / len(R),
            "forced_final": sum(1 for r in R if r.get("info", {}).get("forced_final")) / len(R)})
stats["react_decomposition"] = dec
lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Model & Tool / format & Accuracy & CoT (same run) & Tokens & Calls/problem & Valid calls & Hit step limit \\",
         r"\midrule"]
for model, rows in dec.items():
    for i, r in enumerate(rows):
        hit = r["forced_final"] if r["variant"] != "print-only" else r["no_answer"]
        lines.append(f"{model if i == 0 else ''} & {r['variant']} & {100 * r['accuracy']:.1f} & {100 * r['cot_same_run']:.1f} & "
                     f"{r['tokens']:.0f} & {r['tool_calls_per_problem']:.1f} & {100 * r['tool_validity']:.0f}\\% & {100 * hit:.0f}\\% \\\\")
    lines.append(r"\midrule" if model == "Qwen3-8B" else "")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT_T, "react_decomposition.tex"), "w").write("\n".join(l for l in lines if l) + "\n")

# ---------------------------------------------------------------- verifier ablation (run-008, Qwen)
s8, r8, _ = load("run-008")
cot7 = load("run-007")[1]["cot"]
ver = []
names8 = {"always_on_none": "None", "always_on": "Grounded (code re-derivation)",
          "always_on_intrinsic": "Intrinsic (self-critique)"}
lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"Verifier & Pre-verifier & Final & Flagged & Fixed & Broke & Recall & False-reject & +CoT fallback$^\dagger$ \\",
         r"\midrule"]
for k in ["always_on_none", "always_on", "always_on_intrinsic"]:
    rows = r8[k]
    R = list(rows.values())
    right = [r for r in R if r["cand_correct"] == 1]   # candidates with no answer count as wrong
    wrong = [r for r in R if r["cand_correct"] != 1]
    flagged = sum(1 for r in R if r["flagged"])
    fixed = sum(1 for r in wrong if r["flagged"] and r["correct"] == 1)
    broke = sum(1 for r in right if r["flagged"] and r["correct"] == 0)
    final = sum(r["correct"] for r in R)
    assert len(right) + fixed - broke == final, f"verifier counts do not reconcile for {k}"
    cf = (sum((cot7[i]["correct"] if r["flagged"] else (r["cand_correct"] or 0)) for i, r in rows.items()) / len(rows)
          if k != "always_on_none" else None)
    none = k == "always_on_none"
    rec = None if none else sum(1 for r in wrong if r["flagged"]) / len(wrong)
    frr = None if none else sum(1 for r in right if r["flagged"]) / len(right)
    p = mcnemar(r8["always_on_none"], rows)[2] if not none else None
    ver.append({"verifier": k, "pre_verifier": len(right) / len(R), "accuracy": final / len(R),
                "tokens": s8[k]["tokens_per_problem"], "flagged": flagged, "fixed": fixed, "broke": broke,
                "recall": rec, "false_reject": frr, "detect_plus_cot_fallback": cf, "p_vs_none": p})
    dash = "--"
    lines.append(f"{names8[k]} & {100 * len(right) / len(R):.1f} & {100 * final / len(R):.1f} & {dash if none else flagged} & "
                 f"{dash if none else fixed} & {dash if none else broke} & {dash if rec is None else f'{rec:.3f}'} & "
                 f"{dash if frr is None else f'{frr:.3f}'} & {dash if cf is None else f'{100 * cf:.1f}'} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT_T, "verifier_ablation.tex"), "w").write("\n".join(lines) + "\n")
stats["verifier_ablation"] = ver
sr = list(load("run-007")[1]["self_refine"].values())
sr_fixed = sum(1 for r in sr if r["flagged"] and r["cand_correct"] != 1 and r["correct"] == 1)
sr_broke = sum(1 for r in sr if r["flagged"] and r["cand_correct"] == 1 and r["correct"] == 0)
assert sum(r["cand_correct"] == 1 for r in sr) + sr_fixed - sr_broke == sum(r["correct"] for r in sr)
stats["self_refine_flips"] = {"changed": sum(1 for r in sr if r["flagged"]), "fixed": sr_fixed, "broke": sr_broke,
                              "pre_refine": sum(r["cand_correct"] == 1 for r in sr) / len(sr)}

# ---------------------------------------------------------------- gating: adaptive (run-007) and gate->thinking (run-018)
r7 = load("run-007")[1]
A = load("rebuilt-qwen")[1]["adaptive"]   # gate decisions from independent samples (run-023), escalations from run-007
esc = [i for i, r in A.items() if r["escalated"]]
acc_ids = [i for i in A if i not in esc]
on = lambda by, m, ids: sum(by[m][i]["correct"] for i in ids) / len(ids)
on_a = lambda ids: sum(A[i]["correct"] for i in ids) / len(ids)
stats["adaptive_split"] = {
    "n_accepted": len(acc_ids), "n_escalated": len(esc),
    "by_method": {**{m: {"accepted": on(r7, m, acc_ids), "escalated": on(r7, m, esc)}
                     for m in ["cot", "reflexion", "react", "always_on"]},
                  "adaptive": {"accepted": on_a(acc_ids), "escalated": on_a(esc)}},
    "gate_plus_cot_fallback": (sum(A[i]["correct"] for i in acc_ids) + sum(r7["cot"][i]["correct"] for i in esc)) / len(A)}

s18, r18, _ = load("run-018")
sR, rR, _ = load("rebuilt-qwen")
s23, r23, _ = load("run-023")
G, TH = rR["gate_think"], r18["think"]
Gacc = [i for i, r in G.items() if not r["escalated"]]
Gesc = [i for i, r in G.items() if r["escalated"]]
gstat = lambda m: {"accuracy": sR[m]["accuracy"], "tokens": sR[m]["tokens_per_problem"], "escalation_rate": sR[m]["escalation_rate"],
                   "p_vs_always_think": mcnemar(TH, rR[m])[2]}
stats["gate_think"] = {
    "measured": gstat("gate_think"), "gate3_no_conf": gstat("gate3noconf_think"), "gate2": gstat("gate2_think"),
    "accepted": {"n": len(Gacc), "gate": sum(G[i]["correct"] for i in Gacc) / len(Gacc), "always_think": on(r18, "think", Gacc),
                 "cot": on(r18, "cot", Gacc)},
    "escalated": {"n": len(Gesc), "gate": sum(G[i]["correct"] for i in Gesc) / len(Gesc), "always_think": on(r18, "think", Gesc),
                  "cot": on(r18, "cot", Gesc)},
    "always_think": {"accuracy": s18["think"]["accuracy"], "tokens": s18["think"]["tokens_per_problem"]},
    "cot": {"accuracy": s18["cot"]["accuracy"], "tokens": s18["cot"]["tokens_per_problem"]},
    "sc5": {"accuracy": s23["sc5"]["accuracy"], "tokens": s23["sc5"]["tokens_per_problem"], "cot_same_run": s23["cot"]["accuracy"]},
    "think_truncated": sum(1 for r in TH.values() if r.get("info", {}).get("think_truncated")),
    "distinct5": {run: json.load(open(os.path.join(ROOT, "results", f"rebuilt-{k}", "metrics.json")))["meta"]["distinct5_samples"]
                  for k, run in (("qwen", "run-023"), ("llama", "run-024"), ("gsmhard", "run-025"))},
}
gt = stats["gate_think"]
row = lambda name, d, esc: f"{name} & {100 * d['accuracy']:.1f} & {d['tokens']:.0f} & {esc} \\\\"
lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Strategy & Accuracy & Tokens/problem & Escalated \\", r"\midrule",
         row("CoT (run 018)", gt["cot"], "--"), row("Self-consistency@5 (run 023)", gt["sc5"], "--"),
         row("Always thinking (run 018)", gt["always_think"], "100\\%"),
         row("Gate: 3 agree + confidence $\\rightarrow$ thinking", gt["measured"], f"{100 * gt['measured']['escalation_rate']:.0f}\\%"),
         row("Gate: 3 agree $\\rightarrow$ thinking", gt["gate3_no_conf"], f"{100 * gt['gate3_no_conf']['escalation_rate']:.0f}\\%"),
         row("Gate: 2 agree $\\rightarrow$ thinking", gt["gate2"], f"{100 * gt['gate2']['escalation_rate']:.0f}\\%"),
         r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT_T, "thinking_gate.tex"), "w").write("\n".join(lines) + "\n")

# thinking mode across three sampling seeds, each tested against CoT from its own run
seeds = []
for seed, run in ((1, "run-018"), (2, "run-026"), (3, "run-027")):
    s_, b_, _ = load(run)
    w, l, p = mcnemar(b_["cot"], b_["think"])
    seeds.append({"seed": seed, "run": run, "cot": s_["cot"]["accuracy"], "think": s_["think"]["accuracy"],
                  "delta": s_["think"]["accuracy"] - s_["cot"]["accuracy"], "wins": w, "losses": l, "p": p,
                  "tokens": s_["think"]["tokens_per_problem"],
                  "truncated": sum(1 for r in b_["think"].values() if r.get("info", {}).get("think_truncated"))})
import statistics as st  # noqa: E402
stats["thinking_seeds"] = {"runs": seeds, "mean_think": st.mean(x["think"] for x in seeds), "sd_think": st.stdev(x["think"] for x in seeds),
                           "mean_delta": st.mean(x["delta"] for x in seeds), "sd_delta": st.stdev(x["delta"] for x in seeds)}
lines = [r"\begin{tabular}{lccccc}", r"\toprule", r"Seed (run) & CoT & Thinking & $\Delta$ & $p$ & Truncated \\", r"\midrule"]
for x in seeds:
    lines.append(f"{x['seed']} ({x['run'].split('-')[1]}) & {100 * x['cot']:.1f} & {100 * x['think']:.1f} & {100 * x['delta']:+.1f} & {fmt_p(x['p'])} & {x['truncated']} \\\\")
ts_ = stats["thinking_seeds"]
lines += [r"\midrule", f"Mean $\\pm$ s.d. & & {100 * ts_['mean_think']:.1f} $\\pm$ {100 * ts_['sd_think']:.1f} & {100 * ts_['mean_delta']:+.1f} $\\pm$ {100 * ts_['sd_delta']:.1f} & & \\\\",
          r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT_T, "thinking_seeds.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- failure taxonomy and accuracy by level
fail = json.load(open(os.path.join(ROOT, "analysis", "failures.json")))
CATS = ["truncated", "non_termination", "no_answer_other", "verifier_override", "escalation_harm", "tool_error",
        "reasoning_error"]
FSPEC = [("Qwen3-8B", "CoT", "run-007", "cot"), ("Qwen3-8B", "ReAct, REPL", "run-014", "react_repl"),
         ("Qwen3-8B", "Always-on", "run-007", "always_on"), ("Qwen3-8B", "Adaptive", "rebuilt-qwen", "adaptive"),
         ("Qwen3-8B", "Thinking", "run-018", "think"),
         ("Llama-3.1-8B", "CoT", "run-017", "cot"), ("Llama-3.1-8B", "ReAct, print-only", "run-017", "react"),
         ("Llama-3.1-8B", "ReAct, native", "run-015", "react_native"), ("Llama-3.1-8B", "Always-on", "run-017", "always_on"),
         ("Llama-3.1-8B", "Adaptive", "rebuilt-llama", "adaptive")]
hdr = ["Trunc.", "Loop", "No ans.", "Verif. override", "Escal. harm", "Tool err.", "Reasoning"]
lines = [r"\begin{tabular}{ll" + "c" * (len(CATS) + 1) + "}", r"\toprule",
         "Model & Method & Wrong & " + " & ".join(hdr) + r" \\", r"\midrule"]
stats["failure_taxonomy"] = []
prev = None
for model, label, run, m in FSPEC:
    v = fail[run]["methods"][m]
    stats["failure_taxonomy"].append({"model": model, "method": label, "run": run, **v})
    lines.append(f"{model if model != prev else ''} & {label} & {v['n_wrong']} & " +
                 " & ".join(str(v["categories"][c]) for c in CATS) + r" \\")
    prev = model
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT_T, "failure_taxonomy.tex"), "w").write("\n".join(lines) + "\n")
stats["acc_by_level"] = {f"{run}:{m}": fail[run]["methods"][m]["acc_by_level"]
                         for _, _, run, m in FSPEC + [("", "", "run-018", "cot"), ("", "", "run-023", "sc5"), ("", "", "run-024", "sc5")]}

# level-5 check for Llama scaffolds that looked better than CoT on the hardest problems
import datasets  # noqa: E402
ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="/tmp/hfc")
lvl = {f"math-{r['unique_id']}": int(r["level"]) for r in ds}
r17 = load("run-017")[1]
rL = load("rebuilt-llama")[1]
l5 = [i for i in r17["cot"] if lvl.get(i) == 5]
src5 = {"adaptive": rL["adaptive"], "always_on": r17["always_on"]}
stats["llama_level5"] = {m: dict(zip(("wins", "losses", "p"), mcnemar({i: r17["cot"][i] for i in l5},
                                                                       {i: src5[m][i] for i in l5})))
                         for m in ("adaptive", "always_on")}

# ---------------------------------------------------------------- GSM8K appendix (run-004 valid rows) and run log
agent.set_task("gsm8k")
m4, r4, _ = load("run-004")
lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & Accuracy (95\% CI) & Tokens/problem & $p$ vs CoT \\", r"\midrule"]
stats["gsm8k_valid_rows"] = []
for label, k in (("Direct answer", "direct"), ("Chain-of-thought (CoT)", "cot"), ("Reflection memory", "reflexion"),
                 ("Self-Refine", "self_refine")):
    s = m4[k]
    p = mcnemar(r4["cot"], r4[k])[2] if k != "cot" else None
    stats["gsm8k_valid_rows"].append({"method": k, "accuracy": s["accuracy"], "tokens": s["tokens_per_problem"], "p_vs_cot": p})
    lines.append(f"{label} & {100 * s['accuracy']:.1f} [{100 * s['acc_ci95'][0]:.1f}, {100 * s['acc_ci95'][1]:.1f}] & "
                 f"{s['tokens_per_problem']:.0f} & {fmt_p(p) if p is not None else '--'} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT_T, "gsm8k.tex"), "w").write("\n".join(lines) + "\n")

_log = os.path.join(ROOT, ".rstack", "experiments.jsonl")
if not os.path.exists(_log):  # layout of the public release
    _log = os.path.join(ROOT, "run_log", "experiments.jsonl")
runs = [json.loads(l) for l in open(_log)]
lines = [r"\begin{tabular}{llp{10.5cm}}", r"\toprule", r"Run & Status & Note \\", r"\midrule"]
TEX = {"\\": r"\textbackslash{}", "%": r"\%", "_": r"\_", "&": r"\&", "#": r"\#", "$": r"\$", "^": r"\^{}",
       "~": r"\~{}", "<": r"$<$", ">": r"$>$", "{": r"\{", "}": r"\}"}
for r in sorted(runs, key=lambda r: r["id"]):
    note = r["hypothesis"].split(". ")[0]
    if len(note) > 110:  # cut at a word boundary
        note = note[:110].rsplit(" ", 1)[0] + " ..."
    note = note.replace("->", "\x00")
    note = "".join(TEX.get(c, c) for c in note).replace("\x00", r"$\rightarrow$")
    note = note.encode("ascii", "ignore").decode()
    lines.append(f"{r['id']} & {r['status']} & {note} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT_T, "run_log.tex"), "w").write("\n".join(lines) + "\n")
stats.update(total_runs=len(runs), completed_runs=sum(r["status"] == "completed" for r in runs),
             failed_runs=sum(r["status"] == "failed" for r in runs), seeds=1,
             models=["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"])
stats["figures_generated"] = sorted(f for f in os.listdir(os.path.join(ROOT, "analysis", "figures")) if f.endswith(".pdf"))
stats["tables_generated"] = sorted(os.listdir(OUT_T))
json.dump(stats, open(os.path.join(ROOT, "analysis", "stats.json"), "w"), indent=2, default=float)

# ---------------------------------------------------------------- console summary
for key in ("main_qwen", "main_llama", "gsmhard_qwen"):
    print(f"\n== {key}")
    for r in stats[key]:
        extra = f"  d={100 * r['delta']:+5.1f} +{r['wins']}/-{r['losses']} p_holm={r['p_holm']:.4f}" if "delta" in r else ""
        print(f"  {r['label'][:28]:28s} {100 * r['accuracy']:5.1f} tok {r['tokens']:6.0f} x{r['x_cot']:.2f} [{r['run']}]{extra}")
print("\n== cot reproducibility", stats["cot_reproducibility"])
print("== gate_think", json.dumps({k: v for k, v in stats["gate_think"].items()}, default=float))
print("== llama level-5", stats["llama_level5"])
