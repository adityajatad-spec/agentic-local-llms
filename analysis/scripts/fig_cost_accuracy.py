"""Accuracy vs. inference cost on MATH-500 for both models: external scaffolds vs. internal test-time compute."""
import json
import os

from style import BLUE, INK, INK_2, NEUTRAL, ORANGE, ROOT, SURFACE, plt, save

stats = json.load(open(os.path.join(ROOT, "analysis", "stats.json")))
EXTERNAL = {"Self-Refine": "Self-Refine", "Reflection memory": "Reflection", "ReAct, REPL tool": "ReAct (REPL)",
            "ReAct, native calls": "ReAct (native)", "Always-on pipeline": "Always-on",
            "Always-on, native calls": "Always-on (native)", "Adaptive controller": "Adaptive", "Adaptive controller$^\\ddagger$": "Adaptive"}
INTERNAL = {"Self-consistency@5": "Self-consistency@5", "Thinking mode": "Thinking", "Gate $\\rightarrow$ thinking$^\\ddagger$": "Gate→thinking"}
PANELS = [("Qwen3-8B", stats["main_qwen"], ["Self-Refine", "Reflection memory", "ReAct, REPL tool", "Always-on pipeline",
                                            "Adaptive controller$^\\ddagger$"]),
          ("Llama-3.1-8B", stats["main_llama"], ["Self-Refine", "Reflection memory", "ReAct, native calls",
                                                 "Always-on, native calls", "Adaptive controller$^\\ddagger$"])]
OFF = {  # (dx, dy, ha) label offsets in points, per panel and label
    "Qwen3-8B": {"CoT": (-8, 0, "right"), "Self-Refine": (8, -5, "left"), "Reflection": (-4, 11, "right"),
                 "ReAct (REPL)": (0, -13, "center"), "Always-on": (8, 0, "left"), "Adaptive": (8, 0, "left"),
                 "Self-consistency@5": (8, 2, "left"), "Thinking": (0, 10, "center"), "Gate→thinking": (8, 0, "left"),
                 "2-agree gate": (-8, 0, "right")},
    "Llama-3.1-8B": {"CoT": (-8, 0, "right"), "Self-Refine": (8, 0, "left"), "Reflection": (-8, -4, "right"),
                     "ReAct (native)": (8, 0, "left"), "Always-on (native)": (8, 0, "left"), "Adaptive": (0, 11, "center"),
                     "Self-consistency@5": (0, 11, "center")}}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, (model, rows, keep) in zip(axes, PANELS):
    by = {r["label"]: r for r in rows}
    cot = by["Chain-of-thought (CoT)"]
    ax.axhline(100 * cot["accuracy"], color=NEUTRAL, lw=1, ls=(0, (4, 3)), zorder=1)
    pts = [("CoT", cot, NEUTRAL)] + [(EXTERNAL[k], by[k], BLUE) for k in keep]
    pts += [(INTERNAL[k], by[k], ORANGE) for k in INTERNAL if k in by]
    for name, r, color in pts:
        x, y = r["tokens"], 100 * r["accuracy"]
        lo, hi = 100 * r["ci95"][0], 100 * r["ci95"][1]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color, elinewidth=1.2, zorder=2)
        ax.scatter([x], [y], s=44, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy, ha = OFF[model][name]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK, ha=ha, va="center")
    if model == "Qwen3-8B":  # cheaper 2-agree gate, rebuilt from the same independent samples (hollow marker)
        sim = stats["gate_think"]["gate2"]
        ax.scatter([sim["tokens"]], [100 * sim["accuracy"]], s=44, facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.6, zorder=3)
        dx, dy, ha = OFF[model]["2-agree gate"]
        ax.annotate("2-agree gate", (sim["tokens"], 100 * sim["accuracy"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=INK, ha=ha, va="center")
    ax.set_xscale("log")
    ax.set_xlim(500, 30000)
    ax.set_title(model, fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Tokens per problem (log scale)")
axes[0].set_ylim(25, 95)
axes[0].set_ylabel("MATH-500 accuracy (%)")
h1 = plt.Line2D([], [], marker="o", ls="", color=BLUE, label="External scaffold")
h2 = plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="Internal test-time compute")
h3 = plt.Line2D([], [], color=NEUTRAL, ls=(0, (4, 3)), label="CoT accuracy (same run)")
fig.legend(handles=[h1, h2, h3], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout()
save(fig, "cost_accuracy")
