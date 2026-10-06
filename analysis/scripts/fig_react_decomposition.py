"""ReAct accuracy by tool semantics and call format, against same-run CoT, for both models."""
import json
import os

import numpy as np

from style import AQUA, BLUE, INK, INK_2, NEUTRAL, ORANGE, ROOT, plt, save

stats = json.load(open(os.path.join(ROOT, "analysis", "stats.json")))
dec = stats["react_decomposition"]
models = list(dec)
VAR = [("print-only", "Text format, print-only tool", BLUE), ("REPL", "Text format, REPL tool", ORANGE),
       ("native", "Native calls, REPL tool", AQUA)]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(models))
w, gap = 0.25, 0.02
for j, (key, label, color) in enumerate(VAR):
    vals = [100 * next(r for r in dec[m] if r["variant"] == key)["accuracy"] for m in models]
    pos = x + (j - 1) * (w + gap)
    ax.bar(pos, vals, w, color=color, label=label, zorder=2)
    for p, v in zip(pos, vals):  # value inside the bar top when it fits, else just above (keeps clear of the CoT line)
        inside = v >= 15
        ax.text(p, v - 6 if inside else v + 1.2, f"{v:.0f}", ha="center", fontsize=7.5,
                color="#ffffff" if inside else INK, fontweight="bold" if inside else "normal")
for i, m in enumerate(models):  # CoT reference from the REPL/native run (same backend)
    cot = 100 * next(r for r in dec[m] if r["variant"] == "REPL")["cot_same_run"]
    ax.hlines(cot, x[i] - 1.6 * w, x[i] + 1.6 * w, colors=NEUTRAL, linestyles=(0, (4, 3)), lw=1.2, zorder=3)
    ax.text(x[i] + 1.65 * w, cot, f"CoT {cot:.0f}", va="center", fontsize=7.5, color=INK_2)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 100)
ax.set_ylabel("MATH-500 accuracy (%)")
ax.tick_params(axis="x", length=0)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
save(fig, "react_decomposition")
