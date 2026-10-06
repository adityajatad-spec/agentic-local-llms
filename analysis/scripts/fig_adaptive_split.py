"""Accuracy on problems the adaptive gate accepted vs. escalated (run-007)."""
import json
import os

import numpy as np

from style import BLUE, INK, ORANGE, ROOT, plt, save

stats = json.load(open(os.path.join(ROOT, "analysis", "stats.json")))
split = stats["adaptive_split"]
order = ["cot", "reflexion", "react", "always_on", "adaptive"]
LABEL = {"cot": "CoT", "reflexion": "Reflection\nmemory", "react": "ReAct +\nPython", "always_on": "Always-on\npipeline",
         "adaptive": "Adaptive\n(escalates to\npipeline)"}
acc = [100 * split["by_method"][m]["accepted"] for m in order]
esc = [100 * split["by_method"][m]["escalated"] for m in order]

fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(order))
w = 0.36
gap = 0.02  # surface gap between the paired bars
ax.bar(x - w / 2 - gap / 2, acc, w, color=BLUE, label=f"Accepted by gate (n={split['n_accepted']})", zorder=2)
ax.bar(x + w / 2 + gap / 2, esc, w, color=ORANGE, label=f"Escalated by gate (n={split['n_escalated']})", zorder=2)
for i in (0, 4):  # selective direct labels: the reference method and the controller
    ax.text(x[i] - w / 2 - gap / 2, acc[i] + 1.5, f"{acc[i]:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.text(x[i] + w / 2 + gap / 2, esc[i] + 1.5, f"{esc[i]:.0f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[m] for m in order])
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy on subset (%)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax.tick_params(axis="x", length=0)
save(fig, "adaptive_split")
