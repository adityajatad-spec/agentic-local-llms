"""Accuracy by MATH difficulty level (1-5) for key methods, both models (small multiples, shared y-axis)."""
import json
import os

from style import ENTITY, INK, ROOT, SURFACE, plt, save

stats = json.load(open(os.path.join(ROOT, "analysis", "stats.json")))
L = stats["acc_by_level"]
PANELS = [("Qwen3-8B", [("CoT", "run-018:cot", "cot"), ("Thinking", "run-018:think", "think"),
                        ("Self-consistency@5", "run-023:sc5", "sc5"), ("ReAct (REPL)", "run-014:react_repl", "react"),
                        ("Always-on", "run-007:always_on", "always_on")]),
          ("Llama-3.1-8B", [("CoT", "run-017:cot", "cot"), ("ReAct (native)", "run-015:react_native", "react"),
                            ("Always-on", "run-017:always_on", "always_on"), ("Adaptive", "rebuilt-llama:adaptive", "adaptive"),
                            ("Self-consistency@5", "run-024:sc5", "sc5")])]


def spread(ys, gap=5.5):
    """Push line-end label positions apart so no two are closer than `gap` (keeps original order)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for _ in range(50):
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < gap:
                mid = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = mid - gap / 2, mid + gap / 2
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, (model, series) in zip(axes, PANELS):
    ends = []
    for label, key, ent in series:
        ys = [100 * L[key][str(l)]["acc"] for l in range(1, 6)]
        ax.plot(range(1, 6), ys, color=ENTITY[ent], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=2)
        ends.append(ys[-1])
    for (label, _, _), y in zip(series, spread(ends)):
        ax.text(5.15, y, label, va="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.8, 6.6)
    ax.set_xlabel("MATH difficulty level")
    ax.set_title(model, fontsize=9, color=INK, loc="left")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 105)
fig.tight_layout()
save(fig, "accuracy_by_level")
