"""Answers fixed vs. broken by each self-correction mechanism (run-007 Self-Refine, run-008 verifiers)."""
import json
import os

from style import BLUE, INK, INK_2, NEUTRAL, RED, ROOT, plt, save

stats = json.load(open(os.path.join(ROOT, "analysis", "stats.json")))
ver = {v["verifier"]: v for v in stats["verifier_ablation"]}
sr = stats["self_refine_flips"]
items = [("Self-Refine\n(critique + rewrite)", sr["fixed"], sr["broke"]),
         ("Grounded verifier\n(code re-derivation)", ver["always_on"]["fixed"], ver["always_on"]["broke"]),
         ("Intrinsic verifier\n(self-critique, CoT fallback)", ver["always_on_intrinsic"]["fixed"],
          ver["always_on_intrinsic"]["broke"])]

fig, ax = plt.subplots(figsize=(5.2, 2.4))
ys = list(range(len(items)))[::-1]
for y, (name, fixed, broke) in zip(ys, items):
    ax.barh(y, fixed, height=0.5, color=BLUE, zorder=2)
    ax.barh(y, -broke, height=0.5, color=RED, zorder=2)
    ax.text(fixed + 1, y, f"+{fixed}", va="center", fontsize=8, color=INK)
    ax.text(-broke - 1, y, f"−{broke}", va="center", ha="right", fontsize=8, color=INK)
ax.axvline(0, color=NEUTRAL, lw=1, zorder=3)
ax.set_yticks(ys)
ax.set_yticklabels([i[0] for i in items])
ax.set_xlim(-48, 48)
ax.set_xticks([-40, -20, 0, 20, 40])
ax.set_xticklabels(["40", "20", "0", "20", "40"])
ax.set_xlabel("Problems", color=INK_2)
ax.text(-47, len(items) - 0.35, "broke (right → wrong)", color=INK_2, fontsize=7.5, ha="left")
ax.text(47, len(items) - 0.35, "fixed (wrong → right)", color=INK_2, fontsize=7.5, ha="right")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
save(fig, "verifier_flips")
