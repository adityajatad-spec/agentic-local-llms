"""Shared figure style (reference palette, light mode; validated with the dataviz palette validator)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#c9c8c3"
BLUE = "#2a78d6"     # categorical slot 1 / diverging cool pole
ORANGE = "#eb6834"   # categorical slot 2
RED = "#e34948"      # diverging warm pole
NEUTRAL = "#a8a7a2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": RULE, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "legend.fontsize": 8, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# Categorical slots in fixed order (validated: adjacent CVD >= 9.1, normal-vision >= 19.6; slots 3-5 need direct labels)
AQUA, YELLOW, MAGENTA, GREEN = "#1baf7a", "#eda100", "#e87ba4", "#008300"
# Color follows the entity across every figure
ENTITY = {"cot": BLUE, "think": ORANGE, "sc5": AQUA, "react": YELLOW, "always_on": MAGENTA, "adaptive": GREEN}
