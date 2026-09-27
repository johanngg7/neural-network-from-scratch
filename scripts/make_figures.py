"""Plot validation accuracy per epoch from results/authorship.json.

    python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"MLP (numpy, from scratch)": "#2a78d6", "MLP (Keras)": "#eb6834", "Softmax regression (numpy)": "#898781"}


def main():
    curves = json.loads((ROOT / "results" / "authorship.json").read_text())["valid_accuracy_curves"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=160)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    for name, acc in curves.items():
        epochs = range(1, len(acc) + 1)
        ax.plot(epochs, [a * 100 for a in acc], color=COLORS[name], linewidth=2, label=name)
    ax.set_xlabel("Epoch", color=INK_2)
    ax.set_ylabel("Validation accuracy (%)", color=INK_2)
    ax.set_ylim(55, 95)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=INK_2, loc="lower right")
    ax.set_title("Hand-written MLP vs. the same network in Keras", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "validation_accuracy.png")


if __name__ == "__main__":
    main()
