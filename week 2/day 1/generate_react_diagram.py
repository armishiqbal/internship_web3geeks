"""ReAct diagram for Week 2 Day 1."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent


def box(ax, xy, w, h, text, fc):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2, edgecolor="#0F172A", facecolor=fc)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10, color="#0F172A", fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title("ReAct loop  —  Reason → Act → Observe → repeat", loc="left", fontsize=12, fontweight="bold", color="#0F172A")
    box(ax, (0.3, 1.35), 2.0, 0.9, "REASON\n(model thinks)", "#E0F2FE")
    box(ax, (2.9, 1.35), 2.0, 0.9, "ACT\n(tool_use JSON)", "#FEF3C7")
    box(ax, (5.5, 1.35), 2.0, 0.9, "OBSERVE\n(tool_result)", "#DCFCE7")
    box(ax, (8.1, 1.35), 1.6, 0.9, "FINAL\ntext", "#F1F5F9")
    for x0, x1 in [(2.3, 2.9), (4.9, 5.5), (7.5, 8.1)]:
        ax.add_patch(FancyArrowPatch((x0, 1.8), (x1, 1.8), arrowstyle="-|>", mutation_scale=12, color="#334155"))
    ax.annotate("", xy=(1.3, 1.35), xytext=(6.5, 0.45), arrowprops=dict(arrowstyle="-|>", color="#64748B", connectionstyle="arc3,rad=0.25"))
    ax.text(4.2, 0.35, "repeat until no tool_use  (max_iterations guard)", ha="center", fontsize=8.5, color="#475569")
    fig.tight_layout()
    out = HERE / "react_loop.png"
    fig.savefig(out, dpi=160)
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
