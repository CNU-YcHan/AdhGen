"""Revised Fig. 3 (reviewer comment 1-11), keeping the submitted layout and style.

Changes from the submitted figure (everything else is kept):
  a: "from A" / "from B" labels in the child column; parents labelled as parameter sets.
  a, b: section headers marked "(example)".
  a, b: labels of narrow ranges no longer overlap.
  (draw_cycle() is kept but not used.)
  c: ranges of the best parameter set drawn as min-max bars instead of random points;
     Gen 0 is labelled as the initial search range instead of "Fitness : 0.000".
"""

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})

HISTORY = Path(str(_ADHGEN_ROOT / "AdhGen result/split_seed_60/ga_history.csv"))
OUT_DIR = Path(str(_ADHGEN_ROOT / "Figure/Fig3"))

ROWS = ["R", "G", "B", "Area"]
DOT = {"R": "red", "G": "green", "B": "blue", "Area": "black"}
FILL = {"R": "#ffd6d6", "G": "#d9eedb", "B": "#d6d6fb", "Area": "#d9d9d9"}
TEXT = {"R": "#e01b1b", "G": "#18a33a", "B": "#1a1ae0", "Area": "#111111"}
LIMIT = {"R": 255, "G": 255, "B": 255, "Area": 100}

PARENT_A = {"R": (24, 178), "G": (51, 126), "B": (3, 61), "Area": (13, 49)}
PARENT_B = {"R": (13, 236), "G": (49, 159), "B": (2, 48), "Area": (30, 70)}
SOURCE = {"R": "A", "G": "A", "B": "B", "Area": "B"}
CHILD = {k: (PARENT_A if SOURCE[k] == "A" else PARENT_B)[k] for k in ROWS}
MUTATED = {"R": (8, 141), "G": (67, 158), "B": (2, 29), "Area": (44, 74)}
DELTA = {k: max(1, (CHILD[k][1] - CHILD[k][0]) // 2) for k in ROWS}

rng = np.random.default_rng(0)
POINTS = {k: np.sort(rng.integers(0, LIMIT[k] + 1, 60)) for k in ROWS}


def range_panel(ax, key, lo, hi, show_y, letter):
    top = LIMIT[key]
    ax.axhspan(lo, hi, color=FILL[key], lw=0)
    ax.axhline(lo, color=FILL[key], lw=1)
    ax.axhline(hi, color=FILL[key], lw=1)
    pts = POINTS[key]
    inside = (pts >= lo) & (pts <= hi)
    ax.scatter(np.full((~inside).sum(), 0.5), pts[~inside], s=22, color="#d3d3d3", lw=0, zorder=2)
    ax.scatter(np.full(inside.sum(), 0.5), pts[inside], s=40, color=DOT[key], lw=0, zorder=3)
    pad = 0.03 * top
    if hi - lo < 0.25 * top:  # narrow range: stack labels above the band so they do not overlap
        ax.text(0.08, hi + pad, f"max={hi}\nmin={lo}", color=TEXT[key], va="bottom", fontsize=11,
                linespacing=1.0)
    else:
        ax.text(0.08, hi - pad, f"max={hi}", color=TEXT[key], va="top", fontsize=11)
        ax.text(0.08, lo + pad, f"min={lo}", color=TEXT[key], va="bottom", fontsize=11)
    if letter:
        ax.text(0.02, 0.97, key, transform=ax.transAxes, color=TEXT[key], va="top", fontsize=17)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top)
    ax.set_xticks([])
    ax.set_yticks([0, 50, 100, 150, 200, 250] if top == 255 else [0, 20, 40, 60, 80, 100])
    if not show_y:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=12, length=4)


def red_box(ax):
    for s in ax.spines.values():
        s.set_edgecolor("#e00000")
        s.set_linewidth(2.8)


def tag(ax, text, color):
    ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color=color, fontweight="bold")


def draw_ab(fig, spec):
    grid = spec.subgridspec(4, 6, width_ratios=[1, 1, 1, 0.12, 1, 1], wspace=0.04, hspace=0.06)
    titles = ["Parent A\n(parameter set, Gen 10)", "Parent B\n(parameter set, Gen 10)",
              "Child\n(after crossover)", "Child\n(before mutation)", "Child\n(after mutation)"]
    columns = [PARENT_A, PARENT_B, CHILD, CHILD, MUTATED]
    slots = [0, 1, 2, 4, 5]
    axes = {}
    for i, key in enumerate(ROWS):
        for j, (vals, slot) in enumerate(zip(columns, slots)):
            ax = fig.add_subplot(grid[i, slot])
            range_panel(ax, key, *vals[key], show_y=slot in (0, 4), letter=slot in (0, 4))
            if (j == 0 and SOURCE[key] == "A") or (j == 1 and SOURCE[key] == "B"):
                red_box(ax)
            if j == 2:
                tag(ax, f"from {SOURCE[key]}", "#e00000")
            if i == 0:
                ax.set_title(titles[j], fontsize=14, pad=6)
            axes[(i, j)] = ax
    return axes


def draw_c(fig, spec):
    """Min-max range of the best parameter set at selected generations (split seed 60)."""
    from matplotlib.patches import Rectangle
    hist = pd.read_csv(HISTORY)
    hist["best_individual"] = hist["best_individual"].apply(ast.literal_eval)
    gens = [0, 20, 40, 60, 80, 100]
    grid = spec.subgridspec(1, len(gens), wspace=0.08)
    for i, gen in enumerate(gens):
        ax = fig.add_subplot(grid[0, i])
        if gen == 0:
            ind, title = [0, 255, 0, 255, 0, 255, 1, 100], "Gen 0\nInitial range"
        else:
            r = hist.loc[hist.generation == gen].iloc[0]
            ind, title = r.best_individual, f"Gen {gen}\nFitness : {r.fitness_max:.3f}"
        for x, ch in enumerate(["R", "G", "B"]):
            lo, hi = ind[2 * x], ind[2 * x + 1]
            ax.add_patch(Rectangle((x - 0.3, lo), 0.6, hi - lo, facecolor=DOT[ch], alpha=0.35,
                                   edgecolor=DOT[ch], lw=1.5))
            ax.plot([x - 0.3, x + 0.3], [lo, lo], color=DOT[ch], lw=2.5)
            ax.plot([x - 0.3, x + 0.3], [hi, hi], color=DOT[ch], lw=2.5)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["R", "G", "B"], fontsize=13)
        ax.set_xlim(-0.7, 2.7)
        ax.set_ylim(-10, 265)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)
        if i == 0:
            ax.set_ylabel("Intensity (0-255)", fontsize=13)
        else:
            ax.set_yticklabels([])


def draw_cycle(fig, rect):
    """One-line GA cycle; panels a-c are placed on the steps they illustrate."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    steps = [("Population\n(100 parameter sets)", None),
             ("Fitness evaluation\n(match to reference points)", None),
             ("Parent selection\n(tournament)", None),
             ("Crossover", "a"),
             ("Mutation", "b"),
             ("Next generation\n(best set kept)", None)]
    w, gap, y, h = 14.0, 2.6, 3.8, 5.8
    xs = [1.5 + k * (w + gap) for k in range(len(steps))]
    for x, (label, panel) in zip(xs, steps):
        hl = panel is not None
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#fff1f1" if hl else "#f4f4f4",
                                    ec="#e00000" if hl else "#888888", lw=1.6 if hl else 1.0))
        text = f"{label} ({panel})" if hl else label
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11,
                fontweight="bold" if hl else "normal")
    for k in range(len(steps) - 1):
        ax.add_patch(FancyArrowPatch((xs[k] + w + 0.2, y + h / 2), (xs[k + 1] - 0.2, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#444444"))
    x0, x1 = xs[0] + w / 2, xs[-1] + w / 2
    y_ret = 1.6
    ax.plot([x1, x1, x0], [y - 0.2, y_ret, y_ret], color="#444444", lw=1.3)
    ax.add_patch(FancyArrowPatch((x0, y_ret), (x0, y - 0.2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="#444444"))
    ax.text((x0 + x1) / 2, y_ret, "Repeated for 100 generations (c)",
            ha="center", va="center", fontsize=12, color="#333333", fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=3))


def main():
    fig = plt.figure(figsize=(14.3, 13.8))
    gs = GridSpec(2, 1, height_ratios=[4, 1.2], hspace=0.22, figure=fig,
                  left=0.06, right=0.99, top=0.895, bottom=0.03)
    draw_ab(fig, gs[0])
    draw_c(fig, gs[1])
    fig.text(0.01, 0.95, "a", fontsize=28, fontweight="bold")
    fig.text(0.315, 0.948, "Crossover (example)", fontsize=18, fontweight="bold", ha="center")
    fig.text(0.6, 0.95, "b", fontsize=28, fontweight="bold")
    fig.text(0.81, 0.948, "Mutation (example)", fontsize=18, fontweight="bold", ha="center")
    fig.text(0.01, 0.255, "c", fontsize=28, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "tiff"):
        kw = {"dpi": 300} if ext != "svg" else {}
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT_DIR / f"Fig3_revised.{ext}", **kw)
    print("saved", OUT_DIR / "Fig3_revised.png", DELTA)


if __name__ == "__main__":
    main()
