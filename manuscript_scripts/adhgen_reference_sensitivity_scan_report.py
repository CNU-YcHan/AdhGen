"""
Figures and table for the reference-point sensitivity scan
==========================================================

Reads the output of adhgen_reference_sensitivity_scan.py and writes, into
<output-root>/<base>/report/:

    sensitivity_table.csv               one row per reference setting
    sensitivity_fitness_profiles.png    training fitness as a function of G min,
                                        one panel per factor (small multiples)
    sensitivity_readout_by_g_min.png    held-out FA count per cell and the
                                        Soft/Stiff test as a function of G min

Example
-------
    python adhgen_reference_sensitivity_scan_report.py --base submitted
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_ROOT = Path(str(_SCRIPT_DIR / "reference_sensitivity_scan_results"))

# Chart chrome and ink
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#e1e0d9"

# Ordered levels of one factor: one hue, light -> dark (ordinal steps)
ORDINAL_STEPS = {
    1: ["#3987e5"],
    2: ["#6da7ec", "#1c5cab"],
    3: ["#86b6ef", "#3987e5", "#184f95"],
}
# Identity colours of the two substrate conditions
SOFT = "#2a78d6"
STIFF = "#e34948"

PANELS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Smoothing sigma", [
        ("smooth_sigma_1.0", "1.0 px"), ("baseline", "1.5 px (submitted)"), ("smooth_sigma_2.0", "2.0 px")]),
    ("Minimum separation", [
        ("min_distance_6", "6 px"), ("baseline", "8 px (submitted)"), ("min_distance_10", "10 px")]),
    ("Threshold percentile", [
        ("percentile_99.0", "99.0"), ("baseline", "99.5 (submitted)"), ("percentile_99.75", "99.75")]),
    ("Random removal of points", [
        ("baseline", "none (submitted)"), ("remove_10", "10%"), ("remove_20", "20%"), ("remove_30", "30%")]),
    ("Random shift of points", [
        ("baseline", "none (submitted)"), ("shift_1", "up to 1 px"), ("shift_2", "up to 2 px")]),
]
SETTING_ORDER = [
    "baseline", "smooth_sigma_1.0", "smooth_sigma_2.0", "min_distance_6", "min_distance_10",
    "percentile_99.0", "percentile_99.75", "remove_10", "remove_20", "remove_30", "shift_1", "shift_2",
]


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load_results(base_root: Path) -> dict:
    results = {}
    for name in SETTING_ORDER:
        path = base_root / name / "result.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                results[name] = json.load(file)
    return results


def base_area_curve(base_root: Path, name: str, base_area: Tuple[int, int]) -> pd.DataFrame:
    """G min profile at the starting area bounds (coarse + fine points)."""
    curve = pd.read_csv(base_root / name / "scan_curve.csv")
    curve = curve[
        curve["stage"].isin(["g_coarse", "g_fine"])
        & (curve["area_min"] == base_area[0])
        & (curve["area_max"] == base_area[1])
    ]
    return curve.drop_duplicates("g_min").sort_values("g_min")


def write_table(results: dict, report_dir: Path) -> pd.DataFrame:
    rows = []
    base = results.get("baseline")
    for name in SETTING_ORDER:
        if name not in results:
            continue
        r = results[name]
        row = {
            "setting": name,
            "description": r["description"],
            "reference_points_per_image": round(r["mean_reference_points_per_image"], 1),
            "reference_points_per_cell": round(r["mean_reference_points_per_cell"], 1),
            "best_g_min": r["g_min"],
            "best_area_min": r["area_min"],
            "best_area_max": r["area_max"],
            "train_fitness": round(r["train_fitness"], 4),
            "train_fitness_of_starting_solution": round(r["train_fitness_of_base_solution"], 4),
            "test_fitness_on_submitted_reference": round(r["test_fitness_on_baseline_reference"], 4),
            "test_soft_median_fa_per_cell": round(r["test_soft_median_fa_per_cell"], 2),
            "test_stiff_median_fa_per_cell": round(r["test_stiff_median_fa_per_cell"], 2),
            "test_mann_whitney_u": r["test_mann_whitney_u"],
            "test_mann_whitney_p": r["test_mann_whitney_p"],
            "test_cliffs_delta": round(r["test_cliffs_delta"], 3),
        }
        if base is not None:
            row["g_min_shift"] = r["g_min"] - base["g_min"]
            for key, label in (
                ("test_soft_median_fa_per_cell", "pct_change_soft_median"),
                ("test_stiff_median_fa_per_cell", "pct_change_stiff_median"),
            ):
                row[label] = round(100.0 * (r[key] - base[key]) / base[key], 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "sensitivity_table.csv", index=False)
    return table


def plot_fitness_profiles(base_root: Path, results: dict, report_dir: Path) -> None:
    base_area = (12, 100)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    flat = list(axes.ravel())

    for ax, (title, levels) in zip(flat, PANELS):
        style_axis(ax)
        present = [(name, label) for name, label in levels if name in results]
        varied = [name for name, _ in present if name != "baseline"]
        colors = dict(zip(varied, ORDINAL_STEPS.get(len(varied), ORDINAL_STEPS[3])))

        for name, label in present:
            curve = base_area_curve(base_root, name, base_area)
            if curve.empty:
                continue
            color = INK if name == "baseline" else colors[name]
            ax.plot(
                curve["g_min"], curve["train_fitness"], color=color, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", label=label,
                zorder=3 if name == "baseline" else 2,
            )
            # Optimum along this profile.
            best = curve.loc[curve["train_fitness"].idxmax()]
            ax.plot(
                [best["g_min"]], [best["train_fitness"]], marker="o", markersize=7.5,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.6,
                linestyle="none", zorder=4,
            )

        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
        legend = ax.legend(
            loc="lower center", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY,
            handlelength=1.6, borderaxespad=0.3, ncol=1,
        )
        legend.set_zorder(5)

    for ax in flat[len(PANELS):]:
        ax.axis("off")
        ax.text(
            0.0, 0.98,
            "Each line: fitness on the 143 training\nimages as G min is varied, the other\n"
            "parameters fixed.\n\nDot: G min with the highest fitness\nfor that reference setting.",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK_SECONDARY,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Green lower bound, G min (8-bit intensity)", fontsize=8.5, color=INK_SECONDARY)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fitness (training images)", fontsize=8.5, color=INK_SECONDARY)

    fig.tight_layout(w_pad=1.6, h_pad=1.8)
    for extension in ("png", "svg"):
        fig.savefig(report_dir / f"sensitivity_fitness_profiles.{extension}", dpi=300, facecolor=SURFACE)
    plt.close(fig)


def plot_readout(base_root: Path, results: dict, report_dir: Path) -> None:
    path = base_root / "readout_profile_by_g_min.csv"
    if not path.exists() or not results:
        return
    profile = pd.read_csv(path).sort_values("g_min")
    best_values = [r["g_min"] for r in results.values()]
    low, high = min(best_values), max(best_values)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.patch.set_facecolor(SURFACE)

    for ax in (top, bottom):
        style_axis(ax)
        ax.axvspan(low - 0.5, high + 0.5, color=BAND, zorder=0, linewidth=0)

    for column, color, name in (
        ("test_soft_median_fa_per_cell", SOFT, "Soft"),
        ("test_stiff_median_fa_per_cell", STIFF, "Stiff"),
    ):
        top.plot(profile["g_min"], profile[column], color=color, linewidth=2,
                 solid_capstyle="round", solid_joinstyle="round", label=name)
    top.set_ylabel("Median FA count per cell\n(62 held-out images)", fontsize=8.5, color=INK_SECONDARY)
    top.set_ylim(bottom=0)
    top.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_SECONDARY, handlelength=1.6)
    top.text(
        (low + high) / 2, top.get_ylim()[1] * 0.97,
        f"G min selected across\nall reference settings\n({low}\u2013{high})",
        ha="center", va="top", fontsize=7.5, color=INK_SECONDARY,
    )

    bottom.plot(profile["g_min"], profile["test_mann_whitney_p"], color=INK, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round")
    bottom.axhline(0.05, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    bottom.annotate("p = 0.05", xy=(profile["g_min"].iloc[0], 0.05), xytext=(2, 4),
                    textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)
    bottom.set_yscale("log")
    bottom.set_ylabel("Soft vs Stiff\nMann\u2013Whitney p", fontsize=8.5, color=INK_SECONDARY)
    bottom.set_xlabel("Green lower bound, G min (8-bit intensity)", fontsize=8.5, color=INK_SECONDARY)

    fig.tight_layout(h_pad=1.2)
    fig.subplots_adjust(right=0.9)
    for extension in ("png", "svg"):
        fig.savefig(report_dir / f"sensitivity_readout_by_g_min.{extension}", dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report for the reference sensitivity scan")
    parser.add_argument("--base", default="submitted")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"

    base_root = Path(args.output_root) / args.base
    report_dir = base_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(base_root)
    if not results:
        raise SystemExit(f"No completed settings under {base_root}")

    table = write_table(results, report_dir)
    plot_fitness_profiles(base_root, results, report_dir)
    plot_readout(base_root, results, report_dir)

    print(f"Settings: {len(table)}  ->  {report_dir}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
