"""
Supplementary figure: how reference points are matched to detected objects
==========================================================================

Top row (schematic): the five situations that occur in matching.
    a  point inside an object                  matched, d = 0
    b  point outside, closer than r            matched, d = distance to the edge
    c  point farther than r                    not matched
    d  two points in one object                only one is matched
    e  large object                            matched, but scored lower

Bottom panel (real data): the same situations in a held-out test image,
computed with the AdhGen matching function (split seed 60 solution).

Formulas and parameter values are kept out of the figure; they belong in the
figure legend.

Example
-------
    python make_matching_distance_figure.py
    python make_matching_distance_figure.py --image 1/Soft/Soft_6_M.tif
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon
import numpy as np
import pandas as pd

import adhgen_ga_area_weighted_micro_f1_windows as ga

OUTPUT_DIR = Path(str(_ADHGEN_ROOT / "Figure/Matching_distance"))
RESULT_DIR = Path(str(_ADHGEN_ROOT / "AdhGen result/split_seed_60"))
SEED60_SOLUTION = [30, 255, 109, 247, 0, 69, 12, 100]
R = float(ga.MATCH_RADIUS_PX)
A0 = float(ga.AREA_SOFT_LIMIT_PX)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
FRAME = "#e1e0d9"
OBJ = "#2a78d6"        # detected object (categorical slot 1)
OBJ_FILL = "#cde2fb"
PT = "#eb6834"         # reference point (categorical slot 2)

TITLE_SIZE = 10
TEXT_SIZE = 9


# =============================================================================
# Shared drawing primitives
# =============================================================================

def draw_point(ax, x, y, matched: bool, size: float = 11) -> None:
    ax.plot([x], [y], "o", markersize=size, zorder=6,
            markerfacecolor=PT if matched else "white",
            markeredgecolor=PT, markeredgewidth=2.2)


def draw_line(ax, p, q, label: str = "", side: str = "above") -> None:
    ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, linewidth=1.6, solid_capstyle="round", zorder=5)
    if label:
        dy = 0.9 if side == "above" else -0.9
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label, ha="center",
                va="bottom" if side == "above" else "top", fontsize=TEXT_SIZE, color=INK, style="italic")


# =============================================================================
# Schematic panels
# =============================================================================

def schematic_axis(ax, letter: str, title: str, result: str) -> None:
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME)
    ax.set_facecolor("white")
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=TITLE_SIZE, color=INK, pad=5)
    ax.text(0.5, -0.06, result, transform=ax.transAxes, ha="center", va="top",
            fontsize=TEXT_SIZE, color=INK_2)


def blob(ax, cx, cy, w, h, matched: bool) -> None:
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor=OBJ_FILL, edgecolor=OBJ, linewidth=2,
                         linestyle="-" if matched else (0, (3, 2)), zorder=2))


def radius_circle(ax, x, y, label: bool) -> None:
    ax.add_patch(Circle((x, y), R, fill=False, edgecolor=MUTED, linewidth=1.1,
                        linestyle=(0, (3, 2)), zorder=3))
    if label:
        ax.plot([x, x], [y, y + R], color=MUTED, linewidth=1.1, zorder=3)
        ax.text(x + 0.35, y + R / 2, "r", ha="left", va="center", fontsize=TEXT_SIZE,
                color=INK_2, style="italic")


def draw_schematics(axes) -> None:
    a, b, c, d, e = axes

    schematic_axis(a, "a", "Inside", "Matched, d = 0")
    blob(a, 0, 0, 7, 5, True)
    draw_point(a, 0.3, 0.2, True)

    schematic_axis(b, "b", "Close", "Matched, d < r")
    blob(b, -3.5, 0, 6, 4.4, True)
    radius_circle(b, 2.5, 0, label=True)
    draw_line(b, (2.5, 0), (-0.5, 0), "d", side="below")
    draw_point(b, 2.5, 0, True)

    schematic_axis(c, "c", "Too far", "Not matched, d > r")
    blob(c, -4.6, 0, 5, 4, False)
    radius_circle(c, 3.8, 0, label=False)
    draw_line(c, (3.8, 0), (-2.1, 0), "d", side="below")
    draw_point(c, 3.8, 0, False)

    schematic_axis(d, "d", "Shared object", "One point matched")
    blob(d, 0, 0, 13, 3.6, True)
    draw_point(d, -3.2, 0, True)
    draw_point(d, 3.2, 0, False)

    schematic_axis(e, "e", "Large object", "Matched, lower score")
    blob(e, 0, 0, 13, 8.5, True)
    draw_point(e, 0.5, 0.4, True)


# =============================================================================
# Real example
# =============================================================================

def local_path(path: str, data_root: Path) -> Path:
    parts = str(path).replace("\\", "/").split("/")
    return data_root / parts[-3] / parts[-2] / parts[-1]


def analyse_image(path: Path) -> dict:
    image = ga.preload_images([str(path)])[str(path)]
    reference = ga.build_image_reference(
        str(path), image, ga.POINT_CONFIG_FINAL, ga.NUCLEUS_CONFIG_FINAL,
        ga.CELL_CONFIG_FINAL, ga.REFERENCE_MASK_CONFIG,
    )
    objects, _, index_map = ga.detect_objects_in_union_mask(
        image, SEED60_SOLUTION, reference, ga.REFERENCE_MASK_CONFIG,
        replace(ga.GA_CONFIG, max_workers=1),
    )
    result = ga.match_points_to_objects_local(reference.points, objects, index_map, R, A0)
    return {"path": path, "image": image, "reference": reference, "objects": objects,
            "index_map": index_map, "result": result}


def window_counts(a: dict, y0: int, x0: int, h: int, w: int) -> dict:
    pts, res = a["reference"].points, a["result"]
    inside = (pts[:, 0] >= y0) & (pts[:, 0] < y0 + h) & (pts[:, 1] >= x0) & (pts[:, 1] < x0 + w)
    idx = set(np.nonzero(inside)[0].tolist())
    matched = {m["point_index"]: m for m in res["matches"] if m["point_index"] in idx}
    labels = [int(a["index_map"][int(pts[i, 0]), int(pts[i, 1])]) for i in idx]
    labels = [l for l in labels if l > 0]
    sub = a["index_map"][y0:y0 + h, x0:x0 + w]
    matched_obj = {m["object_index"] + 1 for m in res["matches"]}
    large = {m["object_index"] + 1 for m in matched.values() if a["objects"][m["object_index"]]["area"] > A0}
    return {
        "inside": sum(1 for m in matched.values() if m["distance"] == 0),
        "close": sum(1 for m in matched.values() if m["distance"] > 0),
        "unmatched_point": len(idx) - len(matched),
        "unmatched_object": len(set(np.unique(sub[sub > 0]).tolist()) - matched_obj),
        "shared_object": sum(1 for l in set(labels) if labels.count(l) >= 2),
        "large_object": len(large),
        "n_points": len(idx),
    }


def best_window(a: dict, h: int, w: int, step: int = 8) -> Tuple[int, int, dict, float]:
    mask = a["reference"].analysis_mask
    H, W = mask.shape
    best = (0, 0, {}, -np.inf)
    for y0 in range(0, H - h, step):
        for x0 in range(0, W - w, step):
            if mask[y0:y0 + h, x0:x0 + w].mean() < 0.9:
                continue
            c = window_counts(a, y0, x0, h, w)
            kinds = ("inside", "close", "unmatched_point", "unmatched_object", "shared_object", "large_object")
            score = sum(min(c[k], 3) for k in kinds) - 10 * sum(c[k] == 0 for k in kinds)
            score -= 0.2 * max(0, c["n_points"] - 28)          # keep the panel uncluttered
            if score > best[3]:
                best = (y0, x0, c, score)
    return best


UPSAMPLE = 8    # display only


def display_channel(path: Path) -> np.ndarray:
    """FA-marker plane at the full stored bit depth (the analysis itself uses the 8-bit read)."""
    try:
        import tifffile
        data = tifffile.imread(str(path))
        if data.ndim == 3 and data.shape[-1] >= 3:
            return data[:, :, 1].astype(np.float32)
    except Exception:
        pass
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, 1].astype(np.float32)


def smooth_outlines(index_map: np.ndarray, label: int, y0: int, x0: int, h: int, w: int, pad: int = 3):
    """Smooth display outline(s) of one object, in original pixel coordinates."""
    ya, xa = max(0, y0 - pad), max(0, x0 - pad)
    yb, xb = min(index_map.shape[0], y0 + h + pad), min(index_map.shape[1], x0 + w + pad)
    mask = (index_map[ya:yb, xa:xb] == label).astype(np.float32)
    up = cv2.resize(mask, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_NEAREST)
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=0.38 * UPSAMPLE)
    contours, _ = cv2.findContours((up >= 0.45).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outlines = []
    for c in contours:
        xy = c.reshape(-1, 2).astype(float)
        if len(xy) < 3:
            continue
        xy = (xy + 0.5) / UPSAMPLE - 0.5
        xy[:, 0] += xa
        xy[:, 1] += ya
        outlines.append(xy)
    return outlines


def nearest_on_outlines(x: float, y: float, outlines) -> np.ndarray:
    pts = np.vstack(outlines)
    return pts[int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))]


def draw_real(ax, a: dict, y0: int, x0: int, h: int, w: int) -> None:
    g = display_channel(a["path"])[y0:y0 + h, x0:x0 + w]
    lo, hi = np.percentile(g, 3), np.percentile(g, 99.6)
    norm = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.85
    # Lanczos upsampling followed by a mild unsharp mask: smooth edges without softening the signal.
    up = cv2.resize(norm, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_LANCZOS4)
    up = np.clip(up + 0.6 * (up - cv2.GaussianBlur(up, (0, 0), sigmaX=0.9 * UPSAMPLE)), 0, 1)
    shade = 1.0 - 0.85 * up                       # inverted: background white, signal dark
    ax.imshow(shade, cmap="gray", vmin=0, vmax=1, interpolation="antialiased",
              extent=(x0 - 0.5, x0 + w - 0.5, y0 + h - 0.5, y0 - 0.5))
    ax.set_xlim(x0 - 0.5, x0 + w - 0.5)
    ax.set_ylim(y0 + h - 0.5, y0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME)

    res, objects, index_map, pts = a["result"], a["objects"], a["index_map"], a["reference"].points
    matched_pt = {m["point_index"]: m for m in res["matches"]}
    matched_obj = {m["object_index"] for m in res["matches"]}

    outlines_by_object = {}
    sub = index_map[y0:y0 + h, x0:x0 + w]
    for label in np.unique(sub[sub > 0]):
        k = int(label) - 1
        outlines = smooth_outlines(index_map, int(label), y0, x0, h, w)
        outlines_by_object[k] = outlines
        style = "-" if k in matched_obj else (0, (2.5, 1.8))
        for xy in outlines:
            ax.add_patch(Polygon(xy, closed=True, fill=False, edgecolor=OBJ, linewidth=1.7,
                                 linestyle=style, joinstyle="round", zorder=3))

    for i, (y, x, _) in enumerate(pts):
        if not (y0 <= y < y0 + h and x0 <= x < x0 + w):
            continue
        if i in matched_pt and matched_pt[i]["distance"] > 0:
            k = matched_pt[i]["object_index"]
            outlines = outlines_by_object.get(k) or smooth_outlines(index_map, k + 1, y0, x0, h, w)
            if outlines:
                draw_line(ax, (x, y), nearest_on_outlines(float(x), float(y), outlines))
        draw_point(ax, float(x), float(y), i in matched_pt, size=8)

    ax.set_title("f  Example from a held-out image", loc="left", fontsize=TITLE_SIZE, color=INK, pad=5)


def draw_compact_key(ax) -> None:
    """One-line key grouped by element: what the symbol is, then what its style means."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.5
    fs = TEXT_SIZE - 0.5

    def label(x, text, bold=False):
        return ax.text(x, y, text, ha="left", va="center", fontsize=fs, color=INK,
                       fontweight="bold" if bold else "normal")

    # d and r are labelled in panel b and defined in the figure legend, so the key covers
    # only the two symbol types and their matched / not matched styles.
    x = 14
    label(x, "Reference point", bold=True)
    x += 14.5
    ax.plot([x], [y], "o", markersize=8, markerfacecolor=PT, markeredgecolor=PT, markeredgewidth=2)
    label(x + 1.4, "matched")
    x += 10
    ax.plot([x], [y], "o", markersize=8, markerfacecolor="white", markeredgecolor=PT, markeredgewidth=2)
    label(x + 1.4, "not matched")

    x += 19
    label(x, "Detected object", bold=True)
    x += 14.5
    ax.plot([x, x + 3.2], [y, y], color=OBJ, linewidth=2)
    label(x + 4.2, "matched")
    x += 12.5
    ax.plot([x, x + 3.2], [y, y], color=OBJ, linewidth=2, linestyle=(0, (2.5, 1.6)))
    label(x + 4.2, "not matched")


def legend_handles():
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor=PT,
               markeredgecolor=PT, markeredgewidth=2.2, label="Reference point, matched"),
        Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor="white",
               markeredgecolor=PT, markeredgewidth=2.2, label="Reference point, not matched"),
        Line2D([], [], color=OBJ, linewidth=2, label="Detected object, matched"),
        Line2D([], [], color=OBJ, linewidth=2, linestyle=(0, (3, 2)), label="Detected object, not matched"),
        Line2D([], [], color=INK, linewidth=1.6, label="Distance d to the object edge"),
        Line2D([], [], color=MUTED, linewidth=1.1, linestyle=(0, (3, 2)), label="Matching radius r"),
    ]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Matching-distance supplementary figure")
    parser.add_argument("--image", type=str, default="",
                        help="folder/condition/file of the test image; default: best example in the seed-60 test set")
    parser.add_argument("--window-height", type=int, default=96)
    parser.add_argument("--window-width", type=int, default=176)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--with-example", action="store_true",
                        help="Add panel f (real held-out image). Default: schematic panels a-e only.")
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.with_example:
        fig = plt.figure(figsize=(7.4, 2.05))
        fig.patch.set_facecolor(SURFACE)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.16], hspace=0.42,
                              left=0.02, right=0.98, top=0.88, bottom=0.02)
        top = gs[0].subgridspec(1, 5, wspace=0.16)
        draw_schematics([fig.add_subplot(top[j]) for j in range(5)])
        draw_compact_key(fig.add_subplot(gs[1]))
        for ext in ("png", "svg", "tiff"):
            fig.savefig(out / f"Supp_matching_distance.{ext}", dpi=300, facecolor=SURFACE)
        plt.close(fig)
        stale = out / "real_example.csv"
        if stale.exists():
            stale.unlink()
        print("Saved to", out)
        return

    data_root = ga.DATA_ROOT
    if args.image:
        candidates = [data_root / args.image]
    else:
        manifest = pd.read_csv(RESULT_DIR / "split_manifest.csv")
        candidates = [local_path(p, data_root) for p in manifest.loc[manifest["split"] == "test", "path"]]

    best = None
    for path in candidates:
        a = analyse_image(path)
        y0, x0, counts, score = best_window(a, args.window_height, args.window_width)
        if best is None or score > best[4]:
            best = (a, y0, x0, counts, score)
    a, y0, x0, counts, _ = best
    wh, ww = args.window_height, args.window_width
    print("Image:", a["path"], "window (y0, x0):", (y0, x0), counts)

    fig = plt.figure(figsize=(7.4, 6.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2.7, 0.34], hspace=0.22,
                          left=0.02, right=0.98, top=0.96, bottom=0.01)
    top = gs[0].subgridspec(1, 5, wspace=0.16)
    draw_schematics([fig.add_subplot(top[j]) for j in range(5)])

    draw_real(fig.add_subplot(gs[1]), a, y0, x0, wh, ww)
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis("off")
    ax_leg.legend(handles=legend_handles()[:5], loc="center", ncol=3, frameon=False,
                  fontsize=TEXT_SIZE, labelcolor=INK, handlelength=2.2, columnspacing=1.8)

    for ext in ("png", "svg", "tiff"):
        fig.savefig(out / f"Supp_matching_distance.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)

    pd.DataFrame([{"image": "/".join(str(a["path"]).replace("\\", "/").split("/")[-3:]),
                   "window_y0": y0, "window_x0": x0, "window_height_px": wh, "window_width_px": ww,
                   **counts}]
                 ).to_csv(out / "real_example.csv", index=False)
    print("Saved to", out)


if __name__ == "__main__":
    main()
