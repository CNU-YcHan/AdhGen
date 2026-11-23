"""Supplementary figure for reviewer comment 1-1: reference points on representative soft and stiff cells.

Columns: (a) FA-marker channel (vinculin, grayscale)
         (b) reference points overlaid, with the analysis area (cell area minus nucleus margin) outlined
         (c) magnified view of the boxed area
Same representative images as Fig. 4a (folder 8: Soft_3, Stiff_2). Reference points are generated with the
submitted settings (adhgen_ga_area_weighted_micro_f1_windows.POINT_CONFIG_FINAL etc.); nothing is changed.
"""

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent
import sys
from pathlib import Path

sys.path.insert(0, str(_SCRIPT_DIR))
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import adhgen_ga_area_weighted_micro_f1_windows as ga

plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})

DATA = Path(str(_ADHGEN_ROOT / "merged_data_405_488_actin"))
IMAGES = {"Soft": DATA / "8" / "Soft" / "Soft_3_M.tif", "Stiff": DATA / "8" / "Stiff" / "Stiff_2_M.tif"}
# zoom ROI used in Fig. 4a (x0, y0, x1, y1; Fig4a_individual_source_manifest.csv); the magnified box is chosen inside it
FIG4_ROI = {"Soft": (284, 360, 653, 729), "Stiff": (42, 424, 411, 793)}
OUT_DIR = Path(str(_ADHGEN_ROOT / "Figure/Supp_reference_points"))
BOX = 160          # side of the magnified area (px)
POINT_COLOR = "#ff2e88"
OUTLINE_COLOR = "#33c3ff"


def stretch(gray, mask):
    lo, hi = np.percentile(gray[mask] if mask.any() else gray, [1, 99.7])
    return np.clip((gray.astype(float) - lo) / max(hi - lo, 1), 0, 1)


def pick_box(points, roi):
    """Box (y0, x0) of side BOX inside the Fig. 4a zoom ROI containing the most reference points."""
    rx0, ry0, rx1, ry1 = roi
    best, best_n = (ry0, rx0), -1
    for y0 in range(ry0, ry1 - BOX + 1, 8):
        for x0 in range(rx0, rx1 - BOX + 1, 8):
            n = int(((points[:, 0] >= y0) & (points[:, 0] < y0 + BOX) &
                     (points[:, 1] >= x0) & (points[:, 1] < x0 + BOX)).sum())
            if n > best_n:
                best, best_n = (y0, x0), n
    return best


def outline(ax, mask, **kw):
    ax.contour(mask.astype(float), levels=[0.5], colors=OUTLINE_COLOR, linewidths=0.8, **kw)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.6),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.08, "left": 0.06, "right": 0.99,
                                          "top": 0.91, "bottom": 0.07})
    summary = []
    for row, (cond, path) in enumerate(IMAGES.items()):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        ref = ga.build_image_reference(str(path), img, ga.POINT_CONFIG_FINAL, ga.NUCLEUS_CONFIG_FINAL,
                                       ga.CELL_CONFIG_FINAL, ga.REFERENCE_MASK_CONFIG)
        pts = ref.points
        gray = stretch(img[..., 1], ref.cell_union_mask)
        y0, x0 = pick_box(pts, FIG4_ROI[cond])
        summary.append((cond, path.name, len(pts), ref.n_cells, (y0, x0)))

        ax = axes[row, 0]
        ax.imshow(gray, cmap="gray", interpolation="nearest")

        ax = axes[row, 1]
        ax.imshow(gray, cmap="gray", interpolation="nearest")
        outline(ax, ref.analysis_mask)
        ax.scatter(pts[:, 1], pts[:, 0], s=4, c=POINT_COLOR, lw=0)
        ax.add_patch(Rectangle((x0, y0), BOX, BOX, fill=False, ec="white", lw=1.3))
        ax.text(0.98, 0.03, f"n = {len(pts)} points", transform=ax.transAxes, ha="right", va="bottom",
                color="white", fontsize=11)

        ax = axes[row, 2]
        crop = gray[y0:y0 + BOX, x0:x0 + BOX]
        ax.imshow(crop, cmap="gray", interpolation="nearest", extent=(x0 - 0.5, x0 + BOX - 0.5,
                                                                      y0 + BOX - 0.5, y0 - 0.5))
        outline(ax, ref.analysis_mask)
        inside = ((pts[:, 0] >= y0) & (pts[:, 0] < y0 + BOX) & (pts[:, 1] >= x0) & (pts[:, 1] < x0 + BOX))
        ax.scatter(pts[inside, 1], pts[inside, 0], s=70, facecolors="none", edgecolors=POINT_COLOR, lw=1.3)
        ax.scatter(pts[inside, 1], pts[inside, 0], s=5, c=POINT_COLOR, lw=0)
        ax.set_xlim(x0 - 0.5, x0 + BOX - 0.5)
        ax.set_ylim(y0 + BOX - 0.5, y0 - 0.5)

        axes[row, 0].set_ylabel(cond, fontsize=15, fontweight="bold", labelpad=8)
        for a in axes[row]:
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)

    titles = ["FA-marker channel (Vinculin)", "Reference points", "Magnified view"]
    for j, (t, letter) in enumerate(zip(titles, "abc")):
        axes[0, j].set_title(t, fontsize=13)
        pos = axes[0, j].get_position()
        fig.text(pos.x0, 0.955, letter, fontsize=20, fontweight="bold", va="bottom")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", ms=6, mfc=POINT_COLOR, mec=POINT_COLOR,
                      label="Reference point"),
               Line2D([], [], color=OUTLINE_COLOR, lw=1.5, label="Analysis area")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.52, 0.0), handletextpad=0.4, columnspacing=2.0)

    for ext in ("png", "svg", "tiff"):
        kw = {"dpi": 300} if ext != "svg" else {}
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT_DIR / f"Supp_reference_points.{ext}", **kw)
    for s in summary:
        print(s)
    print("saved", OUT_DIR / "Supp_reference_points.png")


if __name__ == "__main__":
    main()
