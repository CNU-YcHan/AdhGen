#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Revised Fig. 4 for AdhGen

Panel a
-------
Rows:
    Soft
    Stiff

Columns:
    1. Merged image
    2. FA-marker channel (Green)
    3. AdhGen filtered
    4. FA-marker + AdhGen labels
    5. Magnified view

In addition to the composite Fig. 4, every microscopy image used in Panel a
is saved separately under Fig4a_individual_source_images/. The original input
files are also copied unchanged for source-image submission.

Panel b
-------
Held-out test-image FA count per cell for one selected split seed.
Each dot represents ONE held-out test image.
For each image: FA count per cell = n_detected_objects / n_cells.

Required companion module in the same code directory:
    cell_instance_utils_dapi_all_cells.py

The script follows the same cell/nucleus masking logic used by the current
AdhGen visualization code:
    - DAPI-seeded cell instance segmentation
    - nuclear exclusion
    - optional scale-bar exclusion
    - R/G/B chromosome filtering
    - connected-component area filtering

Example
-------
python "(Fig.4)make_fig4_revised.py" ^
  --soft-image "<ADHGEN_ROOT>\\merged_data_405_488_actin\\1\\Soft\\Soft_5_M.tif" ^
  --stiff-image "<ADHGEN_ROOT>\\merged_data_405_488_actin\\1\\Stiff\\Stiff_3_M.tif" ^
  --seed-dir "<ADHGEN_ROOT>\\AdhGen result\\split_seed_61" ^
  --test-predictions-csv "<ADHGEN_ROOT>\\AdhGen result\\all_test_predictions_across_seeds.csv" ^
  --plot-seed 61 ^
  --marker-name "Vinculin" ^
  --output-dir "<ADHGEN_ROOT>\\Figure\\Fig4"

Notes
-----
1. --seed-dir reads seed_summary.json and uses the optimized chromosome
   from the same split seed shown in panel b. The script verifies that
   --seed-dir and --plot-seed refer to the same split seed.

2. The green channel is displayed directly as the FA-marker channel.
   Do not write "Vinculin/Paxillin" unless the representative image truly
   contains both. Set --marker-name to the actual marker for the selected image.

3. Panel b uses the precomputed held-out predictions from all_test_predictions_across_seeds.csv.
   It does not pool observations across split seeds. One selected split seed is plotted,
   and each dot represents one held-out test image.

4. Supplementary Table S5 is generated automatically from all split seeds.
   Soft and Stiff are compared separately within each split using a two-tailed
   Mann-Whitney U test.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.stats import mannwhitneyu

try:
    import cell_instance_utils_dapi_all_cells as cell_utils
except ImportError as exc:
    raise ImportError(
        "\nRequired file not found: cell_instance_utils_dapi_all_cells.py\n"
        "Place it in the same folder as this script.\n"
    ) from exc


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Keep these identical to the current AdhGen visualization code.
SCALE_BAR_X_START_FRACTION = 0.78
SCALE_BAR_HEIGHT_PX = 70


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate revised Fig. 4 for AdhGen."
    )

    parser.add_argument(
        "--soft-image",
        type=Path,
        required=True,
        help="Representative merged image for the Soft condition.",
    )
    parser.add_argument(
        "--stiff-image",
        type=Path,
        required=True,
        help="Representative merged image for the Stiff condition.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Deprecated for panel b. Retained only for backward compatibility; "
            "held-out panel-b values are read from --test-predictions-csv."
        ),
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=None,
        help=(
            "Folder such as 'AdhGen result/split_seed_61'. "
            "If supplied, the chromosome is read from seed_summary.json. "
            "Its split_seed must match --plot-seed."
        ),
    )
    parser.add_argument(
        "--test-predictions-csv",
        type=Path,
        required=True,
        help=(
            "all_test_predictions_across_seeds.csv from the repeated held-out analysis. "
            "Panel b uses only the held-out test rows from --plot-seed."
        ),
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=61,
        help=(
            "Split seed to display in Fig. 4b. "
            "All split seeds are still summarized in Supplementary Table S5."
        ),
    )

    parser.add_argument(
        "--chromosome",
        type=str,
        default=None,
        help=(
            'Manual chromosome if --seed-dir is not used: '
            '"Rmin,Rmax,Gmin,Gmax,Bmin,Bmax,Amin,Amax"'
        ),
    )
    parser.add_argument(
        "--marker-name",
        type=str,
        required=True,
        help='Actual green-channel FA marker, e.g. "Vinculin" or "Paxillin".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./figure/Fig4"),
        help="Output directory.",
    )

    parser.add_argument(
        "--soft-zoom",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Optional fixed zoom ROI for Soft.",
    )
    parser.add_argument(
        "--stiff-zoom",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Optional fixed zoom ROI for Stiff.",
    )
    parser.add_argument(
        "--zoom-fraction",
        type=float,
        default=0.36,
        help="Automatic zoom ROI width/height as a fraction of image size.",
    )
    parser.add_argument(
        "--include-scale-bar",
        action="store_true",
        help="Do not exclude the bottom-right scale-bar region.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Deprecated; panel b now reads precomputed held-out predictions.",
    )

    return parser.parse_args()


# =============================================================================
# Chromosome / file list
# =============================================================================

def parse_chromosome_string(text: str) -> List[int]:
    tokens = [
        token
        for token in re.split(r"[,;\s]+", text.strip())
        if token
    ]
    values = [int(token) for token in tokens]

    if len(values) != 8:
        raise ValueError("Chromosome must contain exactly 8 integers.")

    r_min, r_max, g_min, g_max, b_min, b_max, a_min, a_max = values

    if not (0 <= r_min <= r_max <= 255):
        raise ValueError("Invalid R range.")
    if not (0 <= g_min <= g_max <= 255):
        raise ValueError("Invalid G range.")
    if not (0 <= b_min <= b_max <= 255):
        raise ValueError("Invalid B range.")
    if not (1 <= a_min <= a_max):
        raise ValueError("Invalid Area range.")

    return values


def load_chromosome(args: argparse.Namespace) -> List[int]:
    if args.seed_dir is not None:
        summary_path = args.seed_dir / "seed_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"seed_summary.json not found: {summary_path}"
            )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        chromosome = summary.get("best_individual")

        if chromosome is None or len(chromosome) != 8:
            raise ValueError(
                "best_individual was not found correctly in seed_summary.json."
            )

        return [int(v) for v in chromosome]

    if args.chromosome is None:
        raise ValueError(
            "Provide either --seed-dir or --chromosome."
        )

    return parse_chromosome_string(args.chromosome)


def infer_condition_from_path(path: Path) -> str:
    parts_lower = [part.lower() for part in path.parts]
    if "soft" in parts_lower:
        return "Soft"
    if "stiff" in parts_lower:
        return "Stiff"
    raise ValueError(
        f"Cannot infer Soft/Stiff condition from path: {path}"
    )


def reconstruct_manifest_path(
    dataset_root: Path,
    folder_id: str,
    condition: str,
    original_path: str,
) -> Path:
    basename = Path(original_path).name

    direct = dataset_root / str(folder_id) / condition / basename
    if direct.exists():
        return direct

    # Fallback if local folder organization differs slightly.
    matches = list(dataset_root.rglob(basename))
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Could not locate manifest image locally: {basename}"
        )

    # Prefer a path containing both folder id and condition.
    for candidate in matches:
        if (
            str(folder_id) in candidate.parts
            and condition.lower() in [p.lower() for p in candidate.parts]
        ):
            return candidate

    return matches[0]


def collect_dataset_images(
    dataset_root: Path,
    seed_dir: Path | None,
    max_images: int | None,
) -> List[Tuple[Path, str]]:
    rows: List[Tuple[Path, str]] = []

    if seed_dir is not None:
        manifest_path = seed_dir / "split_manifest.csv"

        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)

            for _, row in manifest.iterrows():
                path = reconstruct_manifest_path(
                    dataset_root=dataset_root,
                    folder_id=str(row["folder_id"]),
                    condition=str(row["condition"]),
                    original_path=str(row["path"]),
                )
                rows.append((path, str(row["condition"])))
        else:
            print(
                f"[WARNING] split_manifest.csv not found: {manifest_path}\n"
                "Falling back to recursive dataset scan."
            )

    if not rows:
        files = sorted(
            path
            for path in dataset_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in files:
            try:
                condition = infer_condition_from_path(path)
            except ValueError:
                continue
            rows.append((path, condition))

    # Remove duplicates.
    seen = set()
    unique_rows: List[Tuple[Path, str]] = []
    for path, condition in rows:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append((path, condition))

    if max_images is not None:
        unique_rows = unique_rows[:max_images]

    if not unique_rows:
        raise ValueError("No dataset images were found.")

    return unique_rows


# =============================================================================
# Held-out test prediction loading / Supplementary Table S5
# =============================================================================

PRIMARY_TEST_METRIC = "fa_count_per_cell"


def _normalize_condition(value: object) -> str:
    text = str(value).strip().lower()
    if text == "soft":
        return "Soft"
    if text == "stiff":
        return "Stiff"
    raise ValueError(f"Unexpected condition: {value!r}")


def load_all_test_predictions(csv_path: Path) -> pd.DataFrame:
    """
    Load the repeated held-out predictions and calculate the image-level
    FA-count-per-cell metric.

    Statistical unit: ONE held-out test image.
    Metric: n_detected_objects / n_cells.
    Rows from different split seeds remain separate and are never pooled.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Test prediction CSV not found: {csv_path}")

    data = pd.read_csv(csv_path).copy()

    required = {
        "split_seed",
        "condition",
        "n_detected_objects",
        "n_cells",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            "Missing required columns in test prediction CSV: "
            + ", ".join(missing)
        )

    # Be explicit even when the input file already contains test rows only.
    if "split" in data.columns:
        data = data.loc[
            data["split"].astype(str).str.lower().eq("test")
        ].copy()

    data["split_seed"] = pd.to_numeric(
        data["split_seed"], errors="raise"
    ).astype(int)
    data["condition"] = data["condition"].map(_normalize_condition)
    data["n_detected_objects"] = pd.to_numeric(
        data["n_detected_objects"], errors="raise"
    )
    data["n_cells"] = pd.to_numeric(
        data["n_cells"], errors="raise"
    )

    if (data["n_cells"] <= 0).any():
        bad = data.loc[data["n_cells"] <= 0]
        raise ValueError(
            "Held-out images with n_cells <= 0 were found; "
            "FA count per cell is undefined for these rows.\n"
            + bad[["split_seed", "condition", "n_cells"]]
            .head(10)
            .to_string(index=False)
        )

    data[PRIMARY_TEST_METRIC] = (
        data["n_detected_objects"].astype(float)
        / data["n_cells"].astype(float)
    )

    # Audit against the precomputed column when present.
    existing = "mean_detected_fa_count_per_cell"
    if existing in data.columns:
        expected = pd.to_numeric(data[existing], errors="coerce")
        diff = np.abs(expected - data[PRIMARY_TEST_METRIC])
        valid = expected.notna()
        if valid.any() and float(diff.loc[valid].max()) > 1e-9:
            raise ValueError(
                "Recomputed FA count per cell does not match "
                f"the existing '{existing}' column."
            )

    # One image should occur at most once within a given split seed.
    if "path" in data.columns:
        duplicated = data.duplicated(
            subset=["split_seed", "path"], keep=False
        )
        if duplicated.any():
            raise ValueError(
                "Duplicate held-out image rows were found within a split seed."
            )

    if data.empty:
        raise ValueError("No held-out test rows were found in the prediction CSV.")

    return data


def select_plot_seed_data(
    all_test: pd.DataFrame,
    plot_seed: int,
) -> pd.DataFrame:
    data = all_test.loc[
        all_test["split_seed"].eq(int(plot_seed))
    ].copy()

    if data.empty:
        available = sorted(all_test["split_seed"].unique().tolist())
        raise ValueError(
            f"Split seed {plot_seed} was not found. Available seeds: {available}"
        )

    counts = data.groupby("condition").size().to_dict()
    if counts.get("Soft", 0) == 0 or counts.get("Stiff", 0) == 0:
        raise ValueError(
            f"Split seed {plot_seed} must contain both Soft and Stiff test images."
        )

    return data


def build_supplementary_table_s5(
    all_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize held-out FA count per cell separately for every split seed.
    No repeated test observations are pooled across seeds.
    """
    rows: List[dict] = []

    for split_seed in sorted(all_test["split_seed"].unique()):
        seed_df = all_test.loc[
            all_test["split_seed"].eq(int(split_seed))
        ]

        soft = seed_df.loc[
            seed_df["condition"].eq("Soft"), PRIMARY_TEST_METRIC
        ].astype(float).to_numpy()
        stiff = seed_df.loc[
            seed_df["condition"].eq("Stiff"), PRIMARY_TEST_METRIC
        ].astype(float).to_numpy()

        if soft.size == 0 or stiff.size == 0:
            raise ValueError(
                f"Split seed {split_seed} is missing Soft or Stiff test values."
            )

        mw = mannwhitneyu(
            stiff,
            soft,
            alternative="two-sided",
            method="auto",
        )

        soft_q1, soft_q3 = np.quantile(soft, [0.25, 0.75])
        stiff_q1, stiff_q3 = np.quantile(stiff, [0.25, 0.75])

        rows.append(
            {
                "split_seed": int(split_seed),
                "soft_n_images": int(soft.size),
                "soft_mean_fa_count_per_cell": float(np.mean(soft)),
                "soft_median_fa_count_per_cell": float(np.median(soft)),
                "soft_q1": float(soft_q1),
                "soft_q3": float(soft_q3),
                "stiff_n_images": int(stiff.size),
                "stiff_mean_fa_count_per_cell": float(np.mean(stiff)),
                "stiff_median_fa_count_per_cell": float(np.median(stiff)),
                "stiff_q1": float(stiff_q1),
                "stiff_q3": float(stiff_q3),
                # U is reported for Stiff as the first sample.
                "mann_whitney_u_stiff_vs_soft": float(mw.statistic),
                "mann_whitney_p_two_sided": float(mw.pvalue),
            }
        )

    return pd.DataFrame(rows)


def validate_seed_dir_matches_plot_seed(
    seed_dir: Path | None,
    plot_seed: int,
) -> None:
    """Ensure the chromosome used in panel a comes from the plotted split seed."""
    if seed_dir is None:
        return

    summary_path = seed_dir / "seed_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_seed = summary.get("split_seed")
    if summary_seed is not None and int(summary_seed) != int(plot_seed):
        raise ValueError(
            f"--seed-dir contains split seed {summary_seed}, but --plot-seed is "
            f"{plot_seed}. Use the same split seed for panels a and b."
        )


# =============================================================================
# AdhGen segmentation
# =============================================================================

def make_scale_bar_mask(
    shape: Sequence[int],
    include_scale_bar: bool,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)

    if not include_scale_bar:
        x_start = int(round(SCALE_BAR_X_START_FRACTION * width))
        y_start = max(0, height - SCALE_BAR_HEIGHT_PX)
        mask[y_start:, x_start:] = True

    return mask


def build_analysis_data(
    image_bgr: np.ndarray,
    include_scale_bar: bool,
):
    """
    Returns:
        analysis_mask
        cell_labels
        cell_union
        nucleus_exclusion
        n_nuclei
        n_cells
    """
    scale_bar_mask = make_scale_bar_mask(
        image_bgr.shape,
        include_scale_bar=include_scale_bar,
    )

    nucleus_labels, nucleus_exclusion, nuclei, _ = (
        cell_utils.segment_nuclei(
            image_bgr,
            cell_utils.NUCLEUS_CONFIG,
        )
    )

    cell_labels, cells, _ = cell_utils.segment_cell_instances(
        image_bgr,
        nucleus_labels,
        nucleus_exclusion,
        cell_utils.CELL_INSTANCE_CONFIG,
    )

    cell_union = cell_labels > 0

    analysis_mask = (
        cell_union
        & ~nucleus_exclusion
        & ~scale_bar_mask
    )

    return (
        analysis_mask.astype(bool),
        cell_labels.astype(np.int32),
        cell_union.astype(bool),
        nucleus_exclusion.astype(bool),
        int(len(nuclei)),
        int(len(cells)),
    )


def detect_objects(
    image_bgr: np.ndarray,
    analysis_mask: np.ndarray,
    chromosome: Sequence[int],
):
    r_min, r_max, g_min, g_max, b_min, b_max, a_min, a_max = chromosome

    red = image_bgr[:, :, 2]
    green = image_bgr[:, :, 1]
    blue = image_bgr[:, :, 0]

    rgb_mask = (
        (red >= r_min) & (red <= r_max)
        & (green >= g_min) & (green <= g_max)
        & (blue >= b_min) & (blue <= b_max)
    )

    candidate_mask = (rgb_mask & analysis_mask).astype(np.uint8)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    accepted_labels = np.zeros_like(labels, dtype=np.int32)
    objects: List[dict] = []

    next_id = 1
    for component_id in range(1, n_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if not (a_min <= area <= a_max):
            continue

        component_mask = labels == component_id
        accepted_labels[component_mask] = next_id

        cx, cy = centroids[component_id]

        objects.append(
            {
                "object_id": next_id,
                "area_px": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
            }
        )

        next_id += 1

    return candidate_mask.astype(bool), accepted_labels, objects


def label_boundary(label_map: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary of accepted FA objects."""
    mask = (label_map > 0).astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return (mask > 0) & (eroded == 0)


def make_filtered_rgb(
    image_bgr: np.ndarray,
    accepted_labels: np.ndarray,
) -> np.ndarray:
    """
    Preserve only accepted AdhGen objects from the original merged image.
    Everything else becomes black.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    filtered = np.zeros_like(image_rgb)
    mask = accepted_labels > 0
    filtered[mask] = image_rgb[mask]
    return filtered


def make_marker_overlay(
    image_bgr: np.ndarray,
    accepted_labels: np.ndarray,
) -> np.ndarray:
    """
    Base: raw Green FA-marker channel shown in grayscale.
    Overlay: AdhGen FA boundaries in yellow.
    """
    green = image_bgr[:, :, 1]

    base = np.stack([green, green, green], axis=-1).astype(np.uint8)

    boundary = label_boundary(accepted_labels)

    overlay = base.copy()

    # Yellow contour in RGB.
    overlay[boundary] = [255, 255, 0]

    return overlay


def count_fa_per_cell(
    cell_labels: np.ndarray,
    accepted_labels: np.ndarray,
) -> List[dict]:
    """
    Assign each accepted FA connected component to exactly one cell,
    based on the most frequent non-zero cell label under that object.

    Every segmented cell is returned, including cells with zero detected FAs.
    """
    cell_ids = np.unique(cell_labels)
    cell_ids = cell_ids[cell_ids > 0]

    counts: Dict[int, int] = {
        int(cell_id): 0
        for cell_id in cell_ids
    }

    object_ids = np.unique(accepted_labels)
    object_ids = object_ids[object_ids > 0]

    for object_id in object_ids:
        object_mask = accepted_labels == object_id

        overlapping_cells = cell_labels[object_mask]
        overlapping_cells = overlapping_cells[overlapping_cells > 0]

        if overlapping_cells.size == 0:
            continue

        ids, frequencies = np.unique(
            overlapping_cells,
            return_counts=True,
        )

        assigned_cell = int(ids[np.argmax(frequencies)])
        counts[assigned_cell] += 1

    rows = [
        {
            "cell_id": int(cell_id),
            "fa_count": int(count),
        }
        for cell_id, count in sorted(counts.items())
    ]

    return rows


# =============================================================================
# Zoom selection
# =============================================================================

def clip_roi(
    roi: Sequence[int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in roi]

    x0 = max(0, min(x0, width - 1))
    x1 = max(x0 + 1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0 + 1, min(y1, height))

    return x0, y0, x1, y1


def auto_zoom_roi(
    accepted_labels: np.ndarray,
    fraction: float,
) -> Tuple[int, int, int, int]:
    """
    Choose a region containing a high density of accepted FA pixels.
    """
    h, w = accepted_labels.shape

    box_w = max(40, int(round(w * fraction)))
    box_h = max(40, int(round(h * fraction)))

    box_w = min(box_w, w)
    box_h = min(box_h, h)

    mask = (accepted_labels > 0).astype(np.float32)

    density = cv2.boxFilter(
        mask,
        ddepth=-1,
        ksize=(box_w, box_h),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )

    cy, cx = np.unravel_index(
        int(np.argmax(density)),
        density.shape,
    )

    x0 = int(round(cx - box_w / 2))
    y0 = int(round(cy - box_h / 2))

    x0 = max(0, min(x0, w - box_w))
    y0 = max(0, min(y0, h - box_h))

    return x0, y0, x0 + box_w, y0 + box_h


# =============================================================================
# Representative image processing
# =============================================================================

def process_representative_image(
    image_path: Path,
    chromosome: Sequence[int],
    include_scale_bar: bool,
    fixed_zoom: Sequence[int] | None,
    zoom_fraction: float,
):
    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            f"OpenCV could not read: {image_path}"
        )

    (
        analysis_mask,
        cell_labels,
        cell_union,
        nucleus_exclusion,
        n_nuclei,
        n_cells,
    ) = build_analysis_data(
        image_bgr,
        include_scale_bar=include_scale_bar,
    )

    candidate_mask, accepted_labels, objects = detect_objects(
        image_bgr,
        analysis_mask,
        chromosome,
    )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    green = image_bgr[:, :, 1]

    filtered_rgb = make_filtered_rgb(
        image_bgr,
        accepted_labels,
    )

    marker_overlay = make_marker_overlay(
        image_bgr,
        accepted_labels,
    )

    h, w = accepted_labels.shape

    if fixed_zoom is None:
        zoom_roi = auto_zoom_roi(
            accepted_labels,
            zoom_fraction,
        )
    else:
        zoom_roi = clip_roi(
            fixed_zoom,
            width=w,
            height=h,
        )

    return {
        "source_path": image_path.expanduser().resolve(),
        "image_rgb": image_rgb,
        "green": green,
        "filtered_rgb": filtered_rgb,
        "marker_overlay": marker_overlay,
        "accepted_labels": accepted_labels,
        "zoom_roi": zoom_roi,
        "n_cells": n_cells,
        "n_fa": len(objects),
    }


# =============================================================================
# Cell-level dataset quantification
# =============================================================================

def build_single_cell_table(
    dataset_rows: Sequence[Tuple[Path, str]],
    chromosome: Sequence[int],
    include_scale_bar: bool,
) -> pd.DataFrame:
    output_rows: List[dict] = []

    total = len(dataset_rows)

    for index, (image_path, condition) in enumerate(
        dataset_rows,
        start=1,
    ):
        print(
            f"[{index:03d}/{total:03d}] "
            f"{condition}: {image_path.name}"
        )

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            print(
                f"[WARNING] Could not read: {image_path}"
            )
            continue

        (
            analysis_mask,
            cell_labels,
            _cell_union,
            _nucleus_exclusion,
            _n_nuclei,
            _n_cells,
        ) = build_analysis_data(
            image_bgr,
            include_scale_bar=include_scale_bar,
        )

        _candidate_mask, accepted_labels, _objects = detect_objects(
            image_bgr,
            analysis_mask,
            chromosome,
        )

        cell_rows = count_fa_per_cell(
            cell_labels,
            accepted_labels,
        )

        for row in cell_rows:
            output_rows.append(
                {
                    "condition": condition,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "cell_id": int(row["cell_id"]),
                    "fa_count": int(row["fa_count"]),
                }
            )

    data = pd.DataFrame(output_rows)

    if data.empty:
        raise ValueError(
            "No single-cell measurements were generated."
        )

    return data


# =============================================================================
# Figure
# =============================================================================

def add_zoom_roi_box(
    ax: plt.Axes,
    roi: Sequence[int],
) -> None:
    """Draw the selected zoom ROI on the full-size overlay image."""
    x0, y0, x1, y1 = roi

    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        linewidth=1.6,
        edgecolor="white",
    )
    ax.add_patch(rect)


def crop_zoom(
    image_rgb: np.ndarray,
    roi: Sequence[int],
) -> np.ndarray:
    """Return the selected zoom crop."""
    x0, y0, x1, y1 = roi
    return image_rgb[y0:y1, x0:x1]


def _safe_filename_text(text: str) -> str:
    """Convert a label to a compact filesystem-safe string."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return safe.strip("_") or "marker"


def _save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    """Save an RGB uint8 image without axes, titles, or rescaling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    if not ok:
        raise IOError(f"Could not save image: {path}")


def _save_gray_image(path: Path, image_gray: np.ndarray) -> None:
    """Save an 8-bit grayscale image without axes, titles, or rescaling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_gray.astype(np.uint8))
    if not ok:
        raise IOError(f"Could not save image: {path}")


def save_panel_a_individual_images(
    soft_data: dict,
    stiff_data: dict,
    marker_name: str,
    output_dir: Path,
) -> Path:
    """
    Save every microscopy image used to assemble Fig. 4a as an individual file.

    For each Soft/Stiff representative image, this saves:
      0. an unchanged copy of the original input file,
      1. merged image,
      2. raw Green FA-marker channel,
      3. AdhGen-filtered image,
      4. FA-marker + AdhGen-label overlay (clean; no ROI rectangle),
      5. magnified view used in Fig. 4a.

    A CSV manifest maps each saved file back to its source image and panel role.
    """
    source_dir = output_dir / "Fig4a_individual_source_images"
    source_dir.mkdir(parents=True, exist_ok=True)

    marker_label = _safe_filename_text(marker_name)
    manifest_rows: List[dict] = []

    for condition, data in [("Soft", soft_data), ("Stiff", stiff_data)]:
        source_path = Path(data["source_path"]).expanduser().resolve()

        # Exact copy of the input image used by the script.
        original_copy = (
            source_dir
            / f"{condition}_00_original_input{source_path.suffix.lower()}"
        )
        shutil.copy2(source_path, original_copy)

        merged_path = source_dir / f"{condition}_01_merged_image.png"
        marker_path = (
            source_dir
            / f"{condition}_02_FA_marker_{marker_label}_green_channel.png"
        )
        filtered_path = source_dir / f"{condition}_03_AdhGen_filtered.png"
        overlay_path = (
            source_dir
            / f"{condition}_04_FA_marker_plus_AdhGen_labels.png"
        )
        zoom_path = source_dir / f"{condition}_05_magnified_view.png"

        _save_rgb_image(merged_path, data["image_rgb"])
        _save_gray_image(marker_path, data["green"])
        _save_rgb_image(filtered_path, data["filtered_rgb"])
        _save_rgb_image(overlay_path, data["marker_overlay"])
        _save_rgb_image(
            zoom_path,
            crop_zoom(data["marker_overlay"], data["zoom_roi"]),
        )

        # Save a lossless TIFF version of every displayed image as well.
        merged_tiff = source_dir / f"{condition}_01_merged_image.tiff"
        marker_tiff = (
            source_dir
            / f"{condition}_02_FA_marker_{marker_label}_green_channel.tiff"
        )
        filtered_tiff = source_dir / f"{condition}_03_AdhGen_filtered.tiff"
        overlay_tiff = (
            source_dir
            / f"{condition}_04_FA_marker_plus_AdhGen_labels.tiff"
        )
        zoom_tiff = source_dir / f"{condition}_05_magnified_view.tiff"

        _save_rgb_image(merged_tiff, data["image_rgb"])
        _save_gray_image(marker_tiff, data["green"])
        _save_rgb_image(filtered_tiff, data["filtered_rgb"])
        _save_rgb_image(overlay_tiff, data["marker_overlay"])
        _save_rgb_image(
            zoom_tiff,
            crop_zoom(data["marker_overlay"], data["zoom_roi"]),
        )

        x0, y0, x1, y1 = [int(v) for v in data["zoom_roi"]]
        common = {
            "condition": condition,
            "original_source_path": str(source_path),
            "n_cells": int(data["n_cells"]),
            "n_fa": int(data["n_fa"]),
            "zoom_x0": x0,
            "zoom_y0": y0,
            "zoom_x1": x1,
            "zoom_y1": y1,
        }

        items = [
            ("original_input", original_copy),
            ("merged_image_png", merged_path),
            ("merged_image_tiff", merged_tiff),
            (f"FA_marker_{marker_name}_green_channel_png", marker_path),
            (f"FA_marker_{marker_name}_green_channel_tiff", marker_tiff),
            ("AdhGen_filtered_png", filtered_path),
            ("AdhGen_filtered_tiff", filtered_tiff),
            ("FA_marker_plus_AdhGen_labels_png", overlay_path),
            ("FA_marker_plus_AdhGen_labels_tiff", overlay_tiff),
            ("magnified_view_png", zoom_path),
            ("magnified_view_tiff", zoom_tiff),
        ]
        for panel_role, saved_path in items:
            manifest_rows.append(
                {
                    **common,
                    "panel_role": panel_role,
                    "saved_path": str(saved_path),
                }
            )

    manifest_path = source_dir / "Fig4a_individual_source_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[INFO] Fig. 4a individual source images saved:")
    print(" ", source_dir)
    print("[INFO] Fig. 4a source manifest:")
    print(" ", manifest_path)

    return source_dir


def format_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def make_figure(
    soft_data: dict,
    stiff_data: dict,
    image_df: pd.DataFrame,
    marker_name: str,
    plot_seed: int,
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    fig = plt.figure(
        figsize=(18.2, 6.4),
        constrained_layout=False,
    )

    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[6.5, 1.35],
        wspace=0.18,
    )

    left = outer[0].subgridspec(
        nrows=2,
        ncols=5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.35],
        hspace=0.035,
        wspace=0.045,
    )

    image_axes = np.empty(
        (2, 5),
        dtype=object,
    )

    for r in range(2):
        for c in range(5):
            image_axes[r, c] = fig.add_subplot(
                left[r, c]
            )

    ax_b = fig.add_subplot(
        outer[1]
    )

    # -------------------------------------------------------------------------
    # Panel a
    # -------------------------------------------------------------------------
    rows = [
        ("Soft", soft_data),
        ("Stiff", stiff_data),
    ]

    column_titles = [
        "Merged image",
        f"FA-marker channel\n({marker_name})",
        "AdhGen filtered",
        "FA-marker + AdhGen labels",
        "Magnified view",
    ]

    for c, title in enumerate(column_titles):
        image_axes[0, c].set_title(
            title,
            fontsize=10.5,
            pad=7,
        )

    for r, (condition, data) in enumerate(rows):
        ax = image_axes[r, 0]
        ax.imshow(data["image_rgb"])
        format_image_axis(ax)

        ax = image_axes[r, 1]
        ax.imshow(
            data["green"],
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        format_image_axis(ax)

        ax = image_axes[r, 2]
        ax.imshow(data["filtered_rgb"])
        format_image_axis(ax)

        ax = image_axes[r, 3]
        ax.imshow(data["marker_overlay"])
        format_image_axis(ax)
        add_zoom_roi_box(
            ax,
            data["zoom_roi"],
        )

        ax = image_axes[r, 4]
        zoom_image = crop_zoom(
            data["marker_overlay"],
            data["zoom_roi"],
        )
        ax.imshow(zoom_image)
        format_image_axis(ax)

        image_axes[r, 0].text(
            -0.07,
            0.5,
            condition,
            transform=image_axes[r, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    image_axes[0, 0].text(
        -0.10,
        1.13,
        "a",
        transform=image_axes[0, 0].transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    image_axes[1, 0].text(
        0.00,
        -0.085,
        f"Red: F-actin   Green: {marker_name}   Blue: DAPI",
        transform=image_axes[1, 0].transAxes,
        fontsize=8.7,
        ha="left",
        va="top",
        clip_on=False,
    )

    # -------------------------------------------------------------------------
    # Panel b: each dot = one held-out TEST IMAGE from one split seed
    # -------------------------------------------------------------------------
    soft = (
        image_df.loc[
            image_df["condition"].eq("Soft"),
            PRIMARY_TEST_METRIC,
        ]
        .astype(float)
        .to_numpy()
    )

    stiff = (
        image_df.loc[
            image_df["condition"].eq("Stiff"),
            PRIMARY_TEST_METRIC,
        ]
        .astype(float)
        .to_numpy()
    )

    if soft.size == 0 or stiff.size == 0:
        raise ValueError(
            "Panel b requires both Soft and Stiff held-out image values."
        )

    violin = ax_b.violinplot(
        [soft, stiff],
        positions=[1, 2],
        widths=0.70,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in violin["bodies"]:
        body.set_alpha(0.25)

    ax_b.boxplot(
        [soft, stiff],
        positions=[1, 2],
        widths=0.26,
        showfliers=False,
        patch_artist=False,
        medianprops={"linewidth": 1.4},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )

    # Deterministic jitter: every dot is one held-out image.
    rng = np.random.default_rng(42)
    x_soft = 1 + rng.normal(
        loc=0.0,
        scale=0.045,
        size=len(soft),
    )
    x_stiff = 2 + rng.normal(
        loc=0.0,
        scale=0.045,
        size=len(stiff),
    )

    ax_b.scatter(
        x_soft,
        soft,
        s=18,
        alpha=0.65,
        linewidths=0,
    )
    ax_b.scatter(
        x_stiff,
        stiff,
        s=18,
        alpha=0.65,
        linewidths=0,
    )

    mw = mannwhitneyu(
        stiff,
        soft,
        alternative="two-sided",
        method="auto",
    )

    y_max = float(max(np.max(soft), np.max(stiff)))
    y_line = y_max * 1.08 + 1.0
    y_text = y_max * 1.11 + 1.0

    ax_b.plot(
        [1, 1, 2, 2],
        [
            y_line - 1.0,
            y_line,
            y_line,
            y_line - 1.0,
        ],
        linewidth=1.0,
    )

    if mw.pvalue < 0.0001:
        p_text = "p < 0.0001"
    else:
        p_text = f"p = {mw.pvalue:.4f}"

    ax_b.text(
        1.5,
        y_text,
        p_text,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
    )

    ax_b.set_xlim(0.55, 2.45)
    ax_b.set_ylim(
        bottom=0,
        top=max(
            y_text * 1.08,
            y_max + 5,
        ),
    )

    ax_b.set_xticks([1, 2])
    ax_b.set_xticklabels(["Soft", "Stiff"])
    ax_b.set_ylabel("Mean FA count per cell")
    ax_b.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax_b.text(
        -0.18,
        1.04,
        "b",
        transform=ax_b.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    ax_b.text(
        0.5,
        -0.12,
        (
            f"Each dot = one held-out test image\n"
            f"Split seed {plot_seed}; Soft n={len(soft)}, Stiff n={len(stiff)}"
        ),
        transform=ax_b.transAxes,
        fontsize=8.3,
        ha="center",
        va="top",
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.91,
        bottom=0.12,
    )

    png_path = output_dir / "Fig4_revised_v3_holdout.png"
    pdf_path = output_dir / "Fig4_revised_v3_holdout.pdf"
    tiff_path = output_dir / "Fig4_revised_v3_holdout.tiff"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        tiff_path,
        dpi=dpi,
        bbox_inches="tight",
        format="tiff",
    )

    plt.close(fig)

    print("\n[INFO] Figure saved:")
    print(" ", png_path)
    print(" ", pdf_path)
    print(" ", tiff_path)
    print(
        f"[INFO] Plot seed {plot_seed}: "
        f"Soft n={len(soft)}, Stiff n={len(stiff)}, "
        f"Mann-Whitney U={mw.statistic:.6g}, p={mw.pvalue:.8g}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_seed_dir_matches_plot_seed(
        seed_dir=args.seed_dir,
        plot_seed=args.plot_seed,
    )

    chromosome = load_chromosome(args)

    print("[INFO] Chromosome:")
    print(" ", chromosome)
    print(f"[INFO] Fig. 4b split seed: {args.plot_seed}")

    # Panel a: representative held-out images visualized with the chromosome
    # selected from the same split seed.
    soft_data = process_representative_image(
        image_path=args.soft_image,
        chromosome=chromosome,
        include_scale_bar=args.include_scale_bar,
        fixed_zoom=args.soft_zoom,
        zoom_fraction=args.zoom_fraction,
    )

    stiff_data = process_representative_image(
        image_path=args.stiff_image,
        chromosome=chromosome,
        include_scale_bar=args.include_scale_bar,
        fixed_zoom=args.stiff_zoom,
        zoom_fraction=args.zoom_fraction,
    )

    # Save the exact input images and every microscopy panel used in Fig. 4a
    # as separate source-image files for manuscript submission / source-data audit.
    save_panel_a_individual_images(
        soft_data=soft_data,
        stiff_data=stiff_data,
        marker_name=args.marker_name,
        output_dir=args.output_dir,
    )

    # Panel b / Supplementary Table S5: use ONLY precomputed held-out test rows.
    all_test = load_all_test_predictions(
        args.test_predictions_csv.expanduser().resolve()
    )

    plot_df = select_plot_seed_data(
        all_test=all_test,
        plot_seed=args.plot_seed,
    )

    # Save exact source data used for Fig. 4b.
    panel_b_columns = [
        column
        for column in [
            "split_seed",
            "path",
            "folder_id",
            "condition",
            "n_detected_objects",
            "n_cells",
            PRIMARY_TEST_METRIC,
        ]
        if column in plot_df.columns
    ]
    panel_b_csv = (
        args.output_dir
        / f"Fig4b_heldout_source_split_seed_{args.plot_seed}.csv"
    )
    plot_df[panel_b_columns].to_csv(
        panel_b_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # Generate Supplementary Table S5 from ALL split seeds, but keep every
    # statistical comparison within its own held-out test split.
    table_s5 = build_supplementary_table_s5(all_test)
    table_s5_csv = args.output_dir / "Supplementary_Table_S5.csv"
    table_s5.to_csv(
        table_s5_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[INFO] Fig. 4b source data: {panel_b_csv}")
    print(f"[INFO] Supplementary Table S5: {table_s5_csv}")
    print("\n[INFO] Supplementary Table S5 preview:")
    print(table_s5.to_string(index=False))

    make_figure(
        soft_data=soft_data,
        stiff_data=stiff_data,
        image_df=plot_df,
        marker_name=args.marker_name,
        plot_seed=args.plot_seed,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
