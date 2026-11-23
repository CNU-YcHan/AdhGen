# -*- coding: utf-8 -*-
"""
Fig. 6 final raw-channel pipeline — two-panel main figure + FA-area supplement
=================================

Panel/folder rules
------------------
Main Fig. 6(a): folders 1,2,3,4,5,6,7,8,10,11
    - F-actin = raw 647 nm for folders 3,4,5,6
    - F-actin = raw 546 nm for folders 1,2,7,8,10,11
Main Fig. 6(b): folders 10,11 only
    - YAP signal = raw 647 nm
    - nucleus mask = raw 405 nm
    - cell mask = raw 546 nm
    - folder 12 contains YAP but is excluded because no valid AdhGen FA result exists

Supplementary figure:
    - direct comparison of image-level total FA count vs total FA area
    - uses the same cross-fitted AdhGen FA outputs as Fig. 6(a)
(e): direct comparison of cross-fitted total FA count and total FA area
    - uses the same valid Fig. 6 FA images as panels (a) and (c)

Merged images are NOT used to quantify F-actin or YAP.
The revised AdhGen repeated-holdout CSV is used only for FA output.
"""

from __future__ import annotations

import argparse
import importlib
import math
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr, t as student_t
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects


# -----------------------------------------------------------------------------
# Fixed experimental design
# -----------------------------------------------------------------------------
CONDITIONS = ("Soft", "Stiff")
FIG6_FACTIN_FOLDERS = {"1", "2", "3", "4", "5", "6", "7", "8", "10", "11"}
F_ACTIN_647_FOLDERS = {"3", "4", "5", "6"}
F_ACTIN_546_FOLDERS = FIG6_FACTIN_FOLDERS - F_ACTIN_647_FOLDERS
YAP_SOURCE_FOLDERS = {"10", "11", "12"}
FIG6_YAP_FOLDERS = {"10", "11"}
ADHGEN_EXCLUDED_FOLDERS = {"9", "12"}
SUPPORTED_EXTENSIONS = {".tif", ".tiff"}

CHANNEL_TOKEN_RE = re.compile(
    r"(?i)(^|[_\-\s])(?:(?:ch(?:annel)?|c)[_\-\s]*)?"
    r"(405|488|546|647)(?:nm)?(?=$|[_\-\s])"
)

SCALE_BAR_X_START_FRACTION = 0.78
SCALE_BAR_HEIGHT_PX = 70
FACTIN_PLOT_SCALE = 1e6
DPI = 300

FA_COUNT_RAW = "fa_count_raw_crossfitted"
FA_AREA_RAW = "total_fa_area_px2_crossfitted"
FA_COUNT_PC = "fa_count_per_raw_dapi_cell"
FA_AREA_PC = "total_fa_area_per_raw_dapi_cell_px2"
FACTIN_RAW = "raw_f_actin_integrated_8bit_equivalent"
FACTIN_PC = "raw_f_actin_integrated_8bit_equivalent_per_dapi_cell"
YAP_NC = "yap_nc_mean_intensity_ratio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create Fig.6 from revised AdhGen FA outputs and raw fluorescence channels."
    )
    p.add_argument("--fa-csv", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument(
        "--cell-utils-dir",
        type=Path,
        default=None,
        help="Directory containing cell_instance_utils_dapi_all_cells.py",
    )
    p.add_argument(
        "--include-scale-bar",
        action="store_true",
        help="Include lower-right scale-bar area in F-actin integration.",
    )
    p.add_argument(
        "--allow-dapi-count-mismatch",
        action="store_true",
        help="Allow raw-405 DAPI count to differ from AdhGen n_cells.",
    )
    p.add_argument(
        "--save-yap-qc",
        action="store_true",
        help="Save YAP nuclear/cell/cytoplasm QC panels.",
    )
    p.add_argument("--dpi", type=int, default=DPI)
    return p.parse_args()


def load_cell_utils(cell_utils_dir: Optional[Path]):
    if cell_utils_dir is not None:
        directory = str(cell_utils_dir.expanduser().resolve())
        if directory not in sys.path:
            sys.path.insert(0, directory)
    try:
        return importlib.import_module("cell_instance_utils_dapi_all_cells")
    except ImportError as exc:
        raise ImportError(
            "Could not import cell_instance_utils_dapi_all_cells.py. "
            "Place it beside this script or pass --cell-utils-dir."
        ) from exc


def normalize_condition(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text == "soft":
        return "Soft"
    if text == "stiff":
        return "Stiff"
    return None


def normalize_folder_id(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    try:
        v = float(text)
        return str(int(v)) if v.is_integer() else text
    except Exception:
        return text or None


def strip_channel_token(path: Path) -> tuple[str, str] | None:
    matches = list(CHANNEL_TOKEN_RE.finditer(path.stem))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Ambiguous wavelength token: {path}")
    match = matches[0]
    wavelength = match.group(2)
    sample_key = path.stem[: match.start()] + "_" + path.stem[match.end() :]
    sample_key = re.sub(r"[_\-\s]+", "_", sample_key).strip("_")
    if not sample_key:
        raise ValueError(f"Could not derive sample key: {path}")
    return sample_key, wavelength


def sample_id_from_sample_key(sample_key: str, condition: str) -> str:
    prefix = f"{condition}_"
    if sample_key.lower().startswith(prefix.lower()):
        return sample_key[len(prefix) :]
    return sample_key


def canonical_key(folder_id: str, condition: str, sample_key: str) -> str:
    return f"{folder_id}_{condition}_{sample_id_from_sample_key(sample_key, condition)}"


def discover_raw_channels(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, ignored = [], []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        parsed = strip_channel_token(path)
        if parsed is None:
            ignored.append({"path": str(path), "reason": "no unique wavelength token"})
            continue
        sample_key, wavelength = parsed
        parts = list(path.parts)
        condition, condition_index = None, None
        for i, part in enumerate(parts):
            c = normalize_condition(part)
            if c is not None:
                condition, condition_index = c, i
        if condition is None or condition_index is None or condition_index == 0:
            ignored.append({"path": str(path), "reason": "could not infer Soft/Stiff"})
            continue
        folder_id = normalize_folder_id(parts[condition_index - 1])
        if folder_id is None:
            ignored.append({"path": str(path), "reason": "could not infer folder"})
            continue
        records.append(
            {
                "folder_id": folder_id,
                "condition": condition,
                "sample_key": sample_key,
                "wavelength": wavelength,
                "path": str(path),
            }
        )

    channels = pd.DataFrame(records)
    if channels.empty:
        raise FileNotFoundError(f"No raw 405/488/546/647 TIFFs found under {data_root}")

    dup = channels[
        channels.duplicated(
            subset=["folder_id", "condition", "sample_key", "wavelength"], keep=False
        )
    ]
    if not dup.empty:
        raise ValueError(
            "Duplicate raw wavelength files for one sample/channel:\n"
            + dup.sort_values(["folder_id", "condition", "sample_key", "wavelength"])
            .head(50)
            .to_string(index=False)
        )
    return channels, pd.DataFrame(ignored)


def channel_inventory_wide(channels: pd.DataFrame) -> pd.DataFrame:
    d = channels.copy()
    d["key"] = d.apply(
        lambda r: canonical_key(str(r["folder_id"]), str(r["condition"]), str(r["sample_key"])),
        axis=1,
    )
    wide = d.pivot(
        index=["key", "folder_id", "condition", "sample_key"],
        columns="wavelength",
        values="path",
    ).reset_index()
    wide.columns.name = None
    for wl in ("405", "488", "546", "647"):
        if wl not in wide.columns:
            wide[wl] = np.nan
    return wide


# -----------------------------------------------------------------------------
# Raw wavelength reading
# -----------------------------------------------------------------------------
def _extract_wavelength_plane(
    image: np.ndarray, wavelength: str, *, color_order: str, path: Path
) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected grayscale/RGB wavelength image, got {image.shape}: {path}")
    if color_order == "BGR":
        blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    elif color_order == "RGB":
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    else:
        raise ValueError(f"Unsupported color order: {color_order}")
    if wavelength == "405":
        plane = blue
    elif wavelength == "488":
        plane = green
    elif wavelength == "546":
        plane = red
    elif wavelength == "647":
        plane = np.maximum(red, blue)  # red or magenta display
    else:
        raise ValueError(f"Unsupported wavelength: {wavelength}")
    return np.ascontiguousarray(plane)


def read_wavelength_plane(path: Path, wavelength: str) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image_bgr is not None:
        return _extract_wavelength_plane(
            image_bgr, wavelength, color_order="BGR", path=path
        )
    image_rgb = tifffile.imread(path)
    return _extract_wavelength_plane(
        image_rgb, wavelength, color_order="RGB", path=path
    )


def to_8bit_equivalent(channel: np.ndarray, *, path: Path) -> np.ndarray:
    if channel.dtype == np.uint8:
        return channel.astype(np.float64)
    if channel.dtype == np.uint16:
        return channel.astype(np.float64) / 257.0
    raise TypeError(f"Expected uint8/uint16 raw image, got {channel.dtype}: {path}")


def to_uint8_for_segmentation(channel: np.ndarray, *, path: Path) -> np.ndarray:
    return np.rint(to_8bit_equivalent(channel, path=path)).clip(0, 255).astype(np.uint8)


def make_scale_bar_mask(shape, include_scale_bar: bool) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    if not include_scale_bar:
        x_start = int(round(SCALE_BAR_X_START_FRACTION * w))
        y_start = max(0, h - SCALE_BAR_HEIGHT_PX)
        mask[y_start:, x_start:] = True
    return mask


# -----------------------------------------------------------------------------
# Revised AdhGen aggregation
# -----------------------------------------------------------------------------
def parse_fa_identity(row: pd.Series) -> tuple[str, str, str, str]:
    folder_id = normalize_folder_id(row.get("folder_id"))
    condition = normalize_condition(row.get("condition"))
    path = Path(str(row["path"]).replace("\\", "/"))
    base = re.sub(r"(?i)_M$", "", path.stem)
    if condition is None:
        m = re.match(r"(?i)^(Soft|Stiff)_", base)
        if m:
            condition = m.group(1).capitalize()
    if folder_id is None:
        parts = list(path.parts)
        for i, part in enumerate(parts):
            c = normalize_condition(part)
            if c is not None and i > 0:
                folder_id = normalize_folder_id(parts[i - 1])
                break
    if folder_id is None or condition is None:
        raise ValueError(f"Could not parse FA identity: {row['path']}")
    sample_key = base
    return folder_id, condition, sample_key, canonical_key(folder_id, condition, sample_key)


def load_crossfitted_fa(fa_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(fa_csv)
    required = {"path", "condition", "n_detected_objects", "n_nuclei", "n_cells", "mean_detected_object_area_px"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"FA CSV missing: {sorted(missing)}")
    if "split" in df.columns:
        df = df.loc[df["split"].astype(str).str.lower().eq("test")].copy()
    if df.empty:
        raise RuntimeError("No TEST rows in FA CSV")

    ids = df.apply(parse_fa_identity, axis=1, result_type="expand")
    ids.columns = ["folder_id_parsed", "condition_parsed", "sample_key", "key"]
    for col in ids.columns:
        df[col] = ids[col]

    unexpected = sorted(set(df["folder_id_parsed"].astype(str)) - FIG6_FACTIN_FOLDERS)
    if unexpected:
        raise ValueError(
            "FA CSV contains folders outside agreed Fig.6 set: " + ", ".join(unexpected)
        )

    for col in ("n_detected_objects", "n_nuclei", "n_cells", "mean_detected_object_area_px"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not (
        np.isfinite(df["n_detected_objects"])
        & np.isfinite(df["n_nuclei"])
        & (df["n_nuclei"] > 0)
        & np.isfinite(df["n_cells"])
        & (df["n_cells"] > 0)
    ).all():
        raise ValueError("Invalid n_detected_objects or n_cells in TEST predictions")

    positive = df["n_detected_objects"] > 0
    if (positive & ~np.isfinite(df["mean_detected_object_area_px"])).any():
        raise ValueError("Positive FA count with invalid mean_detected_object_area_px")
    df["_fa_area_raw"] = np.where(
        positive,
        df["n_detected_objects"].astype(float) * df["mean_detected_object_area_px"].astype(float),
        0.0,
    )

    fa = (
        df.groupby(
            ["key", "folder_id_parsed", "condition_parsed", "sample_key"],
            observed=True,
            dropna=False,
        )
        .agg(
            original_prediction_path=("path", "first"),
            n_oos_predictions=("path", "size"),
            adhgen_n_nuclei_min=("n_nuclei", "min"),
            adhgen_n_nuclei_max=("n_nuclei", "max"),
            adhgen_n_nuclei_median=("n_nuclei", "median"),
            adhgen_n_cells_min=("n_cells", "min"),
            adhgen_n_cells_max=("n_cells", "max"),
            adhgen_n_cells_median=("n_cells", "median"),
            fa_count_raw_crossfitted=("n_detected_objects", "mean"),
            total_fa_area_px2_crossfitted=("_fa_area_raw", "mean"),
        )
        .reset_index()
        .rename(columns={"folder_id_parsed": "folder_id", "condition_parsed": "condition"})
    )
    fa["adhgen_n_nuclei_stable"] = (
        fa["adhgen_n_nuclei_min"] == fa["adhgen_n_nuclei_max"]
    )
    fa["adhgen_n_cells_stable"] = (
        fa["adhgen_n_cells_min"] == fa["adhgen_n_cells_max"]
    )

    if (~fa["adhgen_n_nuclei_stable"]).any():
        raise ValueError("AdhGen n_nuclei changed across repeated TEST appearances")
    if (~fa["adhgen_n_cells_stable"]).any():
        raise ValueError("AdhGen n_cells changed across repeated TEST appearances")

    if fa["key"].duplicated().any():
        raise ValueError("Duplicate cross-fitted FA key")

    audit = pd.DataFrame(
        [{
            "test_prediction_rows_used": len(df),
            "crossfitted_unique_images": len(fa),
            "folders_used": ",".join(sorted(set(fa["folder_id"].astype(str)), key=int)),
            "mean_test_appearances_per_image": float(fa["n_oos_predictions"].mean()),
            "min_test_appearances_per_image": int(fa["n_oos_predictions"].min()),
            "max_test_appearances_per_image": int(fa["n_oos_predictions"].max()),
            "unstable_adhgen_n_nuclei_images": int((~fa["adhgen_n_nuclei_stable"]).sum()),
            "unstable_adhgen_n_cells_images": int((~fa["adhgen_n_cells_stable"]).sum()),
        }]
    )
    return fa, audit


# -----------------------------------------------------------------------------
# Raw DAPI / F-actin
# -----------------------------------------------------------------------------
def segment_raw_dapi_nuclei(
    channel_405: np.ndarray,
    *,
    path_405: Path,
    include_scale_bar: bool,
    cell_utils,
):
    dapi_8 = to_uint8_for_segmentation(channel_405, path=path_405)
    bgr = np.zeros((dapi_8.shape[0], dapi_8.shape[1], 3), dtype=np.uint8)
    bgr[..., 0] = dapi_8
    # IMPORTANT:
    # Revised AdhGen segments nuclei BEFORE the scale-bar exclusion mask is
    # applied to the downstream FA analysis area. Therefore, to reproduce the
    # same DAPI-positive nucleus count from the raw 405-nm image, we must NOT
    # zero the scale-bar region here.
    #
    # The scale-bar mask is still applied later to whole-image F-actin
    # integration; it is simply not part of nucleus-count reproduction.
    nucleus_labels, _, nuclei, _ = cell_utils.segment_nuclei(
        bgr, cell_utils.NUCLEUS_CONFIG
    )
    return nucleus_labels.astype(np.int32), int(len(nuclei))


def expected_factin_wavelength(folder_id: str) -> str:
    if folder_id in F_ACTIN_647_FOLDERS:
        return "647"
    if folder_id in F_ACTIN_546_FOLDERS:
        return "546"
    raise ValueError(f"Folder {folder_id} not permitted in Fig.6a/c")


def require_path(value, description: str) -> Path:
    if pd.isna(value) or not str(value).strip():
        raise FileNotFoundError(f"Missing channel: {description}")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Channel file does not exist: {path}")
    return path


def build_raw_factin_table(
    fa: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    include_scale_bar: bool,
    allow_dapi_count_mismatch: bool,
    cell_utils,
):
    source = fa.merge(
        inventory,
        on=["key", "folder_id", "condition", "sample_key"],
        how="left",
        validate="one_to_one",
    )
    records, failures = [], []

    for _, row in source.iterrows():
        folder_id = str(row["folder_id"])
        key = str(row["key"])
        condition = str(row["condition"])
        factin_wl = expected_factin_wavelength(folder_id)
        base = {
            "key": key,
            "folder_id": folder_id,
            "condition": condition,
            "sample_key": row["sample_key"],
            "f_actin_wavelength": factin_wl,
        }
        try:
            path_405 = require_path(row.get("405"), f"{key} 405 nm")
            path_factin = require_path(row.get(factin_wl), f"{key} {factin_wl} nm F-actin")
            ch405 = read_wavelength_plane(path_405, "405")
            ch_factin = read_wavelength_plane(path_factin, factin_wl)
            if ch405.shape != ch_factin.shape:
                raise ValueError(f"405/F-actin shape mismatch: {ch405.shape} vs {ch_factin.shape}")

            _, raw_dapi_count = segment_raw_dapi_nuclei(
                ch405,
                path_405=path_405,
                include_scale_bar=include_scale_bar,
                cell_utils=cell_utils,
            )
            if raw_dapi_count < 1:
                raise ValueError("No DAPI-positive nucleus detected")

            adhgen_n_nuclei = float(row["adhgen_n_nuclei_median"])
            adhgen_n_cells = float(row["adhgen_n_cells_median"])

            # Raw 405-nm DAPI must reproduce AdhGen's NUCLEUS count, not its
            # watershed-filtered cell-instance count. These are different
            # quantities by design.
            count_match = float(raw_dapi_count) == adhgen_n_nuclei

            if not count_match and not allow_dapi_count_mismatch:
                raise ValueError(
                    f"Raw DAPI nuclei ({raw_dapi_count}) != "
                    f"AdhGen n_nuclei ({adhgen_n_nuclei:g}); "
                    f"AdhGen n_cells={adhgen_n_cells:g}. "
                    "This indicates a raw-405/preprocessing mismatch."
                )

            factin_8 = to_8bit_equivalent(ch_factin, path=path_factin)
            analysis_mask = ~make_scale_bar_mask(factin_8.shape, include_scale_bar)
            values = factin_8[analysis_mask]
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError("Invalid F-actin analysis area")
            integrated = float(np.sum(values, dtype=np.float64))
            if integrated <= 0:
                raise ValueError("F-actin integrated intensity <= 0")

            records.append(
                {
                    **base,
                    "path_405": str(path_405),
                    "path_f_actin": str(path_factin),
                    "raw_dapi_cell_count": raw_dapi_count,
                    "adhgen_n_nuclei": adhgen_n_nuclei,
                    "adhgen_n_cells": adhgen_n_cells,
                    "raw_dapi_matches_adhgen_n_nuclei": count_match,
                    "adhgen_cells_minus_nuclei": adhgen_n_cells - adhgen_n_nuclei,
                    FA_COUNT_RAW: float(row[FA_COUNT_RAW]),
                    FA_AREA_RAW: float(row[FA_AREA_RAW]),
                    FA_COUNT_PC: float(row[FA_COUNT_RAW]) / raw_dapi_count,
                    FA_AREA_PC: float(row[FA_AREA_RAW]) / raw_dapi_count,
                    FACTIN_RAW: integrated,
                    FACTIN_PC: integrated / raw_dapi_count,
                    "f_actin_mean_8bit_equivalent": float(np.mean(values)),
                    "f_actin_max_8bit_equivalent": float(np.max(values)),
                    "f_actin_source_dtype": str(ch_factin.dtype),
                }
            )
        except Exception as exc:
            failures.append({**base, "error_type": type(exc).__name__, "error": str(exc)})

    return pd.DataFrame(records), pd.DataFrame(failures)


# -----------------------------------------------------------------------------
# Raw YAP (folders 10,11 only)
# -----------------------------------------------------------------------------
def region_mask_like_original(
    channel: np.ndarray,
    *,
    path: Path,
    require_single_region: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Region detector from the original YAP/F-actin analysis family.

    For YAP, the original outer loop only called the YAP analyzer when the
    405-nm image produced exactly one region.  We preserve that single-cell-ROI
    inclusion rule for the nuclear channel, while the 546-nm cell-region image
    still uses its largest valid region as in the original implementation.
    """
    image_8 = to_uint8_for_segmentation(channel, path=path)
    image_float = image_8.astype(np.float64) / 255.0

    def _segment(img, clip_limit=None):
        enhanced = equalize_adapthist(img) if clip_limit is None else equalize_adapthist(img, clip_limit=clip_limit)
        threshold = threshold_otsu(enhanced)
        binary = enhanced > threshold
        cleaned = remove_small_objects(binary, min_size=500)
        cleaned = remove_small_holes(cleaned, area_threshold=500)
        labels = label(cleaned)
        regs = regionprops(labels)
        return labels, regs

    labels, regs = _segment(image_float)
    if not regs:
        labels, regs = _segment(image_float, clip_limit=0.04)
    if not regs:
        raise ValueError(f"No valid region detected: {path}")

    n_regions = len(regs)
    if require_single_region and n_regions != 1:
        raise ValueError(
            f"YAP single-cell ROI criterion failed: 405-nm region count={n_regions}, expected 1"
        )

    largest = max(regs, key=lambda region: region.area)
    return labels == largest.label, n_regions


def save_yap_qc_panel(save_path: Path, yap_8, nucleus_mask, cell_mask, cytoplasm_mask, title: str):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(yap_8, cmap="gray"); axes[0].set_title("Raw 647-nm YAP")
    axes[1].imshow(yap_8, cmap="gray"); axes[1].imshow(nucleus_mask, cmap="autumn", alpha=0.45); axes[1].set_title("Nucleus (405)")
    axes[2].imshow(yap_8, cmap="gray"); axes[2].imshow(cell_mask, cmap="winter", alpha=0.35); axes[2].set_title("Cell region (546)")
    axes[3].imshow(yap_8, cmap="gray"); axes[3].imshow(cytoplasm_mask, cmap="summer", alpha=0.40); axes[3].set_title("Cytoplasm")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_raw_yap_table(fa: pd.DataFrame, inventory: pd.DataFrame, *, save_qc: bool, qc_dir: Path):
    fa_yap = fa.loc[fa["folder_id"].astype(str).isin(FIG6_YAP_FOLDERS)].copy()
    source = fa_yap.merge(
        inventory,
        on=["key", "folder_id", "condition", "sample_key"],
        how="left",
        validate="one_to_one",
    )
    records, failures = [], []
    if save_qc:
        qc_dir.mkdir(parents=True, exist_ok=True)

    for _, row in source.iterrows():
        key = str(row["key"])
        folder_id = str(row["folder_id"])
        condition = str(row["condition"])
        base = {"key": key, "folder_id": folder_id, "condition": condition, "sample_key": row["sample_key"]}
        try:
            if folder_id not in FIG6_YAP_FOLDERS:
                raise ValueError("Unexpected folder in YAP subset")
            path_405 = require_path(row.get("405"), f"{key} 405 nm")
            path_546 = require_path(row.get("546"), f"{key} 546 nm cell-region channel")
            path_647 = require_path(row.get("647"), f"{key} 647 nm YAP")
            ch405 = read_wavelength_plane(path_405, "405")
            ch546 = read_wavelength_plane(path_546, "546")
            ch647 = read_wavelength_plane(path_647, "647")
            if not (ch405.shape == ch546.shape == ch647.shape):
                raise ValueError(f"405/546/647 shape mismatch: {ch405.shape}, {ch546.shape}, {ch647.shape}")

            nucleus_mask, n_405_regions = region_mask_like_original(
                ch405, path=path_405, require_single_region=True
            )
            cell_mask, n_546_regions = region_mask_like_original(
                ch546, path=path_546, require_single_region=False
            )
            nucleus_in_cell = nucleus_mask & cell_mask
            cytoplasm_mask = cell_mask & ~nucleus_mask
            n_nuc = int(np.count_nonzero(nucleus_in_cell))
            n_cyto = int(np.count_nonzero(cytoplasm_mask))
            if n_nuc < 1:
                raise ValueError("405 nuclear region does not overlap 546-defined cell region")
            if n_cyto < 1:
                raise ValueError("No cytoplasmic pixels after nucleus subtraction")

            yap_8 = to_8bit_equivalent(ch647, path=path_647)
            nuclear_mean = float(np.mean(yap_8[nucleus_in_cell]))
            cytoplasmic_mean = float(np.mean(yap_8[cytoplasm_mask]))
            if not np.isfinite(nuclear_mean) or not np.isfinite(cytoplasmic_mean) or cytoplasmic_mean <= 0:
                raise ValueError("Invalid YAP compartment mean intensity")
            ratio = nuclear_mean / cytoplasmic_mean
            if not np.isfinite(ratio):
                raise ValueError("Non-finite YAP N/C ratio")

            records.append(
                {
                    **base,
                    "path_405": str(path_405),
                    "path_546_cell_region": str(path_546),
                    "path_647_yap": str(path_647),
                    "n_405_regions_original_method": n_405_regions,
                    "n_546_regions_original_method": n_546_regions,
                    "nuclear_pixels": n_nuc,
                    "cytoplasmic_pixels": n_cyto,
                    "cell_pixels": int(np.count_nonzero(cell_mask)),
                    "nuclear_yap_mean_8bit_equivalent": nuclear_mean,
                    "cytoplasmic_yap_mean_8bit_equivalent": cytoplasmic_mean,
                    YAP_NC: ratio,
                    "yap_source_dtype": str(ch647.dtype),
                }
            )
            if save_qc:
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
                save_yap_qc_panel(
                    qc_dir / f"{safe}_YAP_QC.png",
                    yap_8,
                    nucleus_in_cell,
                    cell_mask,
                    cytoplasm_mask,
                    key,
                )
        except Exception as exc:
            failures.append({**base, "error_type": type(exc).__name__, "error": str(exc)})

    return pd.DataFrame(records), pd.DataFrame(failures)


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
def pearson_stats(x, y):
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(d)
    if n < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return {"n": n, "r": np.nan, "p": np.nan}
    result = pearsonr(d["x"], d["y"])
    return {"n": n, "r": float(result.statistic), "p": float(result.pvalue)}


def residualize(values, covariate):
    y = np.asarray(values, dtype=float)
    c = np.asarray(covariate, dtype=float)
    X = np.column_stack([np.ones_like(c), c])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr_one_covariate(df, *, xcol, ycol, covariate, group):
    d = df[[xcol, ycol, covariate, "condition"]].replace([np.inf, -np.inf], np.nan).dropna()
    if group != "Overall":
        d = d.loc[d["condition"] == group].copy()
    n = len(d)
    if n < 4:
        return {"group": group, "n": n, "partial_r": np.nan, "p": np.nan}
    rx = residualize(d[xcol], d[covariate])
    ry = residualize(d[ycol], d[covariate])
    r = float(pearsonr(rx, ry).statistic)
    dfree = n - 3
    if not np.isfinite(r) or dfree <= 0:
        p = np.nan
    elif abs(r) >= 1:
        p = 0.0
    else:
        t_stat = r * np.sqrt(dfree / max(1e-15, 1.0 - r * r))
        p = float(2.0 * student_t.sf(abs(t_stat), df=dfree))
    return {"group": group, "n": n, "partial_r": r, "p": p}


def build_shared_denominator_sensitivity(factin_df):
    rows = []
    for label_text, xcol in (("FA count", FA_COUNT_RAW), ("Total FA area", FA_AREA_RAW)):
        for group in ("Overall", "Soft", "Stiff"):
            out = partial_corr_one_covariate(
                factin_df,
                xcol=xcol,
                ycol=FACTIN_RAW,
                covariate="raw_dapi_cell_count",
                group=group,
            )
            rows.append(
                {
                    "relationship": f"{label_text} vs raw F-actin",
                    "x_metric": xcol,
                    "y_metric": FACTIN_RAW,
                    "covariate": "raw_dapi_cell_count",
                    **out,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def fmt_r(v):
    return "NA" if not np.isfinite(v) else f"{v:.2f}"


def fmt_p(v):
    if not np.isfinite(v):
        return "NA"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def draw_regression(ax, d, xcol, ycol, *, color, linestyle, linewidth):
    if len(d) < 2 or d[xcol].nunique() < 2:
        return
    x = d[xcol].to_numpy(float)
    y = d[ycol].to_numpy(float)
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(xs, coef[0] * xs + coef[1], color=color, linestyle=linestyle, linewidth=linewidth)


def plot_panel(
    ax,
    df,
    *,
    panel,
    xcol,
    ycol,
    xlabel,
    ylabel,
    y_scale=1.0,
    panel_label=None,
):
    d = df[["condition", xcol, ycol]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d["_yplot"] = d[ycol] / float(y_scale)
    soft = d.loc[d["condition"] == "Soft"].copy()
    stiff = d.loc[d["condition"] == "Stiff"].copy()

    ax.scatter(soft[xcol], soft["_yplot"], s=30, alpha=0.75, edgecolor="none", color="tab:blue", label="Soft")
    ax.scatter(stiff[xcol], stiff["_yplot"], s=30, alpha=0.75, edgecolor="none", color="tab:red", label="Stiff")
    draw_regression(ax, d, xcol, "_yplot", color="black", linestyle="--", linewidth=1.2)
    draw_regression(ax, soft, xcol, "_yplot", color="tab:blue", linestyle="-", linewidth=1.5)
    draw_regression(ax, stiff, xcol, "_yplot", color="tab:red", linestyle="-", linewidth=1.5)

    rows, stat_map = [], {}
    for group in ("Overall", "Soft", "Stiff"):
        sub = d if group == "Overall" else d.loc[d["condition"] == group]
        st = pearson_stats(sub[xcol], sub[ycol])
        stat_map[group] = st
        rows.append({"panel": panel, "x_metric": xcol, "y_metric": ycol, "group": group, **st})

    text = (
        f"Overall: r={fmt_r(stat_map['Overall']['r'])}, p={fmt_p(stat_map['Overall']['p'])}, n={stat_map['Overall']['n']}\n"
        f"Soft: r={fmt_r(stat_map['Soft']['r'])}, p={fmt_p(stat_map['Soft']['p'])}, n={stat_map['Soft']['n']}\n"
        f"Stiff: r={fmt_r(stat_map['Stiff']['r'])}, p={fmt_p(stat_map['Stiff']['p'])}, n={stat_map['Stiff']['n']}"
    )
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if panel_label is None:
        panel_label = panel
    if panel_label:
        ax.text(
            -0.14, 1.06, f"({panel_label})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return rows


def make_figures(factin_df, yap_merged, *, save_dir: Path, dpi: int):
    """
    Create the revised main Fig. 6 and the reviewer-requested supplementary plot.

    Main Fig. 6
    -----------
    (a) FA count per cell vs raw whole-image F-actin intensity per cell
    (b) FA count per cell vs YAP N/C mean-intensity ratio

    Total FA area is not repeated in the main figure because total FA count and
    total FA area are nearly redundant in the present dataset. Their direct
    relationship is shown separately in the supplementary figure.

    Supplementary figure
    --------------------
    Direct comparison of total FA count vs total FA area using image-level raw
    totals, with Overall / Soft / Stiff Pearson correlations.
    """

    # ------------------------------------------------------------------
    # Main Fig. 6: two non-redundant biological association panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    main_rows = []
    main_rows += plot_panel(
        axes[0],
        factin_df,
        panel="a",
        panel_label="a",
        xcol=FA_COUNT_PC,
        ycol=FACTIN_PC,
        xlabel="FA count per cell",
        ylabel=(
            "Raw whole-image F-actin intensity per cell\n"
            "($\\times 10^6$ 8-bit-equivalent a.u.)"
        ),
        y_scale=FACTIN_PLOT_SCALE,
    )
    main_rows += plot_panel(
        axes[1],
        yap_merged,
        panel="b",
        panel_label="b",
        xcol=FA_COUNT_PC,
        ycol=YAP_NC,
        xlabel="FA count per cell",
        ylabel="YAP N/C mean-intensity ratio",
    )

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels_text[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    main_png = save_dir / "Fig6_raw_channels_final.png"
    main_pdf = save_dir / "Fig6_raw_channels_final.pdf"
    main_tiff = save_dir / "Fig6_raw_channels_final.tiff"

    fig.savefig(main_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(main_pdf, bbox_inches="tight")
    fig.savefig(main_tiff, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Supplementary: total FA count vs total FA area
    # ------------------------------------------------------------------
    supp_fig, supp_ax = plt.subplots(1, 1, figsize=(7.2, 5.6))

    supp_rows = plot_panel(
        supp_ax,
        factin_df,
        panel="S_FA_count_vs_area",
        panel_label="",  # standalone supplementary figure: no "(a)" needed
        xcol=FA_COUNT_RAW,
        ycol=FA_AREA_RAW,
        xlabel="Total FA count",
        ylabel=r"Total FA area (px$^2$)",
    )

    handles, labels_text = supp_ax.get_legend_handles_labels()
    supp_fig.legend(
        handles[:2],
        labels_text[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    supp_fig.tight_layout(rect=(0, 0, 1, 0.94))

    supp_png = save_dir / "FigS_FA_count_vs_total_area.png"
    supp_pdf = save_dir / "FigS_FA_count_vs_total_area.pdf"
    supp_tiff = save_dir / "FigS_FA_count_vs_total_area.tiff"

    supp_fig.savefig(supp_png, dpi=dpi, bbox_inches="tight")
    supp_fig.savefig(supp_pdf, bbox_inches="tight")
    supp_fig.savefig(supp_tiff, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(supp_fig)

    return (
        pd.DataFrame(main_rows),
        pd.DataFrame(supp_rows),
        (main_png, main_pdf, main_tiff, supp_png, supp_pdf, supp_tiff),
    )


def main():
    args = parse_args()
    fa_csv = args.fa_csv.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if not fa_csv.exists():
        raise FileNotFoundError(f"FA CSV not found: {fa_csv}")
    if not data_root.exists():
        raise FileNotFoundError(f"Raw data root not found: {data_root}")

    cell_utils = load_cell_utils(args.cell_utils_dir)

    print("=" * 92)
    print("FIG. 6 — FINAL RAW-CHANNEL PIPELINE")
    print("=" * 92)
    print(f"FA CSV   : {fa_csv}")
    print(f"Raw data : {data_root}")
    print(f"Save dir : {save_dir}")
    print("Fig.6a F-actin folders:", ",".join(sorted(FIG6_FACTIN_FOLDERS, key=int)))
    print("  F-actin 647 nm :", ",".join(sorted(F_ACTIN_647_FOLDERS, key=int)))
    print("  F-actin 546 nm :", ",".join(sorted(F_ACTIN_546_FOLDERS, key=int)))
    print("Fig.6b YAP folders:", ",".join(sorted(FIG6_YAP_FOLDERS, key=int)))
    print("Merged data used for biological readouts: NO")
    print("Per-cell denominator: raw 405-nm DAPI-positive nuclei (cell-count proxy)")
    print("QC reference       : revised AdhGen n_nuclei (NOT n_cells)")

    channels, ignored = discover_raw_channels(data_root)
    inventory = channel_inventory_wide(channels)
    channels.to_csv(save_dir / "Fig6_raw_channel_inventory_long.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(save_dir / "Fig6_raw_channel_inventory_wide.csv", index=False, encoding="utf-8-sig")
    ignored.to_csv(save_dir / "Fig6_raw_channel_inventory_ignored.csv", index=False, encoding="utf-8-sig")

    fa, fa_audit = load_crossfitted_fa(fa_csv)
    fa.to_csv(save_dir / "Fig6_FA_crossfitted_raw_metrics.csv", index=False, encoding="utf-8-sig")
    fa_audit.to_csv(save_dir / "Fig6_FA_aggregation_audit.csv", index=False, encoding="utf-8-sig")

    factin, factin_failures = build_raw_factin_table(
        fa,
        inventory,
        include_scale_bar=args.include_scale_bar,
        allow_dapi_count_mismatch=args.allow_dapi_count_mismatch,
        cell_utils=cell_utils,
    )
    factin.to_csv(save_dir / "Fig6_raw_F-actin_per_cell.csv", index=False, encoding="utf-8-sig")
    factin_failures.to_csv(save_dir / "Fig6_raw_F-actin_failures.csv", index=False, encoding="utf-8-sig")
    if not factin_failures.empty:
        print("\n[F-actin QC failures]")
        print(factin_failures.to_string(index=False))
        raise RuntimeError(
            f"Raw F-actin/DAPI QC failed for {len(factin_failures)} images. "
            "Inspect Fig6_raw_F-actin_failures.csv."
        )
    if set(factin["key"]) != set(fa["key"]):
        missing_keys = sorted(set(fa["key"]) - set(factin["key"]))
        raise RuntimeError(f"Not every AdhGen image has valid raw F-actin data: {missing_keys[:30]}")

    yap, yap_failures = build_raw_yap_table(
        fa,
        inventory,
        save_qc=args.save_yap_qc,
        qc_dir=save_dir / "YAP_QC",
    )
    yap.to_csv(save_dir / "Fig6_raw_YAP_NC.csv", index=False, encoding="utf-8-sig")
    yap_failures.to_csv(save_dir / "Fig6_raw_YAP_failures.csv", index=False, encoding="utf-8-sig")
    if not yap_failures.empty:
        print("\n[YAP QC failures]")
        print(yap_failures.to_string(index=False))

    fa_per_cell = factin[
        ["key", "folder_id", "condition", "sample_key", "raw_dapi_cell_count", FA_COUNT_RAW, FA_AREA_RAW, FA_COUNT_PC, FA_AREA_PC]
    ].copy()
    yap_merged = fa_per_cell.loc[
        fa_per_cell["folder_id"].astype(str).isin(FIG6_YAP_FOLDERS)
    ].merge(
        yap[
            [
                "key", "path_405", "path_546_cell_region", "path_647_yap",
                "n_405_regions_original_method", "n_546_regions_original_method",
                "nuclear_pixels", "cytoplasmic_pixels", "cell_pixels",
                "nuclear_yap_mean_8bit_equivalent", "cytoplasmic_yap_mean_8bit_equivalent", YAP_NC,
            ]
        ],
        on="key",
        how="inner",
        validate="one_to_one",
    )
    if not set(yap_merged["folder_id"].astype(str)).issubset(FIG6_YAP_FOLDERS):
        raise RuntimeError("Unexpected folder entered Fig.6b")

    main_corr_df, supp_corr_df, figure_paths = make_figures(
        factin, yap_merged, save_dir=save_dir, dpi=args.dpi
    )
    sensitivity_df = build_shared_denominator_sensitivity(factin)

    main_corr_df.to_csv(
        save_dir / "Fig6_Pearson_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    supp_corr_df.to_csv(
        save_dir / "FigS_FA_count_vs_total_area_Pearson.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_df.to_csv(
        save_dir / "Fig6_shared_denominator_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    yap_merged.to_csv(
        save_dir / "Fig6_YAP_matched_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Source data for the supplementary reviewer-requested comparison.
    factin[
        ["key", "folder_id", "condition", "sample_key", FA_COUNT_RAW, FA_AREA_RAW]
    ].to_csv(
        save_dir / "FigS_FA_count_vs_total_area_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mismatch_n = int((~factin["raw_dapi_matches_adhgen_n_nuclei"]).sum())
    final_audit = pd.DataFrame(
        [{
            "fig6a_factin_folders": ",".join(sorted(FIG6_FACTIN_FOLDERS, key=int)),
            "fig6b_yap_folders": ",".join(sorted(FIG6_YAP_FOLDERS, key=int)),
            "f_actin_647_folders": ",".join(sorted(F_ACTIN_647_FOLDERS, key=int)),
            "f_actin_546_folders": ",".join(sorted(F_ACTIN_546_FOLDERS, key=int)),
            "crossfitted_fa_images": len(fa),
            "fig6a_raw_factin_images": len(factin),
            "fig6b_valid_yap_images": len(yap_merged),
            "yap_qc_failures": len(yap_failures),
            "raw_dapi_vs_adhgen_n_nuclei_mismatches": mismatch_n,
            "merged_data_used_for_factin": False,
            "merged_data_used_for_yap": False,
        }]
    )
    final_audit.to_csv(save_dir / "Fig6_final_audit.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 92)
    print("FINAL AUDIT")
    print("=" * 92)
    print(final_audit.to_string(index=False))
    print("\n" + "=" * 92)
    print("FIG. 6 PEARSON CORRELATIONS")
    print("=" * 92)
    print(main_corr_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print("\n" + "=" * 92)
    print("SUPPLEMENTARY: TOTAL FA COUNT VS TOTAL FA AREA")
    print("=" * 92)
    print(supp_corr_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\n" + "=" * 92)
    print("SHARED-DENOMINATOR SENSITIVITY")
    print("=" * 92)
    print(sensitivity_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nFIGURES")
    for p in figure_paths:
        print(p)
    print("\nDone.")


if __name__ == "__main__":
    main()
