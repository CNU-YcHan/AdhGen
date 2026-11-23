#!/usr/bin/env python3
"""
Fig. 1c: whole-image F-actin intensity and F-actin-positive area
normalized by DAPI-detected nuclei count.

This version DOES NOT:
- segment individual cell regions
- estimate image background
- subtract background from F-actin intensity
- derive an area threshold from background statistics

Analysis unit:
    One image = one observation.

Per image:
1. Detect nuclei from the DAPI channel.
2. Use the number of detected nuclei as a cell-count proxy.
3. Analyze the whole image except the optional scale-bar exclusion.
4. Convert F-actin values to a common 8-bit-equivalent scale and calculate integrated intensity / nucleus count.
5. Segment F-actin-positive pixels using Otsu or a user-specified fixed
   threshold, then calculate positive area / nucleus count.
6. Compare Soft vs Stiff using image-level statistics.
7. Save a two-panel Fig. 1c-style plot and CSV files.

Required in the same directory:
    cell_instance_utils_dapi_all_cells.py

Default channel convention:
    Blue  = DAPI
    Green = vinculin/paxillin
    Red   = F-actin
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats
    from scipy.ndimage import binary_fill_holes
except ImportError as exc:
    raise ImportError(
        "scipy가 필요합니다. Anaconda Prompt에서 "
        "`conda install scipy`를 실행하세요."
    ) from exc

import cell_instance_utils_dapi_all_cells as cell_utils


SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"
}
CONDITION_NAMES = {"soft": "Soft", "stiff": "Stiff"}

SCALE_BAR_X_START_FRACTION = 0.78
SCALE_BAR_HEIGHT_PX = 70

INTENSITY_METRIC = "integrated_f_actin_intensity_8bit_equivalent_per_nucleus"
AREA_METRIC = "f_actin_positive_area_per_nucleus_px"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DAPI nucleus 수를 cell-count proxy로 사용하여, "
            "이미지 전체의 raw F-actin intensity와 F-actin-positive area를 "
            "nucleus 수로 나누고 Soft와 Stiff를 비교합니다."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="하나 이상의 merged image 또는 image folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="결과를 저장할 폴더",
    )
    parser.add_argument(
        "--f-actin-channel",
        choices=("red", "green", "blue"),
        default="red",
        help="F-actin channel. 기본값: red",
    )
    parser.add_argument(
        "--area-threshold-method",
        choices=("otsu", "fixed"),
        default="otsu",
        help=(
            "F-actin-positive area threshold 방식. "
            "otsu: 이미지별 Otsu threshold, "
            "fixed: 모든 이미지에 동일한 threshold. 기본값: otsu"
        ),
    )
    parser.add_argument(
        "--fixed-area-threshold",
        type=float,
        default=None,
        help=(
            "--area-threshold-method fixed일 때 사용할 intensity threshold. "
            "8-bit 이미지 예: 20"
        ),
    )
    parser.add_argument(
        "--opening-radius",
        type=int,
        default=1,
        help="F-actin-positive mask opening radius(px). 기본값: 1",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=2,
        help="F-actin-positive mask closing radius(px). 기본값: 2",
    )
    parser.add_argument(
        "--min-positive-component-area",
        type=int,
        default=50,
        help=(
            "제거할 작은 F-actin-positive component의 최소 면적(px). "
            "기본값: 50"
        ),
    )
    parser.add_argument(
        "--exclude-folder",
        nargs="*",
        default=[],
        help="제외할 acquisition folder 이름. 예: --exclude-folder 9 12",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="폴더 입력 시 하위 폴더를 탐색하지 않음",
    )
    parser.add_argument(
        "--include-scale-bar",
        action="store_true",
        help="오른쪽 아래 scale-bar 영역을 분석에 포함",
    )
    parser.add_argument(
        "--minimum-nuclei",
        type=int,
        default=1,
        help="분석에 포함할 최소 nucleus 수. 기본값: 1",
    )
    parser.add_argument(
        "--primary-test",
        choices=("mannwhitney", "welch"),
        default="mannwhitney",
        help="그림에 표시할 primary test. 기본값: mannwhitney",
    )
    parser.add_argument(
        "--normalize-intensity-to-soft-median",
        action="store_true",
        help=(
            "Intensity panel만 Soft median 기준 상대값으로 표시. "
            "CSV에는 raw a.u./nucleus 값이 저장됩니다."
        ),
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help=(
            "1 pixel의 한 변 길이(µm). 입력하면 area를 µm²/nucleus로도 "
            "계산합니다. 예: --pixel-size-um 0.207"
        ),
    )
    parser.add_argument(
        "--save-qc",
        action="store_true",
        help="각 이미지의 DAPI nucleus와 F-actin-positive mask QC panel 저장",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG 저장 해상도. 기본값: 300",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="점 jitter 재현용 seed. 기본값: 42",
    )
    return parser.parse_args()


def natural_key(path: Path) -> List[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def collect_images(
    input_paths: Sequence[Path],
    recursive: bool,
    excluded_folders: set[str],
) -> List[Path]:
    collected: List[Path] = []

    for raw_path in input_paths:
        input_path = raw_path.expanduser().resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"입력 경로가 없습니다: {input_path}")

        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"지원하지 않는 이미지 확장자입니다: {input_path}"
                )
            collected.append(input_path)
            continue

        iterator: Iterable[Path] = (
            input_path.rglob("*")
            if recursive
            else input_path.glob("*")
        )

        collected.extend(
            path.resolve()
            for path in iterator
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and "_qc" not in path.stem.lower()
        )

    result: List[Path] = []

    for path in sorted(set(collected), key=natural_key):
        parent_names = {part.lower() for part in path.parts}

        if any(
            folder.lower() in parent_names
            for folder in excluded_folders
        ):
            continue

        result.append(path)

    if not result:
        raise ValueError("조건에 맞는 입력 이미지를 찾지 못했습니다.")

    return result


def infer_condition(path: Path) -> str:
    for part in reversed(path.parts):
        normalized = part.strip().lower()

        if normalized in CONDITION_NAMES:
            return CONDITION_NAMES[normalized]

    raise ValueError(
        "경로에서 Soft 또는 Stiff condition을 찾지 못했습니다."
    )


def infer_acquisition_folder(path: Path) -> str:
    parts = list(path.parts)
    condition_index: Optional[int] = None

    for index, part in enumerate(parts):
        if part.strip().lower() in CONDITION_NAMES:
            condition_index = index

    if condition_index is None or condition_index == 0:
        return ""

    return parts[condition_index - 1]


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


def read_image_for_analysis(
    image_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    image_for_segmentation:
        8-bit BGR image used for DAPI nucleus segmentation and Otsu thresholding.
    image_for_measurement:
        Original numeric BGR image. Its dtype is retained for scale conversion.
    """
    original = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if original is None:
        raise ValueError("OpenCV가 이미지를 읽지 못했습니다.")

    if original.ndim != 3 or original.shape[2] < 3:
        raise ValueError("3-channel merged image가 아닙니다.")

    if original.shape[2] > 3:
        original = original[:, :, :3]

    if original.dtype == np.uint8:
        image_8bit = original.copy()

    elif original.dtype == np.uint16:
        image_8bit = np.round(
            original.astype(np.float32) / 257.0
        ).clip(0, 255).astype(np.uint8)

    else:
        array = original.astype(np.float32)
        finite = np.isfinite(array)

        if not np.any(finite):
            raise ValueError("유효한 pixel 값이 없습니다.")

        low = float(np.nanpercentile(array[finite], 0.1))
        high = float(np.nanpercentile(array[finite], 99.9))

        if high <= low:
            raise ValueError("이미지 intensity range가 유효하지 않습니다.")

        image_8bit = (
            (array - low)
            / (high - low)
            * 255.0
        ).clip(0, 255).astype(np.uint8)

    return image_8bit, original


def select_channel(
    image_bgr: np.ndarray,
    channel_name: str,
) -> np.ndarray:
    channel_index = {
        "blue": 0,
        "green": 1,
        "red": 2,
    }[channel_name]

    return image_bgr[:, :, channel_index].astype(np.float64)


def select_channel_8bit_equivalent(
    image_original: np.ndarray,
    image_8bit: np.ndarray,
    channel_name: str,
) -> np.ndarray:
    """
    모든 이미지를 공통 0~255 intensity scale로 변환한다.

    - uint8: 원래 값을 그대로 사용
    - uint16: 257로 나누어 8-bit equivalent 값으로 변환
    - 기타 dtype: segmentation에 사용한 8-bit image 값을 사용

    uint8과 uint16 raw 값을 그대로 합산하면 약 257배의 scale 차이가
    생기므로 직접 비교할 수 없다.
    """
    channel_index = {
        "blue": 0,
        "green": 1,
        "red": 2,
    }[channel_name]

    original_channel = image_original[:, :, channel_index]

    if image_original.dtype == np.uint8:
        return original_channel.astype(np.float64)

    if image_original.dtype == np.uint16:
        return (
            original_channel.astype(np.float64)
            / 257.0
        )

    return image_8bit[:, :, channel_index].astype(np.float64)


def segment_nuclei_only(
    image_bgr_8bit: np.ndarray,
    scale_bar_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    DAPI nucleus만 검출한다.
    Cell instance segmentation은 수행하지 않는다.
    """
    nucleus_input = image_bgr_8bit.copy()
    nucleus_input[scale_bar_mask] = 0

    nucleus_labels, _, nuclei, _ = cell_utils.segment_nuclei(
        nucleus_input,
        cell_utils.NUCLEUS_CONFIG,
    )

    return nucleus_labels.astype(np.int32), int(len(nuclei))


def disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=np.uint8)

    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (x * x + y * y <= radius * radius)

    return disk.astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    minimum_area: int,
) -> np.ndarray:
    if minimum_area <= 1:
        return mask.astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros(mask.shape, dtype=bool)

    for component_id in range(1, n_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area >= minimum_area:
            cleaned[labels == component_id] = True

    return cleaned


def otsu_threshold_on_analysis_area(
    f_actin_8bit: np.ndarray,
    analysis_mask: np.ndarray,
) -> float:
    values = f_actin_8bit[analysis_mask]

    if values.size == 0:
        raise ValueError("Otsu threshold를 계산할 pixel이 없습니다.")

    # OpenCV Otsu는 2D/1D uint8 array에 적용할 수 있다.
    threshold, _ = cv2.threshold(
        values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return float(threshold)


def build_f_actin_positive_mask(
    f_actin_8bit: np.ndarray,
    analysis_mask: np.ndarray,
    threshold_method: str,
    fixed_threshold: Optional[float],
    opening_radius: int,
    closing_radius: int,
    minimum_component_area: int,
) -> Tuple[np.ndarray, float]:
    if threshold_method == "otsu":
        threshold = otsu_threshold_on_analysis_area(
            f_actin_8bit,
            analysis_mask,
        )

    else:
        if fixed_threshold is None:
            raise ValueError(
                "--area-threshold-method fixed를 사용할 때는 "
                "--fixed-area-threshold 값을 입력해야 합니다."
            )

        if not (0.0 <= fixed_threshold <= 255.0):
            raise ValueError(
                "--fixed-area-threshold는 0~255 범위여야 합니다."
            )

        threshold = float(fixed_threshold)

    positive_mask = (
        (f_actin_8bit > threshold)
        & analysis_mask
    )

    if opening_radius > 0:
        positive_mask = cv2.morphologyEx(
            positive_mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            disk_kernel(opening_radius),
        ) > 0

    if closing_radius > 0:
        positive_mask = cv2.morphologyEx(
            positive_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            disk_kernel(closing_radius),
        ) > 0

    positive_mask = binary_fill_holes(
        positive_mask
    ).astype(bool)

    positive_mask = remove_small_components(
        positive_mask,
        minimum_area=minimum_component_area,
    )

    return positive_mask & analysis_mask, threshold


def save_qc_panel(
    image_path: Path,
    image_bgr_8bit: np.ndarray,
    nucleus_labels: np.ndarray,
    positive_mask: np.ndarray,
    n_nuclei: int,
    area_threshold: float,
    f_actin_channel_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    image_rgb = cv2.cvtColor(
        image_bgr_8bit,
        cv2.COLOR_BGR2RGB,
    )

    f_actin_8bit = select_channel(
        image_bgr_8bit,
        f_actin_channel_name,
    ).astype(np.uint8)

    overlay = image_rgb.copy()
    overlay[positive_mask] = (
        0.35 * overlay[positive_mask]
        + 0.65 * np.array([255, 255, 255])
    ).astype(np.uint8)

    nucleus_boundary = np.zeros(
        nucleus_labels.shape,
        dtype=bool,
    )
    nucleus_boundary[1:, :] |= (
        nucleus_labels[1:, :]
        != nucleus_labels[:-1, :]
    )
    nucleus_boundary[:-1, :] |= (
        nucleus_labels[:-1, :]
        != nucleus_labels[1:, :]
    )
    nucleus_boundary[:, 1:] |= (
        nucleus_labels[:, 1:]
        != nucleus_labels[:, :-1]
    )
    nucleus_boundary[:, :-1] |= (
        nucleus_labels[:, :-1]
        != nucleus_labels[:, 1:]
    )
    nucleus_boundary &= nucleus_labels > 0
    overlay[nucleus_boundary] = [0, 255, 255]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 4.2),
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Merged image")

    axes[1].imshow(
        image_bgr_8bit[:, :, 0],
        cmap="gray",
    )
    axes[1].imshow(
        nucleus_boundary,
        cmap="autumn",
        alpha=0.6,
    )
    axes[1].set_title(f"DAPI nuclei (N={n_nuclei})")

    axes[2].imshow(
        f_actin_8bit,
        cmap="gray",
    )
    axes[2].set_title(
        f"F-actin channel\narea threshold={area_threshold:.2f}"
    )

    axes[3].imshow(overlay)
    axes[3].set_title(
        "Whole-image F-actin-positive mask\n"
        f"area={int(np.count_nonzero(positive_mask))} px"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(image_path.name)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def analyze_image(
    image_path: Path,
    args: argparse.Namespace,
    qc_root: Optional[Path],
) -> dict:
    image_8bit, image_original = read_image_for_analysis(
        image_path
    )

    scale_bar_mask = make_scale_bar_mask(
        image_8bit.shape,
        include_scale_bar=args.include_scale_bar,
    )
    analysis_mask = ~scale_bar_mask

    nucleus_labels, n_nuclei = segment_nuclei_only(
        image_8bit,
        scale_bar_mask=scale_bar_mask,
    )

    if n_nuclei < args.minimum_nuclei:
        raise ValueError(
            f"검출 nucleus 수가 {n_nuclei}개로 "
            f"--minimum-nuclei {args.minimum_nuclei}보다 작습니다."
        )

    f_actin_scaled = select_channel_8bit_equivalent(
        image_original=image_original,
        image_8bit=image_8bit,
        channel_name=args.f_actin_channel,
    )
    f_actin_8bit = select_channel(
        image_8bit,
        args.f_actin_channel,
    ).astype(np.uint8)

    # Background subtraction 없이 이미지 전체 intensity를 합산하되,
    # uint8/uint16 혼합으로 인한 scale 차이를 제거하기 위해 모든 값을
    # 공통 8-bit-equivalent scale(0~255)로 변환한다.
    raw_integrated_intensity = float(
        np.sum(f_actin_scaled[analysis_mask])
    )
    raw_intensity_per_nucleus = (
        raw_integrated_intensity / float(n_nuclei)
    )

    mean_raw_intensity_whole_image = float(
        np.mean(f_actin_scaled[analysis_mask])
    )

    positive_mask, area_threshold = (
        build_f_actin_positive_mask(
            f_actin_8bit=f_actin_8bit,
            analysis_mask=analysis_mask,
            threshold_method=args.area_threshold_method,
            fixed_threshold=args.fixed_area_threshold,
            opening_radius=args.opening_radius,
            closing_radius=args.closing_radius,
            minimum_component_area=(
                args.min_positive_component_area
            ),
        )
    )

    total_positive_area_px = int(
        np.count_nonzero(positive_mask)
    )
    positive_area_per_nucleus_px = (
        total_positive_area_px / float(n_nuclei)
    )

    raw_integrated_positive_intensity = float(
        np.sum(f_actin_scaled[positive_mask])
    )
    raw_positive_intensity_per_nucleus = (
        raw_integrated_positive_intensity
        / float(n_nuclei)
    )

    mean_raw_intensity_positive_area = (
        float(np.mean(f_actin_scaled[positive_mask]))
        if total_positive_area_px > 0
        else 0.0
    )

    total_positive_area_um2: object = ""
    positive_area_per_nucleus_um2: object = ""

    if args.pixel_size_um is not None:
        pixel_area_um2 = float(
            args.pixel_size_um ** 2
        )
        total_positive_area_um2 = (
            total_positive_area_px
            * pixel_area_um2
        )
        positive_area_per_nucleus_um2 = (
            positive_area_per_nucleus_px
            * pixel_area_um2
        )

    if qc_root is not None:
        condition = infer_condition(image_path)
        acquisition_folder = infer_acquisition_folder(
            image_path
        )
        qc_dir = qc_root / acquisition_folder / condition
        qc_dir.mkdir(parents=True, exist_ok=True)

        save_qc_panel(
            image_path=image_path,
            image_bgr_8bit=image_8bit,
            nucleus_labels=nucleus_labels,
            positive_mask=positive_mask,
            n_nuclei=n_nuclei,
            area_threshold=area_threshold,
            f_actin_channel_name=args.f_actin_channel,
            output_path=(
                qc_dir / f"{image_path.stem}_qc.png"
            ),
            dpi=min(args.dpi, 200),
        )

    analysis_area_px = int(
        np.count_nonzero(analysis_mask)
    )

    result = {
        "file_name": image_path.name,
        "condition": infer_condition(image_path),
        "acquisition_folder": infer_acquisition_folder(
            image_path
        ),
        "original_dtype": str(image_original.dtype),
        "n_dapi_nuclei": n_nuclei,
        "f_actin_positive_threshold_8bit": area_threshold,
        "integrated_f_actin_intensity_8bit_equivalent_whole_image": (
            raw_integrated_intensity
        ),
        "integrated_f_actin_intensity_8bit_equivalent_per_nucleus": (
            raw_intensity_per_nucleus
        ),
        "f_actin_positive_area_total_px": total_positive_area_px,
        "f_actin_positive_area_per_nucleus_px": (
            positive_area_per_nucleus_px
        ),
    }

    if args.pixel_size_um is not None:
        result[
            "f_actin_positive_area_total_um2"
        ] = total_positive_area_um2
        result[
            "f_actin_positive_area_per_nucleus_um2"
        ] = positive_area_per_nucleus_um2

    return result


def write_csv(
    path: Path,
    rows: Sequence[dict],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def quartile_summary(
    values: np.ndarray,
) -> dict:
    return {
        "n_images": int(values.size),
        "mean": float(np.mean(values)),
        "sd": (
            float(np.std(values, ddof=1))
            if values.size > 1
            else float("nan")
        ),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def cliffs_delta(
    soft: np.ndarray,
    stiff: np.ndarray,
) -> float:
    """
    delta > 0이면 Stiff가 Soft보다 큰 방향.
    """
    greater = 0
    lower = 0

    for soft_value in soft:
        greater += int(
            np.sum(stiff > soft_value)
        )
        lower += int(
            np.sum(stiff < soft_value)
        )

    return float(
        (greater - lower)
        / (soft.size * stiff.size)
    )


def metric_arrays(
    rows: Sequence[dict],
    metric_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    soft = np.asarray(
        [
            float(row[metric_name])
            for row in rows
            if row["condition"] == "Soft"
        ],
        dtype=float,
    )
    stiff = np.asarray(
        [
            float(row[metric_name])
            for row in rows
            if row["condition"] == "Stiff"
        ],
        dtype=float,
    )

    return soft, stiff


def run_statistics_for_metric(
    rows: Sequence[dict],
    metric_name: str,
    metric_label: str,
) -> Tuple[List[dict], List[dict]]:
    soft, stiff = metric_arrays(
        rows,
        metric_name,
    )

    if soft.size == 0 or stiff.size == 0:
        raise RuntimeError(
            f"{metric_label}: Soft와 Stiff 값이 모두 필요합니다."
        )

    summary_rows = [
        {
            "metric": metric_name,
            "metric_label": metric_label,
            "condition": "Soft",
            **quartile_summary(soft),
        },
        {
            "metric": metric_name,
            "metric_label": metric_label,
            "condition": "Stiff",
            **quartile_summary(stiff),
        },
    ]

    test_rows: List[dict] = []

    mann_whitney = stats.mannwhitneyu(
        soft,
        stiff,
        alternative="two-sided",
        method="auto",
    )

    test_rows.append(
        {
            "metric": metric_name,
            "metric_label": metric_label,
            "test": "Mann-Whitney U",
            "statistic": float(
                mann_whitney.statistic
            ),
            "p_value": float(
                mann_whitney.pvalue
            ),
            "effect_size_name": (
                "Cliff_delta_Stiff_greater_positive"
            ),
            "effect_size": cliffs_delta(
                soft,
                stiff,
            ),
        }
    )

    if soft.size >= 2 and stiff.size >= 2:
        welch = stats.ttest_ind(
            soft,
            stiff,
            equal_var=False,
            nan_policy="omit",
        )
        test_rows.append(
            {
                "metric": metric_name,
                "metric_label": metric_label,
                "test": "Welch t-test",
                "statistic": float(
                    welch.statistic
                ),
                "p_value": float(
                    welch.pvalue
                ),
                "effect_size_name": "",
                "effect_size": "",
            }
        )

    if 3 <= soft.size <= 5000:
        shapiro_soft = stats.shapiro(soft)
        test_rows.append(
            {
                "metric": metric_name,
                "metric_label": metric_label,
                "test": "Shapiro-Wilk Soft",
                "statistic": float(
                    shapiro_soft.statistic
                ),
                "p_value": float(
                    shapiro_soft.pvalue
                ),
                "effect_size_name": "",
                "effect_size": "",
            }
        )

    if 3 <= stiff.size <= 5000:
        shapiro_stiff = stats.shapiro(stiff)
        test_rows.append(
            {
                "metric": metric_name,
                "metric_label": metric_label,
                "test": "Shapiro-Wilk Stiff",
                "statistic": float(
                    shapiro_stiff.statistic
                ),
                "p_value": float(
                    shapiro_stiff.pvalue
                ),
                "effect_size_name": "",
                "effect_size": "",
            }
        )

    return summary_rows, test_rows


def select_primary_result(
    test_rows: Sequence[dict],
    metric_name: str,
    primary_test: str,
) -> dict:
    target = (
        "Mann-Whitney U"
        if primary_test == "mannwhitney"
        else "Welch t-test"
    )

    for row in test_rows:
        if (
            row["metric"] == metric_name
            and row["test"] == target
        ):
            return row

    raise ValueError(
        f"{metric_name}의 primary test 결과가 없습니다."
    )


def p_text(p_value: float) -> str:
    if p_value < 0.0001:
        return "p < 0.0001"

    if p_value < 0.001:
        return f"p = {p_value:.4f}"

    return f"p = {p_value:.3f}"


def choose_plot_scale(
    values: np.ndarray,
) -> Tuple[float, str]:
    maximum = float(np.nanmax(values))

    if maximum <= 0:
        return 1.0, ""

    exponent = int(
        math.floor(math.log10(maximum))
    )
    exponent = (exponent // 3) * 3

    if exponent < 3:
        return 1.0, ""

    scale = float(10 ** exponent)

    return scale, f"×10^{exponent} "


def add_significance_bar(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    height: float,
    text: str,
) -> None:
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        linewidth=1.2,
    )
    ax.text(
        (x1 + x2) / 2,
        y + height,
        text,
        ha="center",
        va="bottom",
    )


def draw_violin_panel(
    ax: plt.Axes,
    soft: np.ndarray,
    stiff: np.ndarray,
    y_label: str,
    title: str,
    p_value: float,
    rng: np.random.Generator,
) -> None:
    groups = [soft, stiff]
    positions = [1, 2]

    violin = ax.violinplot(
        groups,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in violin["bodies"]:
        body.set_alpha(0.35)

    ax.boxplot(
        groups,
        positions=positions,
        widths=0.22,
        showfliers=False,
        patch_artist=False,
        medianprops={"linewidth": 1.8},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )

    for position, values in zip(
        positions,
        groups,
    ):
        jitter = rng.normal(
            loc=0.0,
            scale=0.045,
            size=values.size,
        )
        ax.scatter(
            np.full(values.size, position)
            + jitter,
            values,
            s=18,
            alpha=0.75,
            linewidths=0.4,
            edgecolors="black",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"Soft\n(n={soft.size})",
            f"Stiff\n(n={stiff.size})",
        ]
    )
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_values = np.concatenate(groups)
    y_min = float(np.nanmin(all_values))
    y_max = float(np.nanmax(all_values))
    data_range = max(
        y_max - y_min,
        abs(y_max) * 0.15,
        1e-9,
    )

    bar_y = y_max + data_range * 0.12
    bar_height = data_range * 0.04

    add_significance_bar(
        ax,
        1,
        2,
        bar_y,
        bar_height,
        p_text(p_value),
    )

    ax.set_ylim(
        min(
            0.0,
            y_min - data_range * 0.08,
        ),
        bar_y
        + bar_height
        + data_range * 0.18,
    )


def make_figure(
    rows: Sequence[dict],
    test_rows: Sequence[dict],
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    soft_intensity, stiff_intensity = metric_arrays(
        rows,
        INTENSITY_METRIC,
    )
    intensity_primary = select_primary_result(
        test_rows,
        INTENSITY_METRIC,
        args.primary_test,
    )

    if args.normalize_intensity_to_soft_median:
        denominator = float(
            np.median(soft_intensity)
        )

        if denominator <= 0:
            raise ValueError(
                "Soft median이 0 이하라 intensity normalization을 "
                "수행할 수 없습니다."
            )

        soft_intensity_plot = (
            soft_intensity / denominator
        )
        stiff_intensity_plot = (
            stiff_intensity / denominator
        )
        intensity_label = (
            "Whole-image integrated F-actin\n"
            "intensity per DAPI-positive nucleus\n"
            "(relative to Soft median)"
        )

    else:
        combined = np.concatenate(
            [
                soft_intensity,
                stiff_intensity,
            ]
        )
        scale, prefix = choose_plot_scale(
            combined
        )
        soft_intensity_plot = (
            soft_intensity / scale
        )
        stiff_intensity_plot = (
            stiff_intensity / scale
        )
        intensity_label = (
            "Whole-image integrated F-actin intensity\n"
            f"({prefix}8-bit-equivalent a.u./DAPI-positive nucleus)"
        )

    area_metric = AREA_METRIC
    area_label = (
        "Whole-image F-actin-positive area\n"
        "(pixels/DAPI-positive nucleus)"
    )

    if args.pixel_size_um is not None:
        area_metric = (
            "f_actin_positive_area_per_nucleus_um2"
        )
        area_label = (
            "Whole-image F-actin-positive area\n"
            r"($\mu$m$^2$/DAPI-positive nucleus)"
        )

    soft_area, stiff_area = metric_arrays(
        rows,
        area_metric,
    )
    area_primary = select_primary_result(
        test_rows,
        area_metric,
        args.primary_test,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 5.4),
    )
    rng = np.random.default_rng(
        args.random_seed
    )

    draw_violin_panel(
        axes[0],
        soft_intensity_plot,
        stiff_intensity_plot,
        y_label=intensity_label,
        title="F-actin intensity",
        p_value=float(
            intensity_primary["p_value"]
        ),
        rng=rng,
    )

    draw_violin_panel(
        axes[1],
        soft_area,
        stiff_area,
        y_label=area_label,
        title="F-actin-positive area",
        p_value=float(
            area_primary["p_value"]
        ),
        rng=rng,
    )

    fig.tight_layout()
    fig.savefig(
        output_root
        / "fig1c_whole_image_f_actin_common_scale.png",
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        output_root
        / "fig1c_whole_image_f_actin_common_scale.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.minimum_nuclei < 1:
        raise ValueError(
            "--minimum-nuclei는 1 이상이어야 합니다."
        )

    if (
        args.pixel_size_um is not None
        and args.pixel_size_um <= 0
    ):
        raise ValueError(
            "--pixel-size-um은 0보다 커야 합니다."
        )

    if (
        args.area_threshold_method == "fixed"
        and args.fixed_area_threshold is None
    ):
        raise ValueError(
            "--area-threshold-method fixed를 사용하면 "
            "--fixed-area-threshold를 입력해야 합니다."
        )

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    qc_root = None

    if args.save_qc:
        qc_root = output_root / "qc"
        qc_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    image_paths = collect_images(
        input_paths=args.input,
        recursive=not args.non_recursive,
        excluded_folders={
            str(value).lower()
            for value in args.exclude_folder
        },
    )

    image_rows: List[dict] = []
    failures: List[dict] = []

    print("[Fig. 1c whole-image F-actin analysis: no background correction]")
    print("  images:", len(image_paths))
    print("  output:", output_root)
    print("  cell-count proxy: DAPI-detected nuclei")
    print("  cell instance segmentation: NOT USED")
    print("  background estimation/subtraction: NOT USED\n  intensity scale: common 8-bit-equivalent values")
    print(
        "  area threshold method:",
        args.area_threshold_method,
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        try:
            row = analyze_image(
                image_path=image_path,
                args=args,
                qc_root=qc_root,
            )
            image_rows.append(row)

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name}: "
                f"{row['condition']}, "
                f"nuclei={row['n_dapi_nuclei']}, "
                "8-bit-equivalent intensity/nucleus="
                f"{row[INTENSITY_METRIC]:.3f}, "
                "area/nucleus="
                f"{row[AREA_METRIC]:.3f} px, "
                "threshold="
                f"{row['f_actin_positive_threshold_8bit']:.2f}"
            )

        except Exception as exc:
            failures.append(
                {
                    "path": str(image_path),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(exc),
                }
            )
            print(
                f"[FAILED {index}/{len(image_paths)}] "
                f"{image_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if not image_rows:
        write_csv(
            output_root
            / "fig1c_failures.csv",
            failures,
        )
        raise RuntimeError(
            "성공적으로 분석된 이미지가 없습니다."
        )

    metric_specs: List[
        Tuple[str, str]
    ] = [
        (
            INTENSITY_METRIC,
            (
                "Whole-image integrated F-actin intensity "
                "per DAPI-positive nucleus "
                "(8-bit-equivalent a.u.)"
            ),
        ),
        (
            AREA_METRIC,
            (
                "Whole-image F-actin-positive area "
                "per DAPI-positive nucleus (pixels)"
            ),
        ),
    ]

    if args.pixel_size_um is not None:
        metric_specs.append(
            (
                "f_actin_positive_area_per_nucleus_um2",
                (
                    "Whole-image F-actin-positive area "
                    "per DAPI-positive nucleus (um^2)"
                ),
            )
        )

    group_rows: List[dict] = []
    test_rows: List[dict] = []

    for metric_name, metric_label in metric_specs:
        metric_group_rows, metric_test_rows = (
            run_statistics_for_metric(
                rows=image_rows,
                metric_name=metric_name,
                metric_label=metric_label,
            )
        )
        group_rows.extend(
            metric_group_rows
        )
        test_rows.extend(
            metric_test_rows
        )

    make_figure(
        rows=image_rows,
        test_rows=test_rows,
        args=args,
        output_root=output_root,
    )

    write_csv(
        output_root
        / "fig1c_results_compact.csv",
        image_rows,
    )
    write_csv(
        output_root
        / "fig1c_group_summary.csv",
        group_rows,
    )
    write_csv(
        output_root
        / "fig1c_statistical_tests.csv",
        test_rows,
    )
    write_csv(
        output_root
        / "fig1c_failures.csv",
        failures,
    )

    soft_n = sum(
        row["condition"] == "Soft"
        for row in image_rows
    )
    stiff_n = sum(
        row["condition"] == "Stiff"
        for row in image_rows
    )

    print("\n[Completed]")
    print("  successful images:", len(image_rows))
    print("  Soft:", soft_n)
    print("  Stiff:", stiff_n)

    for metric_name, metric_label in metric_specs[:2]:
        primary = select_primary_result(
            test_rows,
            metric_name,
            args.primary_test,
        )
        mann_whitney = select_primary_result(
            test_rows,
            metric_name,
            "mannwhitney",
        )

        print(
            f"  {metric_label}: "
            f"{primary['test']}, "
            f"{p_text(float(primary['p_value']))}, "
            "Cliff's delta="
            f"{float(mann_whitney['effect_size']):.4f}"
        )

    print(
        "  figure:",
        output_root
        / "fig1c_whole_image_f_actin_common_scale.png",
    )


if __name__ == "__main__":
    main()
