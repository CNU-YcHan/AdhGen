"""
DAPI-seeded cell instance segmentation for Point-guided AdhGen
===============================================================

목적
----
1. Blue(DAPI) 채널에서 nucleus instance를 검출한다.
2. Red(F-actin) + Green(FA marker)으로 coarse cell union을 만든다.
3. DAPI nucleus를 marker로 사용한 watershed로 붙어 있는 세포를 분리한다.
4. nucleus가 없는 debris, actin fragment, scale bar 등의 component를 제거한다.
5. nucleus 내부와 nucleus 주변 margin의 Green point를 제외한다.
6. 영상 경계에 닿는 partial cell도 다른 유효 cell과 동일하게 분석에 포함한다.

필수 패키지
-----------
numpy, opencv-python, scipy, scikit-image
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

UTILITY_VERSION = "2026-07-27-all-cells"

try:
    from skimage.filters import sobel, threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import (
        closing,
        dilation,
        opening,
        disk,
        h_maxima,
        remove_small_objects,
    )
    from skimage.segmentation import watershed
except ImportError as exc:
    raise ImportError(
        "이 코드는 scikit-image가 필요합니다. "
        "`pip install scikit-image` 또는 "
        "`conda install -c conda-forge scikit-image`로 설치하세요."
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class NucleusConfig:
    """Blue(DAPI) nucleus instance segmentation 설정."""

    low_percentile: float = 1.0
    high_percentile: float = 99.8
    blur_sigma: float = 1.5

    # 최종 threshold = max(Otsu * otsu_scale, min_normalized_threshold)
    otsu_scale: float = 0.85
    min_normalized_threshold: float = 0.10

    open_radius_px: int = 2
    close_radius_px: int = 3

    min_nucleus_area_px: int = 300
    max_nucleus_area_px: int = 40000

    # scale bar, 긴 fiber, 글자 등을 nucleus로 오인하지 않기 위한 형태 조건
    min_solidity: float = 0.55
    max_eccentricity: float = 0.995
    min_minor_axis_length_px: float = 8.0

    # 유난히 큰 DAPI component만 touching nuclei 분리를 시도한다.
    split_touching_nuclei: bool = True
    split_area_factor: float = 1.8
    h_maxima_height_px: float = 2.0
    max_split_markers_per_component: int = 12

    # Green reference point를 nucleus로부터 제외할 margin
    point_exclusion_margin_px: int = 4


@dataclass(frozen=True)
class CellInstanceConfig:
    """Red+Green coarse mask와 DAPI-seeded watershed 설정."""

    low_percentile: float = 1.0
    high_percentile: float = 99.5
    blur_sigma: float = 3.0

    otsu_scale: float = 0.55
    min_normalized_threshold: float = 0.03

    close_radius_px: int = 5
    close_iterations: int = 2
    open_radius_px: int = 1
    open_iterations: int = 1

    # coarse component 수준 debris 제거
    min_coarse_component_area_px: int = 1000

    # nucleus가 coarse mask의 작은 hole 밖으로 빠지는 것을 방지
    nucleus_support_margin_px: int = 3

    # FA가 cell boundary 바로 밖으로 잘리는 것을 방지
    cell_margin_px: int = 4

    # watershed 후 지나치게 작은 cell instance 제거
    min_cell_instance_area_px: int = 1000

    # 0이면 순수 marker watershed. 작은 양수는 비정상적으로 긴 영역을 완화.
    watershed_compactness: float = 0.001

    # Border cell도 primary analysis 및 GA objective에 포함한다.
    exclude_border_cells_from_primary: bool = False


NUCLEUS_CONFIG = NucleusConfig()
CELL_INSTANCE_CONFIG = CellInstanceConfig()


# =============================================================================
# Generic helpers
# =============================================================================

def percentile_normalize(
    channel: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    channel = channel.astype(np.float32)
    low = float(np.percentile(channel, low_percentile))
    high = float(np.percentile(channel, high_percentile))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(channel, dtype=np.float32)

    normalized = (channel - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def touches_border(mask: np.ndarray) -> bool:
    binary = mask.astype(bool)
    if binary.size == 0:
        return False

    return bool(
        np.any(binary[0, :])
        or np.any(binary[-1, :])
        or np.any(binary[:, 0])
        or np.any(binary[:, -1])
    )


def _safe_threshold_otsu(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(np.max(values)) <= float(np.min(values)):
        return 1.0

    try:
        return float(threshold_otsu(values))
    except ValueError:
        return 1.0


def _remove_objects_smaller_than(
    binary: np.ndarray,
    minimum_size: int,
) -> np.ndarray:
    """
    scikit-image 버전에 관계없이 minimum_size보다 작은 객체를 제거한다.

    scikit-image <= 0.25:
        remove_small_objects(..., min_size=N)

    scikit-image >= 0.26:
        remove_small_objects(..., max_size=N-1)

    버전 차이 때문에 모든 이미지가 실패하는 일을 방지하기 위해
    함수 signature를 런타임에 확인한다.
    """
    binary = np.asarray(binary, dtype=bool)
    minimum_size = max(1, int(minimum_size))

    parameters = inspect.signature(
        remove_small_objects
    ).parameters

    if "max_size" in parameters:
        return remove_small_objects(
            binary,
            max_size=minimum_size - 1,
        ).astype(bool)

    return remove_small_objects(
        binary,
        min_size=minimum_size,
    ).astype(bool)


def _region_axis_minor_length(region) -> float:
    """
    scikit-image 버전에 따른 regionprops property 이름 차이를 흡수한다.
    """
    for attribute in (
        "axis_minor_length",
        "minor_axis_length",
    ):
        if hasattr(region, attribute):
            return float(getattr(region, attribute))
    return 0.0


def _region_axis_major_length(region) -> float:
    for attribute in (
        "axis_major_length",
        "major_axis_length",
    ):
        if hasattr(region, attribute):
            return float(getattr(region, attribute))
    return 0.0


def _region_shape_valid(region, config: NucleusConfig) -> bool:
    return bool(
        config.min_nucleus_area_px
        <= int(region.area)
        <= config.max_nucleus_area_px
        and float(region.solidity) >= config.min_solidity
        and float(region.eccentricity) <= config.max_eccentricity
        and _region_axis_minor_length(region) >= config.min_minor_axis_length_px
    )


def _metadata_from_mask(
    mask: np.ndarray,
    object_id: int,
    prefix: str,
) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("빈 mask의 metadata를 계산할 수 없습니다.")

    return {
        f"{prefix}_id": int(object_id),
        f"{prefix}_area_px": int(len(xs)),
        f"{prefix}_centroid_x": float(np.mean(xs)),
        f"{prefix}_centroid_y": float(np.mean(ys)),
        f"{prefix}_bbox_left": int(np.min(xs)),
        f"{prefix}_bbox_top": int(np.min(ys)),
        f"{prefix}_bbox_width": int(np.max(xs) - np.min(xs) + 1),
        f"{prefix}_bbox_height": int(np.max(ys) - np.min(ys) + 1),
        f"{prefix}_touches_border": touches_border(mask),
    }


# =============================================================================
# DAPI nucleus segmentation
# =============================================================================

def _initial_nucleus_binary(
    image: np.ndarray,
    config: NucleusConfig,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("BGR 3-channel image가 필요합니다.")

    blue = image[:, :, 0]
    normalized = percentile_normalize(
        blue,
        config.low_percentile,
        config.high_percentile,
    )

    if config.blur_sigma > 0:
        smoothed = cv2.GaussianBlur(
            normalized,
            ksize=(0, 0),
            sigmaX=float(config.blur_sigma),
            sigmaY=float(config.blur_sigma),
        )
    else:
        smoothed = normalized

    otsu_value = _safe_threshold_otsu(smoothed)
    applied_threshold = max(
        float(config.min_normalized_threshold),
        otsu_value * float(config.otsu_scale),
    )

    binary = smoothed >= applied_threshold

    if config.open_radius_px > 0:
        binary = opening(
            binary,
            footprint=disk(int(config.open_radius_px)),
        )

    if config.close_radius_px > 0:
        binary = closing(
            binary,
            footprint=disk(int(config.close_radius_px)),
        )

    binary = ndi.binary_fill_holes(binary)
    binary = _remove_objects_smaller_than(
        binary,
        config.min_nucleus_area_px,
    )

    return (
        binary.astype(bool),
        smoothed.astype(np.float32),
        float(otsu_value),
        float(applied_threshold),
    )


def _filter_initial_nucleus_components(
    binary: np.ndarray,
    config: NucleusConfig,
) -> Tuple[np.ndarray, List[dict]]:
    initial_labels = label(binary, connectivity=2)
    accepted_mask = np.zeros_like(binary, dtype=bool)
    candidates: List[dict] = []

    for region in regionprops(initial_labels):
        if not _region_shape_valid(region, config):
            continue

        region_mask = initial_labels == region.label
        accepted_mask |= region_mask
        candidates.append(
            {
                "original_label": int(region.label),
                "area": int(region.area),
                "mask": region_mask,
                "solidity": float(region.solidity),
                "eccentricity": float(region.eccentricity),
                "minor_axis_length": _region_axis_minor_length(region),
            }
        )

    return accepted_mask, candidates


def _split_large_nucleus_component(
    component_mask: np.ndarray,
    config: NucleusConfig,
) -> np.ndarray:
    """
    큰 DAPI component 내부에 명확한 distance maxima가 여러 개 있을 때만 분리한다.
    반환 label map은 component 내부에서 1..K이다.
    """
    distance = ndi.distance_transform_edt(component_mask)

    maxima = h_maxima(
        distance,
        h=float(config.h_maxima_height_px),
    )
    maxima &= component_mask

    markers, n_markers = ndi.label(maxima)

    if n_markers < 2:
        result = np.zeros_like(markers, dtype=np.int32)
        result[component_mask] = 1
        return result

    if n_markers > int(config.max_split_markers_per_component):
        # 비정상적으로 많은 maxima는 noise일 가능성이 높으므로 분리하지 않는다.
        result = np.zeros_like(markers, dtype=np.int32)
        result[component_mask] = 1
        return result

    return watershed(
        -distance,
        markers=markers,
        mask=component_mask,
        watershed_line=False,
    ).astype(np.int32)


def segment_nuclei(
    image: np.ndarray,
    config: NucleusConfig = NUCLEUS_CONFIG,
) -> Tuple[np.ndarray, np.ndarray, List[dict], dict]:
    """
    반환
    ----
    nucleus_label_map:
        0 background, 1..N nucleus instance.
    nucleus_exclusion_mask:
        nucleus + point_exclusion_margin 영역.
    nuclei:
        nucleus별 metadata.
    info:
        threshold 및 QC 정보.
    """
    (
        initial_binary,
        blue_response,
        otsu_value,
        applied_threshold,
    ) = _initial_nucleus_binary(image, config)

    accepted_mask, candidates = _filter_initial_nucleus_components(
        initial_binary,
        config,
    )

    if not candidates:
        raise ValueError("형태 조건을 만족하는 DAPI nucleus를 찾지 못했습니다.")

    candidate_areas = np.asarray(
        [candidate["area"] for candidate in candidates],
        dtype=np.float64,
    )
    median_area = float(np.median(candidate_areas))

    final_label_map = np.zeros_like(initial_binary, dtype=np.int32)
    nuclei: List[dict] = []
    next_id = 1
    n_split_components = 0

    for candidate in candidates:
        component_mask = candidate["mask"]
        should_split = bool(
            config.split_touching_nuclei
            and median_area > 0
            and candidate["area"]
            > float(config.split_area_factor) * median_area
        )

        if should_split:
            local_labels = _split_large_nucleus_component(
                component_mask,
                config,
            )
            if int(local_labels.max()) > 1:
                n_split_components += 1
        else:
            local_labels = np.zeros_like(final_label_map, dtype=np.int32)
            local_labels[component_mask] = 1

        for local_id in range(1, int(local_labels.max()) + 1):
            submask = local_labels == local_id
            if not np.any(submask):
                continue

            sublabels = label(submask, connectivity=2)
            props = regionprops(sublabels)
            if not props:
                continue
            region = max(props, key=lambda item: item.area)

            # Split fragment도 최종 nucleus 형태 조건을 만족해야 한다.
            if not _region_shape_valid(region, config):
                continue

            final_label_map[submask] = next_id

            metadata = _metadata_from_mask(
                submask,
                object_id=next_id,
                prefix="nucleus",
            )
            metadata.update(
                {
                    "nucleus_solidity": float(region.solidity),
                    "nucleus_eccentricity": float(region.eccentricity),
                    "nucleus_major_axis_length": _region_axis_major_length(region),
                    "nucleus_minor_axis_length": _region_axis_minor_length(region),
                }
            )
            nuclei.append(metadata)
            next_id += 1

    if not nuclei:
        raise ValueError("최종 nucleus instance가 없습니다.")

    nucleus_union = final_label_map > 0

    if config.point_exclusion_margin_px > 0:
        nucleus_exclusion_mask = dilation(
            nucleus_union,
            footprint=disk(int(config.point_exclusion_margin_px)),
        )
    else:
        nucleus_exclusion_mask = nucleus_union.copy()

    info = {
        "n_nuclei": int(len(nuclei)),
        "n_initial_nucleus_candidates": int(len(candidates)),
        "n_split_nucleus_components": int(n_split_components),
        "nucleus_otsu_threshold": float(otsu_value),
        "nucleus_applied_threshold": float(applied_threshold),
        "median_initial_nucleus_area_px": float(median_area),
        "nucleus_union_area_px": int(np.count_nonzero(nucleus_union)),
        "nucleus_exclusion_area_px": int(
            np.count_nonzero(nucleus_exclusion_mask)
        ),
        "blue_response_max": float(np.max(blue_response)),
    }

    return (
        final_label_map,
        nucleus_exclusion_mask.astype(bool),
        nuclei,
        info,
    )


# =============================================================================
# Red+Green coarse cell mask and DAPI-seeded watershed
# =============================================================================

def build_coarse_cell_mask(
    image: np.ndarray,
    nucleus_label_map: np.ndarray,
    config: CellInstanceConfig = CELL_INSTANCE_CONFIG,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    red = image[:, :, 2]
    green = image[:, :, 1]

    red_norm = percentile_normalize(
        red,
        config.low_percentile,
        config.high_percentile,
    )
    green_norm = percentile_normalize(
        green,
        config.low_percentile,
        config.high_percentile,
    )

    combined = np.maximum(red_norm, green_norm)

    if config.blur_sigma > 0:
        combined_smooth = cv2.GaussianBlur(
            combined,
            ksize=(0, 0),
            sigmaX=float(config.blur_sigma),
            sigmaY=float(config.blur_sigma),
        )
    else:
        combined_smooth = combined

    otsu_value = _safe_threshold_otsu(combined_smooth)
    applied_threshold = max(
        float(config.min_normalized_threshold),
        otsu_value * float(config.otsu_scale),
    )

    coarse_mask = combined_smooth >= applied_threshold

    for _ in range(max(0, int(config.close_iterations))):
        if config.close_radius_px > 0:
            coarse_mask = closing(
                coarse_mask,
                footprint=disk(int(config.close_radius_px)),
            )

    for _ in range(max(0, int(config.open_iterations))):
        if config.open_radius_px > 0:
            coarse_mask = opening(
                coarse_mask,
                footprint=disk(int(config.open_radius_px)),
            )

    coarse_mask = ndi.binary_fill_holes(coarse_mask)
    coarse_mask = _remove_objects_smaller_than(
        coarse_mask,
        config.min_coarse_component_area_px,
    )

    nucleus_union = nucleus_label_map > 0
    if config.nucleus_support_margin_px > 0:
        nucleus_support = dilation(
            nucleus_union,
            footprint=disk(int(config.nucleus_support_margin_px)),
        )
    else:
        nucleus_support = nucleus_union

    # nucleus가 cell mask의 dark hole 때문에 제외되지 않도록 강제로 포함한다.
    coarse_mask |= nucleus_support

    coarse_labels = label(coarse_mask, connectivity=2)
    valid_union = np.zeros_like(coarse_mask, dtype=bool)
    n_no_nucleus_components = 0
    n_small_components = 0

    for region in regionprops(coarse_labels):
        component = coarse_labels == region.label

        if int(region.area) < int(config.min_coarse_component_area_px):
            n_small_components += 1
            continue

        if not np.any(nucleus_label_map[component] > 0):
            # nucleus가 없는 debris, scale bar, isolated fiber 제거
            n_no_nucleus_components += 1
            continue

        valid_union |= component

    if not np.any(valid_union):
        raise ValueError("nucleus를 포함하는 coarse cell component가 없습니다.")

    if config.cell_margin_px > 0:
        valid_union = dilation(
            valid_union,
            footprint=disk(int(config.cell_margin_px)),
        )

    # dilation 후에도 nucleus가 반드시 mask 안에 존재하도록 보장
    valid_union |= nucleus_union

    info = {
        "cell_otsu_threshold": float(otsu_value),
        "cell_applied_threshold": float(applied_threshold),
        "n_coarse_components": int(coarse_labels.max()),
        "n_discarded_no_nucleus_components": int(n_no_nucleus_components),
        "n_discarded_small_components": int(n_small_components),
        "coarse_valid_area_px": int(np.count_nonzero(valid_union)),
        "coarse_valid_area_fraction": float(np.mean(valid_union)),
    }

    return valid_union.astype(bool), combined_smooth.astype(np.float32), info


def segment_cell_instances(
    image: np.ndarray,
    nucleus_label_map: np.ndarray,
    nucleus_exclusion_mask: np.ndarray,
    config: CellInstanceConfig = CELL_INSTANCE_CONFIG,
) -> Tuple[np.ndarray, List[dict], dict]:
    """
    DAPI nucleus labels를 marker로 사용해 붙어 있는 세포를 watershed로 분리한다.
    """
    valid_union, combined_smooth, coarse_info = build_coarse_cell_mask(
        image,
        nucleus_label_map,
        config,
    )

    markers = nucleus_label_map.astype(np.int32)
    if int(markers.max()) == 0:
        raise ValueError("watershed marker로 사용할 nucleus가 없습니다.")

    elevation = sobel(combined_smooth)

    raw_watershed = watershed(
        elevation,
        markers=markers,
        mask=valid_union,
        compactness=float(config.watershed_compactness),
        watershed_line=True,
    ).astype(np.int32)

    final_label_map = np.zeros_like(raw_watershed, dtype=np.int32)
    cells: List[dict] = []
    next_cell_id = 1
    n_small_instances = 0

    for nucleus_id in range(1, int(nucleus_label_map.max()) + 1):
        cell_mask = raw_watershed == nucleus_id
        if not np.any(cell_mask):
            continue

        cell_area = int(np.count_nonzero(cell_mask))
        if cell_area < int(config.min_cell_instance_area_px):
            n_small_instances += 1
            continue

        nucleus_mask = nucleus_label_map == nucleus_id
        if not np.any(nucleus_mask):
            continue

        final_label_map[cell_mask] = next_cell_id

        metadata = _metadata_from_mask(
            cell_mask,
            object_id=next_cell_id,
            prefix="cell",
        )
        metadata.update(
            {
                "nucleus_id": int(nucleus_id),
                "nucleus_area_px": int(np.count_nonzero(nucleus_mask)),
            }
        )

        cell_nuclear_exclusion = cell_mask & nucleus_exclusion_mask
        point_eligible_area_px = int(
            np.count_nonzero(cell_mask & ~nucleus_exclusion_mask)
        )
        metadata["point_eligible_area_px"] = point_eligible_area_px

        border = bool(metadata["cell_touches_border"])

        # 모든 유효한 DAPI-seeded cell instance를 분석에 포함한다.
        # border 여부는 QC metadata로만 보존한다.
        metadata["analysis_eligible"] = True

        cells.append(metadata)
        next_cell_id += 1

    if not cells:
        raise ValueError("최종 cell instance가 없습니다.")

    total_cell_area = int(np.count_nonzero(final_label_map))
    analysis_cell_ids = [
        int(cell["cell_id"])
        for cell in cells
        if bool(cell["analysis_eligible"])
    ]

    info = dict(coarse_info)
    info.update(
        {
            "n_cells_all": int(len(cells)),
            "n_cells_complete": int(len(analysis_cell_ids)),
            "n_border_cells": int(
                sum(bool(cell["cell_touches_border"]) for cell in cells)
            ),
            "n_discarded_small_watershed_instances": int(n_small_instances),
            "total_cell_instance_area_px": total_cell_area,
            "total_cell_instance_area_fraction": float(
                total_cell_area / final_label_map.size
            ),
        }
    )

    return final_label_map, cells, info


# =============================================================================
# Green point classification
# =============================================================================

def classify_green_points(
    points: np.ndarray,
    cell_label_map: np.ndarray,
    nucleus_exclusion_mask: np.ndarray,
    cells: Sequence[dict],
) -> Tuple[List[dict], Dict[int, np.ndarray]]:
    """
    Point status
    ------------
    eligible:
        유효한 cell 내부이며 nucleus exclusion 영역 밖.
    border_cell:
        border-touching partial cell 내부이며 nucleus 밖.
    nucleus_excluded:
        cell 내부지만 nucleus 또는 nucleus margin에 위치.
    outside_cell:
        모든 cell instance 밖.
    """
    cell_lookup = {
        int(cell["cell_id"]): cell
        for cell in cells
    }

    point_rows: List[dict] = []
    eligible_by_cell: Dict[int, List[np.ndarray]] = {
        int(cell["cell_id"]): []
        for cell in cells
        if bool(cell["analysis_eligible"])
    }

    height, width = cell_label_map.shape

    for point_id, point in enumerate(points):
        y = int(np.clip(round(float(point[0])), 0, height - 1))
        x = int(np.clip(round(float(point[1])), 0, width - 1))
        response = float(point[2])

        cell_id = int(cell_label_map[y, x])

        if cell_id <= 0:
            status = "outside_cell"
            analysis_eligible = False
        elif bool(nucleus_exclusion_mask[y, x]):
            status = "nucleus_excluded"
            analysis_eligible = False
        else:
            cell = cell_lookup[cell_id]
            if bool(cell["analysis_eligible"]):
                status = "eligible"
                analysis_eligible = True
                eligible_by_cell[cell_id].append(
                    np.asarray([float(y), float(x), response], dtype=np.float32)
                )
            else:
                status = "border_cell"
                analysis_eligible = False

        point_rows.append(
            {
                "point_id": int(point_id),
                "y": float(y),
                "x": float(x),
                "green_response": response,
                "cell_id": int(cell_id) if cell_id > 0 else 0,
                "status": status,
                "analysis_eligible": bool(analysis_eligible),
            }
        )

    eligible_arrays: Dict[int, np.ndarray] = {}
    for cell_id, values in eligible_by_cell.items():
        if values:
            eligible_arrays[cell_id] = np.asarray(
                values,
                dtype=np.float32,
            )
        else:
            eligible_arrays[cell_id] = np.empty(
                (0, 3),
                dtype=np.float32,
            )

    return point_rows, eligible_arrays


def summarize_points_by_cell(
    point_rows: Sequence[dict],
    cells: Sequence[dict],
) -> List[dict]:
    status_counts: Dict[Tuple[int, str], int] = {}

    for row in point_rows:
        cell_id = int(row["cell_id"])
        status = str(row["status"])
        if cell_id > 0:
            status_counts[(cell_id, status)] = (
                status_counts.get((cell_id, status), 0) + 1
            )

    rows: List[dict] = []
    for cell in cells:
        cell_id = int(cell["cell_id"])

        eligible_count = int(
            status_counts.get((cell_id, "eligible"), 0)
        )
        border_count = int(
            status_counts.get((cell_id, "border_cell"), 0)
        )
        nucleus_removed_count = int(
            status_counts.get((cell_id, "nucleus_excluded"), 0)
        )

        nonnuclear_count = eligible_count + border_count
        eligible_area = int(cell["point_eligible_area_px"])

        row = dict(cell)
        row.update(
            {
                "eligible_point_count": eligible_count,
                "border_cell_point_count": border_count,
                "nucleus_excluded_point_count": nucleus_removed_count,
                "nonnuclear_point_count": nonnuclear_count,
                "point_density_per_10000px": (
                    nonnuclear_count / eligible_area * 10000.0
                    if eligible_area > 0
                    else np.nan
                ),
                "primary_point_density_per_10000px": (
                    eligible_count / eligible_area * 10000.0
                    if bool(cell["analysis_eligible"])
                    and eligible_area > 0
                    else np.nan
                ),
            }
        )
        rows.append(row)

    return rows


# =============================================================================
# QC visualization
# =============================================================================

def draw_instance_qc(
    image: np.ndarray,
    nucleus_label_map: np.ndarray,
    cell_label_map: np.ndarray,
    cells: Sequence[dict],
    point_rows: Sequence[dict],
) -> np.ndarray:
    """
    Color legend (BGR)
    ------------------
    cyan contour: 분석에 포함되는 cell
    magenta contour: border cell, primary에서 제외
    white contour: nucleus
    yellow point: primary eligible
    orange point: border cell point
    purple point: nucleus-excluded point
    red point: outside-cell point
    """
    overlay = image.copy()

    # Nucleus contours
    for nucleus_id in range(1, int(nucleus_label_map.max()) + 1):
        mask = (nucleus_label_map == nucleus_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    for cell in cells:
        cell_id = int(cell["cell_id"])
        mask = (cell_label_map == cell_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if bool(cell["analysis_eligible"]):
            color = (255, 255, 0)  # cyan
        else:
            color = (255, 0, 255)  # magenta

        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        x = int(round(float(cell["cell_centroid_x"])))
        y = int(round(float(cell["cell_centroid_y"])))
        cv2.putText(
            overlay,
            f"C{cell_id}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    point_colors = {
        "eligible": (0, 255, 255),          # yellow
        "border_cell": (0, 165, 255),       # orange
        "nucleus_excluded": (255, 0, 255),  # purple/magenta
        "outside_cell": (0, 0, 255),        # red
    }

    for row in point_rows:
        cv2.circle(
            overlay,
            (
                int(round(float(row["x"]))),
                int(round(float(row["y"]))),
            ),
            3,
            point_colors[str(row["status"])],
            -1,
            cv2.LINE_AA,
        )

    return overlay


def dependency_versions() -> dict:
    """실행 환경의 핵심 라이브러리 버전을 반환한다."""
    import scipy
    import skimage

    return {
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "scikit_image": str(skimage.__version__),
    }
