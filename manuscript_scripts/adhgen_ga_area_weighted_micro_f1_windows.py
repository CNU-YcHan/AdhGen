"""
Area-weighted Point-guided AdhGen GA
============================

확정된 단순 objective
---------------------
1. DAPI nucleus-seeded cell segmentation을 수행한다.
2. 모든 cell instance를 하나의 union mask로 합친다.
3. nucleus와 nucleus margin을 제외한다.
4. 오른쪽 아래 scale bar 영역을 제외한다.
5. 남은 영역의 Green local maxima를 fixed reference point로 사용한다.
6. 동일 영역에서 R/G/B/Area chromosome으로 AdhGen 객체를 검출한다.
7. 이미지 전체에서 Hungarian one-to-one matching을 수행한다.
8. 모든 이미지의 matched quality, point 수, object 수를 합산한다.
9. 거리 점수에 실제 객체 면적 penalty를 곱한 dataset-level micro F1을 GA fitness로 사용한다.

Soft/Stiff condition은 fitness에 사용하지 않으며, 최종 결과 확인을 위한
post-hoc summary에만 사용한다.

실행 예
-------
Debug:
    python adhgen_ga_area_weighted_micro_f1_windows.py --mode debug

Full:
    python adhgen_ga_area_weighted_micro_f1_windows.py --mode full

필수 파일
---------
- adhgen_point_guided_agg_fix.py
- cell_instance_utils_dapi_all_cells.py
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent

import argparse
import copy
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, ttest_ind

from adhgen_point_guided_agg_fix import (
    EXCLUDED_FOLDERS,
    GA_CONFIG,
    MATCH_RADIUS_PX,
    POINT_CONFIG,
    GAConfig,
    PointConfig,
    collect_image_paths,
    create_individual,
    extract_condition,
    generate_green_points,
    mate_ind,
    mutate_ind,
    preload_images,
    repair_individual,
    seed_everything,
    tournament_select,
)
import cell_instance_utils_dapi_all_cells as cell_utils
from cell_instance_utils_dapi_all_cells import (
    CELL_INSTANCE_CONFIG,
    NUCLEUS_CONFIG,
    UTILITY_VERSION,
    CellInstanceConfig,
    NucleusConfig,
    dependency_versions,
    segment_cell_instances,
    segment_nuclei,
)


SCRIPT_VERSION = "2026-07-29-area-weighted-micro-f1"

DATA_ROOT = Path(
    str(_ADHGEN_ROOT / "merged_data_405_488_actin")
)
OUTPUT_ROOT = Path(
    str(_SCRIPT_DIR / "adhgen_ga_area_weighted_micro_f1_results")
)

POINT_CONFIG_FINAL = replace(
    POINT_CONFIG,
    background_sigma=15.0,
    smooth_sigma=1.5,
    threshold_percentile=99.5,
    mad_multiplier=5.0,
    min_response=8.0,
    min_distance_px=8,
    exclude_border_px=0,
    max_points_per_image=None,
)

NUCLEUS_CONFIG_FINAL = NUCLEUS_CONFIG
CELL_CONFIG_FINAL = replace(
    CELL_INSTANCE_CONFIG,
    exclude_border_cells_from_primary=False,
)


@dataclass(frozen=True)
class ReferenceMaskConfig:
    """
    오른쪽 아래 scale bar와 text 영역을 고정 제외한다.

    1024x1024 영상에서는 대략:
        x >= 0.78 * width
        y >= height - 70
    """

    exclude_scale_bar: bool = True
    scale_bar_x_start_fraction: float = 0.78
    scale_bar_height_px: int = 70



REFERENCE_MASK_CONFIG = ReferenceMaskConfig()

# 실제 검출 객체 면적이 이 값을 넘으면 matching quality를 점진적으로 감점한다.
# w_A(a) = min(1, AREA_SOFT_LIMIT_PX / a)
AREA_SOFT_LIMIT_PX = 60.0


@dataclass
class ImageReference:
    path: str
    analysis_mask: np.ndarray
    cell_union_mask: np.ndarray
    nucleus_exclusion_mask: np.ndarray
    scale_bar_exclusion_mask: np.ndarray
    points: np.ndarray
    n_nuclei: int
    n_cells: int


# =============================================================================
# Reference generation
# =============================================================================

def make_scale_bar_exclusion_mask(
    image_shape: Sequence[int],
    config: ReferenceMaskConfig,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=bool)

    if not config.exclude_scale_bar:
        return mask

    x_start = int(
        round(
            np.clip(
                config.scale_bar_x_start_fraction,
                0.0,
                1.0,
            )
            * width
        )
    )
    y_start = max(
        0,
        height - int(config.scale_bar_height_px),
    )

    mask[y_start:height, x_start:width] = True
    return mask


def filter_points_by_mask(
    points: np.ndarray,
    analysis_mask: np.ndarray,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    height, width = analysis_mask.shape
    y = np.clip(
        np.rint(points[:, 0]).astype(np.int32),
        0,
        height - 1,
    )
    x = np.clip(
        np.rint(points[:, 1]).astype(np.int32),
        0,
        width - 1,
    )
    keep = analysis_mask[y, x]

    return points[keep].astype(np.float32, copy=False)


def build_image_reference(
    path: str,
    image: np.ndarray,
    point_config: PointConfig,
    nucleus_config: NucleusConfig,
    cell_config: CellInstanceConfig,
    mask_config: ReferenceMaskConfig,
) -> ImageReference:
    raw_points = generate_green_points(
        image,
        point_config,
    )

    (
        nucleus_label_map,
        nucleus_exclusion_mask,
        nuclei,
        _,
    ) = segment_nuclei(
        image,
        nucleus_config,
    )

    (
        cell_label_map,
        cells,
        _,
    ) = segment_cell_instances(
        image,
        nucleus_label_map,
        nucleus_exclusion_mask,
        cell_config,
    )

    # Watershed의 세포별 경계는 GA에서 사용하지 않고 union만 사용한다.
    cell_union_mask = cell_label_map > 0

    scale_bar_exclusion_mask = make_scale_bar_exclusion_mask(
        image.shape,
        mask_config,
    )

    analysis_mask = (
        cell_union_mask
        & ~nucleus_exclusion_mask
        & ~scale_bar_exclusion_mask
    )

    points = filter_points_by_mask(
        raw_points,
        analysis_mask,
    )

    return ImageReference(
        path=path,
        analysis_mask=analysis_mask.astype(bool),
        cell_union_mask=cell_union_mask.astype(bool),
        nucleus_exclusion_mask=nucleus_exclusion_mask.astype(bool),
        scale_bar_exclusion_mask=scale_bar_exclusion_mask.astype(bool),
        points=points,
        n_nuclei=int(len(nuclei)),
        n_cells=int(len(cells)),
    )


def precompute_references(
    images: Mapping[str, np.ndarray],
    point_config: PointConfig,
    nucleus_config: NucleusConfig,
    cell_config: CellInstanceConfig,
    mask_config: ReferenceMaskConfig,
    max_workers: int,
) -> Tuple[Dict[str, ImageReference], pd.DataFrame]:
    references: Dict[str, ImageReference] = {}
    failure_rows: List[dict] = []

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {
            executor.submit(
                build_image_reference,
                path,
                image,
                point_config,
                nucleus_config,
                cell_config,
                mask_config,
            ): path
            for path, image in images.items()
        }

        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            path = futures[future]
            completed += 1

            try:
                references[path] = future.result()
            except Exception as exc:
                failure_rows.append(
                    {
                        "path": path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    "[Reference failed] "
                    f"{Path(path).name}: "
                    f"{type(exc).__name__}: {exc}"
                )

            if completed % 25 == 0 or completed == total:
                print(
                    f"Reference 생성: {completed}/{total}"
                )

    return references, pd.DataFrame(failure_rows)


def reference_summary_dataframe(
    references: Mapping[str, ImageReference],
) -> pd.DataFrame:
    rows = []

    for path, reference in references.items():
        rows.append(
            {
                "path": path,
                "condition": extract_condition(path),
                "n_nuclei": int(reference.n_nuclei),
                "n_cells": int(reference.n_cells),
                "n_reference_points": int(len(reference.points)),
                "cell_union_area_px": int(
                    np.count_nonzero(reference.cell_union_mask)
                ),
                "analysis_area_px": int(
                    np.count_nonzero(reference.analysis_mask)
                ),
                "nucleus_exclusion_area_px": int(
                    np.count_nonzero(
                        reference.nucleus_exclusion_mask
                    )
                ),
                "scale_bar_exclusion_area_px": int(
                    np.count_nonzero(
                        reference.scale_bar_exclusion_mask
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_reference_images(
    references: Mapping[str, ImageReference],
    minimum_image_coverage: float = 0.80,
) -> None:
    if not references:
        raise ValueError("Reference가 하나도 없습니다.")

    counts = np.asarray(
        [
            len(reference.points)
            for reference in references.values()
        ],
        dtype=np.int32,
    )
    coverage = float(np.mean(counts > 0))

    print("\n[Fixed reference summary]")
    print("  images:", len(counts))
    print(
        "  images with >=1 point:",
        int(np.count_nonzero(counts > 0)),
    )
    print("  image coverage:", coverage)
    print("  mean points:", float(np.mean(counts)))
    print("  median points:", float(np.median(counts)))
    print("  minimum points:", int(np.min(counts)))
    print("  maximum points:", int(np.max(counts)))

    if coverage < minimum_image_coverage:
        raise ValueError(
            f"Reference point image coverage가 "
            f"{coverage:.1%}로 낮습니다."
        )


# =============================================================================
# Chromosome object detection and objective
# =============================================================================

def detect_objects_in_union_mask(
    image: np.ndarray,
    individual: Sequence[int],
    reference: ImageReference,
    mask_config: ReferenceMaskConfig,
    ga_config: GAConfig,
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """
    Chromosome의 RGB 및 Area 범위를 만족하는 모든 connected component를
    반환한다.

    객체 수 상한은 사용하지 않는다. 모든 객체는 objective 분모에 포함되어
    과검출일수록 fitness가 자연스럽게 감소한다.

    contour는 모든 객체에 대해 즉시 계산하지 않고, point 주변 matching에
    필요한 객체에 대해서만 lazy evaluation한다.
    """
    (
        r_min,
        r_max,
        g_min,
        g_max,
        b_min,
        b_max,
        area_min,
        area_max,
    ) = repair_individual(
        individual,
        ga_config,
    )

    rgb_mask = (
        (image[:, :, 2] >= r_min)
        & (image[:, :, 2] <= r_max)
        & (image[:, :, 1] >= g_min)
        & (image[:, :, 1] <= g_max)
        & (image[:, :, 0] >= b_min)
        & (image[:, :, 0] <= b_max)
    )

    binary_mask = (
        rgb_mask
        & reference.analysis_mask
    ).astype(np.uint8)

    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    objects: List[dict] = []
    component_to_object = np.zeros(
        n_labels,
        dtype=np.int32,
    )

    for component_label in range(1, n_labels):
        area = int(
            stats[
                component_label,
                cv2.CC_STAT_AREA,
            ]
        )
        if not (area_min <= area <= area_max):
            continue

        object_index = len(objects)
        component_to_object[component_label] = (
            object_index + 1
        )

        x = int(
            stats[
                component_label,
                cv2.CC_STAT_LEFT,
            ]
        )
        y = int(
            stats[
                component_label,
                cv2.CC_STAT_TOP,
            ]
        )
        width = int(
            stats[
                component_label,
                cv2.CC_STAT_WIDTH,
            ]
        )
        height = int(
            stats[
                component_label,
                cv2.CC_STAT_HEIGHT,
            ]
        )
        centroid_x, centroid_y = (
            centroids[component_label]
        )

        objects.append(
            {
                "component_label": int(
                    component_label
                ),
                "area": int(area),
                "contour": None,
                "centroid": (
                    float(centroid_x),
                    float(centroid_y),
                ),
                "width": int(width),
                "height": int(height),
                "aspect_ratio": (
                    float(width / height)
                    if height > 0
                    else 0.0
                ),
                "bbox": (
                    x,
                    y,
                    width,
                    height,
                ),
            }
        )

    # 원래 connected-component label을 object index + 1로 변환한다.
    object_index_map = component_to_object[labels]

    return (
        objects,
        binary_mask,
        object_index_map,
    )


def ensure_object_contour(
    obj: dict,
    object_index: int,
    object_index_map: np.ndarray,
) -> np.ndarray:
    """
    필요한 객체에 대해서만 contour를 계산하고 object dict에 cache한다.
    """
    contour = obj.get("contour")
    if contour is not None:
        return contour

    component_mask = (
        object_index_map == (object_index + 1)
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        contour = max(
            contours,
            key=cv2.contourArea,
        )
    else:
        contour = np.empty(
            (0, 1, 2),
            dtype=np.int32,
        )

    obj["contour"] = contour
    return contour


def calculate_area_weight(
    object_area: float,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
) -> float:
    """
    큰 connected component가 Green point를 우연히 포함해 높은 점수를
    받는 것을 막기 위한 면적 가중치.

    a <= A0: 1
    a >  A0: A0 / a
    """
    if area_soft_limit_px <= 0:
        raise ValueError("area_soft_limit_px는 0보다 커야 합니다.")
    if object_area <= 0:
        return 0.0
    return float(min(1.0, area_soft_limit_px / float(object_area)))


def match_points_to_objects_local(
    points: np.ndarray,
    objects: Sequence[dict],
    object_index_map: np.ndarray,
    match_radius_px: float,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
) -> dict:
    """
    Point 주변 match radius 안에 실제 object pixel이 존재하는 객체만
    Hungarian matrix에 포함한다.

    중요:
    - Matching 후보만 줄일 뿐, 전체 predicted object 수는 score 분모에
      그대로 포함한다.
    - 따라서 객체 수 상한 없이도 과검출이 자연스럽게 감점된다.
    """
    n_points = int(len(points))
    n_objects = int(len(objects))

    if n_points == 0:
        return {
            "score": None,
            "matches": [],
            "matched_quality_sum": 0.0,
            "matched_distance_quality_sum": 0.0,
            "mean_matched_area_weight": float("nan"),
            "mean_matched_object_area_px": float("nan"),
            "area_soft_limit_px": float(area_soft_limit_px),
            "unmatched_point_indices": [],
            "unmatched_object_indices": list(
                range(n_objects)
            ),
            "n_points": 0,
            "n_objects": n_objects,
        }

    if n_objects == 0:
        return {
            "score": 0.0,
            "matches": [],
            "matched_quality_sum": 0.0,
            "matched_distance_quality_sum": 0.0,
            "mean_matched_area_weight": float("nan"),
            "mean_matched_object_area_px": float("nan"),
            "area_soft_limit_px": float(area_soft_limit_px),
            "unmatched_point_indices": list(
                range(n_points)
            ),
            "unmatched_object_indices": [],
            "n_points": n_points,
            "n_objects": 0,
        }

    if match_radius_px <= 0:
        raise ValueError(
            "match_radius_px는 0보다 커야 합니다."
        )
    if area_soft_limit_px <= 0:
        raise ValueError(
            "area_soft_limit_px는 0보다 커야 합니다."
        )

    height, width = object_index_map.shape
    radius = int(np.ceil(match_radius_px))

    # (point index, object index) ->
    # (area-weighted quality, distance, distance quality, area weight)
    edge_values: Dict[
        Tuple[int, int],
        Tuple[float, float, float, float],
    ] = {}
    candidate_object_indices = set()

    for point_index, point in enumerate(points):
        y = float(point[0])
        x = float(point[1])

        cy = int(round(y))
        cx = int(round(x))

        y0 = max(0, cy - radius)
        y1 = min(height, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(width, cx + radius + 1)

        local_labels = np.unique(
            object_index_map[
                y0:y1,
                x0:x1,
            ]
        )
        local_labels = local_labels[
            local_labels > 0
        ]

        for map_label in local_labels:
            object_index = int(map_label - 1)
            obj = objects[object_index]

            contour = ensure_object_contour(
                obj,
                object_index,
                object_index_map,
            )
            if contour.size == 0:
                continue

            signed_distance = cv2.pointPolygonTest(
                contour,
                (x, y),
                True,
            )
            distance = (
                0.0
                if signed_distance >= 0
                else float(-signed_distance)
            )

            if distance > match_radius_px:
                continue

            distance_quality = max(
                0.0,
                1.0
                - distance / match_radius_px,
            )
            if distance_quality <= 0:
                continue

            area_weight = calculate_area_weight(
                object_area=float(obj["area"]),
                area_soft_limit_px=area_soft_limit_px,
            )
            quality = distance_quality * area_weight
            if quality <= 0:
                continue

            edge_values[
                (point_index, object_index)
            ] = (
                float(quality),
                float(distance),
                float(distance_quality),
                float(area_weight),
            )
            candidate_object_indices.add(
                object_index
            )

    if not candidate_object_indices:
        return {
            "score": 0.0,
            "matches": [],
            "matched_quality_sum": 0.0,
            "matched_distance_quality_sum": 0.0,
            "mean_matched_area_weight": float("nan"),
            "mean_matched_object_area_px": float("nan"),
            "area_soft_limit_px": float(area_soft_limit_px),
            "unmatched_point_indices": list(
                range(n_points)
            ),
            "unmatched_object_indices": list(
                range(n_objects)
            ),
            "n_points": n_points,
            "n_objects": n_objects,
        }

    candidate_objects = sorted(
        candidate_object_indices
    )
    candidate_column = {
        object_index: column_index
        for column_index, object_index
        in enumerate(candidate_objects)
    }

    quality_matrix = np.zeros(
        (
            n_points,
            len(candidate_objects),
        ),
        dtype=np.float64,
    )
    distance_matrix = np.full(
        quality_matrix.shape,
        np.inf,
        dtype=np.float64,
    )
    distance_quality_matrix = np.zeros(
        quality_matrix.shape,
        dtype=np.float64,
    )
    area_weight_matrix = np.zeros(
        quality_matrix.shape,
        dtype=np.float64,
    )

    for (
        point_index,
        object_index,
    ), (
        quality,
        distance,
        distance_quality,
        area_weight,
    ) in edge_values.items():
        column_index = candidate_column[
            object_index
        ]
        quality_matrix[
            point_index,
            column_index,
        ] = quality
        distance_matrix[
            point_index,
            column_index,
        ] = distance
        distance_quality_matrix[
            point_index,
            column_index,
        ] = distance_quality
        area_weight_matrix[
            point_index,
            column_index,
        ] = area_weight

    row_indices, column_indices = (
        linear_sum_assignment(
            -quality_matrix
        )
    )

    matches: List[dict] = []
    matched_points = set()
    matched_objects = set()
    quality_sum = 0.0
    distance_quality_sum = 0.0
    matched_area_weight_sum = 0.0
    matched_object_area_sum = 0.0

    for point_index, candidate_column_index in zip(
        row_indices,
        column_indices,
    ):
        quality = float(
            quality_matrix[
                point_index,
                candidate_column_index,
            ]
        )
        if quality <= 0:
            continue

        object_index = candidate_objects[
            candidate_column_index
        ]
        distance = float(
            distance_matrix[
                point_index,
                candidate_column_index,
            ]
        )
        distance_quality = float(
            distance_quality_matrix[
                point_index,
                candidate_column_index,
            ]
        )
        area_weight = float(
            area_weight_matrix[
                point_index,
                candidate_column_index,
            ]
        )
        object_area = float(objects[object_index]["area"])

        matches.append(
            {
                "point_index": int(
                    point_index
                ),
                "object_index": int(
                    object_index
                ),
                "distance": distance,
                "distance_quality": distance_quality,
                "object_area_px": object_area,
                "area_weight": area_weight,
                "quality": quality,
            }
        )
        matched_points.add(
            int(point_index)
        )
        matched_objects.add(
            int(object_index)
        )
        quality_sum += quality
        distance_quality_sum += distance_quality
        matched_area_weight_sum += area_weight
        matched_object_area_sum += object_area

    n_matches = len(matches)
    image_score = (
        2.0
        * quality_sum
        / (n_points + n_objects)
    )

    return {
        "score": float(image_score),
        "matches": matches,
        "matched_quality_sum": float(
            quality_sum
        ),
        "matched_distance_quality_sum": float(
            distance_quality_sum
        ),
        "mean_matched_area_weight": (
            float(matched_area_weight_sum / n_matches)
            if n_matches > 0
            else float("nan")
        ),
        "mean_matched_object_area_px": (
            float(matched_object_area_sum / n_matches)
            if n_matches > 0
            else float("nan")
        ),
        "area_soft_limit_px": float(area_soft_limit_px),
        "unmatched_point_indices": sorted(
            set(range(n_points))
            - matched_points
        ),
        "unmatched_object_indices": sorted(
            set(range(n_objects))
            - matched_objects
        ),
        "n_points": n_points,
        "n_objects": n_objects,
    }


def score_single_reference(
    image: np.ndarray,
    individual: Sequence[int],
    reference: ImageReference,
    match_radius_px: float,
    mask_config: ReferenceMaskConfig,
    ga_config: GAConfig,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
) -> Tuple[float, dict, List[dict], np.ndarray]:
    (
        objects,
        _,
        object_index_map,
    ) = detect_objects_in_union_mask(
        image,
        individual,
        reference,
        mask_config,
        ga_config,
    )

    match_result = (
        match_points_to_objects_local(
            reference.points,
            objects,
            object_index_map,
            match_radius_px=match_radius_px,
            area_soft_limit_px=area_soft_limit_px,
        )
    )

    score = match_result["score"]
    score_value = (
        float("nan")
        if score is None
        or not np.isfinite(score)
        else float(score)
    )

    return (
        score_value,
        match_result,
        objects,
        object_index_map,
    )


def evaluate_individual(
    individual: Sequence[int],
    images: Mapping[str, np.ndarray],
    references: Mapping[str, ImageReference],
    match_radius_px: float,
    mask_config: ReferenceMaskConfig,
    ga_config: GAConfig,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
) -> float:
    """
    Dataset-level distance-and-area-weighted micro F1.

    F = 2 * sum_i(Q_i) / [sum_i(P_i) + sum_i(C_i)]

    Q_i:
        image i의 one-to-one matched quality 합
    P_i:
        fixed Green reference point 수
    C_i:
        chromosome으로 검출된 전체 object 수

    어느 이미지에서든 불필요한 객체가 증가하면 전체 분모가 직접 증가한다.
    큰 객체에는 w_A(a)=min(1, A0/a)를 적용하여 우연한 point 포함을 감점한다.
    """
    repaired = repair_individual(
        individual,
        ga_config,
    )

    total_quality_sum = 0.0
    total_points = 0
    total_objects = 0

    def worker(
        path: str,
    ) -> Tuple[float, int, int]:
        (
            _,
            match_result,
            _,
            _,
        ) = score_single_reference(
            images[path],
            repaired,
            references[path],
            match_radius_px,
            mask_config,
            ga_config,
            area_soft_limit_px,
        )

        return (
            float(
                match_result[
                    "matched_quality_sum"
                ]
            ),
            int(
                match_result["n_points"]
            ),
            int(
                match_result["n_objects"]
            ),
        )

    cv2.setNumThreads(1)

    with ThreadPoolExecutor(
        max_workers=ga_config.max_workers
    ) as executor:
        futures = [
            executor.submit(
                worker,
                path,
            )
            for path in references
        ]

        for future in as_completed(
            futures
        ):
            (
                quality_sum,
                n_points,
                n_objects,
            ) = future.result()

            total_quality_sum += quality_sum
            total_points += n_points
            total_objects += n_objects

    denominator = (
        total_points + total_objects
    )
    if denominator <= 0:
        return 0.0

    return float(
        2.0
        * total_quality_sum
        / denominator
    )


# =============================================================================
# Genetic algorithm
# =============================================================================

def run_ga(
    images: Mapping[str, np.ndarray],
    references: Mapping[str, ImageReference],
    match_radius_px: float,
    mask_config: ReferenceMaskConfig,
    ga_config: GAConfig,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
) -> Tuple[List[int], float, pd.DataFrame]:
    seed_everything(ga_config.seed)

    population = [
        create_individual(ga_config)
        for _ in range(ga_config.population_size)
    ]

    fitness_cache: Dict[Tuple[int, ...], float] = {}
    history_rows: List[dict] = []

    global_best: Optional[List[int]] = None
    global_best_fitness = -math.inf

    def cached_fitness(
        individual: Sequence[int],
    ) -> float:
        repaired = repair_individual(
            individual,
            ga_config,
        )
        key = tuple(repaired)

        if key not in fitness_cache:
            fitness_cache[key] = evaluate_individual(
                repaired,
                images,
                references,
                match_radius_px,
                mask_config,
                ga_config,
                area_soft_limit_px,
            )

        return fitness_cache[key]

    for generation in range(
        ga_config.generations
    ):
        fitnesses = [
            cached_fitness(individual)
            for individual in population
        ]

        best_index = int(np.argmax(fitnesses))
        generation_best = repair_individual(
            population[best_index],
            ga_config,
        )
        generation_best_fitness = float(
            fitnesses[best_index]
        )

        if (
            generation_best_fitness
            > global_best_fitness
        ):
            global_best = copy.deepcopy(
                generation_best
            )
            global_best_fitness = (
                generation_best_fitness
            )

        history_rows.append(
            {
                "generation": int(generation + 1),
                "fitness_mean": float(
                    np.mean(fitnesses)
                ),
                "fitness_std": float(
                    np.std(fitnesses)
                ),
                "fitness_min": float(
                    np.min(fitnesses)
                ),
                "fitness_max": float(
                    generation_best_fitness
                ),
                "unique_evaluations": int(
                    len(fitness_cache)
                ),
                "best_individual": json.dumps(
                    generation_best
                ),
            }
        )

        print(
            f"Generation "
            f"{generation + 1:03d}/"
            f"{ga_config.generations}: "
            f"mean={np.mean(fitnesses):.6f}, "
            f"max={generation_best_fitness:.6f}, "
            f"best={generation_best}"
        )

        if (
            generation
            == ga_config.generations - 1
        ):
            break

        elite_indices = np.argsort(
            fitnesses
        )[-ga_config.elite_size:][::-1]

        next_population = [
            copy.deepcopy(population[index])
            for index in elite_indices
        ]

        while (
            len(next_population)
            < ga_config.population_size
        ):
            parent_1 = tournament_select(
                population,
                fitnesses,
                ga_config.tournament_size,
            )
            parent_2 = tournament_select(
                population,
                fitnesses,
                ga_config.tournament_size,
            )

            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)

            if (
                np.random.random()
                < ga_config.crossover_probability
            ):
                child_1, child_2 = mate_ind(
                    parent_1,
                    parent_2,
                    ga_config,
                )

            if (
                np.random.random()
                < ga_config.mutation_probability
            ):
                child_1 = mutate_ind(
                    child_1,
                    ga_config,
                )

            if (
                np.random.random()
                < ga_config.mutation_probability
            ):
                child_2 = mutate_ind(
                    child_2,
                    ga_config,
                )

            next_population.append(
                repair_individual(
                    child_1,
                    ga_config,
                )
            )

            if (
                len(next_population)
                < ga_config.population_size
            ):
                next_population.append(
                    repair_individual(
                        child_2,
                        ga_config,
                    )
                )

        population = next_population

    if global_best is None:
        raise RuntimeError(
            "유효한 best chromosome이 없습니다."
        )

    return (
        global_best,
        float(global_best_fitness),
        pd.DataFrame(history_rows),
    )


# =============================================================================
# Best-solution output
# =============================================================================

def evaluate_best_solution(
    images: Mapping[str, np.ndarray],
    references: Mapping[str, ImageReference],
    individual: Sequence[int],
    match_radius_px: float,
    mask_config: ReferenceMaskConfig,
    ga_config: GAConfig,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
) -> pd.DataFrame:
    rows: List[dict] = []

    for path in sorted(references):
        (
            score,
            match_result,
            objects,
            object_index_map,
        ) = score_single_reference(
            images[path],
            individual,
            references[path],
            match_radius_px,
            mask_config,
            ga_config,
            area_soft_limit_px,
        )

        n_points = int(
            len(references[path].points)
        )
        n_objects = int(len(objects))
        n_matched = int(
            len(match_result["matches"])
        )

        rows.append(
            {
                "path": path,
                "condition": extract_condition(path),
                "image_score": score,
                "n_reference_points": n_points,
                "n_detected_objects": n_objects,
                "n_matched": n_matched,
                "matched_quality_sum": float(
                    match_result[
                        "matched_quality_sum"
                    ]
                ),
                "matched_distance_quality_sum": float(
                    match_result[
                        "matched_distance_quality_sum"
                    ]
                ),
                "mean_matched_area_weight": float(
                    match_result[
                        "mean_matched_area_weight"
                    ]
                ),
                "mean_matched_object_area_px": float(
                    match_result[
                        "mean_matched_object_area_px"
                    ]
                ),
                "mean_detected_object_area_px": (
                    float(np.mean([obj["area"] for obj in objects]))
                    if objects
                    else np.nan
                ),
                "max_detected_object_area_px": (
                    int(max(obj["area"] for obj in objects))
                    if objects
                    else 0
                ),
                "n_objects_above_area_soft_limit": int(
                    sum(
                        obj["area"] > area_soft_limit_px
                        for obj in objects
                    )
                ),
                "area_soft_limit_px": float(area_soft_limit_px),
                "match_rate_points": (
                    n_matched / n_points
                    if n_points > 0
                    else np.nan
                ),
                "match_rate_objects": (
                    n_matched / n_objects
                    if n_objects > 0
                    else np.nan
                ),
                "n_nuclei": int(
                    references[path].n_nuclei
                ),
                "n_cells": int(
                    references[path].n_cells
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_best_solution(
    result_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        result_df.groupby(
            "condition",
            observed=True,
        )
        .agg(
            n_images=("path", "count"),
            mean_score=("image_score", "mean"),
            median_score=("image_score", "median"),
            mean_reference_points=(
                "n_reference_points",
                "mean",
            ),
            mean_detected_objects=(
                "n_detected_objects",
                "mean",
            ),
            median_detected_objects=(
                "n_detected_objects",
                "median",
            ),
            mean_matched=("n_matched", "mean"),
        )
        .reset_index()
    )

    soft = (
        result_df.loc[
            result_df["condition"] == "Soft",
            "n_detected_objects",
        ]
        .dropna()
        .to_numpy(dtype=float)
    )
    stiff = (
        result_df.loc[
            result_df["condition"] == "Stiff",
            "n_detected_objects",
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    statistics_rows: List[dict] = []

    if len(soft) > 0 and len(stiff) > 0:
        mw = mannwhitneyu(
            stiff,
            soft,
            alternative="two-sided",
            method="auto",
        )
        welch = ttest_ind(
            stiff,
            soft,
            equal_var=False,
            nan_policy="omit",
        )

        statistics_rows.append(
            {
                "metric": "n_detected_objects",
                "comparison": "Stiff - Soft",
                "n_soft": int(len(soft)),
                "n_stiff": int(len(stiff)),
                "soft_mean": float(np.mean(soft)),
                "stiff_mean": float(np.mean(stiff)),
                "soft_median": float(
                    np.median(soft)
                ),
                "stiff_median": float(
                    np.median(stiff)
                ),
                "mean_difference": float(
                    np.mean(stiff)
                    - np.mean(soft)
                ),
                "mann_whitney_u": float(
                    mw.statistic
                ),
                "mann_whitney_p": float(
                    mw.pvalue
                ),
                "cliffs_delta_stiff_vs_soft": float(
                    2.0
                    * float(mw.statistic)
                    / (len(stiff) * len(soft))
                    - 1.0
                ),
                "welch_t": float(
                    welch.statistic
                ),
                "welch_p": float(
                    welch.pvalue
                ),
            }
        )

    return summary, pd.DataFrame(
        statistics_rows
    )


def plot_ga_history(
    history_df: pd.DataFrame,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        history_df["generation"],
        history_df["fitness_mean"],
        label="Population mean",
    )
    ax.plot(
        history_df["generation"],
        history_df["fitness_max"],
        label="Generation best",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel(
        "Image-level point-guided fitness"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def choose_preview_paths(
    result_df: pd.DataFrame,
    n_per_condition: int = 4,
) -> List[str]:
    selected: List[str] = []

    for condition in ("Soft", "Stiff"):
        subset = (
            result_df[
                result_df["condition"]
                == condition
            ]
            .dropna(subset=["image_score"])
            .sort_values("image_score")
            .reset_index(drop=True)
        )

        if subset.empty:
            continue

        positions = np.linspace(
            0,
            len(subset) - 1,
            num=min(n_per_condition, len(subset)),
        ).round().astype(int)

        for position in np.unique(positions):
            selected.append(
                str(
                    subset.iloc[
                        int(position)
                    ]["path"]
                )
            )

    return selected


def save_best_previews(
    images: Mapping[str, np.ndarray],
    references: Mapping[str, ImageReference],
    result_df: pd.DataFrame,
    individual: Sequence[int],
    match_radius_px: float,
    mask_config: ReferenceMaskConfig,
    ga_config: GAConfig,
    output_dir: Path,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_lookup = result_df.set_index(
        "path"
    ).to_dict(orient="index")

    for path in choose_preview_paths(
        result_df
    ):
        reference = references[path]
        image = images[path]

        (
            score,
            match_result,
            objects,
            object_index_map,
        ) = score_single_reference(
            image,
            individual,
            reference,
            match_radius_px,
            mask_config,
            ga_config,
            area_soft_limit_px,
        )

        overlay = image.copy()

        # Analysis union boundary
        union_mask = (
            reference.analysis_mask.astype(
                np.uint8
            )
            * 255
        )
        contours, _ = cv2.findContours(
            union_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Predicted objects
        for object_index, obj in enumerate(
            objects
        ):
            contour = ensure_object_contour(
                obj,
                object_index,
                object_index_map,
            )
            if contour.size == 0:
                continue

            cv2.drawContours(
                overlay,
                [contour],
                -1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Reference points
        for y, x, _ in reference.points:
            cv2.circle(
                overlay,
                (
                    int(round(x)),
                    int(round(y)),
                ),
                3,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )

        # Matching lines
        for match in match_result["matches"]:
            point = reference.points[
                match["point_index"]
            ]
            obj = objects[
                match["object_index"]
            ]
            cx, cy = obj["centroid"]

            cv2.line(
                overlay,
                (
                    int(round(point[1])),
                    int(round(point[0])),
                ),
                (
                    int(round(cx)),
                    int(round(cy)),
                ),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Scale-bar exclusion rectangle
        ys, xs = np.nonzero(
            reference.scale_bar_exclusion_mask
        )
        if len(xs) > 0:
            cv2.rectangle(
                overlay,
                (
                    int(np.min(xs)),
                    int(np.min(ys)),
                ),
                (
                    int(np.max(xs)),
                    int(np.max(ys)),
                ),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        info = result_lookup[path]
        text = (
            f"{info['condition']} | "
            f"score={score:.3f} | "
            f"points={int(info['n_reference_points'])} | "
            f"objects={int(info['n_detected_objects'])} | "
            f"matched={int(info['n_matched'])}"
        )
        cv2.putText(
            overlay,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        filename = (
            f"{info['condition']}_"
            f"score-{score:.3f}_"
            f"{Path(path).stem}.png"
        )
        cv2.imwrite(
            str(output_dir / filename),
            overlay,
        )


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final image-level point-guided AdhGen GA"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("debug", "full"),
        default="debug",
        help=(
            "debug: population 10, generation 5; "
            "full: population 50, generation 50"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=MATCH_RADIUS_PX,
    )
    parser.add_argument(
        "--area-soft-limit-px",
        type=float,
        default=AREA_SOFT_LIMIT_PX,
    )
    return parser.parse_args()


def main(
    mode: str = "debug",
    seed: int = 42,
    workers: int = 4,
    match_radius_px: float = MATCH_RADIUS_PX,
    area_soft_limit_px: float = AREA_SOFT_LIMIT_PX,
    data_root: str | Path = DATA_ROOT,
    output_root: str | Path = OUTPUT_ROOT,
    point_config: PointConfig = POINT_CONFIG_FINAL,
    nucleus_config: NucleusConfig = NUCLEUS_CONFIG_FINAL,
    cell_config: CellInstanceConfig = CELL_CONFIG_FINAL,
    mask_config: ReferenceMaskConfig = REFERENCE_MASK_CONFIG,
    excluded_folders: Iterable[str] = EXCLUDED_FOLDERS,
) -> Tuple[List[int], float, pd.DataFrame]:
    if match_radius_px <= 0:
        raise ValueError("match_radius_px는 0보다 커야 합니다.")
    if area_soft_limit_px <= 0:
        raise ValueError("area_soft_limit_px는 0보다 커야 합니다.")

    if mode == "debug":
        ga_config = replace(
            GA_CONFIG,
            population_size=10,
            generations=5,
            seed=int(seed),
            max_workers=int(workers),
        )
    elif mode == "full":
        ga_config = replace(
            GA_CONFIG,
            population_size=50,
            generations=50,
            seed=int(seed),
            max_workers=int(workers),
        )
    else:
        raise ValueError(
            "mode는 debug 또는 full이어야 합니다."
        )

    output_dir = (
        Path(output_root)
        / f"{mode}_seed_{seed}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[Runtime identity]")
    print("  script_version:", SCRIPT_VERSION)
    print("  script_file:", Path(__file__).resolve())
    print("  utility_version:", UTILITY_VERSION)
    print(
        "  utility_file:",
        Path(cell_utils.__file__).resolve(),
    )

    print("\n[Paths]")
    print("  data_root:", Path(data_root))
    print("  output_dir:", output_dir)

    print("\n[GA configuration]")
    for key, value in asdict(ga_config).items():
        print(f"  {key}: {value}")

    print("\n[Reference mask configuration]")
    for key, value in asdict(mask_config).items():
        print(f"  {key}: {value}")

    print("\n[Matching configuration]")
    print("  green_point_min_distance_px:", point_config.min_distance_px)
    print("  match_radius_px:", match_radius_px)
    print("  area_soft_limit_px:", area_soft_limit_px)
    print("  area_weight: min(1, A0 / detected_object_area)")

    print("\n[Dependency versions]")
    for name, version in dependency_versions().items():
        print(f"  {name}: {version}")

    seed_everything(ga_config.seed)

    image_paths = collect_image_paths(
        data_root=data_root,
        excluded_folders=excluded_folders,
    )
    if not image_paths:
        raise FileNotFoundError(
            f"이미지를 찾지 못했습니다: "
            f"{data_root}"
        )

    print(
        f"\n수집된 이미지: {len(image_paths)}"
    )
    images = preload_images(image_paths)

    if not images:
        raise RuntimeError(
            "불러온 이미지가 없습니다."
        )

    first_path = sorted(images)[0]
    try:
        preflight = build_image_reference(
            first_path,
            images[first_path],
            point_config,
            nucleus_config,
            cell_config,
            mask_config,
        )
        print(
            "[Preflight OK] "
            f"{Path(first_path).name}: "
            f"nuclei={preflight.n_nuclei}, "
            f"cells={preflight.n_cells}, "
            f"points={len(preflight.points)}"
        )
    except Exception as exc:
        raise RuntimeError(
            "Preflight reference 생성 실패: "
            f"{Path(first_path).name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    print("\nFixed reference 생성 중...")
    references, failure_df = (
        precompute_references(
            images,
            point_config,
            nucleus_config,
            cell_config,
            mask_config,
            ga_config.max_workers,
        )
    )

    failure_df.to_csv(
        output_dir / "reference_failures.csv",
        index=False,
    )

    validate_reference_images(references)

    reference_df = (
        reference_summary_dataframe(
            references
        )
    )
    reference_df.to_csv(
        output_dir / "fixed_reference_summary.csv",
        index=False,
    )

    print("\nGA 실행 시작...")
    (
        best_individual,
        best_fitness,
        history_df,
    ) = run_ga(
        images,
        references,
        match_radius_px,
        mask_config,
        ga_config,
        area_soft_limit_px,
    )

    history_df.to_csv(
        output_dir / "ga_history.csv",
        index=False,
    )
    plot_ga_history(
        history_df,
        output_dir / "ga_history.png",
    )

    result_df = evaluate_best_solution(
        images,
        references,
        best_individual,
        match_radius_px,
        mask_config,
        ga_config,
        area_soft_limit_px,
    )
    result_df.to_csv(
        output_dir / "best_solution_by_image.csv",
        index=False,
    )

    global_quality_sum = float(
        result_df["matched_quality_sum"].sum()
    )
    global_reference_points = int(
        result_df["n_reference_points"].sum()
    )
    global_detected_objects = int(
        result_df["n_detected_objects"].sum()
    )
    global_micro_f1 = (
        2.0 * global_quality_sum
        / (
            global_reference_points
            + global_detected_objects
        )
        if (
            global_reference_points
            + global_detected_objects
        ) > 0
        else 0.0
    )

    global_df = pd.DataFrame(
        [
            {
                "matched_quality_sum": global_quality_sum,
                "n_reference_points": global_reference_points,
                "n_detected_objects": global_detected_objects,
                "dataset_micro_f1": global_micro_f1,
            }
        ]
    )
    global_df.to_csv(
        output_dir / "best_solution_global_micro_f1.csv",
        index=False,
    )

    summary_df, statistics_df = (
        summarize_best_solution(result_df)
    )
    summary_df.to_csv(
        output_dir
        / "best_solution_condition_summary.csv",
        index=False,
    )
    statistics_df.to_csv(
        output_dir
        / "best_solution_posthoc_statistics.csv",
        index=False,
    )

    save_best_previews(
        images,
        references,
        result_df,
        best_individual,
        match_radius_px,
        mask_config,
        ga_config,
        output_dir / "best_solution_previews",
        area_soft_limit_px,
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "mode": mode,
        "best_individual": [
            int(value)
            for value in best_individual
        ],
        "best_fitness": float(best_fitness),
        "objective": (
            "dataset_level_distance_and_area_weighted_micro_f1"
        ),
        "green_point_min_distance_px": int(point_config.min_distance_px),
        "area_soft_limit_px": float(area_soft_limit_px),
        "area_weight_formula": "min(1, A0 / detected_object_area)",
        "match_radius_px": float(
            match_radius_px
        ),
        "n_images_loaded": int(len(images)),
        "n_reference_images": int(
            len(references)
        ),
        "point_config": asdict(
            point_config
        ),
        "nucleus_config": asdict(
            nucleus_config
        ),
        "cell_config": asdict(
            cell_config
        ),
        "mask_config": asdict(
            mask_config
        ),
        "ga_config": asdict(
            ga_config
        ),
    }

    with open(
        output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[GA completed]")
    print(
        "  best_individual:",
        best_individual,
    )
    print(
        "  best_fitness:",
        best_fitness,
    )
    print(
        "  result_dir:",
        output_dir.resolve(),
    )

    print("\n[Dataset-level objective components]")
    print(
        "  matched_quality_sum:",
        global_quality_sum,
    )
    print(
        "  reference_points:",
        global_reference_points,
    )
    print(
        "  detected_objects:",
        global_detected_objects,
    )
    print(
        "  dataset_micro_f1:",
        global_micro_f1,
    )

    print("\n[Object-count distribution]")
    print(
        result_df["n_detected_objects"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print("\n[Post-hoc condition summary]")
    print(summary_df.to_string(index=False))

    if not statistics_df.empty:
        print(
            "\n[Post-hoc object-count statistics]"
        )
        print(
            statistics_df.to_string(index=False)
        )

    return (
        best_individual,
        best_fitness,
        history_df,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        mode=args.mode,
        seed=args.seed,
        workers=args.workers,
        match_radius_px=args.match_radius,
        area_soft_limit_px=args.area_soft_limit_px,
    )
