"""
Point-guided AdhGen
===================

Green 채널에서 자동으로 FA reference point를 생성하고,
R/G/B/Area chromosome으로 검출한 connected component와 point를
일대일 matching하여 거리 가중 F1 score를 최대화한다.

필수 패키지:
    numpy, pandas, opencv-python, scipy, matplotlib

데이터 기본 구조:
    ./data/*/Soft/*_M.tif
    ./data/*/Stiff/*_M.tif
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass(frozen=True)
class PointConfig:
    """Green reference point 생성 설정."""

    # Green 채널의 느린 배경 성분을 제거하기 위한 Gaussian sigma
    background_sigma: float = 10.0

    # background subtraction 이후 local maximum 안정화를 위한 sigma
    smooth_sigma: float = 1.0

    # 이미지별 robust threshold의 percentile 항
    threshold_percentile: float = 97.5

    # threshold = max(percentile, median + mad_multiplier * robust_sigma)
    mad_multiplier: float = 3.0

    # response가 지나치게 낮을 때 point가 생성되는 것을 막는 절대 하한
    min_response: float = 4.0

    # point 사이 최소 거리. 이미지 해상도가 다르면 µm 기준으로 환산할 것
    min_distance_px: int = 8

    # 영상 가장자리에서 제외할 폭
    exclude_border_px: int = 0

    # 비정상적으로 많은 point 발생 시의 안전장치. None이면 제한하지 않음
    # 고정된 top-K를 강제하는 용도가 아니라 오류 방지용이다.
    max_points_per_image: Optional[int] = 300


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 50
    generations: int = 50
    crossover_probability: float = 0.8
    mutation_probability: float = 0.5
    elite_size: int = 1
    tournament_size: int = 3
    seed: int = 42
    max_workers: int = 4

    rgb_low: int = 0
    rgb_high: int = 255
    area_low: int = 1
    area_high: int = 100

    # 각 min-max 쌍의 최소 폭
    min_rgb_span: int = 1
    min_area_span: int = 1


# 기본 설정: 기존 notebook과 동일하게 9, 12 폴더 제외
EXCLUDED_FOLDERS = {"9", "12"}
POINT_CONFIG = PointConfig()
GA_CONFIG = GAConfig()
MATCH_RADIUS_PX = 4.0
MIN_POINT_COVERAGE = 0.80
OUTPUT_DIR = Path("./point_guided_results")


# =============================================================================
# 2. Reproducibility and image I/O
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_image_paths(
    data_root: str | Path = "./data",
    excluded_folders: Iterable[str] = EXCLUDED_FOLDERS,
) -> List[str]:
    """Soft/Stiff 이미지를 수집하되 지정한 폴더는 제외한다."""

    data_root = Path(data_root)
    excluded = {str(x) for x in excluded_folders}

    paths = list(data_root.glob("*/Soft/*_M.tif"))
    paths += list(data_root.glob("*/Stiff/*_M.tif"))

    selected = [p for p in paths if not excluded.intersection(p.parts)]
    return sorted(str(p) for p in selected)


def preload_images(image_paths: Sequence[str]) -> Dict[str, np.ndarray]:
    """이미지를 BGR uint8 형식으로 메모리에 미리 로드한다."""

    images: Dict[str, np.ndarray] = {}
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARNING] 이미지를 읽지 못했습니다: {path}")
            continue
        images[path] = image
    return images


def extract_condition(path: str) -> str:
    parts = {part.lower() for part in Path(path).parts}
    if "soft" in parts:
        return "Soft"
    if "stiff" in parts:
        return "Stiff"
    return "Unknown"


# =============================================================================
# 3. Automatic Green reference points
# =============================================================================

def _robust_green_response(
    image: np.ndarray,
    config: PointConfig,
) -> Tuple[np.ndarray, float]:
    """
    Green 채널에서 느린 배경을 제거하고 robust threshold를 계산한다.

    반환:
        response: background-subtracted and smoothed response
        threshold: point 후보가 되기 위한 intensity threshold
    """

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("BGR 3-channel image가 필요합니다.")

    green = image[:, :, 1].astype(np.float32)

    background = cv2.GaussianBlur(
        green,
        ksize=(0, 0),
        sigmaX=max(config.background_sigma, 0.1),
        sigmaY=max(config.background_sigma, 0.1),
    )
    response = np.clip(green - background, 0.0, None)

    if config.smooth_sigma > 0:
        response = cv2.GaussianBlur(
            response,
            ksize=(0, 0),
            sigmaX=config.smooth_sigma,
            sigmaY=config.smooth_sigma,
        )

    median = float(np.median(response))
    mad = float(np.median(np.abs(response - median)))
    robust_sigma = 1.4826 * mad

    percentile_threshold = float(
        np.percentile(response, config.threshold_percentile)
    )
    mad_threshold = median + config.mad_multiplier * robust_sigma

    threshold = max(
        percentile_threshold,
        mad_threshold,
        config.min_response,
    )
    return response, threshold


def generate_green_points(
    image: np.ndarray,
    config: PointConfig = POINT_CONFIG,
) -> np.ndarray:
    """
    Green 채널의 local maxima를 자동 reference point로 생성한다.

    반환 배열 shape = (N, 3):
        [y, x, response]

    point 밀집도는 min_distance_px로 제한한다.
    이미지마다 point 개수를 고정하지 않는다.
    """

    response, threshold = _robust_green_response(image, config)
    height, width = response.shape

    radius = max(1, int(config.min_distance_px))
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    local_max = cv2.dilate(response, kernel)
    candidate_mask = (
        (response >= threshold)
        & (response >= local_max - 1e-6)
    ).astype(np.uint8)

    border = max(0, int(config.exclude_border_px))
    if border > 0:
        candidate_mask[:border, :] = 0
        candidate_mask[-border:, :] = 0
        candidate_mask[:, :border] = 0
        candidate_mask[:, -border:] = 0

    # local maximum plateau 하나에서 point 하나만 선택
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )

    candidates: List[Tuple[int, int, float]] = []
    for label_id in range(1, n_labels):
        ys, xs = np.where(labels == label_id)
        if len(ys) == 0:
            continue
        values = response[ys, xs]
        best_idx = int(np.argmax(values))
        candidates.append(
            (int(ys[best_idx]), int(xs[best_idx]), float(values[best_idx]))
        )

    # 강한 peak부터 minimum-distance non-maximum suppression
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: List[Tuple[int, int, float]] = []
    min_distance_sq = float(config.min_distance_px ** 2)

    for y, x, value in candidates:
        if all(
            (y - sy) ** 2 + (x - sx) ** 2 >= min_distance_sq
            for sy, sx, _ in selected
        ):
            selected.append((y, x, value))

        if (
            config.max_points_per_image is not None
            and len(selected) >= config.max_points_per_image
        ):
            break

    if not selected:
        return np.empty((0, 3), dtype=np.float32)

    return np.asarray(selected, dtype=np.float32)


def precompute_green_points(
    images: Mapping[str, np.ndarray],
    config: PointConfig = POINT_CONFIG,
    max_workers: int = 4,
) -> Dict[str, np.ndarray]:
    """GA 실행 전에 모든 Green point를 한 번만 생성하여 고정한다."""

    point_cache: Dict[str, np.ndarray] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_green_points, image, config): path
            for path, image in images.items()
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                point_cache[path] = future.result()
            except Exception as exc:
                print(f"[WARNING] point 생성 실패: {path}: {exc}")
                point_cache[path] = np.empty((0, 3), dtype=np.float32)

    return point_cache


def point_cache_to_dataframe(
    point_cache: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for path, points in point_cache.items():
        for point_id, (y, x, response) in enumerate(points):
            rows.append(
                {
                    "path": path,
                    "condition": extract_condition(path),
                    "point_id": point_id,
                    "y": float(y),
                    "x": float(x),
                    "green_response": float(response),
                }
            )
    return pd.DataFrame(rows)


def validate_point_coverage(
    point_cache: Mapping[str, np.ndarray],
    minimum_coverage: float = MIN_POINT_COVERAGE,
) -> Dict[str, float]:
    counts = np.asarray([len(points) for points in point_cache.values()], dtype=float)
    if len(counts) == 0:
        raise ValueError("point cache가 비어 있습니다.")

    coverage = float(np.mean(counts > 0))
    summary = {
        "n_images": int(len(counts)),
        "n_images_with_points": int(np.sum(counts > 0)),
        "coverage": coverage,
        "mean_points": float(np.mean(counts)),
        "median_points": float(np.median(counts)),
        "max_points": int(np.max(counts)),
    }

    print("\n[Green point summary]")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if coverage < minimum_coverage:
        raise ValueError(
            f"Green point coverage가 {coverage:.1%}로 너무 낮습니다. "
            f"POINT_CONFIG의 threshold_percentile, mad_multiplier, "
            f"min_response를 먼저 조정하세요. 요구 coverage={minimum_coverage:.1%}"
        )

    return summary


# =============================================================================
# 4. AdhGen object detection
# =============================================================================

def repair_individual(
    individual: Sequence[int],
    config: GAConfig = GA_CONFIG,
) -> List[int]:
    """min/max 정렬, 범위 제한, 최소 span을 보장한다."""

    if len(individual) != 8:
        raise ValueError("chromosome은 8개 값이어야 합니다.")

    repaired: List[int] = []
    pair_specs = [
        (config.rgb_low, config.rgb_high, config.min_rgb_span),
        (config.rgb_low, config.rgb_high, config.min_rgb_span),
        (config.rgb_low, config.rgb_high, config.min_rgb_span),
        (config.area_low, config.area_high, config.min_area_span),
    ]

    for pair_index, (low, high, min_span) in enumerate(pair_specs):
        a = int(round(individual[2 * pair_index]))
        b = int(round(individual[2 * pair_index + 1]))
        lo, hi = sorted((a, b))
        lo = max(low, min(lo, high))
        hi = max(low, min(hi, high))

        if hi - lo < min_span:
            hi = min(high, lo + min_span)
            if hi - lo < min_span:
                lo = max(low, hi - min_span)

        repaired.extend([lo, hi])

    return repaired


def detect_adhesion_objects(
    image: np.ndarray,
    individual: Sequence[int],
) -> Tuple[List[dict], np.ndarray]:
    """
    R/G/B threshold와 area 범위를 만족하는 connected component를 반환한다.
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
    ) = repair_individual(individual)

    mask = (
        (image[:, :, 2] >= r_min)
        & (image[:, :, 2] <= r_max)
        & (image[:, :, 1] >= g_min)
        & (image[:, :, 1] <= g_max)
        & (image[:, :, 0] >= b_min)
        & (image[:, :, 0] <= b_max)
    )

    binary = np.zeros(image.shape[:2], dtype=np.uint8)
    binary[mask] = 255

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    objects: List[dict] = []
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if not (area_min <= area <= area_max):
            continue

        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        centroid_x, centroid_y = centroids[label_id]

        objects.append(
            {
                "label": label_id,
                "area": area,
                "contour": contour,
                "centroid": (float(centroid_x), float(centroid_y)),
                "width": int(width),
                "height": int(height),
                "aspect_ratio": float(width / height) if height > 0 else 0.0,
            }
        )

    return objects, binary


def point_to_object_distance(
    point_yx: Sequence[float],
    contour: np.ndarray,
) -> float:
    """
    point와 객체 영역 사이 최소 거리.
    point가 객체 내부 또는 경계에 있으면 0이다.
    """

    y, x = float(point_yx[0]), float(point_yx[1])
    signed_distance = cv2.pointPolygonTest(contour, (x, y), True)
    return 0.0 if signed_distance >= 0 else float(-signed_distance)


# =============================================================================
# 5. One-to-one matching and distance-weighted F1 objective
# =============================================================================

def match_points_to_objects(
    points: np.ndarray,
    objects: Sequence[dict],
    match_radius_px: float = MATCH_RADIUS_PX,
) -> dict:
    """
    Hungarian algorithm으로 point와 predicted object를 일대일 matching한다.

    q = max(0, 1 - distance / match_radius)
    image score = 2 * sum(q_matched) / (N_points + N_objects)
    """

    n_points = int(len(points))
    n_objects = int(len(objects))

    if n_points == 0:
        return {
            "score": None,
            "matches": [],
            "matched_quality_sum": 0.0,
            "unmatched_point_indices": [],
            "unmatched_object_indices": list(range(n_objects)),
        }

    if n_objects == 0:
        return {
            "score": 0.0,
            "matches": [],
            "matched_quality_sum": 0.0,
            "unmatched_point_indices": list(range(n_points)),
            "unmatched_object_indices": [],
        }

    if match_radius_px <= 0:
        raise ValueError("match_radius_px는 0보다 커야 합니다.")

    quality = np.zeros((n_points, n_objects), dtype=np.float64)
    distances = np.zeros_like(quality)

    for point_index, point in enumerate(points):
        for object_index, obj in enumerate(objects):
            distance = point_to_object_distance(point[:2], obj["contour"])
            distances[point_index, object_index] = distance
            quality[point_index, object_index] = max(
                0.0,
                1.0 - distance / match_radius_px,
            )

    # quality 최대화 = -quality 비용 최소화
    row_indices, col_indices = linear_sum_assignment(-quality)

    matches = []
    matched_points = set()
    matched_objects = set()
    quality_sum = 0.0

    for point_index, object_index in zip(row_indices, col_indices):
        q = float(quality[point_index, object_index])
        if q <= 0:
            continue

        distance = float(distances[point_index, object_index])
        matches.append(
            {
                "point_index": int(point_index),
                "object_index": int(object_index),
                "distance": distance,
                "quality": q,
            }
        )
        matched_points.add(int(point_index))
        matched_objects.add(int(object_index))
        quality_sum += q

    score = 2.0 * quality_sum / (n_points + n_objects)

    return {
        "score": float(score),
        "matches": matches,
        "matched_quality_sum": float(quality_sum),
        "unmatched_point_indices": sorted(set(range(n_points)) - matched_points),
        "unmatched_object_indices": sorted(set(range(n_objects)) - matched_objects),
    }


def score_single_image(
    image: np.ndarray,
    points: np.ndarray,
    individual: Sequence[int],
    match_radius_px: float = MATCH_RADIUS_PX,
) -> Tuple[Optional[float], dict, List[dict], np.ndarray]:
    objects, binary = detect_adhesion_objects(image, individual)
    match_result = match_points_to_objects(points, objects, match_radius_px)
    return match_result["score"], match_result, objects, binary


def evaluate_individual(
    individual: Sequence[int],
    images: Mapping[str, np.ndarray],
    point_cache: Mapping[str, np.ndarray],
    match_radius_px: float = MATCH_RADIUS_PX,
    max_workers: int = 4,
) -> float:
    """이미지별 score의 macro mean. Soft/Stiff label은 사용하지 않는다."""

    individual = repair_individual(individual)
    scores: List[float] = []

    def worker(path: str) -> Optional[float]:
        points = point_cache.get(path, np.empty((0, 3), dtype=np.float32))
        if len(points) == 0:
            # Green reference가 없는 이미지는 최고 점수로 처리하지 않고
            # development fitness에서 제외한다.
            return None
        score, _, _, _ = score_single_image(
            images[path],
            points,
            individual,
            match_radius_px,
        )
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, path) for path in images]
        for future in as_completed(futures):
            score = future.result()
            if score is not None and np.isfinite(score):
                scores.append(float(score))

    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# 6. GA operators (DEAP 없이 기존 구조를 재현)
# =============================================================================

def valid_range(low: int, high: int) -> Tuple[int, int]:
    a = random.randint(low, high)
    b = random.randint(low, high)
    return min(a, b), max(a, b)


def create_individual(config: GAConfig = GA_CONFIG) -> List[int]:
    r_min, r_max = valid_range(config.rgb_low, config.rgb_high)
    g_min, g_max = valid_range(config.rgb_low, config.rgb_high)
    b_min, b_max = valid_range(config.rgb_low, config.rgb_high)
    area_min, area_max = valid_range(config.area_low, config.area_high)
    return repair_individual(
        [
            r_min,
            r_max,
            g_min,
            g_max,
            b_min,
            b_max,
            area_min,
            area_max,
        ],
        config,
    )


def mate_ind(
    individual_1: Sequence[int],
    individual_2: Sequence[int],
    config: GAConfig = GA_CONFIG,
) -> Tuple[List[int], List[int]]:
    """기존 partial-swap crossover: min-max pair 단위 교환."""

    child_1 = list(individual_1)
    child_2 = list(individual_2)

    for start in range(0, 8, 2):
        if random.random() < 0.5:
            child_1[start : start + 2], child_2[start : start + 2] = (
                child_2[start : start + 2],
                child_1[start : start + 2],
            )

    return repair_individual(child_1, config), repair_individual(child_2, config)


def mutate_pair(
    pair: Sequence[int],
    low: int,
    high: int,
    small_expansion: int,
) -> List[int]:
    """현재 range 폭을 이용하는 adaptive-range mutation."""

    current_low, current_high = sorted((int(pair[0]), int(pair[1])))
    half_range = max(1, (current_high - current_low) // 2)

    new_low = random.randint(
        max(low, current_low - half_range),
        min(high, current_low + half_range),
    )
    new_high = random.randint(
        max(low, current_high - half_range),
        min(high, current_high + half_range),
    )
    new_low, new_high = sorted((new_low, new_high))

    if new_high - new_low <= 3:
        if random.random() < 0.5:
            new_low = max(low, new_low - small_expansion)
        if random.random() < 0.5:
            new_high = min(high, new_high + small_expansion)

    return [new_low, new_high]


def mutate_ind(
    individual: Sequence[int],
    config: GAConfig = GA_CONFIG,
) -> List[int]:
    mutated = list(individual)

    if random.random() < 0.5:
        mutated[0:2] = mutate_pair(
            mutated[0:2], config.rgb_low, config.rgb_high, 10
        )
    if random.random() < 0.5:
        mutated[2:4] = mutate_pair(
            mutated[2:4], config.rgb_low, config.rgb_high, 10
        )
    if random.random() < 0.5:
        mutated[4:6] = mutate_pair(
            mutated[4:6], config.rgb_low, config.rgb_high, 10
        )
    if random.random() < 0.5:
        mutated[6:8] = mutate_pair(
            mutated[6:8], config.area_low, config.area_high, 5
        )

    return repair_individual(mutated, config)


def tournament_select(
    population: Sequence[Sequence[int]],
    fitnesses: Sequence[float],
    tournament_size: int,
) -> List[int]:
    candidate_indices = random.sample(
        range(len(population)),
        k=min(tournament_size, len(population)),
    )
    best_index = max(candidate_indices, key=lambda idx: fitnesses[idx])
    return list(population[best_index])


def run_genetic_algorithm(
    images: Mapping[str, np.ndarray],
    point_cache: Mapping[str, np.ndarray],
    match_radius_px: float = MATCH_RADIUS_PX,
    config: GAConfig = GA_CONFIG,
) -> Tuple[List[int], float, pd.DataFrame]:
    """Point-guided objective를 최대화하는 GA."""

    seed_everything(config.seed)
    population = [create_individual(config) for _ in range(config.population_size)]
    fitness_cache: Dict[Tuple[int, ...], float] = {}
    history_rows = []

    global_best: Optional[List[int]] = None
    global_best_fitness = -math.inf

    def cached_fitness(individual: Sequence[int]) -> float:
        repaired = repair_individual(individual, config)
        key = tuple(repaired)
        if key not in fitness_cache:
            fitness_cache[key] = evaluate_individual(
                repaired,
                images,
                point_cache,
                match_radius_px=match_radius_px,
                max_workers=config.max_workers,
            )
        return fitness_cache[key]

    for generation in range(config.generations):
        fitnesses = [cached_fitness(ind) for ind in population]

        best_index = int(np.argmax(fitnesses))
        generation_best = list(population[best_index])
        generation_best_fitness = float(fitnesses[best_index])

        if generation_best_fitness > global_best_fitness:
            global_best = copy.deepcopy(generation_best)
            global_best_fitness = generation_best_fitness

        history_rows.append(
            {
                "generation": generation,
                "fitness_mean": float(np.mean(fitnesses)),
                "fitness_std": float(np.std(fitnesses)),
                "fitness_min": float(np.min(fitnesses)),
                "fitness_max": generation_best_fitness,
                "unique_evaluations": len(fitness_cache),
                "best_individual": json.dumps(generation_best),
            }
        )

        print(
            f"Generation {generation + 1:03d}/{config.generations}: "
            f"mean={np.mean(fitnesses):.4f}, "
            f"max={generation_best_fitness:.4f}, "
            f"best={generation_best}"
        )

        # 마지막 세대에서는 다음 population을 만들 필요가 없다.
        if generation == config.generations - 1:
            break

        elite_indices = np.argsort(fitnesses)[-config.elite_size :][::-1]
        next_population = [copy.deepcopy(population[idx]) for idx in elite_indices]

        while len(next_population) < config.population_size:
            parent_1 = tournament_select(
                population,
                fitnesses,
                config.tournament_size,
            )
            parent_2 = tournament_select(
                population,
                fitnesses,
                config.tournament_size,
            )

            child_1, child_2 = parent_1, parent_2
            if random.random() < config.crossover_probability:
                child_1, child_2 = mate_ind(parent_1, parent_2, config)

            if random.random() < config.mutation_probability:
                child_1 = mutate_ind(child_1, config)
            if random.random() < config.mutation_probability:
                child_2 = mutate_ind(child_2, config)

            next_population.append(repair_individual(child_1, config))
            if len(next_population) < config.population_size:
                next_population.append(repair_individual(child_2, config))

        population = next_population

    if global_best is None:
        raise RuntimeError("GA가 유효한 best individual을 생성하지 못했습니다.")

    return global_best, float(global_best_fitness), pd.DataFrame(history_rows)


# =============================================================================
# 7. Result table and visualization
# =============================================================================

def evaluate_solution(
    images: Mapping[str, np.ndarray],
    point_cache: Mapping[str, np.ndarray],
    individual: Sequence[int],
    match_radius_px: float = MATCH_RADIUS_PX,
) -> pd.DataFrame:
    rows = []

    for path, image in images.items():
        points = point_cache.get(path, np.empty((0, 3), dtype=np.float32))
        score, match_result, objects, _ = score_single_image(
            image,
            points,
            individual,
            match_radius_px,
        )

        object_areas = [obj["area"] for obj in objects]
        aspect_ratios = [obj["aspect_ratio"] for obj in objects]

        rows.append(
            {
                "path": path,
                "condition": extract_condition(path),
                "point_count": len(points),
                "fa_count": len(objects),
                "matched_count": len(match_result["matches"]),
                "point_match_score": np.nan if score is None else score,
                "total_area": int(np.sum(object_areas)) if object_areas else 0,
                "mean_area": float(np.mean(object_areas)) if object_areas else 0.0,
                "mean_ar": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
            }
        )

    return pd.DataFrame(rows)


def draw_points_only(
    image: np.ndarray,
    points: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    overlay = image.copy()
    for y, x, _ in points:
        cv2.circle(
            overlay,
            (int(round(x)), int(round(y))),
            radius,
            (0, 255, 255),  # yellow in BGR
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def draw_matching_result(
    image: np.ndarray,
    points: np.ndarray,
    individual: Sequence[int],
    match_radius_px: float = MATCH_RADIUS_PX,
) -> Tuple[np.ndarray, dict]:
    _, match_result, objects, _ = score_single_image(
        image,
        points,
        individual,
        match_radius_px,
    )

    overlay = image.copy()
    matched_object_indices = {
        match["object_index"] for match in match_result["matches"]
    }

    # unmatched predicted objects: red, matched objects: green
    for object_index, obj in enumerate(objects):
        color = (0, 255, 0) if object_index in matched_object_indices else (0, 0, 255)
        cv2.drawContours(overlay, [obj["contour"]], -1, color, 2)

    # reference points: yellow
    for y, x, _ in points:
        cv2.circle(
            overlay,
            (int(round(x)), int(round(y))),
            3,
            (0, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # matched point-object centroid connection: green line
    for match in match_result["matches"]:
        point = points[match["point_index"]]
        obj = objects[match["object_index"]]
        y, x = point[:2]
        centroid_x, centroid_y = obj["centroid"]
        cv2.line(
            overlay,
            (int(round(x)), int(round(y))),
            (int(round(centroid_x)), int(round(centroid_y))),
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    score_text = match_result["score"]
    text = "score=N/A" if score_text is None else f"score={score_text:.3f}"
    cv2.putText(
        overlay,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay, match_result


def save_preview_images(
    images: Mapping[str, np.ndarray],
    point_cache: Mapping[str, np.ndarray],
    output_dir: str | Path,
    individual: Optional[Sequence[int]] = None,
    n_images: int = 8,
    match_radius_px: float = MATCH_RADIUS_PX,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(images.keys())[:n_images]
    for index, path in enumerate(paths):
        image = images[path]
        points = point_cache.get(path, np.empty((0, 3), dtype=np.float32))

        if individual is None:
            overlay = draw_points_only(image, points)
            prefix = "points"
        else:
            overlay, _ = draw_matching_result(
                image,
                points,
                individual,
                match_radius_px,
            )
            prefix = "matching"

        filename = output_dir / f"{prefix}_{index:03d}_{Path(path).stem}.png"
        cv2.imwrite(str(filename), overlay)


def plot_ga_history(history_df: pd.DataFrame, save_path: str | Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["generation"], history_df["fitness_mean"], label="Mean")
    plt.plot(history_df["generation"], history_df["fitness_max"], label="Best")
    plt.xlabel("Generation")
    plt.ylabel("Point-guided fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 8. Main
# =============================================================================

def main(
    data_root: str | Path = "./data",
    output_dir: str | Path = OUTPUT_DIR,
    excluded_folders: Iterable[str] = EXCLUDED_FOLDERS,
    point_config: PointConfig = POINT_CONFIG,
    ga_config: GAConfig = GA_CONFIG,
    match_radius_px: float = MATCH_RADIUS_PX,
) -> Tuple[List[int], float, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(ga_config.seed)

    image_paths = collect_image_paths(data_root, excluded_folders)
    if not image_paths:
        raise FileNotFoundError(
            f"이미지를 찾지 못했습니다. data_root={data_root}, "
            "예상 구조: data/*/Soft/*_M.tif, data/*/Stiff/*_M.tif"
        )

    print(f"Loading {len(image_paths)} images...")
    images = preload_images(image_paths)
    if not images:
        raise RuntimeError("로드된 이미지가 없습니다.")

    print("Generating fixed Green reference points...")
    point_cache = precompute_green_points(
        images,
        point_config,
        max_workers=ga_config.max_workers,
    )
    point_summary = validate_point_coverage(point_cache, MIN_POINT_COVERAGE)

    point_df = point_cache_to_dataframe(point_cache)
    point_df.to_csv(output_dir / "green_reference_points.csv", index=False)
    save_preview_images(
        images,
        point_cache,
        output_dir / "point_previews",
        individual=None,
        n_images=8,
    )

    print("\nStarting point-guided GA...")
    best_individual, best_fitness, history_df = run_genetic_algorithm(
        images,
        point_cache,
        match_radius_px=match_radius_px,
        config=ga_config,
    )

    print("\nBest Individual:", best_individual)
    print("Best point-guided fitness:", best_fitness)

    history_df.to_csv(output_dir / "ga_history.csv", index=False)
    plot_ga_history(history_df, output_dir / "ga_history.png")

    result_df = evaluate_solution(
        images,
        point_cache,
        best_individual,
        match_radius_px,
    )
    result_df.to_csv(output_dir / "adhgen_point_guided_results.csv", index=False)

    save_preview_images(
        images,
        point_cache,
        output_dir / "matching_previews",
        individual=best_individual,
        n_images=8,
        match_radius_px=match_radius_px,
    )

    metadata = {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "match_radius_px": match_radius_px,
        "excluded_folders": sorted(str(x) for x in excluded_folders),
        "point_config": asdict(point_config),
        "ga_config": asdict(ga_config),
        "point_summary": point_summary,
        "objective": (
            "Macro mean of image-level distance-weighted F1: "
            "2*sum(max(0,1-distance/r))/(N_points+N_objects), "
            "using one-to-one Hungarian matching"
        ),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return best_individual, best_fitness, result_df


if __name__ == "__main__":
    main()
