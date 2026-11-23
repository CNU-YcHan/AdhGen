from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Dict
import os
import random
import cv2
import numpy as np

EXCLUDED_FOLDERS = {"9", "12"}
MATCH_RADIUS_PX = 4.0

@dataclass(frozen=True)
class PointConfig:
    background_sigma: float = 15.0
    smooth_sigma: float = 1.5
    threshold_percentile: float = 99.5
    mad_multiplier: float = 5.0
    min_response: float = 8.0
    min_distance_px: int = 8
    exclude_border_px: int = 0
    max_points_per_image: Optional[int] = None

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    generations: int = 100
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
    min_rgb_span: int = 1
    min_area_span: int = 1

POINT_CONFIG = PointConfig()
GA_CONFIG = GAConfig()

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def collect_image_paths(data_root: str | Path='./data', excluded_folders: Iterable[str]=EXCLUDED_FOLDERS) -> List[str]:
    data_root = Path(data_root)
    excluded = {str(x) for x in excluded_folders}
    paths = list(data_root.glob('*/Soft/*_M.tif'))
    paths += list(data_root.glob('*/Stiff/*_M.tif'))
    selected = [p for p in paths if not excluded.intersection(p.parts)]
    return sorted((str(p) for p in selected))

def preload_images(image_paths: Sequence[str]) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f'[WARNING] Could not read image: {path}')
            continue
        images[path] = image
    return images

def extract_condition(path: str) -> str:
    parts = {part.lower() for part in Path(path).parts}
    if 'soft' in parts:
        return 'Soft'
    if 'stiff' in parts:
        return 'Stiff'
    return 'Unknown'

def _robust_green_response(image: np.ndarray, config: PointConfig) -> Tuple[np.ndarray, float]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('A three-channel BGR image is required.')
    green = image[:, :, 1].astype(np.float32)
    background = cv2.GaussianBlur(green, ksize=(0, 0), sigmaX=max(config.background_sigma, 0.1), sigmaY=max(config.background_sigma, 0.1))
    response = np.clip(green - background, 0.0, None)
    if config.smooth_sigma > 0:
        response = cv2.GaussianBlur(response, ksize=(0, 0), sigmaX=config.smooth_sigma, sigmaY=config.smooth_sigma)
    median = float(np.median(response))
    mad = float(np.median(np.abs(response - median)))
    robust_sigma = 1.4826 * mad
    percentile_threshold = float(np.percentile(response, config.threshold_percentile))
    mad_threshold = median + config.mad_multiplier * robust_sigma
    threshold = max(percentile_threshold, mad_threshold, config.min_response)
    return (response, threshold)

def generate_green_points(image: np.ndarray, config: PointConfig=POINT_CONFIG) -> np.ndarray:
    response, threshold = _robust_green_response(image, config)
    height, width = response.shape
    radius = max(1, int(config.min_distance_px))
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(response, kernel)
    candidate_mask = ((response >= threshold) & (response >= local_max - 1e-06)).astype(np.uint8)
    border = max(0, int(config.exclude_border_px))
    if border > 0:
        candidate_mask[:border, :] = 0
        candidate_mask[-border:, :] = 0
        candidate_mask[:, :border] = 0
        candidate_mask[:, -border:] = 0
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    candidates: List[Tuple[int, int, float]] = []
    for label_id in range(1, n_labels):
        ys, xs = np.where(labels == label_id)
        if len(ys) == 0:
            continue
        values = response[ys, xs]
        best_idx = int(np.argmax(values))
        candidates.append((int(ys[best_idx]), int(xs[best_idx]), float(values[best_idx])))
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: List[Tuple[int, int, float]] = []
    min_distance_sq = float(config.min_distance_px ** 2)
    for y, x, value in candidates:
        if all(((y - sy) ** 2 + (x - sx) ** 2 >= min_distance_sq for sy, sx, _ in selected)):
            selected.append((y, x, value))
        if config.max_points_per_image is not None and len(selected) >= config.max_points_per_image:
            break
    if not selected:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)

def repair_individual(individual: Sequence[int], config: GAConfig=GA_CONFIG) -> List[int]:
    if len(individual) != 8:
        raise ValueError('The chromosome must contain eight values.')
    repaired: List[int] = []
    pair_specs = [(config.rgb_low, config.rgb_high, config.min_rgb_span), (config.rgb_low, config.rgb_high, config.min_rgb_span), (config.rgb_low, config.rgb_high, config.min_rgb_span), (config.area_low, config.area_high, config.min_area_span)]
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

def valid_range(low: int, high: int) -> Tuple[int, int]:
    a = random.randint(low, high)
    b = random.randint(low, high)
    return (min(a, b), max(a, b))

def create_individual(config: GAConfig=GA_CONFIG) -> List[int]:
    r_min, r_max = valid_range(config.rgb_low, config.rgb_high)
    g_min, g_max = valid_range(config.rgb_low, config.rgb_high)
    b_min, b_max = valid_range(config.rgb_low, config.rgb_high)
    area_min, area_max = valid_range(config.area_low, config.area_high)
    return repair_individual([r_min, r_max, g_min, g_max, b_min, b_max, area_min, area_max], config)

def mate_ind(individual_1: Sequence[int], individual_2: Sequence[int], config: GAConfig=GA_CONFIG) -> Tuple[List[int], List[int]]:
    child_1 = list(individual_1)
    child_2 = list(individual_2)
    for start in range(0, 8, 2):
        if random.random() < 0.5:
            child_1[start:start + 2], child_2[start:start + 2] = (child_2[start:start + 2], child_1[start:start + 2])
    return (repair_individual(child_1, config), repair_individual(child_2, config))

def mutate_pair(pair: Sequence[int], low: int, high: int, small_expansion: int) -> List[int]:
    current_low, current_high = sorted((int(pair[0]), int(pair[1])))
    half_range = max(1, (current_high - current_low) // 2)
    new_low = random.randint(max(low, current_low - half_range), min(high, current_low + half_range))
    new_high = random.randint(max(low, current_high - half_range), min(high, current_high + half_range))
    new_low, new_high = sorted((new_low, new_high))
    if new_high - new_low <= 3:
        if random.random() < 0.5:
            new_low = max(low, new_low - small_expansion)
        if random.random() < 0.5:
            new_high = min(high, new_high + small_expansion)
    return [new_low, new_high]

def mutate_ind(individual: Sequence[int], config: GAConfig=GA_CONFIG) -> List[int]:
    mutated = list(individual)
    if random.random() < 0.5:
        mutated[0:2] = mutate_pair(mutated[0:2], config.rgb_low, config.rgb_high, 10)
    if random.random() < 0.5:
        mutated[2:4] = mutate_pair(mutated[2:4], config.rgb_low, config.rgb_high, 10)
    if random.random() < 0.5:
        mutated[4:6] = mutate_pair(mutated[4:6], config.rgb_low, config.rgb_high, 10)
    if random.random() < 0.5:
        mutated[6:8] = mutate_pair(mutated[6:8], config.area_low, config.area_high, 5)
    return repair_individual(mutated, config)

def tournament_select(population: Sequence[Sequence[int]], fitnesses: Sequence[float], tournament_size: int) -> List[int]:
    candidate_indices = random.sample(range(len(population)), k=min(tournament_size, len(population)))
    best_index = max(candidate_indices, key=lambda idx: fitnesses[idx])
    return list(population[best_index])
