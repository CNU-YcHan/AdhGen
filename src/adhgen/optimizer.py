from __future__ import annotations
import argparse
import copy
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .ga_utils import (EXCLUDED_FOLDERS, GA_CONFIG, MATCH_RADIUS_PX, POINT_CONFIG, GAConfig, PointConfig,
    collect_image_paths, create_individual, extract_condition, generate_green_points, mate_ind, mutate_ind,
    preload_images, repair_individual, seed_everything, tournament_select)
from .cell_segmentation import (CELL_INSTANCE_CONFIG, NUCLEUS_CONFIG, CellInstanceConfig, NucleusConfig,
    segment_cell_instances, segment_nuclei)

SCRIPT_VERSION = "1.0.0"
AREA_SOFT_LIMIT_PX = 60.0
POINT_CONFIG_FINAL = POINT_CONFIG
NUCLEUS_CONFIG_FINAL = NUCLEUS_CONFIG
CELL_CONFIG_FINAL = CELL_INSTANCE_CONFIG

@dataclass(frozen=True)
class ReferenceMaskConfig:
    exclude_scale_bar: bool = True
    scale_bar_x_start_fraction: float = 0.78
    scale_bar_height_px: int = 70

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

REFERENCE_MASK_CONFIG = ReferenceMaskConfig()

def make_scale_bar_exclusion_mask(image_shape: Sequence[int], config: ReferenceMaskConfig) -> np.ndarray:
    height, width = (int(image_shape[0]), int(image_shape[1]))
    mask = np.zeros((height, width), dtype=bool)
    if not config.exclude_scale_bar:
        return mask
    x_start = int(round(np.clip(config.scale_bar_x_start_fraction, 0.0, 1.0) * width))
    y_start = max(0, height - int(config.scale_bar_height_px))
    mask[y_start:height, x_start:width] = True
    return mask

def filter_points_by_mask(points: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    height, width = analysis_mask.shape
    y = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, height - 1)
    x = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, width - 1)
    keep = analysis_mask[y, x]
    return points[keep].astype(np.float32, copy=False)

def build_image_reference(path: str, image: np.ndarray, point_config: PointConfig, nucleus_config: NucleusConfig, cell_config: CellInstanceConfig, mask_config: ReferenceMaskConfig) -> ImageReference:
    raw_points = generate_green_points(image, point_config)
    nucleus_label_map, nucleus_exclusion_mask, nuclei, _ = segment_nuclei(image, nucleus_config)
    cell_label_map, cells, _ = segment_cell_instances(image, nucleus_label_map, nucleus_exclusion_mask, cell_config)
    cell_union_mask = cell_label_map > 0
    scale_bar_exclusion_mask = make_scale_bar_exclusion_mask(image.shape, mask_config)
    analysis_mask = cell_union_mask & ~nucleus_exclusion_mask & ~scale_bar_exclusion_mask
    points = filter_points_by_mask(raw_points, analysis_mask)
    return ImageReference(path=path, analysis_mask=analysis_mask.astype(bool), cell_union_mask=cell_union_mask.astype(bool), nucleus_exclusion_mask=nucleus_exclusion_mask.astype(bool), scale_bar_exclusion_mask=scale_bar_exclusion_mask.astype(bool), points=points, n_nuclei=int(len(nuclei)), n_cells=int(len(cells)))

def precompute_references(images: Mapping[str, np.ndarray], point_config: PointConfig, nucleus_config: NucleusConfig, cell_config: CellInstanceConfig, mask_config: ReferenceMaskConfig, max_workers: int) -> Tuple[Dict[str, ImageReference], pd.DataFrame]:
    references: Dict[str, ImageReference] = {}
    failure_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(build_image_reference, path, image, point_config, nucleus_config, cell_config, mask_config): path for path, image in images.items()}
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            path = futures[future]
            completed += 1
            try:
                references[path] = future.result()
            except Exception as exc:
                failure_rows.append({'path': path, 'error_type': type(exc).__name__, 'error': str(exc)})
                print(f'[Reference failed] {Path(path).name}: {type(exc).__name__}: {exc}')
            if completed % 25 == 0 or completed == total:
                print(f'Reference generation: {completed}/{total}')
    return (references, pd.DataFrame(failure_rows))

def reference_summary_dataframe(references: Mapping[str, ImageReference]) -> pd.DataFrame:
    rows = []
    for path, reference in references.items():
        rows.append({'path': path, 'condition': extract_condition(path), 'n_nuclei': int(reference.n_nuclei), 'n_cells': int(reference.n_cells), 'n_reference_points': int(len(reference.points)), 'cell_union_area_px': int(np.count_nonzero(reference.cell_union_mask)), 'analysis_area_px': int(np.count_nonzero(reference.analysis_mask)), 'nucleus_exclusion_area_px': int(np.count_nonzero(reference.nucleus_exclusion_mask)), 'scale_bar_exclusion_area_px': int(np.count_nonzero(reference.scale_bar_exclusion_mask))})
    return pd.DataFrame(rows)

def validate_reference_images(references: Mapping[str, ImageReference], minimum_image_coverage: float=0.8) -> None:
    if not references:
        raise ValueError('No valid image references were generated.')
    counts = np.asarray([len(reference.points) for reference in references.values()], dtype=np.int32)
    coverage = float(np.mean(counts > 0))
    print('\n[Fixed reference summary]')
    print('  images:', len(counts))
    print('  images with >=1 point:', int(np.count_nonzero(counts > 0)))
    print('  image coverage:', coverage)
    print('  mean points:', float(np.mean(counts)))
    print('  median points:', float(np.median(counts)))
    print('  minimum points:', int(np.min(counts)))
    print('  maximum points:', int(np.max(counts)))
    if coverage < minimum_image_coverage:
        raise ValueError(f'Reference-point image coverage is too low: {coverage:.1%}.')

def detect_objects_in_union_mask(image: np.ndarray, individual: Sequence[int], reference: ImageReference, mask_config: ReferenceMaskConfig, ga_config: GAConfig) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    r_min, r_max, g_min, g_max, b_min, b_max, area_min, area_max = repair_individual(individual, ga_config)
    rgb_mask = (image[:, :, 2] >= r_min) & (image[:, :, 2] <= r_max) & (image[:, :, 1] >= g_min) & (image[:, :, 1] <= g_max) & (image[:, :, 0] >= b_min) & (image[:, :, 0] <= b_max)
    binary_mask = (rgb_mask & reference.analysis_mask).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    objects: List[dict] = []
    component_to_object = np.zeros(n_labels, dtype=np.int32)
    for component_label in range(1, n_labels):
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        if not area_min <= area <= area_max:
            continue
        object_index = len(objects)
        component_to_object[component_label] = object_index + 1
        x = int(stats[component_label, cv2.CC_STAT_LEFT])
        y = int(stats[component_label, cv2.CC_STAT_TOP])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        centroid_x, centroid_y = centroids[component_label]
        objects.append({'component_label': int(component_label), 'area': int(area), 'contour': None, 'centroid': (float(centroid_x), float(centroid_y)), 'width': int(width), 'height': int(height), 'aspect_ratio': float(width / height) if height > 0 else 0.0, 'bbox': (x, y, width, height)})
    object_index_map = component_to_object[labels]
    return (objects, binary_mask, object_index_map)

def ensure_object_contour(obj: dict, object_index: int, object_index_map: np.ndarray) -> np.ndarray:
    contour = obj.get('contour')
    if contour is not None:
        return contour
    component_mask = (object_index_map == object_index + 1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
    else:
        contour = np.empty((0, 1, 2), dtype=np.int32)
    obj['contour'] = contour
    return contour

def calculate_area_weight(object_area: float, area_soft_limit_px: float=AREA_SOFT_LIMIT_PX) -> float:
    if area_soft_limit_px <= 0:
        raise ValueError('area_soft_limit_px must be greater than 0.')
    if object_area <= 0:
        return 0.0
    return float(min(1.0, area_soft_limit_px / float(object_area)))

def match_points_to_objects_local(points: np.ndarray, objects: Sequence[dict], object_index_map: np.ndarray, match_radius_px: float, area_soft_limit_px: float=AREA_SOFT_LIMIT_PX) -> dict:
    n_points = int(len(points))
    n_objects = int(len(objects))
    if n_points == 0:
        return {'score': None, 'matches': [], 'matched_quality_sum': 0.0, 'matched_distance_quality_sum': 0.0, 'mean_matched_area_weight': float('nan'), 'mean_matched_object_area_px': float('nan'), 'area_soft_limit_px': float(area_soft_limit_px), 'unmatched_point_indices': [], 'unmatched_object_indices': list(range(n_objects)), 'n_points': 0, 'n_objects': n_objects}
    if n_objects == 0:
        return {'score': 0.0, 'matches': [], 'matched_quality_sum': 0.0, 'matched_distance_quality_sum': 0.0, 'mean_matched_area_weight': float('nan'), 'mean_matched_object_area_px': float('nan'), 'area_soft_limit_px': float(area_soft_limit_px), 'unmatched_point_indices': list(range(n_points)), 'unmatched_object_indices': [], 'n_points': n_points, 'n_objects': 0}
    if match_radius_px <= 0:
        raise ValueError('match_radius_px must be greater than 0.')
    if area_soft_limit_px <= 0:
        raise ValueError('area_soft_limit_px must be greater than 0.')
    height, width = object_index_map.shape
    radius = int(np.ceil(match_radius_px))
    edge_values: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
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
        local_labels = np.unique(object_index_map[y0:y1, x0:x1])
        local_labels = local_labels[local_labels > 0]
        for map_label in local_labels:
            object_index = int(map_label - 1)
            obj = objects[object_index]
            contour = ensure_object_contour(obj, object_index, object_index_map)
            if contour.size == 0:
                continue
            signed_distance = cv2.pointPolygonTest(contour, (x, y), True)
            distance = 0.0 if signed_distance >= 0 else float(-signed_distance)
            if distance > match_radius_px:
                continue
            distance_quality = max(0.0, 1.0 - distance / match_radius_px)
            if distance_quality <= 0:
                continue
            area_weight = calculate_area_weight(object_area=float(obj['area']), area_soft_limit_px=area_soft_limit_px)
            quality = distance_quality * area_weight
            if quality <= 0:
                continue
            edge_values[point_index, object_index] = (float(quality), float(distance), float(distance_quality), float(area_weight))
            candidate_object_indices.add(object_index)
    if not candidate_object_indices:
        return {'score': 0.0, 'matches': [], 'matched_quality_sum': 0.0, 'matched_distance_quality_sum': 0.0, 'mean_matched_area_weight': float('nan'), 'mean_matched_object_area_px': float('nan'), 'area_soft_limit_px': float(area_soft_limit_px), 'unmatched_point_indices': list(range(n_points)), 'unmatched_object_indices': list(range(n_objects)), 'n_points': n_points, 'n_objects': n_objects}
    candidate_objects = sorted(candidate_object_indices)
    candidate_column = {object_index: column_index for column_index, object_index in enumerate(candidate_objects)}
    quality_matrix = np.zeros((n_points, len(candidate_objects)), dtype=np.float64)
    distance_matrix = np.full(quality_matrix.shape, np.inf, dtype=np.float64)
    distance_quality_matrix = np.zeros(quality_matrix.shape, dtype=np.float64)
    area_weight_matrix = np.zeros(quality_matrix.shape, dtype=np.float64)
    for (point_index, object_index), (quality, distance, distance_quality, area_weight) in edge_values.items():
        column_index = candidate_column[object_index]
        quality_matrix[point_index, column_index] = quality
        distance_matrix[point_index, column_index] = distance
        distance_quality_matrix[point_index, column_index] = distance_quality
        area_weight_matrix[point_index, column_index] = area_weight
    row_indices, column_indices = linear_sum_assignment(-quality_matrix)
    matches: List[dict] = []
    matched_points = set()
    matched_objects = set()
    quality_sum = 0.0
    distance_quality_sum = 0.0
    matched_area_weight_sum = 0.0
    matched_object_area_sum = 0.0
    for point_index, candidate_column_index in zip(row_indices, column_indices):
        quality = float(quality_matrix[point_index, candidate_column_index])
        if quality <= 0:
            continue
        object_index = candidate_objects[candidate_column_index]
        distance = float(distance_matrix[point_index, candidate_column_index])
        distance_quality = float(distance_quality_matrix[point_index, candidate_column_index])
        area_weight = float(area_weight_matrix[point_index, candidate_column_index])
        object_area = float(objects[object_index]['area'])
        matches.append({'point_index': int(point_index), 'object_index': int(object_index), 'distance': distance, 'distance_quality': distance_quality, 'object_area_px': object_area, 'area_weight': area_weight, 'quality': quality})
        matched_points.add(int(point_index))
        matched_objects.add(int(object_index))
        quality_sum += quality
        distance_quality_sum += distance_quality
        matched_area_weight_sum += area_weight
        matched_object_area_sum += object_area
    n_matches = len(matches)
    image_score = 2.0 * quality_sum / (n_points + n_objects)
    return {'score': float(image_score), 'matches': matches, 'matched_quality_sum': float(quality_sum), 'matched_distance_quality_sum': float(distance_quality_sum), 'mean_matched_area_weight': float(matched_area_weight_sum / n_matches) if n_matches > 0 else float('nan'), 'mean_matched_object_area_px': float(matched_object_area_sum / n_matches) if n_matches > 0 else float('nan'), 'area_soft_limit_px': float(area_soft_limit_px), 'unmatched_point_indices': sorted(set(range(n_points)) - matched_points), 'unmatched_object_indices': sorted(set(range(n_objects)) - matched_objects), 'n_points': n_points, 'n_objects': n_objects}

def score_single_reference(image: np.ndarray, individual: Sequence[int], reference: ImageReference, match_radius_px: float, mask_config: ReferenceMaskConfig, ga_config: GAConfig, area_soft_limit_px: float=AREA_SOFT_LIMIT_PX) -> Tuple[float, dict, List[dict], np.ndarray]:
    objects, _, object_index_map = detect_objects_in_union_mask(image, individual, reference, mask_config, ga_config)
    match_result = match_points_to_objects_local(reference.points, objects, object_index_map, match_radius_px=match_radius_px, area_soft_limit_px=area_soft_limit_px)
    score = match_result['score']
    score_value = float('nan') if score is None or not np.isfinite(score) else float(score)
    return (score_value, match_result, objects, object_index_map)

def evaluate_individual(individual: Sequence[int], images: Mapping[str, np.ndarray], references: Mapping[str, ImageReference], match_radius_px: float, mask_config: ReferenceMaskConfig, ga_config: GAConfig, area_soft_limit_px: float=AREA_SOFT_LIMIT_PX) -> float:
    repaired = repair_individual(individual, ga_config)
    total_quality_sum = 0.0
    total_points = 0
    total_objects = 0

    def worker(path: str) -> Tuple[float, int, int]:
        _, match_result, _, _ = score_single_reference(images[path], repaired, references[path], match_radius_px, mask_config, ga_config, area_soft_limit_px)
        return (float(match_result['matched_quality_sum']), int(match_result['n_points']), int(match_result['n_objects']))
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=ga_config.max_workers) as executor:
        futures = [executor.submit(worker, path) for path in references]
        for future in as_completed(futures):
            quality_sum, n_points, n_objects = future.result()
            total_quality_sum += quality_sum
            total_points += n_points
            total_objects += n_objects
    denominator = total_points + total_objects
    if denominator <= 0:
        return 0.0
    return float(2.0 * total_quality_sum / denominator)

def run_ga(images: Mapping[str, np.ndarray], references: Mapping[str, ImageReference], match_radius_px: float, mask_config: ReferenceMaskConfig, ga_config: GAConfig, area_soft_limit_px: float=AREA_SOFT_LIMIT_PX) -> Tuple[List[int], float, pd.DataFrame]:
    seed_everything(ga_config.seed)
    population = [create_individual(ga_config) for _ in range(ga_config.population_size)]
    fitness_cache: Dict[Tuple[int, ...], float] = {}
    history_rows: List[dict] = []
    global_best: Optional[List[int]] = None
    global_best_fitness = -math.inf

    def cached_fitness(individual: Sequence[int]) -> float:
        repaired = repair_individual(individual, ga_config)
        key = tuple(repaired)
        if key not in fitness_cache:
            fitness_cache[key] = evaluate_individual(repaired, images, references, match_radius_px, mask_config, ga_config, area_soft_limit_px)
        return fitness_cache[key]
    for generation in range(ga_config.generations):
        fitnesses = [cached_fitness(individual) for individual in population]
        best_index = int(np.argmax(fitnesses))
        generation_best = repair_individual(population[best_index], ga_config)
        generation_best_fitness = float(fitnesses[best_index])
        if generation_best_fitness > global_best_fitness:
            global_best = copy.deepcopy(generation_best)
            global_best_fitness = generation_best_fitness
        history_rows.append({'generation': int(generation + 1), 'fitness_mean': float(np.mean(fitnesses)), 'fitness_std': float(np.std(fitnesses)), 'fitness_min': float(np.min(fitnesses)), 'fitness_max': float(generation_best_fitness), 'unique_evaluations': int(len(fitness_cache)), 'best_individual': json.dumps(generation_best)})
        print(f'Generation {generation + 1:03d}/{ga_config.generations}: mean={np.mean(fitnesses):.6f}, max={generation_best_fitness:.6f}, best={generation_best}')
        if generation == ga_config.generations - 1:
            break
        elite_indices = np.argsort(fitnesses)[-ga_config.elite_size:][::-1]
        next_population = [copy.deepcopy(population[index]) for index in elite_indices]
        while len(next_population) < ga_config.population_size:
            parent_1 = tournament_select(population, fitnesses, ga_config.tournament_size)
            parent_2 = tournament_select(population, fitnesses, ga_config.tournament_size)
            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)
            if np.random.random() < ga_config.crossover_probability:
                child_1, child_2 = mate_ind(parent_1, parent_2, ga_config)
            if np.random.random() < ga_config.mutation_probability:
                child_1 = mutate_ind(child_1, ga_config)
            if np.random.random() < ga_config.mutation_probability:
                child_2 = mutate_ind(child_2, ga_config)
            next_population.append(repair_individual(child_1, ga_config))
            if len(next_population) < ga_config.population_size:
                next_population.append(repair_individual(child_2, ga_config))
        population = next_population
    if global_best is None:
        raise RuntimeError('No valid best chromosome was found.')
    return (global_best, float(global_best_fitness), pd.DataFrame(history_rows))

def evaluate_best_solution(images: Mapping[str, np.ndarray], references: Mapping[str, ImageReference], individual: Sequence[int], match_radius_px: float, mask_config: ReferenceMaskConfig, ga_config: GAConfig, area_soft_limit_px: float=AREA_SOFT_LIMIT_PX) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(references):
        score, match_result, objects, object_index_map = score_single_reference(images[path], individual, references[path], match_radius_px, mask_config, ga_config, area_soft_limit_px)
        n_points = int(len(references[path].points))
        n_objects = int(len(objects))
        n_matched = int(len(match_result['matches']))
        rows.append({'path': path, 'condition': extract_condition(path), 'image_score': score, 'n_reference_points': n_points, 'n_detected_objects': n_objects, 'n_matched': n_matched, 'matched_quality_sum': float(match_result['matched_quality_sum']), 'matched_distance_quality_sum': float(match_result['matched_distance_quality_sum']), 'mean_matched_area_weight': float(match_result['mean_matched_area_weight']), 'mean_matched_object_area_px': float(match_result['mean_matched_object_area_px']), 'mean_detected_object_area_px': float(np.mean([obj['area'] for obj in objects])) if objects else np.nan, 'max_detected_object_area_px': int(max((obj['area'] for obj in objects))) if objects else 0, 'n_objects_above_area_soft_limit': int(sum((obj['area'] > area_soft_limit_px for obj in objects))), 'area_soft_limit_px': float(area_soft_limit_px), 'match_rate_points': n_matched / n_points if n_points > 0 else np.nan, 'match_rate_objects': n_matched / n_objects if n_objects > 0 else np.nan, 'n_nuclei': int(references[path].n_nuclei), 'n_cells': int(references[path].n_cells)})
    return pd.DataFrame(rows)

def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_optimization(data_root: str | Path, output_root: str | Path, seed: int = 42,
                     population_size: int = 100, generations: int = 100, workers: int = 4,
                     match_radius_px: float = MATCH_RADIUS_PX,
                     area_soft_limit_px: float = AREA_SOFT_LIMIT_PX) -> tuple[list[int], float, pd.DataFrame]:
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = collect_image_paths(data_root, EXCLUDED_FOLDERS)
    if not paths:
        raise FileNotFoundError(f"No eligible *_M.tif images found under {data_root}")
    images = preload_images(paths)
    references, failures = precompute_references(
        images, POINT_CONFIG_FINAL, NUCLEUS_CONFIG_FINAL, CELL_CONFIG_FINAL,
        REFERENCE_MASK_CONFIG, max_workers=workers,
    )
    failures.to_csv(output_root / "reference_failures.csv", index=False)
    if not failures.empty:
        raise RuntimeError("Reference generation failed for one or more images; see reference_failures.csv")
    validate_reference_images(references)
    reference_summary_dataframe(references).to_csv(output_root / "fixed_reference_summary.csv", index=False)
    ga_config = replace(GA_CONFIG, population_size=population_size, generations=generations,
                        seed=seed, max_workers=workers)
    best, best_fitness, history = run_ga(
        images, references, match_radius_px, REFERENCE_MASK_CONFIG, ga_config, area_soft_limit_px
    )
    history.to_csv(output_root / "ga_history.csv", index=False)
    results = evaluate_best_solution(
        images, references, best, match_radius_px, REFERENCE_MASK_CONFIG, ga_config, area_soft_limit_px
    )
    results.to_csv(output_root / "best_solution_by_image.csv", index=False)
    _save_json(output_root / "best_chromosome.json", {
        "chromosome": [int(x) for x in best],
        "fitness": float(best_fitness),
        "seed": int(seed),
        "population_size": int(population_size),
        "generations": int(generations),
        "match_radius_px": float(match_radius_px),
        "area_soft_limit_px": float(area_soft_limit_px),
        "point_config": asdict(POINT_CONFIG_FINAL),
        "nucleus_config": asdict(NUCLEUS_CONFIG_FINAL),
        "cell_config": asdict(CELL_CONFIG_FINAL),
        "mask_config": asdict(REFERENCE_MASK_CONFIG),
        "note": "Soft/Stiff labels are not used in the GA fitness."
    })
    return best, float(best_fitness), results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AdhGen optimization on merged fluorescence images.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("adhgen_results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--match-radius", type=float, default=MATCH_RADIUS_PX)
    p.add_argument("--area-soft-limit-px", type=float, default=AREA_SOFT_LIMIT_PX)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run_optimization(a.data_root, a.output_root, a.seed, a.population_size, a.generations,
                     a.workers, a.match_radius, a.area_soft_limit_px)


if __name__ == "__main__":
    main()
