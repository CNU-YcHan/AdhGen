from __future__ import annotations
from dataclasses import dataclass
import inspect
from typing import List, Tuple
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, dilation, opening, disk, h_maxima, remove_small_objects
from skimage.segmentation import watershed

UTILITY_VERSION = "1.0.0"

@dataclass(frozen=True)
class NucleusConfig:
    low_percentile: float = 1.0
    high_percentile: float = 99.8
    blur_sigma: float = 1.5
    otsu_scale: float = 0.85
    min_normalized_threshold: float = 0.1
    open_radius_px: int = 2
    close_radius_px: int = 3
    min_nucleus_area_px: int = 300
    max_nucleus_area_px: int = 40000
    min_solidity: float = 0.55
    max_eccentricity: float = 0.995
    min_minor_axis_length_px: float = 8.0
    split_touching_nuclei: bool = True
    split_area_factor: float = 1.8
    h_maxima_height_px: float = 2.0
    max_split_markers_per_component: int = 12
    point_exclusion_margin_px: int = 4

@dataclass(frozen=True)
class CellInstanceConfig:
    low_percentile: float = 1.0
    high_percentile: float = 99.5
    blur_sigma: float = 3.0
    otsu_scale: float = 0.55
    min_normalized_threshold: float = 0.03
    close_radius_px: int = 5
    close_iterations: int = 2
    open_radius_px: int = 1
    open_iterations: int = 1
    min_coarse_component_area_px: int = 1000
    nucleus_support_margin_px: int = 3
    cell_margin_px: int = 4
    min_cell_instance_area_px: int = 1000
    watershed_compactness: float = 0.001
    exclude_border_cells_from_primary: bool = False

NUCLEUS_CONFIG = NucleusConfig()
CELL_INSTANCE_CONFIG = CellInstanceConfig()

def percentile_normalize(channel: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
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
    return bool(np.any(binary[0, :]) or np.any(binary[-1, :]) or np.any(binary[:, 0]) or np.any(binary[:, -1]))

def _safe_threshold_otsu(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(np.max(values)) <= float(np.min(values)):
        return 1.0
    try:
        return float(threshold_otsu(values))
    except ValueError:
        return 1.0

def _remove_objects_smaller_than(binary: np.ndarray, minimum_size: int) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    minimum_size = max(1, int(minimum_size))
    parameters = inspect.signature(remove_small_objects).parameters
    if 'max_size' in parameters:
        return remove_small_objects(binary, max_size=minimum_size - 1).astype(bool)
    return remove_small_objects(binary, min_size=minimum_size).astype(bool)

def _region_axis_minor_length(region) -> float:
    for attribute in ('axis_minor_length', 'minor_axis_length'):
        if hasattr(region, attribute):
            return float(getattr(region, attribute))
    return 0.0

def _region_axis_major_length(region) -> float:
    for attribute in ('axis_major_length', 'major_axis_length'):
        if hasattr(region, attribute):
            return float(getattr(region, attribute))
    return 0.0

def _region_shape_valid(region, config: NucleusConfig) -> bool:
    return bool(config.min_nucleus_area_px <= int(region.area) <= config.max_nucleus_area_px and float(region.solidity) >= config.min_solidity and (float(region.eccentricity) <= config.max_eccentricity) and (_region_axis_minor_length(region) >= config.min_minor_axis_length_px))

def _metadata_from_mask(mask: np.ndarray, object_id: int, prefix: str) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError('Cannot calculate metadata from an empty mask.')
    return {f'{prefix}_id': int(object_id), f'{prefix}_area_px': int(len(xs)), f'{prefix}_centroid_x': float(np.mean(xs)), f'{prefix}_centroid_y': float(np.mean(ys)), f'{prefix}_bbox_left': int(np.min(xs)), f'{prefix}_bbox_top': int(np.min(ys)), f'{prefix}_bbox_width': int(np.max(xs) - np.min(xs) + 1), f'{prefix}_bbox_height': int(np.max(ys) - np.min(ys) + 1), f'{prefix}_touches_border': touches_border(mask)}

def _initial_nucleus_binary(image: np.ndarray, config: NucleusConfig) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('A three-channel BGR image is required.')
    blue = image[:, :, 0]
    normalized = percentile_normalize(blue, config.low_percentile, config.high_percentile)
    if config.blur_sigma > 0:
        smoothed = cv2.GaussianBlur(normalized, ksize=(0, 0), sigmaX=float(config.blur_sigma), sigmaY=float(config.blur_sigma))
    else:
        smoothed = normalized
    otsu_value = _safe_threshold_otsu(smoothed)
    applied_threshold = max(float(config.min_normalized_threshold), otsu_value * float(config.otsu_scale))
    binary = smoothed >= applied_threshold
    if config.open_radius_px > 0:
        binary = opening(binary, footprint=disk(int(config.open_radius_px)))
    if config.close_radius_px > 0:
        binary = closing(binary, footprint=disk(int(config.close_radius_px)))
    binary = ndi.binary_fill_holes(binary)
    binary = _remove_objects_smaller_than(binary, config.min_nucleus_area_px)
    return (binary.astype(bool), smoothed.astype(np.float32), float(otsu_value), float(applied_threshold))

def _filter_initial_nucleus_components(binary: np.ndarray, config: NucleusConfig) -> Tuple[np.ndarray, List[dict]]:
    initial_labels = label(binary, connectivity=2)
    accepted_mask = np.zeros_like(binary, dtype=bool)
    candidates: List[dict] = []
    for region in regionprops(initial_labels):
        if not _region_shape_valid(region, config):
            continue
        region_mask = initial_labels == region.label
        accepted_mask |= region_mask
        candidates.append({'original_label': int(region.label), 'area': int(region.area), 'mask': region_mask, 'solidity': float(region.solidity), 'eccentricity': float(region.eccentricity), 'minor_axis_length': _region_axis_minor_length(region)})
    return (accepted_mask, candidates)

def _split_large_nucleus_component(component_mask: np.ndarray, config: NucleusConfig) -> np.ndarray:
    distance = ndi.distance_transform_edt(component_mask)
    maxima = h_maxima(distance, h=float(config.h_maxima_height_px))
    maxima &= component_mask
    markers, n_markers = ndi.label(maxima)
    if n_markers < 2:
        result = np.zeros_like(markers, dtype=np.int32)
        result[component_mask] = 1
        return result
    if n_markers > int(config.max_split_markers_per_component):
        result = np.zeros_like(markers, dtype=np.int32)
        result[component_mask] = 1
        return result
    return watershed(-distance, markers=markers, mask=component_mask, watershed_line=False).astype(np.int32)

def segment_nuclei(image: np.ndarray, config: NucleusConfig=NUCLEUS_CONFIG) -> Tuple[np.ndarray, np.ndarray, List[dict], dict]:
    initial_binary, blue_response, otsu_value, applied_threshold = _initial_nucleus_binary(image, config)
    accepted_mask, candidates = _filter_initial_nucleus_components(initial_binary, config)
    if not candidates:
        raise ValueError('No DAPI nucleus satisfied the configured shape criteria.')
    candidate_areas = np.asarray([candidate['area'] for candidate in candidates], dtype=np.float64)
    median_area = float(np.median(candidate_areas))
    final_label_map = np.zeros_like(initial_binary, dtype=np.int32)
    nuclei: List[dict] = []
    next_id = 1
    n_split_components = 0
    for candidate in candidates:
        component_mask = candidate['mask']
        should_split = bool(config.split_touching_nuclei and median_area > 0 and (candidate['area'] > float(config.split_area_factor) * median_area))
        if should_split:
            local_labels = _split_large_nucleus_component(component_mask, config)
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
            if not _region_shape_valid(region, config):
                continue
            final_label_map[submask] = next_id
            metadata = _metadata_from_mask(submask, object_id=next_id, prefix='nucleus')
            metadata.update({'nucleus_solidity': float(region.solidity), 'nucleus_eccentricity': float(region.eccentricity), 'nucleus_major_axis_length': _region_axis_major_length(region), 'nucleus_minor_axis_length': _region_axis_minor_length(region)})
            nuclei.append(metadata)
            next_id += 1
    if not nuclei:
        raise ValueError('No final nucleus instance was generated.')
    nucleus_union = final_label_map > 0
    if config.point_exclusion_margin_px > 0:
        nucleus_exclusion_mask = dilation(nucleus_union, footprint=disk(int(config.point_exclusion_margin_px)))
    else:
        nucleus_exclusion_mask = nucleus_union.copy()
    info = {'n_nuclei': int(len(nuclei)), 'n_initial_nucleus_candidates': int(len(candidates)), 'n_split_nucleus_components': int(n_split_components), 'nucleus_otsu_threshold': float(otsu_value), 'nucleus_applied_threshold': float(applied_threshold), 'median_initial_nucleus_area_px': float(median_area), 'nucleus_union_area_px': int(np.count_nonzero(nucleus_union)), 'nucleus_exclusion_area_px': int(np.count_nonzero(nucleus_exclusion_mask)), 'blue_response_max': float(np.max(blue_response))}
    return (final_label_map, nucleus_exclusion_mask.astype(bool), nuclei, info)

def build_coarse_cell_mask(image: np.ndarray, nucleus_label_map: np.ndarray, config: CellInstanceConfig=CELL_INSTANCE_CONFIG) -> Tuple[np.ndarray, np.ndarray, dict]:
    red = image[:, :, 2]
    green = image[:, :, 1]
    red_norm = percentile_normalize(red, config.low_percentile, config.high_percentile)
    green_norm = percentile_normalize(green, config.low_percentile, config.high_percentile)
    combined = np.maximum(red_norm, green_norm)
    if config.blur_sigma > 0:
        combined_smooth = cv2.GaussianBlur(combined, ksize=(0, 0), sigmaX=float(config.blur_sigma), sigmaY=float(config.blur_sigma))
    else:
        combined_smooth = combined
    otsu_value = _safe_threshold_otsu(combined_smooth)
    applied_threshold = max(float(config.min_normalized_threshold), otsu_value * float(config.otsu_scale))
    coarse_mask = combined_smooth >= applied_threshold
    for _ in range(max(0, int(config.close_iterations))):
        if config.close_radius_px > 0:
            coarse_mask = closing(coarse_mask, footprint=disk(int(config.close_radius_px)))
    for _ in range(max(0, int(config.open_iterations))):
        if config.open_radius_px > 0:
            coarse_mask = opening(coarse_mask, footprint=disk(int(config.open_radius_px)))
    coarse_mask = ndi.binary_fill_holes(coarse_mask)
    coarse_mask = _remove_objects_smaller_than(coarse_mask, config.min_coarse_component_area_px)
    nucleus_union = nucleus_label_map > 0
    if config.nucleus_support_margin_px > 0:
        nucleus_support = dilation(nucleus_union, footprint=disk(int(config.nucleus_support_margin_px)))
    else:
        nucleus_support = nucleus_union
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
            n_no_nucleus_components += 1
            continue
        valid_union |= component
    if not np.any(valid_union):
        raise ValueError('No coarse cell component containing a nucleus was found.')
    if config.cell_margin_px > 0:
        valid_union = dilation(valid_union, footprint=disk(int(config.cell_margin_px)))
    valid_union |= nucleus_union
    info = {'cell_otsu_threshold': float(otsu_value), 'cell_applied_threshold': float(applied_threshold), 'n_coarse_components': int(coarse_labels.max()), 'n_discarded_no_nucleus_components': int(n_no_nucleus_components), 'n_discarded_small_components': int(n_small_components), 'coarse_valid_area_px': int(np.count_nonzero(valid_union)), 'coarse_valid_area_fraction': float(np.mean(valid_union))}
    return (valid_union.astype(bool), combined_smooth.astype(np.float32), info)

def segment_cell_instances(image: np.ndarray, nucleus_label_map: np.ndarray, nucleus_exclusion_mask: np.ndarray, config: CellInstanceConfig=CELL_INSTANCE_CONFIG) -> Tuple[np.ndarray, List[dict], dict]:
    valid_union, combined_smooth, coarse_info = build_coarse_cell_mask(image, nucleus_label_map, config)
    markers = nucleus_label_map.astype(np.int32)
    if int(markers.max()) == 0:
        raise ValueError('No nucleus is available as a watershed marker.')
    elevation = sobel(combined_smooth)
    raw_watershed = watershed(elevation, markers=markers, mask=valid_union, compactness=float(config.watershed_compactness), watershed_line=True).astype(np.int32)
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
        metadata = _metadata_from_mask(cell_mask, object_id=next_cell_id, prefix='cell')
        metadata.update({'nucleus_id': int(nucleus_id), 'nucleus_area_px': int(np.count_nonzero(nucleus_mask))})
        cell_nuclear_exclusion = cell_mask & nucleus_exclusion_mask
        point_eligible_area_px = int(np.count_nonzero(cell_mask & ~nucleus_exclusion_mask))
        metadata['point_eligible_area_px'] = point_eligible_area_px
        border = bool(metadata['cell_touches_border'])
        metadata['analysis_eligible'] = True
        cells.append(metadata)
        next_cell_id += 1
    if not cells:
        raise ValueError('No final cell instance was generated.')
    total_cell_area = int(np.count_nonzero(final_label_map))
    analysis_cell_ids = [int(cell['cell_id']) for cell in cells if bool(cell['analysis_eligible'])]
    info = dict(coarse_info)
    info.update({'n_cells_all': int(len(cells)), 'n_cells_complete': int(len(analysis_cell_ids)), 'n_border_cells': int(sum((bool(cell['cell_touches_border']) for cell in cells))), 'n_discarded_small_watershed_instances': int(n_small_instances), 'total_cell_instance_area_px': total_cell_area, 'total_cell_instance_area_fraction': float(total_cell_area / final_label_map.size)})
    return (final_label_map, cells, info)

def dependency_versions() -> dict:
    import scipy
    import skimage
    return {'opencv': str(cv2.__version__), 'numpy': str(np.__version__), 'scipy': str(scipy.__version__), 'scikit_image': str(skimage.__version__)}
