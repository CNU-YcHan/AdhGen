#!/usr/bin/env python3
"""
Create a green-only point-marker comparison image for the AdhGen framework.

Comparison definition
---------------------
1. AdhGen candidates:
   Rmin <= R <= Rmax
   Gmin <= G <= Gmax
   Bmin <= B <= Bmax
   Amin <= connected-component area <= Amax

2. Green point markers:
   Gmin <= G <= Gmax
   Amin <= connected-component area <= Amax

The green marker extraction deliberately ignores the red and blue channels.
The same G and Area genes from the AdhGen chromosome are used in both branches.

Visualization
-------------
- Yellow contours: AdhGen candidates detected with R + G + B + Area
- Green circles: green-only point-marker centroids
- Optional red circles: green-only markers that do not match an AdhGen candidate

Required in the same code directory
-----------------------------------
cell_instance_utils_dapi_all_cells.py

Example
-------
python 03_make_green_only_marker_comparison.py ^
  --input "<ADHGEN_ROOT>\merged_data\1\Stiff\Stiff_1_M.tif" ^
  --chromosome "20,255,80,255,0,100,30,60" ^
  --output "<ADHGEN_ROOT>\figures\green_marker_comparison"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import cell_instance_utils_dapi_all_cells as cell_utils


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SCALE_BAR_X_START_FRACTION = 0.78
SCALE_BAR_HEIGHT_PX = 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AdhGen RGB-Area candidates with green-only point markers "
            "using the same G and Area genes."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more merged images or folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory.",
    )

    parser.add_argument(
        "--chromosome",
        type=str,
        default=None,
        help='Eight values: "Rmin,Rmax,Gmin,Gmax,Bmin,Bmax,Amin,Amax"',
    )
    parser.add_argument("--r-min", type=int)
    parser.add_argument("--r-max", type=int)
    parser.add_argument("--g-min", type=int)
    parser.add_argument("--g-max", type=int)
    parser.add_argument("--b-min", type=int)
    parser.add_argument("--b-max", type=int)
    parser.add_argument("--a-min", type=int)
    parser.add_argument("--a-max", type=int)

    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Analyze the whole image without the DAPI/cell analysis mask.",
    )
    parser.add_argument(
        "--include-scale-bar",
        action="store_true",
        help="Do not exclude the lower-right scale-bar region.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Do not search subfolders.",
    )
    parser.add_argument("--max-images", type=int, default=None)

    parser.add_argument(
        "--background-dim",
        type=float,
        default=0.30,
        help="Brightness multiplier for the merged-image background. Default: 0.30",
    )
    parser.add_argument(
        "--candidate-thickness",
        type=int,
        default=2,
        help="Yellow AdhGen contour thickness. Default: 2",
    )
    parser.add_argument(
        "--marker-radius",
        type=int,
        default=4,
        help="Radius of green point markers. Default: 4",
    )
    parser.add_argument(
        "--match-radius",
        type=int,
        default=3,
        help=(
            "Pixel distance allowed when deciding whether a green-only marker "
            "matches an AdhGen candidate. Default: 3"
        ),
    )
    parser.add_argument(
        "--show-unmatched",
        action="store_true",
        help=(
            "Draw unmatched green-only markers in red. "
            "By default, every green-only marker is shown in green."
        ),
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Write candidate, marker, and matched counts in the image.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save the AdhGen and green-only binary masks.",
    )

    return parser.parse_args()


def parse_chromosome(args: argparse.Namespace) -> List[int]:
    if args.chromosome:
        tokens = [
            token
            for token in re.split(r"[,;\s]+", args.chromosome.strip())
            if token
        ]
        values = [int(token) for token in tokens]
    else:
        values = [
            args.r_min,
            args.r_max,
            args.g_min,
            args.g_max,
            args.b_min,
            args.b_max,
            args.a_min,
            args.a_max,
        ]
        if any(value is None for value in values):
            raise ValueError(
                "Use --chromosome or provide all eight R/G/B/Area min/max values."
            )

    if len(values) != 8:
        raise ValueError("The chromosome must contain exactly eight integers.")

    r_min, r_max, g_min, g_max, b_min, b_max, a_min, a_max = values

    if any(value < 0 or value > 255 for value in values[:6]):
        raise ValueError("RGB values must be between 0 and 255.")
    if r_min > r_max or g_min > g_max or b_min > b_max:
        raise ValueError("Each RGB minimum must not exceed its maximum.")
    if a_min < 1 or a_min > a_max:
        raise ValueError("Area must satisfy 1 <= Amin <= Amax.")

    return [int(value) for value in values]


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.background_dim <= 1.0:
        raise ValueError("--background-dim must be between 0 and 1.")
    if args.candidate_thickness < 1:
        raise ValueError("--candidate-thickness must be at least 1.")
    if args.marker_radius < 1:
        raise ValueError("--marker-radius must be at least 1.")
    if args.match_radius < 0:
        raise ValueError("--match-radius must be non-negative.")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be at least 1.")


def natural_key(path: Path) -> List[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def collect_images(
    input_paths: Sequence[Path],
    recursive: bool,
    max_images: Optional[int],
) -> List[Path]:
    collected: List[Path] = []

    for raw_path in input_paths:
        input_path = raw_path.expanduser().resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported extension: {input_path.suffix}")
            collected.append(input_path)
            continue

        iterator: Iterable[Path] = (
            input_path.rglob("*") if recursive else input_path.glob("*")
        )
        collected.extend(
            path.resolve()
            for path in iterator
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and "_green_marker_comparison" not in path.stem.lower()
        )

    paths = sorted(set(collected), key=natural_key)

    if max_images is not None:
        paths = paths[:max_images]

    if not paths:
        raise ValueError("No supported images were found.")

    return paths


def common_input_root(input_paths: Sequence[Path]) -> Path:
    roots: List[str] = []
    for raw_path in input_paths:
        path = raw_path.expanduser().resolve()
        roots.append(str(path.parent if path.is_file() else path))
    return Path(os.path.commonpath(roots)).resolve()


def output_subdir(image_path: Path, input_root: Path) -> Path:
    try:
        return image_path.parent.relative_to(input_root)
    except ValueError:
        return Path(".")


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


def build_analysis_mask(
    image_bgr: np.ndarray,
    full_image: bool,
    include_scale_bar: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Use the same cell/nucleus analysis mask logic as the previous AdhGen
    visualization code.
    """
    height, width = image_bgr.shape[:2]
    scale_bar_mask = make_scale_bar_mask(
        image_bgr.shape,
        include_scale_bar,
    )

    if full_image:
        analysis_mask = np.ones((height, width), dtype=bool)
        analysis_mask &= ~scale_bar_mask
        empty = np.zeros((height, width), dtype=bool)
        return analysis_mask, empty, empty, 0, 0

    nucleus_labels, nucleus_exclusion, nuclei, _ = cell_utils.segment_nuclei(
        image_bgr,
        cell_utils.NUCLEUS_CONFIG,
    )
    cell_labels, cells, _ = cell_utils.segment_cell_instances(
        image_bgr,
        nucleus_labels,
        nucleus_exclusion,
        cell_utils.CELL_INSTANCE_CONFIG,
    )

    cell_union = cell_labels > 0
    analysis_mask = cell_union & ~nucleus_exclusion & ~scale_bar_mask

    return (
        analysis_mask.astype(bool),
        cell_union.astype(bool),
        nucleus_exclusion.astype(bool),
        int(len(nuclei)),
        int(len(cells)),
    )


def area_filter_mask(
    pixel_mask: np.ndarray,
    a_min: int,
    a_max: int,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Apply one identical connected-component area rule to either branch.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pixel_mask.astype(np.uint8),
        connectivity=8,
    )

    accepted_labels = np.zeros_like(labels, dtype=np.int32)
    objects: List[dict] = []
    next_id = 1

    for component_id in range(1, n_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if not (a_min <= area <= a_max):
            continue

        accepted_labels[labels == component_id] = next_id
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

    return accepted_labels, objects


def detect_adhgen_candidates(
    image_bgr: np.ndarray,
    analysis_mask: np.ndarray,
    chromosome: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    AdhGen branch: R + G + B constraints followed by the Area constraint.
    """
    r_min, r_max, g_min, g_max, b_min, b_max, a_min, a_max = chromosome

    red = image_bgr[:, :, 2]
    green = image_bgr[:, :, 1]
    blue = image_bgr[:, :, 0]

    rgb_mask = (
        (red >= r_min)
        & (red <= r_max)
        & (green >= g_min)
        & (green <= g_max)
        & (blue >= b_min)
        & (blue <= b_max)
        & analysis_mask
    )

    accepted_labels, objects = area_filter_mask(
        rgb_mask,
        a_min,
        a_max,
    )
    return rgb_mask.astype(bool), accepted_labels, objects


def detect_green_only_markers(
    image_bgr: np.ndarray,
    analysis_mask: np.ndarray,
    chromosome: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Green-marker branch: use only Gmin/Gmax and the same Amin/Amax.

    Red and blue channels are intentionally ignored.
    """
    _, _, g_min, g_max, _, _, a_min, a_max = chromosome
    green = image_bgr[:, :, 1]

    green_mask = (
        (green >= g_min)
        & (green <= g_max)
        & analysis_mask
    )

    accepted_labels, objects = area_filter_mask(
        green_mask,
        a_min,
        a_max,
    )
    return green_mask.astype(bool), accepted_labels, objects


def label_boundary(label_map: np.ndarray) -> np.ndarray:
    edge = np.zeros(label_map.shape, dtype=bool)
    edge[1:, :] |= label_map[1:, :] != label_map[:-1, :]
    edge[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    edge[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    edge[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    return edge & (label_map > 0)


def match_green_markers(
    adhgen_labels: np.ndarray,
    green_objects: Sequence[dict],
    match_radius: int,
) -> List[bool]:
    """
    Match each green-only marker centroid to the AdhGen accepted-object mask.
    """
    adhgen_mask = (adhgen_labels > 0).astype(np.uint8)

    if match_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * match_radius + 1, 2 * match_radius + 1),
        )
        comparison_mask = cv2.dilate(adhgen_mask, kernel)
    else:
        comparison_mask = adhgen_mask

    matched: List[bool] = []

    for obj in green_objects:
        x = int(round(obj["centroid_x"]))
        y = int(round(obj["centroid_y"]))
        x = max(0, min(x, comparison_mask.shape[1] - 1))
        y = max(0, min(y, comparison_mask.shape[0] - 1))
        matched.append(bool(comparison_mask[y, x]))

    return matched


def make_comparison_overlay(
    image_bgr: np.ndarray,
    analysis_mask: np.ndarray,
    adhgen_labels: np.ndarray,
    green_objects: Sequence[dict],
    marker_matches: Sequence[bool],
    background_dim: float,
    candidate_thickness: int,
    marker_radius: int,
    show_unmatched: bool,
    show_counts: bool,
) -> np.ndarray:
    """
    Create the publication-style comparison image.

    Yellow = AdhGen RGB-Area candidates
    Green  = green-only point markers
    Red    = optional unmatched green-only markers
    """
    overlay = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)

    overlay *= float(background_dim)
    overlay[~analysis_mask] *= 0.35
    result = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw accepted AdhGen object boundaries in yellow.
    candidate_edge = label_boundary(adhgen_labels)
    result[candidate_edge] = [255, 255, 0]

    # Increase contour visibility if requested.
    if candidate_thickness > 1:
        edge_u8 = candidate_edge.astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (candidate_thickness, candidate_thickness),
        )
        thick_edge = cv2.dilate(edge_u8, kernel) > 0
        result[thick_edge] = [255, 255, 0]

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    matched_count = 0
    for obj, is_matched in zip(green_objects, marker_matches):
        x = int(round(obj["centroid_x"]))
        y = int(round(obj["centroid_y"]))

        if is_matched:
            matched_count += 1

        if show_unmatched and not is_matched:
            color_bgr = (0, 0, 255)      # red
        else:
            color_bgr = (0, 255, 0)      # green

        # Black outer ring improves visibility on yellow/green fluorescence.
        cv2.circle(
            result_bgr,
            (x, y),
            marker_radius + 1,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            result_bgr,
            (x, y),
            marker_radius,
            color_bgr,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            result_bgr,
            (x, y),
            1,
            color_bgr,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    if show_counts:
        candidate_count = int(adhgen_labels.max())
        marker_count = len(green_objects)
        lines = [
            f"RGBA candidates: {candidate_count}",
            f"Green-only markers: {marker_count}",
            f"Matched markers: {matched_count}",
        ]

        y = 24
        for text in lines:
            cv2.putText(
                result_bgr,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                result_bgr,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 22

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise IOError(f"Failed to save image: {path}")


def process_image(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    chromosome: Sequence[int],
    args: argparse.Namespace,
) -> dict:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("OpenCV could not read the image.")

    (
        analysis_mask,
        _cell_union,
        _nucleus_exclusion,
        n_nuclei,
        n_cells,
    ) = build_analysis_mask(
        image_bgr,
        full_image=args.full_image,
        include_scale_bar=args.include_scale_bar,
    )

    rgba_pixel_mask, adhgen_labels, adhgen_objects = detect_adhgen_candidates(
        image_bgr,
        analysis_mask,
        chromosome,
    )

    green_pixel_mask, green_labels, green_objects = detect_green_only_markers(
        image_bgr,
        analysis_mask,
        chromosome,
    )

    marker_matches = match_green_markers(
        adhgen_labels,
        green_objects,
        args.match_radius,
    )

    overlay_rgb = make_comparison_overlay(
        image_bgr=image_bgr,
        analysis_mask=analysis_mask,
        adhgen_labels=adhgen_labels,
        green_objects=green_objects,
        marker_matches=marker_matches,
        background_dim=args.background_dim,
        candidate_thickness=args.candidate_thickness,
        marker_radius=args.marker_radius,
        show_unmatched=args.show_unmatched,
        show_counts=args.show_counts,
    )

    out_dir = output_root / output_subdir(image_path, input_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"{image_path.stem}_green_marker_comparison.png"
    save_rgb(output_path, overlay_rgb)

    if args.save_masks:
        cv2.imwrite(
            str(out_dir / f"{image_path.stem}_adhgen_mask.png"),
            ((adhgen_labels > 0) * 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(out_dir / f"{image_path.stem}_green_only_marker_mask.png"),
            ((green_labels > 0) * 255).astype(np.uint8),
        )

    matched_count = int(sum(marker_matches))
    green_count = len(green_objects)

    return {
        "path": str(image_path),
        "output": str(output_path),
        "n_nuclei": n_nuclei,
        "n_cells": n_cells,
        "adhgen_candidates": len(adhgen_objects),
        "green_only_markers": green_count,
        "matched_green_markers": matched_count,
        "marker_match_fraction": (
            float(matched_count / green_count) if green_count else 0.0
        ),
        "rgba_candidate_pixels_before_area": int(np.count_nonzero(rgba_pixel_mask)),
        "green_pixels_before_area": int(np.count_nonzero(green_pixel_mask)),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    chromosome = parse_chromosome(args)

    input_paths = [
        path.expanduser().resolve()
        for path in args.input
    ]
    input_root = common_input_root(input_paths)

    image_paths = collect_images(
        input_paths,
        recursive=not args.non_recursive,
        max_images=args.max_images,
    )

    if args.output is not None:
        output_root = args.output.expanduser().resolve()
    else:
        label = (
            f"R{chromosome[0]}-{chromosome[1]}_"
            f"G{chromosome[2]}-{chromosome[3]}_"
            f"B{chromosome[4]}-{chromosome[5]}_"
            f"A{chromosome[6]}-{chromosome[7]}"
        )
        output_root = input_root / f"green_marker_comparison_{label}"

    output_root.mkdir(parents=True, exist_ok=True)

    print("[AdhGen vs green-only marker comparison]")
    print("  chromosome:", chromosome)
    print("  AdhGen branch: R + G + B + Area")
    print("  marker branch: G + Area only")
    print("  output:", output_root)
    print("  images:", len(image_paths))

    failures = 0

    for index, image_path in enumerate(image_paths, start=1):
        try:
            result = process_image(
                image_path,
                input_root,
                output_root,
                chromosome,
                args,
            )
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"RGBA={result['adhgen_candidates']}, "
                f"G-only={result['green_only_markers']}, "
                f"matched={result['matched_green_markers']}, "
                f"fraction={result['marker_match_fraction']:.3f}"
            )
        except Exception as exc:
            failures += 1
            print(
                f"[FAILED {index}/{len(image_paths)}] {image_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print("\n[Completed]")
    print("  successful:", len(image_paths) - failures)
    print("  failed:", failures)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
