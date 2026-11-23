#!/usr/bin/env python3
"""
Input image/folder + [Rmin,Rmax,Gmin,Gmax,Bmin,Bmax,Amin,Amax]
=> AdhGen object visualization with numbered FA objects.

Required in the same code directory:
    cell_instance_utils_dapi_all_cells.py

Multiple image example:
    python visualize_adhgen_from_chromosome.py \
        --input image1.tif image2.tif image3.tif \
        --chromosome "13,253,104,255,0,88,11,98"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import cell_instance_utils_dapi_all_cells as cell_utils


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SCALE_BAR_X_START_FRACTION = 0.78
SCALE_BAR_HEIGHT_PX = 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R/G/B/Area chromosome으로 AdhGen 결과를 시각화합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="하나 이상의 merged image 또는 image folder",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="결과 폴더")
    parser.add_argument(
        "--chromosome",
        type=str,
        default=None,
        help='8개 값: "Rmin,Rmax,Gmin,Gmax,Bmin,Bmax,Amin,Amax"',
    )
    parser.add_argument("--r-min", type=int)
    parser.add_argument("--r-max", type=int)
    parser.add_argument("--g-min", type=int)
    parser.add_argument("--g-max", type=int)
    parser.add_argument("--b-min", type=int)
    parser.add_argument("--b-max", type=int)
    parser.add_argument("--a-min", type=int)
    parser.add_argument("--a-max", type=int)
    parser.add_argument("--non-recursive", action="store_true",
                        help="폴더 입력 시 하위 폴더를 탐색하지 않음")
    parser.add_argument("--full-image", action="store_true",
                        help="DAPI/cell analysis mask 없이 전체 영상에서 검출")
    parser.add_argument("--include-scale-bar", action="store_true",
                        help="오른쪽 아래 scale-bar 영역을 제외하지 않음")
    parser.add_argument(
        "--show-object-id",
        dest="show_object_id",
        action="store_true",
        default=True,
        help="검출된 모든 FA에 객체 번호 표시 (기본값: 표시)",
    )
    parser.add_argument(
        "--hide-object-id",
        dest="show_object_id",
        action="store_false",
        help="FA 객체 번호를 표시하지 않음",
    )
    parser.add_argument(
        "--id-font-scale",
        type=float,
        default=0.32,
        help="FA 번호 글자 크기 (기본값: 0.32)",
    )
    parser.add_argument("--save-label-map", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def parse_chromosome(args: argparse.Namespace) -> List[int]:
    if args.chromosome:
        tokens = [
            token for token in re.split(r"[,;\s]+", args.chromosome.strip())
            if token
        ]
        values = [int(token) for token in tokens]
    else:
        values = [
            args.r_min, args.r_max,
            args.g_min, args.g_max,
            args.b_min, args.b_max,
            args.a_min, args.a_max,
        ]
        if any(value is None for value in values):
            raise ValueError(
                "--chromosome 또는 R/G/B/A min/max 8개를 모두 입력하세요."
            )

    if len(values) != 8:
        raise ValueError("chromosome은 정확히 8개 정수여야 합니다.")

    r_min, r_max, g_min, g_max, b_min, b_max, a_min, a_max = values

    if any(value < 0 or value > 255 for value in values[:6]):
        raise ValueError("RGB 값은 0~255 범위여야 합니다.")
    if r_min > r_max or g_min > g_max or b_min > b_max:
        raise ValueError("각 RGB min은 max보다 클 수 없습니다.")
    if a_min < 1 or a_min > a_max:
        raise ValueError("Area는 1 <= A_min <= A_max여야 합니다.")

    return [int(value) for value in values]


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
    """여러 image/folder 입력을 하나의 중복 없는 image 목록으로 변환."""
    collected: List[Path] = []

    for raw_path in input_paths:
        input_path = raw_path.expanduser().resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"입력 경로가 없습니다: {input_path}")

        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"지원하지 않는 확장자: {input_path.suffix}"
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
            and "_adhgen_" not in path.stem.lower()
        )

    # 동일 이미지가 파일과 폴더 양쪽에서 지정돼도 한 번만 처리.
    paths = sorted(set(collected), key=natural_key)

    if max_images is not None:
        if max_images < 1:
            raise ValueError("--max-images는 1 이상이어야 합니다.")
        paths = paths[:max_images]

    if not paths:
        raise ValueError("입력 경로에서 이미지를 찾지 못했습니다.")

    return paths


def common_input_root(input_paths: Sequence[Path]) -> Path:
    """
    여러 입력의 공통 기준 경로를 계산한다.

    명시적으로 입력된 파일은 부모 폴더를, 폴더는 해당 폴더 자체를
    기준으로 사용한다. 출력 시 이 경로 아래의 상대 폴더 구조를 보존한다.
    """
    roots: List[str] = []

    for raw_path in input_paths:
        path = raw_path.expanduser().resolve()
        roots.append(str(path.parent if path.is_file() else path))

    return Path(os.path.commonpath(roots)).resolve()


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
    height, width = image_bgr.shape[:2]
    scale_bar_mask = make_scale_bar_mask(
        image_bgr.shape, include_scale_bar
    )

    if full_image:
        analysis_mask = np.ones((height, width), dtype=bool)
        analysis_mask &= ~scale_bar_mask
        empty = np.zeros((height, width), dtype=bool)
        return analysis_mask, empty, empty, 0, 0

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
        cell_union.astype(bool),
        nucleus_exclusion.astype(bool),
        int(len(nuclei)),
        int(len(cells)),
    )


def detect_objects(
    image_bgr: np.ndarray,
    analysis_mask: np.ndarray,
    chromosome: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
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

        accepted_labels[labels == component_id] = next_id

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[component_id]

        objects.append(
            {
                "object_id": next_id,
                "area_px": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "bbox_left": x,
                "bbox_top": y,
                "bbox_width": width,
                "bbox_height": height,
                "aspect_ratio": float(width / height) if height else 0.0,
            }
        )
        next_id += 1

    return candidate_mask.astype(bool), accepted_labels, objects


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8))
    return (mask_u8 > 0) & (eroded == 0)


def label_boundary(label_map: np.ndarray) -> np.ndarray:
    edge = np.zeros(label_map.shape, dtype=bool)
    edge[1:, :] |= label_map[1:, :] != label_map[:-1, :]
    edge[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    edge[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    edge[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    return edge & (label_map > 0)


def make_overlay(
    image_bgr: np.ndarray,
    analysis_mask: np.ndarray,
    cell_union: np.ndarray,
    accepted_labels: np.ndarray,
    objects: Sequence[dict],
    show_object_id: bool,
    id_font_scale: float,
) -> np.ndarray:
    overlay = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    # 분석 영역 밖은 어둡게 표시.
    overlay[~analysis_mask] *= 0.50

    # 검출된 FA 객체는 순수한 흰색으로 표시.
    object_mask = accepted_labels > 0
    overlay[object_mask] = [255, 255, 255]

    # Cell union 경계: cyan, FA object 경계: white.
    if np.any(cell_union):
        overlay[binary_boundary(cell_union)] = [0, 255, 255]
    overlay[label_boundary(accepted_labels)] = [255, 255, 255]

    result = np.clip(overlay, 0, 255).astype(np.uint8)

    if show_object_id:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.10, float(id_font_scale))
        text_thickness = 1
        outline_thickness = 3

        for obj in objects:
            text = str(obj["object_id"])
            cx = int(round(obj["centroid_x"]))
            cy = int(round(obj["centroid_y"]))

            # 텍스트 중심이 FA centroid에 오도록 위치 조정
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, text_thickness
            )
            x = int(round(cx - text_width / 2))
            y = int(round(cy + text_height / 2))

            # 영상 경계 밖으로 나가지 않도록 제한
            x = max(0, min(x, result.shape[1] - text_width - 1))
            y = max(text_height, min(y, result.shape[0] - baseline - 1))

            # 검은 외곽선
            cv2.putText(
                result, text, (x, y), font, font_scale,
                (0, 0, 0), outline_thickness, cv2.LINE_AA,
            )

            # 노란색 번호
            cv2.putText(
                result, text, (x, y), font, font_scale,
                (255, 255, 0), text_thickness, cv2.LINE_AA,
            )

    return result


def colorize_labels(label_map: np.ndarray) -> np.ndarray:
    if int(label_map.max()) == 0:
        return np.zeros((*label_map.shape, 3), dtype=np.uint8)

    normalized = (
        label_map.astype(np.float32)
        / float(label_map.max())
        * 255.0
    ).astype(np.uint8)

    colored_bgr = cv2.applyColorMap(
        normalized, cv2.COLORMAP_TURBO
    )
    colored_bgr[label_map == 0] = 0
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def save_panel(
    image_path: Path,
    image_bgr: np.ndarray,
    analysis_mask: np.ndarray,
    candidate_mask: np.ndarray,
    accepted_labels: np.ndarray,
    overlay_rgb: np.ndarray,
    chromosome: Sequence[int],
    n_nuclei: int,
    n_cells: int,
    objects: Sequence[dict],
    output_path: Path,
    dpi: int,
) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fa_per_cell = (
        float(len(objects) / n_cells) if n_cells > 0 else float("nan")
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("1. Merged image")

    axes[0, 1].imshow(analysis_mask, cmap="gray")
    axes[0, 1].set_title("2. Analysis mask")

    axes[0, 2].imshow(candidate_mask, cmap="gray")
    axes[0, 2].set_title("3. RGB-range candidate pixels")

    axes[1, 0].imshow(accepted_labels > 0, cmap="gray")
    axes[1, 0].set_title(
        f"4. Area-filtered mask (N={len(objects)})"
    )

    axes[1, 1].imshow(colorize_labels(accepted_labels))
    axes[1, 1].set_title("5. Object labels")

    axes[1, 2].imshow(overlay_rgb)
    axes[1, 2].set_title(
        "6. Final AdhGen result\n"
        f"nuclei={n_nuclei}, cells={n_cells}, "
        f"FA={len(objects)}, FA/cell={fa_per_cell:.2f}"
    )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"{image_path.name}\nChromosome={list(chromosome)}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def output_subdir(image_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(".")
    try:
        return image_path.parent.relative_to(input_path)
    except ValueError:
        return Path(".")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
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

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_image(
    image_path: Path,
    input_path: Path,
    output_root: Path,
    chromosome: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[dict, List[dict]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("OpenCV가 이미지를 읽지 못했습니다.")

    (
        analysis_mask,
        cell_union,
        nucleus_exclusion,
        n_nuclei,
        n_cells,
    ) = build_analysis_mask(
        image_bgr,
        full_image=args.full_image,
        include_scale_bar=args.include_scale_bar,
    )

    candidate_mask, accepted_labels, objects = detect_objects(
        image_bgr,
        analysis_mask,
        chromosome,
    )

    overlay_rgb = make_overlay(
        image_bgr,
        analysis_mask,
        cell_union,
        accepted_labels,
        objects,
        show_object_id=args.show_object_id,
        id_font_scale=args.id_font_scale,
    )

    out_dir = output_root / output_subdir(image_path, input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = out_dir / f"{image_path.stem}_adhgen_overlay.png"
    panel_path = out_dir / f"{image_path.stem}_adhgen_panel.png"

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )
    save_panel(
        image_path=image_path,
        image_bgr=image_bgr,
        analysis_mask=analysis_mask,
        candidate_mask=candidate_mask,
        accepted_labels=accepted_labels,
        overlay_rgb=overlay_rgb,
        chromosome=chromosome,
        n_nuclei=n_nuclei,
        n_cells=n_cells,
        objects=objects,
        output_path=panel_path,
        dpi=args.dpi,
    )

    label_map_path = ""
    if args.save_label_map:
        label_path = out_dir / f"{image_path.stem}_adhgen_labels.tif"
        if int(accepted_labels.max()) > np.iinfo(np.uint16).max:
            raise ValueError("객체 수가 uint16 범위를 초과했습니다.")
        cv2.imwrite(str(label_path), accepted_labels.astype(np.uint16))
        label_map_path = str(label_path)

    n_objects = len(objects)
    fa_per_cell = (
        float(n_objects / n_cells) if n_cells > 0 else float("nan")
    )
    r_min, r_max, g_min, g_max, b_min, b_max, a_min, a_max = chromosome

    summary = {
        "path": str(image_path),
        "n_nuclei": n_nuclei,
        "n_cells": n_cells,
        "n_detected_objects": n_objects,
        "mean_detected_fa_count_per_cell": fa_per_cell,
        "analysis_area_px": int(np.count_nonzero(analysis_mask)),
        "cell_union_area_px": int(np.count_nonzero(cell_union)),
        "nucleus_exclusion_area_px": int(
            np.count_nonzero(nucleus_exclusion)
        ),
        "rgb_candidate_area_px": int(np.count_nonzero(candidate_mask)),
        "detected_object_area_px": int(
            np.count_nonzero(accepted_labels)
        ),
        "mean_detected_object_area_px": (
            float(np.mean([obj["area_px"] for obj in objects]))
            if objects else 0.0
        ),
        "overlay_path": str(overlay_path),
        "panel_path": str(panel_path),
        "label_map_path": label_map_path,
        "r_min": r_min, "r_max": r_max,
        "g_min": g_min, "g_max": g_max,
        "b_min": b_min, "b_max": b_max,
        "a_min": a_min, "a_max": a_max,
    }

    object_rows = []
    for obj in objects:
        object_rows.append(
            {
                "path": str(image_path),
                **obj,
                "r_min": r_min, "r_max": r_max,
                "g_min": g_min, "g_max": g_max,
                "b_min": b_min, "b_max": b_max,
                "a_min": a_min, "a_max": a_max,
            }
        )

    return summary, object_rows


def main() -> None:
    args = parse_args()
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

    if args.output:
        output_root = args.output.expanduser().resolve()
    else:
        label = (
            f"R{chromosome[0]}-{chromosome[1]}_"
            f"G{chromosome[2]}-{chromosome[3]}_"
            f"B{chromosome[4]}-{chromosome[5]}_"
            f"A{chromosome[6]}-{chromosome[7]}"
        )

        if len(input_paths) == 1:
            only_input = input_paths[0]
            base = only_input.parent
            stem = only_input.stem if only_input.is_file() else only_input.name
            output_root = base / f"{stem}_adhgen_{label}"
        else:
            output_root = input_root / f"adhgen_multi_input_{label}"

    output_root.mkdir(parents=True, exist_ok=True)

    summaries: List[dict] = []
    object_rows: List[dict] = []
    failures: List[dict] = []

    print("[AdhGen visualization]")
    print("  inputs:")
    for input_path in input_paths:
        print("   -", input_path)
    print("  common input root:", input_root)
    print("  output:", output_root)
    print("  chromosome:", chromosome)
    print("  images:", len(image_paths))

    for index, image_path in enumerate(image_paths, start=1):
        try:
            summary, rows = process_image(
                image_path,
                input_root,
                output_root,
                chromosome,
                args,
            )
            summaries.append(summary)
            object_rows.extend(rows)
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"cells={summary['n_cells']}, "
                f"FA={summary['n_detected_objects']}, "
                f"FA/cell={summary['mean_detected_fa_count_per_cell']:.3f}"
            )
        except Exception as exc:
            failures.append(
                {
                    "path": str(image_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"[FAILED {index}/{len(image_paths)}] {image_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    write_csv(output_root / "adhgen_image_summary.csv", summaries)
    write_csv(output_root / "adhgen_object_details.csv", object_rows)
    write_csv(output_root / "adhgen_failures.csv", failures)

    print("\n[Completed]")
    print("  successful:", len(summaries))
    print("  failed:", len(failures))
    print("  total objects:", len(object_rows))
    print("  summary:", output_root / "adhgen_image_summary.csv")


if __name__ == "__main__":
    main()
