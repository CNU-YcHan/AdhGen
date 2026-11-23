#!/usr/bin/env python3
"""
Red(F-actin) + Green(FA marker)을 이용한 cell area 추정 시각화.

실행:
python visualize_red_green_cell_area.py \
    --image "/home/schbigdata/Desktop/AdhGen/merged_data_405_488_actin/1/Soft/Soft_1_M.tif"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import cell_instance_utils_dapi_all_cells as cell_utils


def boundaries(label_map: np.ndarray) -> np.ndarray:
    """Cell instance 경계만 추출."""
    edge = np.zeros(label_map.shape, dtype=bool)
    edge[1:] |= label_map[1:] != label_map[:-1]
    edge[:-1] |= label_map[:-1] != label_map[1:]
    edge[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    edge[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    return edge & (label_map > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # OpenCV BGR: Blue=DAPI, Green=FA marker, Red=F-actin
    red = image_bgr[:, :, 2]
    green = image_bgr[:, :, 1]

    cfg = cell_utils.CELL_INSTANCE_CONFIG

    red_norm = cell_utils.percentile_normalize(
        red, cfg.low_percentile, cfg.high_percentile
    )
    green_norm = cell_utils.percentile_normalize(
        green, cfg.low_percentile, cfg.high_percentile
    )

    # 실제 cell mask 생성에서 사용하는 결합 방식
    combined = np.maximum(red_norm, green_norm)

    if cfg.blur_sigma > 0:
        combined_smooth = cv2.GaussianBlur(
            combined,
            (0, 0),
            sigmaX=cfg.blur_sigma,
            sigmaY=cfg.blur_sigma,
        )
    else:
        combined_smooth = combined

    # DAPI nucleus 검출
    nucleus_labels, nucleus_exclusion, nuclei, _ = cell_utils.segment_nuclei(
        image_bgr
    )

    # Red+Green 기반 coarse cell mask
    coarse_mask, _, coarse_info = cell_utils.build_coarse_cell_mask(
        image_bgr,
        nucleus_labels,
    )

    # DAPI nucleus-seeded watershed 기반 최종 cell instance
    cell_labels, cells, cell_info = cell_utils.segment_cell_instances(
        image_bgr,
        nucleus_labels,
        nucleus_exclusion,
    )

    # 최종 overlay
    overlay = image_rgb.copy()
    overlay[boundaries(cell_labels)] = [255, 255, 0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("1. Merged image")

    axes[0, 1].imshow(red, cmap="gray")
    axes[0, 1].set_title("2. Red: F-actin")

    axes[0, 2].imshow(green, cmap="gray")
    axes[0, 2].set_title("3. Green: FA marker")

    axes[1, 0].imshow(combined_smooth, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("4. max(Red, Green) + smoothing")

    axes[1, 1].imshow(coarse_mask, cmap="gray")
    axes[1, 1].set_title(
        "5. Coarse cell mask\n"
        f"threshold={coarse_info['cell_applied_threshold']:.3f}"
    )

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(
        "6. Final DAPI-seeded cell instances\n"
        f"cells={len(cells)}"
    )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(image_path.name)
    fig.tight_layout()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else image_path.with_name(
            f"{image_path.stem}_red_green_cell_area.png"
        )
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"Saved: {output_path}")
    print(f"Nuclei: {len(nuclei)}")
    print(f"Cells: {len(cells)}")
    print(
        "Cell area fraction: "
        f"{cell_info['total_cell_instance_area_fraction']:.4f}"
    )

    plt.show()


if __name__ == "__main__":
    main()
