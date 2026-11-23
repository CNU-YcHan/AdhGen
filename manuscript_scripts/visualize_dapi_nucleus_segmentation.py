#!/usr/bin/env python3
"""
Blue(DAPI) 기반 nucleus 검출 단계 시각화

실행 예:
python visualize_dapi_nucleus_segmentation.py \
    --image "/home/schbigdata/Desktop/AdhGen/merged_data_405_488_actin/1/Soft/Soft_1_M.tif"

결과:
- 화면에 6단계 시각화
- 입력 영상 옆에 *_dapi_nucleus_visualization.png 저장
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import cell_instance_utils_dapi_all_cells as cell_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdhGen의 Blue(DAPI) nucleus 검출 과정을 단계별로 시각화합니다."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Merged TIFF 경로 (*_M.tif)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="저장할 PNG 경로. 생략하면 입력 영상 옆에 자동 저장합니다.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="저장 이미지 DPI (기본값: 180)",
    )
    return parser.parse_args()


def label_boundaries(label_map: np.ndarray) -> np.ndarray:
    """Label map에서 instance 경계만 추출한다."""
    boundaries = np.zeros(label_map.shape, dtype=bool)

    boundaries[1:, :] |= label_map[1:, :] != label_map[:-1, :]
    boundaries[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    boundaries[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    boundaries[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]

    return boundaries & (label_map > 0)


def make_final_overlay(
    image_bgr: np.ndarray,
    nucleus_label_map: np.ndarray,
    nucleus_exclusion_mask: np.ndarray,
) -> np.ndarray:
    """
    최종 nucleus 경계와 nucleus+4 px exclusion 영역을 merged image 위에 표시한다.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.astype(np.float32).copy()

    # nucleus + margin 영역을 반투명하게 표시
    exclusion = nucleus_exclusion_mask.astype(bool)
    tint = np.zeros_like(overlay)
    tint[..., 0] = 255.0
    tint[..., 1] = 255.0
    overlay[exclusion] = 0.65 * overlay[exclusion] + 0.35 * tint[exclusion]

    # 최종 nucleus instance 경계
    boundaries = label_boundaries(nucleus_label_map)
    overlay[boundaries] = np.array([255.0, 255.0, 0.0])

    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {image_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV가 이미지를 읽지 못했습니다: {image_path}")

    config = cell_utils.NUCLEUS_CONFIG

    # 1) 정규화 + smoothing + threshold + morphology가 적용된 초기 binary
    (
        initial_binary,
        blue_response,
        otsu_value,
        applied_threshold,
    ) = cell_utils._initial_nucleus_binary(
        image_bgr,
        config,
    )

    # 2) 면적/solidity/eccentricity/minor-axis 조건을 통과한 후보
    accepted_mask, candidates = cell_utils._filter_initial_nucleus_components(
        initial_binary,
        config,
    )

    # 3) 큰 component 선택적 watershed 분리 + 최종 nucleus instance
    (
        nucleus_label_map,
        nucleus_exclusion_mask,
        nuclei,
        info,
    ) = cell_utils.segment_nuclei(
        image_bgr,
        config,
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    blue_raw = image_bgr[:, :, 0]
    final_overlay = make_final_overlay(
        image_bgr,
        nucleus_label_map,
        nucleus_exclusion_mask,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("1. Merged image")

    axes[0, 1].imshow(blue_raw, cmap="gray")
    axes[0, 1].set_title("2. Raw Blue (DAPI)")

    im = axes[0, 2].imshow(blue_response, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(
        "3. Normalized + Gaussian smoothed\n"
        f"Otsu={otsu_value:.3f}, applied={applied_threshold:.3f}"
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(initial_binary, cmap="gray")
    axes[1, 0].set_title(
        "4. Threshold + morphology\n"
        f"initial foreground={np.count_nonzero(initial_binary):,} px"
    )

    axes[1, 1].imshow(accepted_mask, cmap="gray")
    axes[1, 1].set_title(
        "5. Size/shape-filtered candidates\n"
        f"candidates={len(candidates)}"
    )

    axes[1, 2].imshow(final_overlay)
    axes[1, 2].set_title(
        "6. Final nuclei and exclusion margin\n"
        f"nuclei={len(nuclei)}, split components={info['n_split_nucleus_components']}"
    )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(image_path.name, fontsize=14)
    fig.tight_layout()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else image_path.with_name(
            f"{image_path.stem}_dapi_nucleus_visualization.png"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")

    print("\n[DAPI nucleus visualization]")
    print(f"  image: {image_path}")
    print(f"  output: {output_path}")
    print(f"  nuclei: {len(nuclei)}")
    print(f"  initial candidates: {len(candidates)}")
    print(f"  split components: {info['n_split_nucleus_components']}")
    print(f"  Otsu threshold: {otsu_value:.6f}")
    print(f"  applied threshold: {applied_threshold:.6f}")
    print(f"  nucleus area: {info['nucleus_union_area_px']:,} px")
    print(f"  exclusion area: {info['nucleus_exclusion_area_px']:,} px")

    plt.show()


if __name__ == "__main__":
    main()
