#!/usr/bin/env python
r"""
Create a publication-ready Fig. 2b showing RGBA-based filtering.

Here, RGBA means:
    R = red intensity range
    G = green intensity range
    B = blue intensity range
    A = connected-component area range in pixels

Example
-------
python 02_make_fig2b_rgba_filtering.py ^
  --input "<ADHGEN_ROOT>\merged_data_405_488_actin\1\Stiff\Stiff_1_M.tif" ^
  --r-min 20 --r-max 255 ^
  --g-min 80 --g-max 255 ^
  --b-min 0 --b-max 100 ^
  --a-min 30 --a-max 60 ^
  --output "<ADHGEN_ROOT>\figures\Fig2b_RGBA_filtering.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply RGB intensity and connected-component area constraints to a "
            "merged fluorescence image and create a publication-ready Fig. 2b."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input merged RGB TIFF/PNG/JPG image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Fig2b_RGBA_filtering.png"),
        help="Output figure path. Default: Fig2b_RGBA_filtering.png",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=None,
        help=(
            "Optional path for saving only the filtered merged image. "
            "Example: filtered_merged.tif"
        ),
    )

    parser.add_argument("--r-min", type=float, required=True)
    parser.add_argument("--r-max", type=float, required=True)
    parser.add_argument("--g-min", type=float, required=True)
    parser.add_argument("--g-max", type=float, required=True)
    parser.add_argument("--b-min", type=float, required=True)
    parser.add_argument("--b-max", type=float, required=True)
    parser.add_argument(
        "--a-min",
        type=int,
        required=True,
        help="Minimum connected-component area in pixels.",
    )
    parser.add_argument(
        "--a-max",
        type=int,
        required=True,
        help="Maximum connected-component area in pixels.",
    )

    parser.add_argument(
        "--threshold-domain",
        choices=("8bit", "native"),
        default="8bit",
        help=(
            "Intensity domain used by the RGB thresholds. '8bit' applies values "
            "from 0 to 255 even when the TIFF is uint16. 'native' uses the image's "
            "original numeric values. Default: 8bit"
        ),
    )
    parser.add_argument(
        "--result-style",
        choices=("overlay", "black"),
        default="overlay",
        help=(
            "'overlay': detected regions over a dimmed merged image; "
            "'black': detected pixels on a black background. Default: overlay"
        ),
    )
    parser.add_argument(
        "--contour-thickness",
        type=int,
        default=2,
        help="Yellow contour thickness in display pixels. Default: 2",
    )
    parser.add_argument(
        "--dim-factor",
        type=float,
        default=0.22,
        help="Brightness of the original image in overlay mode, from 0 to 1.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure resolution. Default: 600 dpi",
    )
    parser.add_argument(
        "--panel-label",
        default="b",
        help="Panel label shown at the upper-left. Use an empty string to hide it.",
    )
    parser.add_argument(
        "--show-count",
        action="store_true",
        help="Show the number of retained connected components.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    ranges = [
        ("R", args.r_min, args.r_max),
        ("G", args.g_min, args.g_max),
        ("B", args.b_min, args.b_max),
        ("Area", args.a_min, args.a_max),
    ]
    for name, lower, upper in ranges:
        if lower > upper:
            raise ValueError(f"{name} minimum must not exceed its maximum.")

    if args.a_min < 0:
        raise ValueError("Area minimum must be non-negative.")
    if not 0.0 <= args.dim_factor <= 1.0:
        raise ValueError("--dim-factor must be between 0 and 1.")
    if args.contour_thickness < 1:
        raise ValueError("--contour-thickness must be at least 1.")


def read_rgb_image(path: Path) -> np.ndarray:
    """Read an RGB image without per-image min-max normalization."""
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
        image = np.asarray(image)
        image = np.squeeze(image)

        # Convert channel-first arrays such as (3, H, W) to (H, W, 3).
        if (
            image.ndim == 3
            and image.shape[0] in (3, 4)
            and image.shape[-1] not in (3, 4)
        ):
            image = np.moveaxis(image, 0, -1)

        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(
                f"Expected an RGB/RGBA TIFF, but received shape {image.shape}."
            )
        return np.ascontiguousarray(image[..., :3])

    # cv2 reads standard color images in BGR order.
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    if bgr.ndim != 3 or bgr.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected an RGB/RGBA image, but received shape {bgr.shape}."
        )
    return np.ascontiguousarray(cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB))


def dtype_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to uint8 without independently normalizing each image.

    uint16 is converted by dividing the full 16-bit range by 257.
    Floating-point values are interpreted as 0-1, 0-255, or 0-65535.
    """
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image.copy()

    if image.dtype == np.uint16:
        return np.rint(image.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)

    if np.issubdtype(image.dtype, np.unsignedinteger):
        maximum = float(np.iinfo(image.dtype).max)
        return np.rint(image.astype(np.float64) * (255.0 / maximum)).clip(
            0, 255
        ).astype(np.uint8)

    if np.issubdtype(image.dtype, np.signedinteger):
        maximum = float(np.iinfo(image.dtype).max)
        values = np.clip(image.astype(np.float64), 0, None)
        return np.rint(values * (255.0 / maximum)).clip(0, 255).astype(np.uint8)

    if np.issubdtype(image.dtype, np.floating):
        values = np.nan_to_num(
            image.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0
        )
        max_value = float(values.max(initial=0.0))
        if max_value <= 1.0:
            values *= 255.0
        elif max_value > 255.0:
            values /= 257.0
        return np.rint(values).clip(0, 255).astype(np.uint8)

    raise TypeError(f"Unsupported image dtype: {image.dtype}")


def make_initial_rgb_mask(
    rgb_values: np.ndarray,
    r_range: tuple[float, float],
    g_range: tuple[float, float],
    b_range: tuple[float, float],
) -> np.ndarray:
    """Return a binary mask satisfying all three RGB intensity constraints."""
    red = rgb_values[..., 0]
    green = rgb_values[..., 1]
    blue = rgb_values[..., 2]

    mask = (
        (red >= r_range[0])
        & (red <= r_range[1])
        & (green >= g_range[0])
        & (green <= g_range[1])
        & (blue >= b_range[0])
        & (blue <= b_range[1])
    )
    return mask.astype(np.uint8)


def apply_area_filter(
    binary_mask: np.ndarray,
    area_min: int,
    area_max: int,
) -> tuple[np.ndarray, list[int]]:
    """Retain only 8-connected components whose area is within the given range."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    retained_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    retained_areas: list[int] = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_min <= area <= area_max:
            retained_mask[labels == label_id] = 255
            retained_areas.append(area)

    return retained_mask, retained_areas


def make_filtered_visualization(
    display_rgb: np.ndarray,
    retained_mask: np.ndarray,
    result_style: str,
    dim_factor: float,
    contour_thickness: int,
) -> np.ndarray:
    """Create the right-side filtered image with yellow component contours."""
    mask_bool = retained_mask > 0

    if result_style == "black":
        result = np.zeros_like(display_rgb)
        result[mask_bool] = display_rgb[mask_bool]
    else:
        result = np.rint(display_rgb.astype(np.float32) * dim_factor).clip(
            0, 255
        ).astype(np.uint8)
        # Restore detected pixels at their original brightness.
        result[mask_bool] = display_rgb[mask_bool]

    contours, _ = cv2.findContours(
        retained_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # cv2 uses BGR color order. Yellow in BGR is (0, 255, 255).
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.drawContours(
        result_bgr,
        contours,
        contourIdx=-1,
        color=(0, 255, 255),
        thickness=contour_thickness,
        lineType=cv2.LINE_AA,
    )
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def save_filtered_image(path: Path, filtered_rgb: np.ndarray) -> None:
    """Save only the filtered visualization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        tifffile.imwrite(
            path,
            filtered_rgb,
            photometric="rgb",
            compression=None,
            metadata=None,
        )
        return

    bgr = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(suffix or ".png", bgr)
    if not success:
        raise ValueError(f"Could not encode filtered image as {suffix}.")
    encoded.tofile(path)


def add_center_diagram(
    ax: plt.Axes,
    args: argparse.Namespace,
    retained_count: int,
) -> None:
    """Draw the threshold box and arrows in the central panel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("RGBA filtering", fontsize=15, fontweight="bold", pad=12)

    # Left and right process arrows.
    left_arrow = FancyArrowPatch(
        (0.00, 0.56),
        (0.15, 0.56),
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=1.8,
        color="black",
        transform=ax.transAxes,
        clip_on=False,
    )
    right_arrow = FancyArrowPatch(
        (0.85, 0.56),
        (1.00, 0.56),
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=1.8,
        color="black",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(left_arrow)
    ax.add_patch(right_arrow)

    box = FancyBboxPatch(
        (0.17, 0.34),
        0.66,
        0.44,
        boxstyle="round,pad=0.035,rounding_size=0.035",
        linewidth=1.5,
        edgecolor="#255b96",
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(box)

    # Each line is split into colored and black segments for reliable rendering.
    y_positions = [0.69, 0.59, 0.49, 0.39]
    channel_specs = [
        ("R", args.r_min, args.r_max, "red"),
        ("G", args.g_min, args.g_max, "green"),
        ("B", args.b_min, args.b_max, "blue"),
    ]

    for y, (channel, lower, upper, color) in zip(y_positions[:3], channel_specs):
        ax.text(
            0.50,
            y,
            rf"${channel}_{{\min}}\leq {channel}\leq {channel}_{{\max}}$",
            ha="center",
            va="center",
            fontsize=13,
            color=color,
            transform=ax.transAxes,
        )
        ax.text(
            0.50,
            y - 0.038,
            f"({lower:g}–{upper:g})",
            ha="center",
            va="center",
            fontsize=8.5,
            color="black",
            transform=ax.transAxes,
        )

    ax.text(
        0.50,
        y_positions[3],
        r"$A_{\min}\leq \mathrm{Area}\leq A_{\max}$",
        ha="center",
        va="center",
        fontsize=13,
        color="black",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        y_positions[3] - 0.038,
        f"({args.a_min}–{args.a_max} pixels)",
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
        transform=ax.transAxes,
    )

    note = "Candidate connected components\nsatisfying RGB + area constraints"
    if args.show_count:
        note += f"\nRetained components: {retained_count}"

    ax.text(
        0.50,
        0.23,
        note,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.3,
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.12,
        r"($A$ = area, not alpha)",
        ha="center",
        va="center",
        fontsize=9,
        fontstyle="italic",
        transform=ax.transAxes,
    )


def create_figure(
    original_rgb: np.ndarray,
    filtered_rgb: np.ndarray,
    args: argparse.Namespace,
    retained_count: int,
) -> None:
    """Assemble and save the complete Fig. 2b panel."""
    height, width = original_rgb.shape[:2]
    image_ratio = width / height

    # Figure width adapts mildly to image aspect ratio.
    figure_width = max(10.5, 7.0 * image_ratio + 5.0)
    figure_height = 5.2

    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 0.78, 1.0],
        left=0.035,
        right=0.985,
        top=0.88,
        bottom=0.08,
        wspace=0.06,
    )

    ax_original = fig.add_subplot(grid[0, 0])
    ax_center = fig.add_subplot(grid[0, 1])
    ax_filtered = fig.add_subplot(grid[0, 2])

    ax_original.imshow(original_rgb)
    ax_original.set_title("Merged image", fontsize=15, fontweight="bold", pad=12)
    ax_original.set_xticks([])
    ax_original.set_yticks([])
    for spine in ax_original.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    add_center_diagram(ax_center, args, retained_count)

    ax_filtered.imshow(filtered_rgb)
    ax_filtered.set_title(
        "Filtered merged image", fontsize=15, fontweight="bold", pad=12
    )
    ax_filtered.set_xticks([])
    ax_filtered.set_yticks([])
    for spine in ax_filtered.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    if args.panel_label:
        fig.text(
            0.012,
            0.94,
            args.panel_label,
            fontsize=19,
            fontweight="bold",
            ha="left",
            va="top",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    validate_args(args)

    original_native = read_rgb_image(args.input)
    display_rgb = dtype_to_uint8(original_native)

    if args.threshold_domain == "8bit":
        threshold_rgb = display_rgb
    else:
        threshold_rgb = original_native.astype(np.float64, copy=False)

    initial_mask = make_initial_rgb_mask(
        threshold_rgb,
        (args.r_min, args.r_max),
        (args.g_min, args.g_max),
        (args.b_min, args.b_max),
    )
    retained_mask, retained_areas = apply_area_filter(
        initial_mask,
        args.a_min,
        args.a_max,
    )

    filtered_rgb = make_filtered_visualization(
        display_rgb,
        retained_mask,
        result_style=args.result_style,
        dim_factor=args.dim_factor,
        contour_thickness=args.contour_thickness,
    )

    create_figure(
        display_rgb,
        filtered_rgb,
        args,
        retained_count=len(retained_areas),
    )

    if args.filtered_output is not None:
        save_filtered_image(args.filtered_output, filtered_rgb)

    initial_pixels = int(np.count_nonzero(initial_mask))
    retained_pixels = int(np.count_nonzero(retained_mask))

    print("RGBA filtering complete")
    print("-----------------------")
    print(f"Input: {args.input.resolve()}")
    print(f"Input dtype: {original_native.dtype}")
    print(f"Threshold domain: {args.threshold_domain}")
    print(
        "RGB ranges: "
        f"R={args.r_min:g}-{args.r_max:g}, "
        f"G={args.g_min:g}-{args.g_max:g}, "
        f"B={args.b_min:g}-{args.b_max:g}"
    )
    print(f"Area range: {args.a_min}-{args.a_max} pixels")
    print(f"Pixels passing RGB constraints: {initial_pixels}")
    print(f"Retained connected components: {len(retained_areas)}")
    print(f"Pixels retained after area filtering: {retained_pixels}")
    print(f"Figure saved to: {args.output.resolve()}")
    if args.filtered_output is not None:
        print(f"Filtered image saved to: {args.filtered_output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
