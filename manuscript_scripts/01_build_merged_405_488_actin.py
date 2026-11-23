#!/usr/bin/env python
"""Create standardized three-channel merged TIFFs from raw fluorescence channels.

Merge rule
----------
- Blue plane: 405-nm channel
- Green plane: 488-nm channel (vinculin or paxillin)
- Red plane:
    * folders 3, 4, 5, 6 -> 647-nm channel (F-actin), pseudocolored red
    * all other eligible folders -> 546-nm channel (F-actin), pseudocolored red
- Folders 9 and 12 are excluded because their 488-nm channels are not FA markers.

The output folder structure mirrors the input structure:
    <output-root>/<folder>/<Soft|Stiff>/<sample>_M.tif

Source wavelength TIFFs may be stored either as 2D grayscale images or as RGB
pseudocolored images. For RGB inputs, 405 nm is extracted from blue, 488 nm
from green, 546 nm from red, and 647 nm from red/magenta before being
pseudocolored red in the output. Raw intensities are not independently min-max
normalized per image. If all source channels are uint8, the output remains
uint8. Otherwise, all channels are converted to uint16 using dtype-aware scaling.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile


EXCLUDED_FOLDERS = {}
ACTIN_647_FOLDERS = {"3", "4", "5", "6"}
CONDITIONS = {"Soft", "Stiff"}
TIFF_EXTENSIONS = {".tif", ".tiff"}

# Matches filenames such as Soft_1_488.tif, Soft_1_ch488.tif,
# Soft_1_channel_488nm.tif, etc.
CHANNEL_TOKEN_RE = re.compile(
    r"(?i)(^|[_\-\s])(?:(?:ch(?:annel)?|c)[_\-\s]*)?"
    r"(405|488|546|647)(?:nm)?(?=$|[_\-\s])"
)


@dataclass(frozen=True)
class SampleChannels:
    folder_id: str
    condition: str
    sample_key: str
    channels: dict[str, Path]


@dataclass(frozen=True)
class MergeRecord:
    folder_id: str
    condition: str
    sample_key: str
    actin_wavelength: str
    channel_405_path: str
    channel_488_path: str
    channel_actin_path: str
    output_path: str
    output_dtype: str
    height: int
    width: int
    channel_405_min: int
    channel_405_max: int
    channel_488_min: int
    channel_488_max: int
    channel_actin_min: int
    channel_actin_max: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge 405-nm, 488-nm, and 546/647-nm fluorescence channels into "
            "a standardized blue/green/red TIFF. Folders 9 and 12 are excluded; "
            "folders 3-6 use 647 nm as F-actin."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing <folder>/<Soft|Stiff>/ raw TIFF files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: a sibling directory named "
            "'merged_data_405_488_actin' next to data-root."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged TIFF files.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help=(
            "Delete the entire output directory before writing. Use only when "
            "the output directory contains no files that must be retained."
        ),
    )
    return parser.parse_args()


def parse_channel_filename(path: Path) -> tuple[str, str] | None:
    """Return (sample_key, wavelength) for a raw channel TIFF."""
    matches = list(CHANNEL_TOKEN_RE.finditer(path.stem))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Ambiguous wavelength token in filename: {path}")

    match = matches[0]
    wavelength = match.group(2)
    sample_key = path.stem[: match.start()] + "_" + path.stem[match.end() :]
    sample_key = re.sub(r"[_\-\s]+", "_", sample_key).strip("_")
    if not sample_key:
        raise ValueError(f"Could not derive sample key from filename: {path}")
    return sample_key, wavelength


def discover_samples(data_root: Path) -> tuple[list[SampleChannels], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], dict[str, Path]] = {}
    audit: list[dict[str, str]] = []

    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TIFF_EXTENSIONS:
            continue

        parsed = parse_channel_filename(path)
        if parsed is None:
            # Ignore existing merged files and unrelated TIFFs.
            continue
        sample_key, wavelength = parsed

        relative_parts = path.relative_to(data_root).parts
        condition_index = next(
            (i for i, part in enumerate(relative_parts) if part in CONDITIONS),
            None,
        )
        if condition_index is None or condition_index == 0:
            audit.append(
                {
                    "folder_id": "",
                    "condition": "",
                    "sample_key": sample_key,
                    "status": "ignored",
                    "reason": "Could not identify <folder>/<Soft|Stiff> from path",
                    "path": str(path),
                }
            )
            continue

        condition = relative_parts[condition_index]
        folder_id = relative_parts[condition_index - 1]
        key = (folder_id, condition, sample_key)
        channel_map = grouped.setdefault(key, {})
        if wavelength in channel_map:
            raise ValueError(
                "Duplicate channel found for one sample: "
                f"sample={key}, wavelength={wavelength}, "
                f"files={channel_map[wavelength]} and {path}"
            )
        channel_map[wavelength] = path.resolve()

    samples = [
        SampleChannels(folder_id, condition, sample_key, channels)
        for (folder_id, condition, sample_key), channels in sorted(grouped.items())
    ]
    if not samples:
        raise FileNotFoundError(
            f"No raw channel TIFFs containing 405/488/546/647 tokens were found under {data_root}"
        )
    return samples, audit


def _extract_wavelength_plane(
    image: np.ndarray,
    wavelength: str,
    *,
    color_order: str,
    path: Path,
) -> np.ndarray:
    """Extract fluorescence intensity from a 2D or pseudocolored TIFF.

    Many source files are stored as RGB TIFFs even though each filename refers to
    only one fluorescence wavelength. For those files, 405 nm is read from the
    blue display plane, 488 nm from the green display plane, 546 nm from the
    red display plane, and 647 nm from the
    maximum of the red/blue planes because 647-nm images may be displayed either
    red or magenta.
    """
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim == 2:
        return image

    # Convert channel-first arrays such as (3, H, W) to (H, W, 3).
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)

    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected a 2D channel or RGB/RGBA image, but got shape {image.shape}: {path}"
        )

    if color_order == "BGR":
        blue = image[..., 0]
        green = image[..., 1]
        red = image[..., 2]
    elif color_order == "RGB":
        red = image[..., 0]
        green = image[..., 1]
        blue = image[..., 2]
    else:
        raise ValueError(f"Unsupported color order: {color_order}")

    if wavelength == "405":
        plane = blue
    elif wavelength == "488":
        plane = green
    elif wavelength == "546":
        plane = red
    elif wavelength == "647":
        # 647-nm images may be pseudocolored red or magenta. Taking the maximum
        # of red and blue recovers the displayed fluorescence intensity in both
        # cases, after which it is placed in the red plane of the merged output.
        plane = np.maximum(red, blue)
    else:
        raise ValueError(f"Unsupported wavelength: {wavelength}")

    return np.ascontiguousarray(plane)


def read_channel_intensity(path: Path, wavelength: str) -> np.ndarray:
    """Read one wavelength image while supporting grayscale and RGB TIFFs.

    OpenCV is tried first because it can usually decode LZW-compressed TIFF files
    without the optional Python ``imagecodecs`` package. tifffile is retained as
    a fallback for TIFF variants OpenCV cannot decode.
    """
    encoded = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image_bgr is not None:
        return _extract_wavelength_plane(
            image_bgr, wavelength, color_order="BGR", path=path
        )

    try:
        image_rgb = tifffile.imread(path)
    except Exception as exc:
        raise ValueError(
            f"Could not decode TIFF {path}. Install imagecodecs if the file uses "
            f"LZW or another optional TIFF compression codec. Original error: {exc}"
        ) from exc

    return _extract_wavelength_plane(
        image_rgb, wavelength, color_order="RGB", path=path
    )


def to_uint16(channel: np.ndarray) -> np.ndarray:
    """Convert a channel to uint16 without per-image min-max normalization."""
    if channel.dtype == np.uint16:
        return channel
    if channel.dtype == np.uint8:
        return channel.astype(np.uint16) * np.uint16(257)

    if np.issubdtype(channel.dtype, np.unsignedinteger):
        source_max = float(np.iinfo(channel.dtype).max)
        values = channel.astype(np.float64)
        return np.rint(values * (65535.0 / source_max)).clip(0, 65535).astype(np.uint16)

    if np.issubdtype(channel.dtype, np.signedinteger):
        values = np.clip(channel.astype(np.float64), 0, None)
        source_max = float(np.iinfo(channel.dtype).max)
        return np.rint(values * (65535.0 / source_max)).clip(0, 65535).astype(np.uint16)

    if np.issubdtype(channel.dtype, np.floating):
        values = np.nan_to_num(
            channel.astype(np.float64), nan=0.0, posinf=65535.0, neginf=0.0
        )
        if float(values.min(initial=0.0)) < 0:
            raise ValueError("Negative fluorescence intensities are not supported.")
        maximum = float(values.max(initial=0.0))
        if maximum <= 1.0:
            values = values * 65535.0
        elif maximum <= 255.0:
            values = values * 257.0
        # Values above 255 are treated as already being in a higher-bit-depth
        # acquisition range and are clipped, not independently normalized.
        return np.rint(values).clip(0, 65535).astype(np.uint16)

    raise TypeError(f"Unsupported channel dtype: {channel.dtype}")


def harmonize_channels(*channels: np.ndarray) -> tuple[np.ndarray, ...]:
    """Match shapes and dtypes without per-image intensity normalization."""
    if not channels:
        raise ValueError("At least one channel is required.")
    reference_shape = channels[0].shape
    if any(channel.shape != reference_shape for channel in channels[1:]):
        shapes = [channel.shape for channel in channels]
        raise ValueError(f"Channel dimensions differ: {shapes}")
    if all(channel.dtype == np.uint8 for channel in channels):
        return tuple(np.ascontiguousarray(channel) for channel in channels)
    return tuple(to_uint16(channel) for channel in channels)


def write_rgb_tiff(output_path: Path, rgb: np.ndarray, overwrite: bool) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return "skipped_existing"

    temporary_path = output_path.with_name(output_path.name + ".tmp.tif")
    tifffile.imwrite(
        temporary_path,
        rgb,
        photometric="rgb",
        compression=None,
        metadata=None,
    )
    os.replace(temporary_path, output_path)
    return "overwritten" if overwrite else "written"


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {data_root}")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (data_root.parent / "merged_data_405_488_actin").resolve()
    )
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError("Output root must not be the data directory or a child of it.")

    if args.clean_output and output_root.exists():
        print(f"Removing existing output directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples, discovery_audit = discover_samples(data_root)
    merge_records: list[dict[str, object]] = []
    excluded_records: list[dict[str, object]] = list(discovery_audit)
    errors: list[str] = []

    eligible_candidates = [s for s in samples if s.folder_id not in EXCLUDED_FOLDERS]
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Discovered samples: {len(samples)}")
    print(f"Eligible candidates after excluding folders 9 and 12: {len(eligible_candidates)}")

    for index, sample in enumerate(samples, start=1):
        if sample.folder_id in EXCLUDED_FOLDERS:
            excluded_records.append(
                {
                    "folder_id": sample.folder_id,
                    "condition": sample.condition,
                    "sample_key": sample.sample_key,
                    "status": "excluded",
                    "reason": "Folder 9/12 excluded because the 488-nm marker is not vinculin/paxillin",
                    "path": str(data_root / sample.folder_id / sample.condition),
                }
            )
            continue

        actin_wavelength = "647" if sample.folder_id in ACTIN_647_FOLDERS else "546"
        missing = [wave for wave in ("405", "488", actin_wavelength) if wave not in sample.channels]
        if missing:
            excluded_records.append(
                {
                    "folder_id": sample.folder_id,
                    "condition": sample.condition,
                    "sample_key": sample.sample_key,
                    "status": "excluded",
                    "reason": "Missing required channel(s): " + ", ".join(missing),
                    "path": str(data_root / sample.folder_id / sample.condition),
                }
            )
            continue

        try:
            path_405 = sample.channels["405"]
            path_488 = sample.channels["488"]
            path_actin = sample.channels[actin_wavelength]
            blue_405_raw = read_channel_intensity(path_405, "405")
            green_488_raw = read_channel_intensity(path_488, "488")
            red_actin_raw = read_channel_intensity(path_actin, actin_wavelength)
            blue_405, green_488, red_actin = harmonize_channels(
                blue_405_raw, green_488_raw, red_actin_raw
            )

            # tifffile expects RGB order. 405 is blue, 488 is green, and the
            # selected 546/647 F-actin signal is pseudocolored red.
            rgb = np.stack([red_actin, green_488, blue_405], axis=-1)

            output_path = (
                output_root
                / sample.folder_id
                / sample.condition
                / f"{sample.sample_key}_M.tif"
            )
            status = write_rgb_tiff(output_path, rgb, overwrite=args.overwrite)

            merge_records.append(
                {
                    "folder_id": sample.folder_id,
                    "condition": sample.condition,
                    "sample_key": sample.sample_key,
                    "actin_wavelength_used": actin_wavelength,
                    "channel_405_path": str(path_405),
                    "channel_488_path": str(path_488),
                    "channel_actin_path": str(path_actin),
                    "output_path": str(output_path),
                    "output_dtype": str(rgb.dtype),
                    "height": int(rgb.shape[0]),
                    "width": int(rgb.shape[1]),
                    "channel_405_min": int(blue_405.min(initial=0)),
                    "channel_405_max": int(blue_405.max(initial=0)),
                    "channel_488_min": int(green_488.min(initial=0)),
                    "channel_488_max": int(green_488.max(initial=0)),
                    "channel_actin_min": int(red_actin.min(initial=0)),
                    "channel_actin_max": int(red_actin.max(initial=0)),
                    "status": status,
                }
            )
        except Exception as exc:  # keep a complete audit, then fail at the end
            message = (
                f"folder={sample.folder_id}, condition={sample.condition}, "
                f"sample={sample.sample_key}: {type(exc).__name__}: {exc}"
            )
            errors.append(message)
            excluded_records.append(
                {
                    "folder_id": sample.folder_id,
                    "condition": sample.condition,
                    "sample_key": sample.sample_key,
                    "status": "error",
                    "reason": message,
                    "path": str(data_root / sample.folder_id / sample.condition),
                }
            )

        if index % 20 == 0 or index == len(samples):
            print(f"Processed {index}/{len(samples)} samples")

    manifest_path = output_root / "manifest_merged_405_488_actin.csv"
    excluded_path = output_root / "manifest_excluded_or_failed.csv"
    write_csv(manifest_path, merge_records)
    write_csv(excluded_path, excluded_records)

    n_written = sum(r["status"] in {"written", "overwritten"} for r in merge_records)
    n_skipped = sum(r["status"] == "skipped_existing" for r in merge_records)
    soft_count = sum(r["condition"] == "Soft" for r in merge_records)
    stiff_count = sum(r["condition"] == "Stiff" for r in merge_records)

    print("\nMerge summary")
    print(f"  Merged samples: {len(merge_records)} (Soft={soft_count}, Stiff={stiff_count})")
    print(f"  Written/overwritten: {n_written}")
    print(f"  Skipped existing: {n_skipped}")
    print(f"  Excluded/audit rows: {len(excluded_records)}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Exclusion/error manifest: {excluded_path}")

    if errors:
        print("\nErrors occurred:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("\nCompleted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
