"""
Reference-point sensitivity analysis for AdhGen
===============================================

Purpose
-------
Quantify how the choice of reference points affects the final AdhGen result
(Reviewer 1, comment 1-3). On one hold-out split (default: split seed 60) the
GA is re-optimized from scratch for each setting below, and the resulting
parameter set is applied unchanged to the held-out test images.

Design
------
One factor is changed at a time relative to the submitted configuration:

    reference-point detection : smoothing sigma, minimum separation,
                                threshold percentile (optional: background sigma)
    matching                  : matching radius r
    area weighting            : area soft limit
    direct perturbation       : random removal of points, random shift of points

For every setting the script stores the optimized chromosome, the training
and test fitness, and the held-out FA count per cell with the Soft/Stiff
comparison. Each solution is additionally scored against the *baseline*
reference points with the baseline matching settings, so that all solutions
can be compared on one common scale.

The existing AdhGen modules are imported unchanged; nothing in them is edited.

Examples
--------
List the settings:
    python adhgen_reference_sensitivity.py --list

Fill the baseline row with the submitted solution, without re-running its GA:
    python adhgen_reference_sensitivity.py --settings baseline ^
        --baseline-chromosome 30,255,109,247,0,69,12,100

Run everything in the default set (resumes, skips completed settings):
    python adhgen_reference_sensitivity.py

Run a subset, e.g. in a second terminal:
    python adhgen_reference_sensitivity.py --settings remove_10,remove_20,remove_30

Only rebuild the summary table from completed settings:
    python adhgen_reference_sensitivity.py --summary-only
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent

import argparse
import json
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

import adhgen_ga_area_weighted_micro_f1_windows as ga
import adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_windows as holdout


SCRIPT_VERSION = "2026-11-21-reference-sensitivity"

DATA_ROOT = ga.DATA_ROOT
OUTPUT_ROOT = Path(str(_SCRIPT_DIR / "reference_sensitivity_results"))

DEFAULT_SPLIT_SEED = 60
DEFAULT_TEST_SIZE = holdout.DEFAULT_TEST_SIZE
DEFAULT_GA_SEED = holdout.DEFAULT_GA_SEED
DEFAULT_POPULATION_SIZE = holdout.DEFAULT_POPULATION_SIZE
DEFAULT_GENERATIONS = holdout.DEFAULT_GENERATIONS
DEFAULT_MAX_WORKERS = holdout.DEFAULT_MAX_WORKERS
DEFAULT_PERTURB_SEED = 0

BASELINE_MATCH_RADIUS_PX = float(ga.MATCH_RADIUS_PX)
BASELINE_AREA_SOFT_LIMIT_PX = float(ga.AREA_SOFT_LIMIT_PX)
METRIC = holdout.PRIMARY_TEST_METRIC


# =============================================================================
# Settings
# =============================================================================

@dataclass(frozen=True)
class Setting:
    name: str
    group: str
    description: str
    point_overrides: Mapping[str, float] = field(default_factory=dict)
    match_radius_px: float = BASELINE_MATCH_RADIUS_PX
    area_soft_limit_px: float = BASELINE_AREA_SOFT_LIMIT_PX
    remove_fraction: float = 0.0
    shift_px: int = 0
    in_default_set: bool = True


SETTINGS: List[Setting] = [
    Setting("baseline", "baseline", "Submitted configuration"),
    # --- reference-point detection ---
    Setting("smooth_sigma_1.0", "detection", "Smoothing sigma 1.0 px (baseline 1.5)",
            point_overrides={"smooth_sigma": 1.0}),
    Setting("smooth_sigma_2.0", "detection", "Smoothing sigma 2.0 px (baseline 1.5)",
            point_overrides={"smooth_sigma": 2.0}),
    Setting("min_distance_6", "detection", "Minimum separation 6 px (baseline 8)",
            point_overrides={"min_distance_px": 6}),
    Setting("min_distance_10", "detection", "Minimum separation 10 px (baseline 8)",
            point_overrides={"min_distance_px": 10}),
    Setting("percentile_99.0", "detection", "Threshold percentile 99.0 (baseline 99.5)",
            point_overrides={"threshold_percentile": 99.0}),
    Setting("percentile_99.75", "detection", "Threshold percentile 99.75 (baseline 99.5)",
            point_overrides={"threshold_percentile": 99.75}),
    Setting("background_sigma_10", "detection", "Background sigma 10 px (baseline 15)",
            point_overrides={"background_sigma": 10.0}, in_default_set=False),
    Setting("background_sigma_20", "detection", "Background sigma 20 px (baseline 15)",
            point_overrides={"background_sigma": 20.0}, in_default_set=False),
    # --- matching ---
    Setting("match_radius_2", "matching", "Matching radius 2 px (baseline 4)",
            match_radius_px=2.0),
    Setting("match_radius_6", "matching", "Matching radius 6 px (baseline 4)",
            match_radius_px=6.0),
    # --- area weighting ---
    Setting("area_soft_limit_40", "area", "Area soft limit 40 px (baseline 60)",
            area_soft_limit_px=40.0),
    Setting("area_soft_limit_80", "area", "Area soft limit 80 px (baseline 60)",
            area_soft_limit_px=80.0),
    # --- direct perturbation of the reference set ---
    Setting("remove_10", "removal", "Random removal of 10% of reference points",
            remove_fraction=0.10),
    Setting("remove_20", "removal", "Random removal of 20% of reference points",
            remove_fraction=0.20),
    Setting("remove_30", "removal", "Random removal of 30% of reference points",
            remove_fraction=0.30),
    Setting("shift_1", "shift", "Random shift of every point by up to 1 px",
            shift_px=1),
    Setting("shift_2", "shift", "Random shift of every point by up to 2 px",
            shift_px=2),
]
SETTING_BY_NAME: Dict[str, Setting] = {s.name: s for s in SETTINGS}


# =============================================================================
# Reference construction
# =============================================================================

def relative_key(path: str) -> str:
    """folder/condition/filename, identical on Windows and Linux."""
    parts = holdout.split_path_parts(path)
    return "/".join(parts[-3:])


def image_rng(perturb_seed: int, path: str, salt: int) -> np.random.Generator:
    """Per-image generator that does not depend on processing order or OS."""
    return np.random.default_rng(
        [int(perturb_seed), int(salt), zlib.crc32(relative_key(path).encode("utf-8"))]
    )


def remove_points(points: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    n = int(len(points))
    n_remove = int(round(n * float(fraction)))
    if n == 0 or n_remove <= 0:
        return points
    keep = np.sort(rng.choice(n, size=n - n_remove, replace=False))
    return points[keep]


def shift_points(
    points: np.ndarray,
    shift_px: int,
    image_shape: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Move every point by an integer offset with |dy|, |dx| <= shift_px, never (0, 0)."""
    n = int(len(points))
    s = int(shift_px)
    if n == 0 or s <= 0:
        return points

    dy = rng.integers(-s, s + 1, size=n)
    dx = rng.integers(-s, s + 1, size=n)
    still = (dy == 0) & (dx == 0)
    while np.any(still):
        dy[still] = rng.integers(-s, s + 1, size=int(still.sum()))
        dx[still] = rng.integers(-s, s + 1, size=int(still.sum()))
        still = (dy == 0) & (dx == 0)

    shifted = points.copy()
    height, width = int(image_shape[0]), int(image_shape[1])
    shifted[:, 0] = np.clip(shifted[:, 0] + dy, 0, height - 1)
    shifted[:, 1] = np.clip(shifted[:, 1] + dx, 0, width - 1)
    return shifted


def build_setting_references(
    setting: Setting,
    images: Mapping[str, np.ndarray],
    baseline_references: Mapping[str, ga.ImageReference],
    perturb_seed: int,
) -> Dict[str, ga.ImageReference]:
    """
    Derive the references of one setting from the baseline references.

    The cell, nucleus and scale-bar masks never change, so only the points are
    regenerated (detection settings) or perturbed (removal / shift).
    """
    point_config = replace(ga.POINT_CONFIG_FINAL, **dict(setting.point_overrides))
    references: Dict[str, ga.ImageReference] = {}

    for path, base in baseline_references.items():
        points = base.points

        if setting.point_overrides:
            raw_points = ga.generate_green_points(images[path], point_config)
            points = ga.filter_points_by_mask(raw_points, base.analysis_mask)

        if setting.remove_fraction > 0:
            points = remove_points(
                points, setting.remove_fraction, image_rng(perturb_seed, path, salt=1)
            )

        if setting.shift_px > 0:
            points = shift_points(
                points, setting.shift_px, images[path].shape, image_rng(perturb_seed, path, salt=2)
            )

        references[path] = replace(base, points=np.asarray(points, dtype=np.float32))

    return references


# =============================================================================
# One setting
# =============================================================================

def micro_components(result_df: pd.DataFrame) -> dict:
    micro = holdout.calculate_micro_metrics(result_df)
    q = micro["matched_quality_sum"]
    p = micro["n_reference_points"]
    c = micro["n_detected_objects"]
    return {
        "micro_f1": micro["micro_f1"],
        "precision_like": float(q / c) if c > 0 else float("nan"),
        "recall_like": float(q / p) if p > 0 else float("nan"),
        "n_reference_points": int(p),
        "n_detected_objects": int(c),
        "n_matched": int(micro["n_matched"]),
    }


def run_setting(
    setting: Setting,
    setting_dir: Path,
    images: Mapping[str, np.ndarray],
    baseline_references: Mapping[str, ga.ImageReference],
    train_paths: Sequence[str],
    test_paths: Sequence[str],
    manifest: pd.DataFrame,
    ga_config: ga.GAConfig,
    split_seed: int,
    perturb_seed: int,
    fixed_individual: Sequence[int] | None = None,
) -> dict:
    """
    Optimize and evaluate one setting.

    If `fixed_individual` is given, the GA is skipped and that chromosome is
    scored instead. This is used to fill the baseline row with the solution
    of the submitted run without repeating its optimization.
    """
    setting_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    references = build_setting_references(setting, images, baseline_references, perturb_seed)

    reference_df = pd.DataFrame(
        [
            {
                "key": relative_key(path),
                "condition": ga.extract_condition(path),
                "n_cells": int(ref.n_cells),
                "n_reference_points": int(len(ref.points)),
                "n_reference_points_baseline": int(len(baseline_references[path].points)),
            }
            for path, ref in sorted(references.items())
        ]
    )
    reference_df.to_csv(setting_dir / "reference_summary.csv", index=False)

    export_manifest = manifest.copy()
    export_manifest.insert(0, "key", export_manifest["path"].map(relative_key))
    export_manifest.to_csv(setting_dir / "split_manifest.csv", index=False)

    train_images = holdout.select_mapping(images, train_paths)
    test_images = holdout.select_mapping(images, test_paths)
    train_refs = holdout.select_mapping(references, train_paths)
    test_refs = holdout.select_mapping(references, test_paths)

    print(
        f"\n{'=' * 76}\n"
        f"Setting: {setting.name}  ({setting.description})\n"
        f"  match_radius_px={setting.match_radius_px}, "
        f"area_soft_limit_px={setting.area_soft_limit_px}, "
        f"mean reference points/image={reference_df['n_reference_points'].mean():.1f} "
        f"(baseline {reference_df['n_reference_points_baseline'].mean():.1f})\n"
        f"{'=' * 76}"
    )

    if fixed_individual is not None:
        best = ga.repair_individual(list(fixed_individual), ga_config)
        train_best_fitness = ga.evaluate_individual(
            best,
            train_images,
            train_refs,
            setting.match_radius_px,
            ga.REFERENCE_MASK_CONFIG,
            ga_config,
            setting.area_soft_limit_px,
        )
        history_df = pd.DataFrame(
            [
                {
                    "generation": 0,
                    "fitness_mean": float("nan"),
                    "fitness_std": float("nan"),
                    "fitness_min": float("nan"),
                    "fitness_max": float(train_best_fitness),
                    "unique_evaluations": 0,
                    "best_individual": json.dumps([int(v) for v in best]),
                }
            ]
        )
        print("GA skipped: scoring the supplied chromosome", best)
    else:
        best, train_best_fitness, history_df = ga.run_ga(
            images=train_images,
            references=train_refs,
            match_radius_px=setting.match_radius_px,
            mask_config=ga.REFERENCE_MASK_CONFIG,
            ga_config=ga_config,
            area_soft_limit_px=setting.area_soft_limit_px,
        )
    history_df.insert(0, "setting", setting.name)
    history_df.to_csv(setting_dir / "ga_history.csv", index=False)

    def evaluate(imgs, refs, radius, soft_limit) -> pd.DataFrame:
        return ga.evaluate_best_solution(
            images=imgs,
            references=refs,
            individual=best,
            match_radius_px=radius,
            mask_config=ga.REFERENCE_MASK_CONFIG,
            ga_config=ga_config,
            area_soft_limit_px=soft_limit,
        )

    # Scored with this setting's own references and matching settings.
    train_df = holdout.add_fa_count_per_cell(
        evaluate(train_images, train_refs, setting.match_radius_px, setting.area_soft_limit_px)
    )
    test_df = holdout.add_fa_count_per_cell(
        evaluate(test_images, test_refs, setting.match_radius_px, setting.area_soft_limit_px)
    )
    for frame in (train_df, test_df):
        frame.insert(0, "key", frame["path"].map(relative_key))
    train_df.to_csv(setting_dir / "train_by_image.csv", index=False)
    test_df.to_csv(setting_dir / "test_by_image.csv", index=False)

    # Common scale: baseline references with baseline matching settings.
    base_train = evaluate(
        train_images, holdout.select_mapping(baseline_references, train_paths),
        BASELINE_MATCH_RADIUS_PX, BASELINE_AREA_SOFT_LIMIT_PX,
    )
    base_test = evaluate(
        test_images, holdout.select_mapping(baseline_references, test_paths),
        BASELINE_MATCH_RADIUS_PX, BASELINE_AREA_SOFT_LIMIT_PX,
    )

    condition_summary, statistics = holdout.summarize_fa_count_per_cell(test_df)
    condition_summary.to_csv(setting_dir / "test_condition_summary.csv", index=False)
    statistics.to_csv(setting_dir / "test_statistics.csv", index=False)
    stats = statistics.iloc[0].to_dict() if not statistics.empty else {}

    train_micro = micro_components(train_df)
    test_micro = micro_components(test_df)
    base_train_micro = micro_components(base_train)
    base_test_micro = micro_components(base_test)

    soft = test_df.loc[test_df["condition"] == "Soft", METRIC].dropna()
    stiff = test_df.loc[test_df["condition"] == "Stiff", METRIC].dropna()

    result = {
        "completed": True,
        "ga_skipped": bool(fixed_individual is not None),
        "script_version": SCRIPT_VERSION,
        "base_ga_version": ga.SCRIPT_VERSION,
        "setting": setting.name,
        "group": setting.group,
        "description": setting.description,
        "split_seed": int(split_seed),
        "ga_seed": int(ga_config.seed),
        "perturb_seed": int(perturb_seed),
        "population_size": int(ga_config.population_size),
        "generations": int(ga_config.generations),
        "match_radius_px": float(setting.match_radius_px),
        "area_soft_limit_px": float(setting.area_soft_limit_px),
        "remove_fraction": float(setting.remove_fraction),
        "shift_px": int(setting.shift_px),
        "point_config": asdict(replace(ga.POINT_CONFIG_FINAL, **dict(setting.point_overrides))),
        "n_train": int(len(train_paths)),
        "n_test": int(len(test_paths)),
        "mean_reference_points_per_image": float(reference_df["n_reference_points"].mean()),
        "mean_reference_points_per_cell": float(
            (reference_df["n_reference_points"] / reference_df["n_cells"]).mean()
        ),
        "best_individual": [int(v) for v in best],
        "r_min": int(best[0]), "r_max": int(best[1]),
        "g_min": int(best[2]), "g_max": int(best[3]),
        "b_min": int(best[4]), "b_max": int(best[5]),
        "area_min": int(best[6]), "area_max": int(best[7]),
        "train_best_fitness": float(train_best_fitness),
        "train_micro_f1": train_micro["micro_f1"],
        "test_micro_f1": test_micro["micro_f1"],
        "test_precision_like": test_micro["precision_like"],
        "test_recall_like": test_micro["recall_like"],
        "train_micro_f1_on_baseline_reference": base_train_micro["micro_f1"],
        "test_micro_f1_on_baseline_reference": base_test_micro["micro_f1"],
        "test_total_detected_objects": test_micro["n_detected_objects"],
        "test_n_soft": int(len(soft)),
        "test_n_stiff": int(len(stiff)),
        "test_soft_median_fa_per_cell": float(soft.median()),
        "test_stiff_median_fa_per_cell": float(stiff.median()),
        "test_soft_mean_fa_per_cell": float(soft.mean()),
        "test_stiff_mean_fa_per_cell": float(stiff.mean()),
        "test_median_fa_per_cell_all": float(test_df[METRIC].median()),
        "test_mean_difference": stats.get("mean_difference", float("nan")),
        "test_mann_whitney_u": stats.get("mann_whitney_u", float("nan")),
        "test_mann_whitney_p": stats.get("mann_whitney_p", float("nan")),
        "test_welch_p": stats.get("welch_p", float("nan")),
        "test_cliffs_delta": stats.get("cliffs_delta_stiff_vs_soft", float("nan")),
        "unique_evaluations": int(history_df["unique_evaluations"].iloc[-1]),
        "duration_seconds": float(time.time() - start_time),
    }
    holdout.save_json(setting_dir / "result.json", result)

    print(
        f"[{setting.name}] best={result['best_individual']} "
        f"train F={result['train_best_fitness']:.4f}, test F={result['test_micro_f1']:.4f}, "
        f"FA/cell median Soft={result['test_soft_median_fa_per_cell']:.1f} "
        f"Stiff={result['test_stiff_median_fa_per_cell']:.1f}, "
        f"MW p={result['test_mann_whitney_p']:.4g}, "
        f"{result['duration_seconds'] / 60:.1f} min"
    )
    return result


# =============================================================================
# Summary across settings
# =============================================================================

def build_summary(output_root: Path) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        result_path = output_root / setting.name / "result.json"
        if not result_path.exists():
            continue
        with open(result_path, "r", encoding="utf-8") as file:
            result = json.load(file)
        if not result.get("completed"):
            continue
        result = dict(result)
        result["best_individual"] = json.dumps(result["best_individual"])
        result.pop("point_config", None)
        rows.append(result)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    baseline = summary[summary["setting"] == "baseline"]
    if not baseline.empty:
        base = baseline.iloc[0]
        reference_value = float(base["test_median_fa_per_cell_all"])
        summary["pct_change_median_fa_per_cell_vs_baseline"] = (
            100.0 * (summary["test_median_fa_per_cell_all"] - reference_value) / reference_value
        )
        summary["g_min_change_vs_baseline"] = summary["g_min"] - int(base["g_min"])

    summary.to_csv(output_root / "sensitivity_summary.csv", index=False)
    return summary


# =============================================================================
# Main
# =============================================================================

def resolve_settings(value: str) -> List[Setting]:
    token = value.strip().lower()
    if token == "default":
        return [s for s in SETTINGS if s.in_default_set]
    if token == "all":
        return list(SETTINGS)

    selected: List[Setting] = []
    for name in value.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in SETTING_BY_NAME:
            raise ValueError(
                f"Unknown setting '{name}'. Use --list to see the available names."
            )
        if SETTING_BY_NAME[name] not in selected:
            selected.append(SETTING_BY_NAME[name])
    if not selected:
        raise ValueError("No setting selected.")
    return selected


def is_completed(setting_dir: Path, ga_config: ga.GAConfig, split_seed: int, perturb_seed: int) -> bool:
    result_path = setting_dir / "result.json"
    if not result_path.exists():
        return False
    try:
        with open(result_path, "r", encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        result.get("completed")
        and result.get("population_size") == ga_config.population_size
        and result.get("generations") == ga_config.generations
        and result.get("ga_seed") == ga_config.seed
        and result.get("split_seed") == split_seed
        and result.get("perturb_seed") == perturb_seed
    )


def run(
    data_root: str | Path,
    output_root: str | Path,
    settings: Sequence[Setting],
    split_seed: int,
    test_size: float,
    ga_seed: int,
    population_size: int,
    generations: int,
    workers: int,
    perturb_seed: int,
    overwrite: bool,
    baseline_chromosome: Sequence[int] | None = None,
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ga_config = replace(
        ga.GA_CONFIG,
        population_size=int(population_size),
        generations=int(generations),
        seed=int(ga_seed),
        max_workers=int(workers),
    )

    print("[Reference sensitivity]")
    print("  script_version:", SCRIPT_VERSION)
    print("  base_ga_version:", ga.SCRIPT_VERSION)
    print("  data_root:", Path(data_root))
    print("  output_root:", output_root)
    print("  split_seed:", split_seed, "| ga_seed:", ga_seed, "| perturb_seed:", perturb_seed)
    print("  population_size:", population_size, "| generations:", generations, "| workers:", workers)
    print("  settings:", ", ".join(s.name for s in settings))

    pending = [
        s for s in settings
        if overwrite or not is_completed(output_root / s.name, ga_config, split_seed, perturb_seed)
    ]
    for s in settings:
        if s not in pending:
            print(f"  [skip] {s.name}: already completed with the same configuration")

    if pending:
        image_paths = ga.collect_image_paths(
            data_root=data_root, excluded_folders=ga.EXCLUDED_FOLDERS
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found under {data_root}")

        images = ga.preload_images(image_paths)
        if len(images) != len(image_paths):
            raise RuntimeError("Some images could not be loaded.")

        print("\nBuilding baseline references ...")
        baseline_references, failure_df = ga.precompute_references(
            images=images,
            point_config=ga.POINT_CONFIG_FINAL,
            nucleus_config=ga.NUCLEUS_CONFIG_FINAL,
            cell_config=ga.CELL_CONFIG_FINAL,
            mask_config=ga.REFERENCE_MASK_CONFIG,
            max_workers=workers,
        )
        if not failure_df.empty or len(baseline_references) != len(images):
            raise RuntimeError("Baseline reference generation failed for some images.")
        ga.validate_reference_images(baseline_references)

        manifest = holdout.build_stratified_holdout_manifest(
            image_paths=image_paths, test_size=test_size, split_seed=split_seed
        )
        split_summary = holdout.validate_holdout_manifest(manifest)
        print("Split:", split_summary)
        train_paths = manifest.loc[manifest["split"] == "train", "path"].tolist()
        test_paths = manifest.loc[manifest["split"] == "test", "path"].tolist()

        for setting in pending:
            run_setting(
                setting=setting,
                setting_dir=output_root / setting.name,
                images=images,
                baseline_references=baseline_references,
                train_paths=train_paths,
                test_paths=test_paths,
                manifest=manifest,
                ga_config=ga_config,
                split_seed=split_seed,
                perturb_seed=perturb_seed,
                fixed_individual=(
                    baseline_chromosome if setting.name == "baseline" else None
                ),
            )
            build_summary(output_root)

    summary = build_summary(output_root)
    if not summary.empty:
        columns = [
            "setting", "g_min", "g_max", "area_min", "area_max",
            "train_best_fitness", "test_micro_f1_on_baseline_reference",
            "test_soft_median_fa_per_cell", "test_stiff_median_fa_per_cell",
            "test_mann_whitney_p", "test_cliffs_delta",
        ]
        print("\n[Summary]")
        print(summary[[c for c in columns if c in summary.columns]].to_string(index=False))
        print("\nSaved:", output_root / "sensitivity_summary.csv")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdhGen reference-point sensitivity analysis on one hold-out split"
    )
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--settings", type=str, default="default",
        help="'default', 'all', or a comma-separated list of setting names",
    )
    parser.add_argument("--list", action="store_true", help="Print the settings and exit")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only rebuild sensitivity_summary.csv from completed settings")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--ga-seed", type=int, default=DEFAULT_GA_SEED)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--perturb-seed", type=int, default=DEFAULT_PERTURB_SEED,
                        help="Seed for random removal / shift of reference points")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--baseline-chromosome", type=str, default="",
        help=(
            "Eight comma-separated integers (R min,max, G min,max, B min,max, "
            "area min,max). If given, the 'baseline' setting scores this "
            "chromosome instead of re-running the GA, e.g. the submitted "
            "split-seed-60 solution 30,255,109,247,0,69,12,100"
        ),
    )
    return parser.parse_args()


def parse_chromosome(value: str) -> List[int] | None:
    if not value.strip():
        return None
    genes = [int(token) for token in value.split(",") if token.strip()]
    if len(genes) != 8:
        raise ValueError("--baseline-chromosome needs exactly eight integers.")
    return genes


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        for s in SETTINGS:
            flag = "default" if s.in_default_set else "optional"
            print(f"{s.name:22s} {s.group:10s} {flag:9s} {s.description}")
        raise SystemExit(0)

    if args.summary_only:
        table = build_summary(Path(args.output_root))
        print(f"Completed settings: {len(table)}")
        raise SystemExit(0)

    run(
        data_root=args.data_root,
        output_root=args.output_root,
        settings=resolve_settings(args.settings),
        split_seed=args.split_seed,
        test_size=args.test_size,
        ga_seed=args.ga_seed,
        population_size=args.population_size,
        generations=args.generations,
        workers=args.workers,
        perturb_seed=args.perturb_seed,
        overwrite=args.overwrite,
        baseline_chromosome=parse_chromosome(args.baseline_chromosome),
    )
