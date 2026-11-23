"""
Reference-point sensitivity analysis by direct parameter scan
=============================================================

Purpose
-------
Quantify how the choice of reference points affects the AdhGen result
(Reviewer 1, comment 1-3) without repeating the genetic algorithm.

Rationale
---------
On the training images the fitness is governed by the green lower bound
(G min) and the area bounds; the red range, the green upper bound and the blue
range leave it unchanged or lower it. Scanning G min and the area bounds
therefore locates the best solution of each reference setting exactly and
deterministically, so that differences between settings are caused by the
reference points alone and not by the stochastic search.

Procedure for each reference setting (hold-out split seed 60)
-------------------------------------------------------------
1. Build the reference points of that setting.
2. Scan G min on the training images (coarse grid, then every integer around
   the best coarse value) with the other parameters fixed.
3. At the best G min, scan the area lower bound, then the area upper bound.
4. If an area bound moved, re-scan G min around the current best value.
5. Apply the resulting parameter set unchanged to the held-out test images and
   compare FA count per cell between Soft and Stiff.

`--base submitted` keeps the remaining parameters at the submitted seed-60
solution; `--base relaxed` opens the red range, green upper bound and blue
range completely (0-255) as a robustness check.

Examples
--------
    python adhgen_reference_sensitivity_scan.py --base submitted
    python adhgen_reference_sensitivity_scan.py --base relaxed --settings baseline,remove_30
    python adhgen_reference_sensitivity_scan.py --base submitted --summary-only
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import adhgen_ga_area_weighted_micro_f1_windows as ga
import adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_windows as holdout
import adhgen_reference_sensitivity as sens


SCRIPT_VERSION = "2026-11-21-reference-sensitivity-scan"
OUTPUT_ROOT = Path(str(_SCRIPT_DIR / "reference_sensitivity_scan_results"))

SUBMITTED_SEED60 = [30, 255, 109, 247, 0, 69, 12, 100]
BASE_CHROMOSOMES: Dict[str, List[int]] = {
    "submitted": list(SUBMITTED_SEED60),
    "relaxed": [0, 255, 109, 255, 0, 255, 12, 100],
}

# Settings that change the reference points themselves.
DEFAULT_SETTINGS = [
    "baseline",
    "smooth_sigma_1.0", "smooth_sigma_2.0",
    "min_distance_6", "min_distance_10",
    "percentile_99.0", "percentile_99.75",
    "remove_10", "remove_20", "remove_30",
    "shift_1", "shift_2",
]

G_COARSE = list(range(60, 171, 5))
G_FINE_HALF_WIDTH = 4
AREA_MIN_GRID = [6, 8, 10, 11, 12, 13, 14, 16, 20]
AREA_MAX_GRID = [60, 70, 80, 90, 100]

QUICK_G_COARSE = [90, 120, 150]
QUICK_AREA_MIN_GRID = [8, 12]
QUICK_AREA_MAX_GRID = [80, 100]

METRIC = holdout.PRIMARY_TEST_METRIC


def make_individual(base: Sequence[int], g_min: int, area_min: int, area_max: int) -> List[int]:
    individual = list(base)
    individual[2] = int(g_min)
    individual[6] = int(area_min)
    individual[7] = int(area_max)
    return individual


def argmax_first(values: Sequence[int], scores: Sequence[float]) -> int:
    """Value with the highest score; ties go to the first (smallest) value."""
    return int(values[int(np.argmax(np.asarray(scores, dtype=float)))])


def scan_setting(
    fitness: Callable[[Sequence[int]], float],
    base: Sequence[int],
    g_coarse: Sequence[int],
    g_fine_half_width: int,
    area_min_grid: Sequence[int],
    area_max_grid: Sequence[int],
) -> Tuple[List[int], float, pd.DataFrame]:
    cache: Dict[Tuple[int, int, int], float] = {}
    rows: List[dict] = []

    def score(stage: str, g_min: int, area_min: int, area_max: int) -> float:
        key = (int(g_min), int(area_min), int(area_max))
        if key not in cache:
            cache[key] = float(fitness(make_individual(base, *key)))
        rows.append(
            {
                "stage": stage,
                "g_min": key[0],
                "area_min": key[1],
                "area_max": key[2],
                "train_fitness": cache[key],
            }
        )
        return cache[key]

    area_min, area_max = int(base[6]), int(base[7])
    g_low, g_high = int(min(g_coarse)), int(max(g_coarse))

    def fine_values(center: int) -> List[int]:
        return list(
            range(max(g_low, center - g_fine_half_width),
                  min(g_high, center + g_fine_half_width) + 1)
        )

    g_min = argmax_first(g_coarse, [score("g_coarse", g, area_min, area_max) for g in g_coarse])
    values = fine_values(g_min)
    g_min = argmax_first(values, [score("g_fine", g, area_min, area_max) for g in values])

    grid = sorted(set(list(area_min_grid) + [area_min]))
    new_area_min = argmax_first(grid, [score("area_min", g_min, a, area_max) for a in grid])
    grid = sorted(set(list(area_max_grid) + [area_max]))
    new_area_max = argmax_first(grid, [score("area_max", g_min, new_area_min, a) for a in grid])

    if (new_area_min, new_area_max) != (area_min, area_max):
        area_min, area_max = new_area_min, new_area_max
        values = fine_values(g_min)
        g_min = argmax_first(values, [score("g_refine", g, area_min, area_max) for g in values])

    best = make_individual(base, g_min, area_min, area_max)
    return best, cache[(g_min, area_min, area_max)], pd.DataFrame(rows)


def test_statistics(test_df: pd.DataFrame) -> dict:
    _, statistics = holdout.summarize_fa_count_per_cell(test_df)
    stats = statistics.iloc[0].to_dict() if not statistics.empty else {}
    data = holdout.add_fa_count_per_cell(test_df)
    soft = data.loc[data["condition"] == "Soft", METRIC].dropna()
    stiff = data.loc[data["condition"] == "Stiff", METRIC].dropna()
    return {
        "test_n_soft": int(len(soft)),
        "test_n_stiff": int(len(stiff)),
        "test_soft_median_fa_per_cell": float(soft.median()),
        "test_stiff_median_fa_per_cell": float(stiff.median()),
        "test_soft_mean_fa_per_cell": float(soft.mean()),
        "test_stiff_mean_fa_per_cell": float(stiff.mean()),
        "test_median_fa_per_cell_all": float(data[METRIC].median()),
        "test_mann_whitney_u": stats.get("mann_whitney_u", float("nan")),
        "test_mann_whitney_p": stats.get("mann_whitney_p", float("nan")),
        "test_welch_p": stats.get("welch_p", float("nan")),
        "test_cliffs_delta": stats.get("cliffs_delta_stiff_vs_soft", float("nan")),
    }


def build_summary(base_root: Path) -> pd.DataFrame:
    rows = []
    for name in [s.name for s in sens.SETTINGS]:
        path = base_root / name / "result.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as file:
            result = json.load(file)
        if result.get("completed"):
            result["best_individual"] = json.dumps(result["best_individual"])
            rows.append(result)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    baseline = summary[summary["setting"] == "baseline"]
    if not baseline.empty:
        base = baseline.iloc[0]
        summary["g_min_shift_vs_baseline"] = summary["g_min"] - int(base["g_min"])
        for column in (
            "test_median_fa_per_cell_all",
            "test_soft_median_fa_per_cell",
            "test_stiff_median_fa_per_cell",
        ):
            reference_value = float(base[column])
            summary["pct_change_" + column] = (
                100.0 * (summary[column] - reference_value) / reference_value
            )
    summary.to_csv(base_root / "scan_summary.csv", index=False)
    return summary


def run(
    base_name: str,
    setting_names: Sequence[str],
    output_root: Path,
    split_seed: int,
    workers: int,
    perturb_seed: int,
    quick: bool,
    overwrite: bool,
) -> pd.DataFrame:
    base = BASE_CHROMOSOMES[base_name]
    base_root = Path(output_root) / base_name
    base_root.mkdir(parents=True, exist_ok=True)

    g_coarse = QUICK_G_COARSE if quick else G_COARSE
    area_min_grid = QUICK_AREA_MIN_GRID if quick else AREA_MIN_GRID
    area_max_grid = QUICK_AREA_MAX_GRID if quick else AREA_MAX_GRID
    fine_half_width = 1 if quick else G_FINE_HALF_WIDTH

    settings = [sens.SETTING_BY_NAME[name] for name in setting_names]
    pending = [
        s for s in settings
        if overwrite or not (base_root / s.name / "result.json").exists()
    ]

    print("[Reference sensitivity scan]")
    print("  script_version:", SCRIPT_VERSION, "| base:", base_name, base)
    print("  output:", base_root)
    print("  settings:", ", ".join(s.name for s in settings))
    print("  pending :", ", ".join(s.name for s in pending) or "(none)")

    if pending:
        ga_config = replace(ga.GA_CONFIG, max_workers=int(workers))
        image_paths = ga.collect_image_paths(
            data_root=ga.DATA_ROOT, excluded_folders=ga.EXCLUDED_FOLDERS
        )
        images = ga.preload_images(image_paths)
        if len(images) != len(image_paths):
            raise RuntimeError("Some images could not be loaded.")

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

        manifest = holdout.build_stratified_holdout_manifest(
            image_paths=image_paths, test_size=holdout.DEFAULT_TEST_SIZE, split_seed=split_seed
        )
        holdout.validate_holdout_manifest(manifest)
        train_paths = manifest.loc[manifest["split"] == "train", "path"].tolist()
        test_paths = manifest.loc[manifest["split"] == "test", "path"].tolist()
        train_images = holdout.select_mapping(images, train_paths)
        test_images = holdout.select_mapping(images, test_paths)
        base_train_refs = holdout.select_mapping(baseline_references, train_paths)
        base_test_refs = holdout.select_mapping(baseline_references, test_paths)

        def evaluate(individual, imgs, refs) -> float:
            return ga.evaluate_individual(
                individual, imgs, refs, ga.MATCH_RADIUS_PX,
                ga.REFERENCE_MASK_CONFIG, ga_config, ga.AREA_SOFT_LIMIT_PX,
            )

        def evaluate_by_image(individual, imgs, refs) -> pd.DataFrame:
            return ga.evaluate_best_solution(
                images=imgs, references=refs, individual=individual,
                match_radius_px=ga.MATCH_RADIUS_PX, mask_config=ga.REFERENCE_MASK_CONFIG,
                ga_config=ga_config, area_soft_limit_px=ga.AREA_SOFT_LIMIT_PX,
            )

        for setting in pending:
            start = time.time()
            setting_dir = base_root / setting.name
            setting_dir.mkdir(parents=True, exist_ok=True)

            references = sens.build_setting_references(
                setting, images, baseline_references, perturb_seed
            )
            train_refs = holdout.select_mapping(references, train_paths)
            test_refs = holdout.select_mapping(references, test_paths)
            counts = np.asarray([len(r.points) for r in references.values()], dtype=float)
            cells = np.asarray([r.n_cells for r in references.values()], dtype=float)

            best, best_fitness, curve = scan_setting(
                fitness=lambda ind: evaluate(ind, train_images, train_refs),
                base=base,
                g_coarse=g_coarse,
                g_fine_half_width=fine_half_width,
                area_min_grid=area_min_grid,
                area_max_grid=area_max_grid,
            )
            curve.insert(0, "setting", setting.name)
            curve.to_csv(setting_dir / "scan_curve.csv", index=False)

            test_df = holdout.add_fa_count_per_cell(evaluate_by_image(best, test_images, test_refs))
            test_df.insert(0, "key", test_df["path"].map(sens.relative_key))
            test_df.to_csv(setting_dir / "test_by_image.csv", index=False)
            test_micro = sens.micro_components(test_df)

            result = {
                "completed": True,
                "script_version": SCRIPT_VERSION,
                "base": base_name,
                "quick": bool(quick),
                "setting": setting.name,
                "group": setting.group,
                "description": setting.description,
                "split_seed": int(split_seed),
                "perturb_seed": int(perturb_seed),
                "mean_reference_points_per_image": float(counts.mean()),
                "mean_reference_points_per_cell": float((counts / cells).mean()),
                "best_individual": [int(v) for v in best],
                "g_min": int(best[2]),
                "area_min": int(best[6]),
                "area_max": int(best[7]),
                "g_min_at_grid_edge": bool(best[2] in (min(g_coarse), max(g_coarse))),
                "train_fitness": float(best_fitness),
                "train_fitness_of_base_solution": float(evaluate(base, train_images, train_refs)),
                "test_micro_f1": test_micro["micro_f1"],
                "test_precision_like": test_micro["precision_like"],
                "test_recall_like": test_micro["recall_like"],
                "train_fitness_on_baseline_reference": float(evaluate(best, train_images, base_train_refs)),
                "test_fitness_on_baseline_reference": float(evaluate(best, test_images, base_test_refs)),
                "test_total_detected_objects": test_micro["n_detected_objects"],
                **test_statistics(test_df),
                "n_evaluations": int(curve[["g_min", "area_min", "area_max"]].drop_duplicates().shape[0]),
                "duration_seconds": float(time.time() - start),
            }

            # Readout as a function of G min alone (does not depend on the references).
            if setting.name == "baseline":
                profile_rows = []
                for g in g_coarse:
                    individual = make_individual(base, g, base[6], base[7])
                    frame = holdout.add_fa_count_per_cell(
                        evaluate_by_image(individual, test_images, base_test_refs)
                    )
                    profile_rows.append({"g_min": int(g), **test_statistics(frame)})
                pd.DataFrame(profile_rows).to_csv(base_root / "readout_profile_by_g_min.csv", index=False)

            holdout.save_json(setting_dir / "result.json", result)
            print(
                f"[{base_name} | {setting.name}] points/image={counts.mean():.1f} "
                f"best G min={result['g_min']} area={result['area_min']}-{result['area_max']} "
                f"train F={result['train_fitness']:.4f} | test FA/cell median "
                f"Soft={result['test_soft_median_fa_per_cell']:.1f} "
                f"Stiff={result['test_stiff_median_fa_per_cell']:.1f} "
                f"MW p={result['test_mann_whitney_p']:.4g} | "
                f"{result['n_evaluations']} evaluations, {result['duration_seconds'] / 60:.1f} min",
                flush=True,
            )
            build_summary(base_root)

    return build_summary(base_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdhGen reference-point sensitivity by direct scan of G min and area bounds"
    )
    parser.add_argument("--base", choices=sorted(BASE_CHROMOSOMES), default="submitted")
    parser.add_argument("--settings", type=str, default=",".join(DEFAULT_SETTINGS))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--split-seed", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Tiny grids, for a smoke test only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    names = [name.strip() for name in args.settings.split(",") if name.strip()]
    for name in names:
        if name not in sens.SETTING_BY_NAME:
            raise SystemExit(f"Unknown setting: {name}")

    if args.summary_only:
        table = build_summary(Path(args.output_root) / args.base)
        print(f"Completed settings: {len(table)}")
        raise SystemExit(0)

    table = run(
        base_name=args.base,
        setting_names=names,
        output_root=Path(args.output_root),
        split_seed=args.split_seed,
        workers=args.workers,
        perturb_seed=args.perturb_seed,
        quick=args.quick,
        overwrite=args.overwrite,
    )
    if not table.empty:
        columns = [
            "setting", "mean_reference_points_per_image", "g_min", "area_min", "area_max",
            "train_fitness", "test_soft_median_fa_per_cell", "test_stiff_median_fa_per_cell",
            "test_mann_whitney_p", "test_cliffs_delta",
        ]
        print("\n[Summary]")
        print(table[columns].to_string(index=False))
