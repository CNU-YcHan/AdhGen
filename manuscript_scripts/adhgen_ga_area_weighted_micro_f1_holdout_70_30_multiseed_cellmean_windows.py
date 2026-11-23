r"""
AdhGen Dataset-level Micro-F1: Repeated 70:30 Hold-out
=======================================================

목적
----
1. 이미지를 Train 70% / Test 30%로 image-level hold-out 분할한다.
2. `condition + folder` 조합을 기준으로 층화한다.
3. 여러 data split seed에 대해 독립적으로 hold-out 실험을 반복한다.
4. GA objective는 거리 점수와 실제 검출 객체 면적 penalty를 결합한 dataset-level micro F1을 사용한다.
5. Test의 Soft/Stiff 사후 통계는 이미지 총 FA 수가 아니라 아래 값을 사용한다.

    mean_detected_fa_count_per_cell
        = n_detected_objects / n_cells

중요
----
- Test FA/cell 값은 각 FA 객체를 개별 cell에 직접 배정한 값이 아니다.
- 이미지 전체 AdhGen FA 수를 해당 이미지의 cell 수로 나눈
  "이미지별 세포당 평균 검출 FA 수"이다.
- 여러 split seed에서 같은 이미지가 test에 반복 포함될 수 있으므로,
  모든 seed의 test row를 하나로 합쳐 독립 표본처럼 통계 검정하지 않는다.
  통계는 split seed별로 따로 계산하고, seed 간 결과 분포를 요약한다.
- Soft/Stiff label, FA/cell 차이, ANOVA, t-test, p-value는 GA objective에
  절대 사용되지 않는다. 이들은 최적 chromosome을 test에 고정 적용한 뒤의
  사후 분석에만 사용된다.

기본 실행
---------
    python adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_windows.py

split seed 직접 지정
--------------------
    python adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_windows.py ^
        --split-seeds 42,43,44,45,46,47,48,49,50,51

빠른 동작 점검
--------------
    python adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_windows.py ^
        --split-seeds 42 ^
        --population-size 10 ^
        --generations 3 ^
        --output-root <ADHGEN_ROOT>\code\holdout_debug

필수 파일
---------
- adhgen_ga_area_weighted_micro_f1_windows.py
- adhgen_point_guided_agg_fix.py
- cell_instance_utils_dapi_all_cells.py
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent

import argparse
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, mannwhitneyu, ttest_ind

import adhgen_ga_area_weighted_micro_f1_windows as ga


SCRIPT_VERSION = "2026-07-29-area-weighted-micro-f1-holdout-70-30-multiseed-cellmean"

DATA_ROOT = ga.DATA_ROOT
OUTPUT_ROOT = Path(
    str(_SCRIPT_DIR / "adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_results")
)

DEFAULT_SPLIT_SEEDS = list(range(42, 52))
DEFAULT_TEST_SIZE = 0.30
DEFAULT_GA_SEED = 42
DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 100
DEFAULT_MAX_WORKERS = 4
DEFAULT_AREA_SOFT_LIMIT_PX = ga.AREA_SOFT_LIMIT_PX

PRIMARY_TEST_METRIC = "mean_detected_fa_count_per_cell"


# =============================================================================
# General utilities
# =============================================================================


def split_path_parts(path: str) -> List[str]:
    """Windows/Unix 경로 구분자를 모두 인식한다."""
    return [part for part in re.split(r"[\\/]+", str(path)) if part]


def extract_condition_from_path(path: str) -> str:
    parts = split_path_parts(path)
    for part in parts:
        lowered = part.lower()
        if lowered == "soft":
            return "Soft"
        if lowered == "stiff":
            return "Stiff"
    return "Unknown"


def extract_folder_id(path: str) -> str:
    """Soft/Stiff 바로 위 numbered folder 이름을 반환한다."""
    parts = split_path_parts(path)
    lowered = [part.lower() for part in parts]

    for condition_name in ("soft", "stiff"):
        if condition_name in lowered:
            index = lowered.index(condition_name)
            if index > 0:
                return str(parts[index - 1])

    return "Unknown"


def parse_integer_list(value: str) -> List[int]:
    """쉼표 구분 정수 목록을 중복 없이 입력 순서대로 반환한다."""
    parsed: List[int] = []
    seen = set()

    for token in value.split(","):
        token = token.strip()
        if not token:
            continue

        number = int(token)
        if number not in seen:
            parsed.append(number)
            seen.add(number)

    if not parsed:
        raise ValueError("하나 이상의 split seed가 필요합니다.")

    return parsed


def select_mapping(
    mapping: Mapping[str, object],
    paths: Iterable[str],
) -> Dict[str, object]:
    return {path: mapping[path] for path in sorted(paths)}


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def calculate_micro_metrics(result_df: pd.DataFrame) -> dict:
    quality_sum = float(result_df["matched_quality_sum"].sum())
    n_points = int(result_df["n_reference_points"].sum())
    n_objects = int(result_df["n_detected_objects"].sum())
    n_matched = int(result_df["n_matched"].sum())

    denominator = n_points + n_objects
    micro_f1 = (
        2.0 * quality_sum / denominator
        if denominator > 0
        else 0.0
    )

    return {
        "matched_quality_sum": quality_sum,
        "n_reference_points": n_points,
        "n_detected_objects": n_objects,
        "n_matched": n_matched,
        "micro_f1": float(micro_f1),
        "macro_image_score": float(result_df["image_score"].mean()),
        "median_image_score": float(result_df["image_score"].median()),
    }


def add_result_metadata(
    result_df: pd.DataFrame,
    split_seed: int,
    split_name: str,
    chromosome: Sequence[int],
) -> pd.DataFrame:
    output = result_df.copy()
    output.insert(0, "split_seed", int(split_seed))
    output.insert(1, "split", split_name)

    chromosome_names = [
        "r_min",
        "r_max",
        "g_min",
        "g_max",
        "b_min",
        "b_max",
        "area_min",
        "area_max",
    ]

    for name, value in zip(chromosome_names, chromosome):
        output[name] = int(value)

    output["folder_id"] = output["path"].map(extract_folder_id)
    return add_fa_count_per_cell(output)


# =============================================================================
# Hold-out split
# =============================================================================


def build_stratified_holdout_manifest(
    image_paths: Sequence[str],
    test_size: float,
    split_seed: int,
) -> pd.DataFrame:
    """
    `condition + folder`를 기준으로 층화된 image-level hold-out을 만든다.

    전체 test 이미지 수는 round(N * test_size)로 맞추고, 각 stratum의
    test 수는 비례 배분 후 largest-remainder 방식으로 조정한다. 각
    stratum에는 최소 1개의 train과 1개의 test가 남는다.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size는 0과 1 사이여야 합니다.")

    rows = []
    for path in sorted(image_paths):
        condition = extract_condition_from_path(path)
        folder_id = extract_folder_id(path)
        rows.append(
            {
                "path": str(path),
                "condition": condition,
                "folder_id": folder_id,
                "stratum": f"{condition}_folder_{folder_id}",
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("Split할 이미지가 없습니다.")

    if (manifest["condition"] == "Unknown").any():
        unknown_paths = manifest.loc[
            manifest["condition"] == "Unknown", "path"
        ]
        raise ValueError(
            "Soft/Stiff condition을 경로에서 확인할 수 없는 이미지가 "
            "있습니다.\n" + unknown_paths.to_string(index=False)
        )

    if manifest["path"].duplicated().any():
        duplicates = manifest.loc[
            manifest["path"].duplicated(keep=False), "path"
        ]
        raise RuntimeError(
            "중복 이미지 경로가 발견됐습니다:\n"
            + duplicates.to_string(index=False)
        )

    stratum_counts = (
        manifest.groupby("stratum", observed=True)
        .size()
        .sort_index()
    )
    insufficient = stratum_counts[stratum_counts < 2]
    if not insufficient.empty:
        raise ValueError(
            "Hold-out 층화를 위해 train/test에 하나씩 배정할 수 없는 "
            "stratum이 있습니다.\n"
            + insufficient.to_string()
        )

    target_n_test = int(np.rint(len(manifest) * test_size))
    min_possible = int(len(stratum_counts))
    max_possible = int((stratum_counts - 1).sum())
    if not min_possible <= target_n_test <= max_possible:
        raise ValueError(
            "요청한 전체 test 수를 모든 stratum에 train/test를 하나씩 "
            "유지하면서 만들 수 없습니다. "
            f"target={target_n_test}, feasible={min_possible}-{max_possible}"
        )

    allocation = pd.DataFrame(
        {
            "n_total": stratum_counts.astype(int),
            "ideal_test": stratum_counts.astype(float) * float(test_size),
        }
    )
    allocation["n_test"] = np.floor(
        allocation["ideal_test"]
    ).astype(int)
    allocation["n_test"] = np.maximum(
        allocation["n_test"], 1
    )
    allocation["n_test"] = np.minimum(
        allocation["n_test"], allocation["n_total"] - 1
    ).astype(int)

    # 전체 test 수가 정확히 target_n_test가 되도록 조정한다.
    while int(allocation["n_test"].sum()) < target_n_test:
        candidates = allocation[
            allocation["n_test"] < allocation["n_total"] - 1
        ].copy()
        if candidates.empty:
            raise RuntimeError("Test allocation을 증가시킬 수 없습니다.")
        candidates["priority"] = (
            candidates["ideal_test"] - candidates["n_test"]
        )
        selected = candidates.sort_values(
            ["priority", "ideal_test", "n_total"],
            ascending=[False, False, False],
        ).index[0]
        allocation.loc[selected, "n_test"] += 1

    while int(allocation["n_test"].sum()) > target_n_test:
        candidates = allocation[allocation["n_test"] > 1].copy()
        if candidates.empty:
            raise RuntimeError("Test allocation을 감소시킬 수 없습니다.")
        candidates["priority"] = (
            candidates["n_test"] - candidates["ideal_test"]
        )
        selected = candidates.sort_values(
            ["priority", "ideal_test", "n_total"],
            ascending=[False, True, True],
        ).index[0]
        allocation.loc[selected, "n_test"] -= 1

    manifest["split"] = "train"
    rng = np.random.default_rng(int(split_seed))

    for stratum, group in manifest.groupby(
        "stratum", sort=True, observed=True
    ):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_test = int(allocation.loc[stratum, "n_test"])
        manifest.loc[indices[:n_test], "split"] = "test"

    return manifest.sort_values(
        ["split", "stratum", "path"]
    ).reset_index(drop=True)


def validate_holdout_manifest(manifest: pd.DataFrame) -> dict:
    train_df = manifest[manifest["split"] == "train"]
    test_df = manifest[manifest["split"] == "test"]

    train_paths = set(train_df["path"])
    test_paths = set(test_df["path"])
    all_paths = set(manifest["path"])

    if train_paths & test_paths:
        raise RuntimeError("Train/Test image overlap이 있습니다.")
    if train_paths | test_paths != all_paths:
        raise RuntimeError("Train/Test union이 전체 이미지와 다릅니다.")
    if not train_paths or not test_paths:
        raise RuntimeError("Train 또는 Test가 비어 있습니다.")

    stratum_table = (
        manifest.groupby(["stratum", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    if "train" not in stratum_table or "test" not in stratum_table:
        raise RuntimeError("층화 결과에 train 또는 test가 없습니다.")
    if (stratum_table["train"] == 0).any() or (
        stratum_table["test"] == 0
    ).any():
        raise RuntimeError(
            "일부 stratum이 train 또는 test에 포함되지 않았습니다."
        )

    return {
        "n_total": int(len(manifest)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_fraction": float(len(train_df) / len(manifest)),
        "test_fraction": float(len(test_df) / len(manifest)),
        "train_soft": int((train_df["condition"] == "Soft").sum()),
        "train_stiff": int((train_df["condition"] == "Stiff").sum()),
        "test_soft": int((test_df["condition"] == "Soft").sum()),
        "test_stiff": int((test_df["condition"] == "Stiff").sum()),
        "n_strata": int(manifest["stratum"].nunique()),
    }


# =============================================================================
# FA count per cell and post-hoc statistics
# =============================================================================


def add_fa_count_per_cell(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    이미지별 세포당 평균 AdhGen FA 수를 추가한다.

    mean_detected_fa_count_per_cell
        = n_detected_objects / n_cells

    n_cells <= 0인 행은 NaN으로 처리하여 0으로 나누는 오류와 잘못된
    생물학적 해석을 방지한다.
    """
    required = {"n_detected_objects", "n_cells"}
    missing = required - set(result_df.columns)
    if missing:
        raise KeyError(f"필수 열이 없습니다: {sorted(missing)}")

    output = result_df.copy()
    n_objects = pd.to_numeric(
        output["n_detected_objects"], errors="coerce"
    ).astype(float)
    n_cells = pd.to_numeric(output["n_cells"], errors="coerce").astype(
        float
    )

    output[PRIMARY_TEST_METRIC] = np.where(
        n_cells > 0,
        n_objects / n_cells,
        np.nan,
    )
    return output


def summarize_fa_count_per_cell(
    result_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    이미지별 세포당 평균 FA 수를 조건별로 요약하고 Soft/Stiff를 비교한다.

    출력 통계:
    - One-way ANOVA p-value만 저장
    - Welch's t-test
    - Mann-Whitney U test
    - Cliff's delta (Stiff vs Soft)
    """
    data = add_fa_count_per_cell(result_df)

    summary = (
        data.groupby("condition", observed=True)
        .agg(
            n_images=("path", "count"),
            n_valid_images=(PRIMARY_TEST_METRIC, "count"),
            mean_fa_count_per_cell=(PRIMARY_TEST_METRIC, "mean"),
            sd_fa_count_per_cell=(PRIMARY_TEST_METRIC, "std"),
            median_fa_count_per_cell=(PRIMARY_TEST_METRIC, "median"),
            mean_total_detected_objects=("n_detected_objects", "mean"),
            median_total_detected_objects=("n_detected_objects", "median"),
            mean_n_cells=("n_cells", "mean"),
            median_n_cells=("n_cells", "median"),
        )
        .reset_index()
    )

    soft = (
        data.loc[data["condition"] == "Soft", PRIMARY_TEST_METRIC]
        .dropna()
        .to_numpy(dtype=float)
    )
    stiff = (
        data.loc[data["condition"] == "Stiff", PRIMARY_TEST_METRIC]
        .dropna()
        .to_numpy(dtype=float)
    )

    rows: List[dict] = []
    if len(soft) > 0 and len(stiff) > 0:
        anova = f_oneway(soft, stiff)
        welch = ttest_ind(
            stiff,
            soft,
            equal_var=False,
            nan_policy="omit",
        )
        mw = mannwhitneyu(
            stiff,
            soft,
            alternative="two-sided",
            method="auto",
        )

        rows.append(
            {
                "metric": PRIMARY_TEST_METRIC,
                "comparison": "Stiff - Soft",
                "n_soft": int(len(soft)),
                "n_stiff": int(len(stiff)),
                "soft_mean": float(np.mean(soft)),
                "stiff_mean": float(np.mean(stiff)),
                "soft_sd": float(np.std(soft, ddof=1))
                if len(soft) > 1
                else np.nan,
                "stiff_sd": float(np.std(stiff, ddof=1))
                if len(stiff) > 1
                else np.nan,
                "soft_median": float(np.median(soft)),
                "stiff_median": float(np.median(stiff)),
                "mean_difference": float(np.mean(stiff) - np.mean(soft)),
                # 사용자 요청에 따라 ANOVA는 p-value만 출력한다.
                "anova_p": float(anova.pvalue),
                "welch_t": float(welch.statistic),
                "welch_p": float(welch.pvalue),
                "mann_whitney_u": float(mw.statistic),
                "mann_whitney_p": float(mw.pvalue),
                "cliffs_delta_stiff_vs_soft": float(
                    2.0
                    * float(mw.statistic)
                    / (len(stiff) * len(soft))
                    - 1.0
                ),
            }
        )

    return summary, pd.DataFrame(rows)


# =============================================================================
# Output aggregation and plots
# =============================================================================


def plot_seed_metrics(summary_df: pd.DataFrame, output_root: Path) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df.sort_values("split_seed")
    x = np.arange(len(plot_df))
    labels = plot_df["split_seed"].astype(str).tolist()

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x, plot_df["train_micro_f1"], marker="o", label="Train")
    axis.plot(x, plot_df["test_micro_f1"], marker="o", label="Test")
    axis.set_xlabel("Data split seed")
    axis.set_ylabel("Dataset-level micro F1")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_root / "train_test_micro_f1_by_split_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        x,
        plot_df["test_fa_per_cell_mean_difference"],
        marker="o",
    )
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Data split seed")
    axis.set_ylabel("Stiff - Soft mean FA count/cell")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_root / "test_fa_per_cell_difference_by_split_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_multiseed_aggregate_summary(seed_summary_df: pd.DataFrame) -> dict:
    """Seed별 p-value를 평균하지 않고 반복 실험의 안정성을 요약한다."""
    if seed_summary_df.empty:
        return {}

    return {
        "n_split_seeds": int(len(seed_summary_df)),
        "test_micro_f1_mean": float(seed_summary_df["test_micro_f1"].mean()),
        "test_micro_f1_sd": float(seed_summary_df["test_micro_f1"].std(ddof=1))
        if len(seed_summary_df) > 1
        else 0.0,
        "test_fa_per_cell_difference_mean": float(
            seed_summary_df["test_fa_per_cell_mean_difference"].mean()
        ),
        "test_fa_per_cell_difference_sd": float(
            seed_summary_df["test_fa_per_cell_mean_difference"].std(ddof=1)
        )
        if len(seed_summary_df) > 1
        else 0.0,
        "n_positive_stiff_minus_soft": int(
            (seed_summary_df["test_fa_per_cell_mean_difference"] > 0).sum()
        ),
        "n_anova_p_lt_0_05": int(
            (seed_summary_df["test_anova_p"] < 0.05).sum()
        ),
        "n_welch_p_lt_0_05": int(
            (seed_summary_df["test_welch_p"] < 0.05).sum()
        ),
        "n_mann_whitney_p_lt_0_05": int(
            (seed_summary_df["test_mann_whitney_p"] < 0.05).sum()
        ),
        "median_anova_p": float(seed_summary_df["test_anova_p"].median()),
        "median_welch_p": float(seed_summary_df["test_welch_p"].median()),
        "median_mann_whitney_p": float(
            seed_summary_df["test_mann_whitney_p"].median()
        ),
        "note": (
            "p-values are calculated separately for each split seed and are "
            "not pooled across repeated test appearances"
        ),
    }


def load_completed_seed(seed_dir: Path) -> Tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    test_path = seed_dir / "test_best_solution_by_image.csv"
    seed_summary_path = seed_dir / "seed_summary.json"
    condition_summary_path = seed_dir / "test_condition_summary.csv"
    statistics_path = seed_dir / "test_posthoc_statistics.csv"

    required = [
        test_path,
        seed_summary_path,
        condition_summary_path,
        statistics_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "완료된 split seed 결과가 불완전합니다:\n" + "\n".join(missing)
        )

    test_df = pd.read_csv(test_path)
    with open(seed_summary_path, "r", encoding="utf-8") as file:
        seed_summary = json.load(file)
    condition_summary = pd.read_csv(condition_summary_path)
    statistics = pd.read_csv(statistics_path)
    return test_df, seed_summary, condition_summary, statistics


def completed_seed_is_compatible(
    seed_dir: Path,
    area_soft_limit_px: float,
) -> bool:
    """이전 objective/settings 결과가 현재 실행에 섞이는 것을 방지한다."""
    metadata_path = seed_dir / "seed_metadata.json"
    if not metadata_path.exists():
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    point_config = metadata.get("point_config", {})
    return bool(
        metadata.get("script_version") == SCRIPT_VERSION
        and metadata.get("base_ga_version") == ga.SCRIPT_VERSION
        and np.isclose(
            float(metadata.get("match_radius_px", np.nan)),
            float(ga.MATCH_RADIUS_PX),
        )
        and int(point_config.get("min_distance_px", -1))
        == int(ga.POINT_CONFIG_FINAL.min_distance_px)
        and np.isclose(
            float(metadata.get("area_soft_limit_px", np.nan)),
            float(area_soft_limit_px),
        )
        and metadata.get("training_objective")
        == "dataset_level_distance_and_area_weighted_micro_f1"
    )


# =============================================================================
# Main repeated hold-out routine
# =============================================================================


def run_repeated_holdout(
    data_root: str | Path = DATA_ROOT,
    output_root: str | Path = OUTPUT_ROOT,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    test_size: float = DEFAULT_TEST_SIZE,
    ga_seed: int = DEFAULT_GA_SEED,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    workers: int = DEFAULT_MAX_WORKERS,
    area_soft_limit_px: float = DEFAULT_AREA_SOFT_LIMIT_PX,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_seeds = [int(seed) for seed in split_seeds]
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split_seeds에 중복이 있습니다.")
    if area_soft_limit_px <= 0:
        raise ValueError("area_soft_limit_px는 0보다 커야 합니다.")

    ga_config = replace(
        ga.GA_CONFIG,
        population_size=int(population_size),
        generations=int(generations),
        seed=int(ga_seed),
        max_workers=int(workers),
    )

    print("[Runtime identity]")
    print("  script_version:", SCRIPT_VERSION)
    print("  script_file:", Path(__file__).resolve())
    print("  base_ga_version:", ga.SCRIPT_VERSION)
    print("  base_ga_file:", Path(ga.__file__).resolve())
    print("  utility_file:", Path(ga.cell_utils.__file__).resolve())

    print("\n[Repeated hold-out configuration]")
    print("  train:test =", f"{1.0 - test_size:.2f}:{test_size:.2f}")
    print("  split_seeds:", split_seeds)
    print("  stratification: condition + folder")
    print("  split_unit: image")
    print("  primary_test_metric:", PRIMARY_TEST_METRIC)
    print("  green_point_min_distance_px:", ga.POINT_CONFIG_FINAL.min_distance_px)
    print("  match_radius_px:", ga.MATCH_RADIUS_PX)
    print("  area_soft_limit_px:", area_soft_limit_px)
    print("  area_weight: min(1, area_soft_limit_px / detected_object_area)")
    print("  overwrite:", overwrite)

    print("\n[GA configuration]")
    for key, value in asdict(ga_config).items():
        print(f"  {key}: {value}")

    print("\n[Paths]")
    print("  data_root:", Path(data_root))
    print("  output_root:", output_root)

    image_paths = ga.collect_image_paths(
        data_root=data_root,
        excluded_folders=ga.EXCLUDED_FOLDERS,
    )
    if not image_paths:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {data_root}")

    print("\n수집된 이미지:", len(image_paths))

    # 이미지와 fixed reference는 모든 seed에서 동일하므로 한 번만 준비한다.
    images = ga.preload_images(image_paths)
    if len(images) != len(image_paths):
        raise RuntimeError(
            "일부 이미지를 불러오지 못했습니다. "
            f"paths={len(image_paths)}, loaded={len(images)}"
        )

    first_path = sorted(images)[0]
    preflight = ga.build_image_reference(
        first_path,
        images[first_path],
        ga.POINT_CONFIG_FINAL,
        ga.NUCLEUS_CONFIG_FINAL,
        ga.CELL_CONFIG_FINAL,
        ga.REFERENCE_MASK_CONFIG,
    )
    print(
        "\n[Preflight OK] "
        f"{Path(first_path).name}: "
        f"nuclei={preflight.n_nuclei}, "
        f"cells={preflight.n_cells}, "
        f"points={len(preflight.points)}"
    )

    print("\nFixed reference 생성 중...")
    references, failure_df = ga.precompute_references(
        images=images,
        point_config=ga.POINT_CONFIG_FINAL,
        nucleus_config=ga.NUCLEUS_CONFIG_FINAL,
        cell_config=ga.CELL_CONFIG_FINAL,
        mask_config=ga.REFERENCE_MASK_CONFIG,
        max_workers=workers,
    )
    failure_df.to_csv(output_root / "reference_failures.csv", index=False)

    if not failure_df.empty:
        raise RuntimeError(
            "Reference 생성 실패 이미지가 있습니다. "
            "reference_failures.csv를 확인하세요."
        )
    if len(references) != len(images):
        raise RuntimeError("Reference 수와 image 수가 다릅니다.")

    ga.validate_reference_images(references)
    reference_df = ga.reference_summary_dataframe(references)
    reference_df["folder_id"] = reference_df["path"].map(extract_folder_id)
    reference_df.to_csv(
        output_root / "fixed_reference_summary.csv", index=False
    )

    all_test_frames: List[pd.DataFrame] = []
    seed_summary_rows: List[dict] = []
    all_test_condition_frames: List[pd.DataFrame] = []
    all_test_statistics_frames: List[pd.DataFrame] = []

    for split_seed in split_seeds:
        seed_dir = output_root / f"split_seed_{split_seed}"
        completed_marker = seed_dir / "SEED_COMPLETED.json"

        if completed_marker.exists() and not overwrite:
            if completed_seed_is_compatible(seed_dir, area_soft_limit_px):
                print(
                    f"\n[Split seed {split_seed}] 현재 설정과 호환되는 "
                    "완료 결과를 불러옵니다."
                )
                (
                    test_df,
                    seed_summary,
                    test_condition_summary,
                    test_statistics,
                ) = load_completed_seed(seed_dir)

                all_test_frames.append(test_df)
                seed_summary_rows.append(seed_summary)
                test_condition_summary.insert(0, "split_seed", int(split_seed))
                test_statistics.insert(0, "split_seed", int(split_seed))
                all_test_condition_frames.append(test_condition_summary)
                all_test_statistics_frames.append(test_statistics)
                continue

            print(
                f"\n[Split seed {split_seed}] 기존 완료 결과가 현재 "
                "area-weighted 설정과 호환되지 않아 다시 실행합니다."
            )

        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_start_time = time.time()

        manifest = build_stratified_holdout_manifest(
            image_paths=image_paths,
            test_size=test_size,
            split_seed=split_seed,
        )
        split_summary = validate_holdout_manifest(manifest)
        manifest.to_csv(seed_dir / "split_manifest.csv", index=False)
        save_json(seed_dir / "split_summary.json", split_summary)

        train_paths = manifest.loc[
            manifest["split"] == "train", "path"
        ].tolist()
        test_paths = manifest.loc[
            manifest["split"] == "test", "path"
        ].tolist()

        train_images = select_mapping(images, train_paths)
        test_images = select_mapping(images, test_paths)
        train_references = select_mapping(references, train_paths)
        test_references = select_mapping(references, test_paths)

        print(
            f"\n{'=' * 76}\n"
            f"Data split seed: {split_seed}\n"
            f"Train images: {len(train_paths)} "
            f"({split_summary['train_fraction']:.3f})\n"
            f"Test images: {len(test_paths)} "
            f"({split_summary['test_fraction']:.3f})\n"
            f"{'=' * 76}"
        )

        best_individual, train_best_fitness, history_df = ga.run_ga(
            images=train_images,
            references=train_references,
            match_radius_px=ga.MATCH_RADIUS_PX,
            mask_config=ga.REFERENCE_MASK_CONFIG,
            ga_config=ga_config,
            area_soft_limit_px=area_soft_limit_px,
        )

        history_df.insert(0, "split_seed", int(split_seed))
        history_df.to_csv(seed_dir / "ga_history.csv", index=False)
        ga.plot_ga_history(history_df, seed_dir / "ga_history.png")

        train_result_df = ga.evaluate_best_solution(
            images=train_images,
            references=train_references,
            individual=best_individual,
            match_radius_px=ga.MATCH_RADIUS_PX,
            mask_config=ga.REFERENCE_MASK_CONFIG,
            ga_config=ga_config,
            area_soft_limit_px=area_soft_limit_px,
        )
        test_result_df = ga.evaluate_best_solution(
            images=test_images,
            references=test_references,
            individual=best_individual,
            match_radius_px=ga.MATCH_RADIUS_PX,
            mask_config=ga.REFERENCE_MASK_CONFIG,
            ga_config=ga_config,
            area_soft_limit_px=area_soft_limit_px,
        )

        train_result_df = add_result_metadata(
            train_result_df,
            split_seed=split_seed,
            split_name="train",
            chromosome=best_individual,
        )
        test_result_df = add_result_metadata(
            test_result_df,
            split_seed=split_seed,
            split_name="test",
            chromosome=best_individual,
        )

        train_result_df.to_csv(
            seed_dir / "train_best_solution_by_image.csv", index=False
        )
        test_result_df.to_csv(
            seed_dir / "test_best_solution_by_image.csv", index=False
        )

        # Primary biological post-hoc analysis: FA count / cell / image
        train_condition_summary, train_statistics = (
            summarize_fa_count_per_cell(train_result_df)
        )
        test_condition_summary, test_statistics = (
            summarize_fa_count_per_cell(test_result_df)
        )

        train_condition_summary.to_csv(
            seed_dir / "train_condition_summary.csv", index=False
        )
        test_condition_summary.to_csv(
            seed_dir / "test_condition_summary.csv", index=False
        )
        train_statistics.to_csv(
            seed_dir / "train_posthoc_statistics.csv", index=False
        )
        test_statistics.to_csv(
            seed_dir / "test_posthoc_statistics.csv", index=False
        )

        # 총 FA count 기반 결과도 보조 파일로 보존한다.
        train_total_summary, train_total_statistics = ga.summarize_best_solution(
            train_result_df
        )
        test_total_summary, test_total_statistics = ga.summarize_best_solution(
            test_result_df
        )
        train_total_summary.to_csv(
            seed_dir / "train_total_count_condition_summary.csv", index=False
        )
        test_total_summary.to_csv(
            seed_dir / "test_total_count_condition_summary.csv", index=False
        )
        train_total_statistics.to_csv(
            seed_dir / "train_total_count_posthoc_statistics.csv", index=False
        )
        test_total_statistics.to_csv(
            seed_dir / "test_total_count_posthoc_statistics.csv", index=False
        )

        train_metrics = calculate_micro_metrics(train_result_df)
        test_metrics = calculate_micro_metrics(test_result_df)

        ga.save_best_previews(
            images=test_images,
            references=test_references,
            result_df=test_result_df,
            individual=best_individual,
            match_radius_px=ga.MATCH_RADIUS_PX,
            mask_config=ga.REFERENCE_MASK_CONFIG,
            ga_config=ga_config,
            output_dir=seed_dir / "test_previews",
            area_soft_limit_px=area_soft_limit_px,
        )

        duration_seconds = float(time.time() - seed_start_time)

        if test_statistics.empty:
            raise RuntimeError(
                f"Split seed {split_seed}: Test Soft/Stiff 통계를 계산하지 "
                "못했습니다."
            )

        test_stats_row = test_statistics.iloc[0]
        train_stats_row = (
            train_statistics.iloc[0] if not train_statistics.empty else None
        )

        seed_summary = {
            "split_seed": int(split_seed),
            "ga_seed": int(ga_seed),
            "green_point_min_distance_px": int(
                ga.POINT_CONFIG_FINAL.min_distance_px
            ),
            "match_radius_px": float(ga.MATCH_RADIUS_PX),
            "area_soft_limit_px": float(area_soft_limit_px),
            **split_summary,
            "best_individual": [int(value) for value in best_individual],
            "r_min": int(best_individual[0]),
            "r_max": int(best_individual[1]),
            "g_min": int(best_individual[2]),
            "g_max": int(best_individual[3]),
            "b_min": int(best_individual[4]),
            "b_max": int(best_individual[5]),
            "area_min": int(best_individual[6]),
            "area_max": int(best_individual[7]),
            "train_best_fitness": float(train_best_fitness),
            "train_micro_f1": float(train_metrics["micro_f1"]),
            "test_micro_f1": float(test_metrics["micro_f1"]),
            "train_macro_image_score": float(
                train_metrics["macro_image_score"]
            ),
            "test_macro_image_score": float(test_metrics["macro_image_score"]),
            "train_fa_per_cell_mean_difference": float(
                train_stats_row["mean_difference"]
            )
            if train_stats_row is not None
            else np.nan,
            "test_soft_mean_fa_per_cell": float(test_stats_row["soft_mean"]),
            "test_stiff_mean_fa_per_cell": float(test_stats_row["stiff_mean"]),
            "test_fa_per_cell_mean_difference": float(
                test_stats_row["mean_difference"]
            ),
            "test_anova_p": float(test_stats_row["anova_p"]),
            "test_welch_p": float(test_stats_row["welch_p"]),
            "test_mann_whitney_p": float(
                test_stats_row["mann_whitney_p"]
            ),
            "test_cliffs_delta": float(
                test_stats_row["cliffs_delta_stiff_vs_soft"]
            ),
            "duration_seconds": duration_seconds,
        }
        save_json(seed_dir / "seed_summary.json", seed_summary)

        seed_metadata = {
            "script_version": SCRIPT_VERSION,
            "base_ga_version": ga.SCRIPT_VERSION,
            "split_seed": int(split_seed),
            "test_size_requested": float(test_size),
            "split_summary": split_summary,
            "ga_seed": int(ga_seed),
            "best_individual": [int(value) for value in best_individual],
            "train_best_fitness": float(train_best_fitness),
            "green_point_min_distance_px": int(
                ga.POINT_CONFIG_FINAL.min_distance_px
            ),
            "match_radius_px": float(ga.MATCH_RADIUS_PX),
            "area_soft_limit_px": float(area_soft_limit_px),
            "area_weight_formula": "min(1, A0 / detected_object_area)",
            "training_objective": (
                "dataset_level_distance_and_area_weighted_micro_f1"
            ),
            "primary_test_metric": PRIMARY_TEST_METRIC,
            "primary_test_metric_formula": (
                "n_detected_objects / n_cells for each image"
            ),
            "population_size": int(ga_config.population_size),
            "generations": int(ga_config.generations),
            "ga_config": asdict(ga_config),
            "point_config": asdict(ga.POINT_CONFIG_FINAL),
            "nucleus_config": asdict(ga.NUCLEUS_CONFIG_FINAL),
            "cell_config": asdict(ga.CELL_CONFIG_FINAL),
            "mask_config": asdict(ga.REFERENCE_MASK_CONFIG),
            "stratification": "condition_plus_folder",
            "split_unit": "image",
            "train_paths": train_paths,
            "test_paths": test_paths,
        }
        save_json(seed_dir / "seed_metadata.json", seed_metadata)

        save_json(
            completed_marker,
            {
                "split_seed": int(split_seed),
                "completed": True,
                "duration_seconds": duration_seconds,
            },
        )

        all_test_frames.append(test_result_df)
        seed_summary_rows.append(seed_summary)

        test_condition_for_all = test_condition_summary.copy()
        test_condition_for_all.insert(0, "split_seed", int(split_seed))
        all_test_condition_frames.append(test_condition_for_all)

        test_statistics_for_all = test_statistics.copy()
        test_statistics_for_all.insert(0, "split_seed", int(split_seed))
        all_test_statistics_frames.append(test_statistics_for_all)

        print(f"\n[Split seed {split_seed} completed]")
        print("  best_individual:", best_individual)
        print("  train_micro_f1:", train_metrics["micro_f1"])
        print("  test_micro_f1:", test_metrics["micro_f1"])
        print(
            "  test Soft mean FA/cell:",
            test_stats_row["soft_mean"],
        )
        print(
            "  test Stiff mean FA/cell:",
            test_stats_row["stiff_mean"],
        )
        print("  test ANOVA p:", test_stats_row["anova_p"])
        print("  test Welch p:", test_stats_row["welch_p"])
        print(
            "  test Mann-Whitney p:",
            test_stats_row["mann_whitney_p"],
        )
        print("  duration_seconds:", duration_seconds)

    if not seed_summary_rows:
        raise RuntimeError("완료되거나 실행된 split seed가 없습니다.")

    seed_summary_df = pd.DataFrame(seed_summary_rows).sort_values(
        "split_seed"
    )
    seed_summary_df.to_csv(
        output_root / "holdout_seed_summary.csv", index=False
    )

    all_test_df = pd.concat(all_test_frames, ignore_index=True).sort_values(
        ["split_seed", "condition", "path"]
    )
    # 동일 이미지는 서로 다른 split seed의 test에 반복 포함될 수 있다.
    all_test_df.to_csv(
        output_root / "all_test_predictions_across_seeds.csv", index=False
    )

    all_test_condition_df = pd.concat(
        all_test_condition_frames, ignore_index=True
    ).sort_values(["split_seed", "condition"])
    all_test_condition_df.to_csv(
        output_root / "test_condition_summary_across_seeds.csv", index=False
    )

    all_test_statistics_df = pd.concat(
        all_test_statistics_frames, ignore_index=True
    ).sort_values("split_seed")
    all_test_statistics_df.to_csv(
        output_root / "test_statistics_across_seeds.csv", index=False
    )

    chromosome_columns = [
        "split_seed",
        "r_min",
        "r_max",
        "g_min",
        "g_max",
        "b_min",
        "b_max",
        "area_min",
        "area_max",
        "train_best_fitness",
        "test_micro_f1",
    ]
    seed_summary_df[chromosome_columns].to_csv(
        output_root / "chromosomes_across_split_seeds.csv", index=False
    )

    aggregate_summary = build_multiseed_aggregate_summary(seed_summary_df)
    save_json(
        output_root / "multiseed_aggregate_summary.json",
        aggregate_summary,
    )

    plot_seed_metrics(seed_summary_df, output_root)

    run_metadata = {
        "script_version": SCRIPT_VERSION,
        "base_ga_version": ga.SCRIPT_VERSION,
        "split_method": "repeated_stratified_image_level_holdout",
        "train_fraction_requested": float(1.0 - test_size),
        "test_fraction_requested": float(test_size),
        "split_seeds": [int(seed) for seed in split_seeds],
        "ga_seed": int(ga_seed),
        "population_size": int(population_size),
        "generations": int(generations),
        "workers": int(workers),
        "green_point_min_distance_px": int(
            ga.POINT_CONFIG_FINAL.min_distance_px
        ),
        "match_radius_px": float(ga.MATCH_RADIUS_PX),
        "area_soft_limit_px": float(area_soft_limit_px),
        "area_weight_formula": "min(1, A0 / detected_object_area)",
        "training_objective": (
            "dataset_level_distance_and_area_weighted_micro_f1"
        ),
        "primary_test_metric": PRIMARY_TEST_METRIC,
        "primary_test_metric_formula": (
            "n_detected_objects / n_cells for each image"
        ),
        "stratification": "condition_plus_folder",
        "split_unit": "image",
        "warning": (
            "Do not pool all_test_predictions_across_seeds.csv rows as "
            "independent observations because images may repeat across seeds."
        ),
    }
    save_json(output_root / "run_metadata.json", run_metadata)

    print("\n[Repeated hold-out completed]")
    print("  output_root:", output_root.resolve())
    print("\n[Seed summary]")
    display_columns = [
        "split_seed",
        "n_train",
        "n_test",
        "train_micro_f1",
        "test_micro_f1",
        "test_soft_mean_fa_per_cell",
        "test_stiff_mean_fa_per_cell",
        "test_fa_per_cell_mean_difference",
        "test_anova_p",
        "test_welch_p",
        "test_mann_whitney_p",
        "test_cliffs_delta",
    ]
    print(seed_summary_df[display_columns].to_string(index=False))

    print("\n[Across-seed aggregate summary]")
    for key, value in aggregate_summary.items():
        print(f"  {key}: {value}")

    return seed_summary_df, all_test_df


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AdhGen repeated stratified 70:30 image-level hold-out with "
            "test FA count normalized by image cell count"
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DATA_ROOT),
        help="Merged image dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
        help="결과 저장 경로",
    )
    parser.add_argument(
        "--split-seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SPLIT_SEEDS),
        help=(
            "쉼표로 구분한 data split seed 목록. 예: "
            "42,43,44,45,46,47,48,49,50,51"
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test 비율. 기본값 0.30",
    )
    parser.add_argument(
        "--ga-seed",
        type=int,
        default=DEFAULT_GA_SEED,
        help="모든 split에서 사용할 GA random seed",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
    )
    parser.add_argument(
        "--area-soft-limit-px",
        type=float,
        default=DEFAULT_AREA_SOFT_LIMIT_PX,
        help=(
            "이 면적까지 penalty 없음. 초과 객체는 "
            "min(1, A0/area) 가중치 적용. 기본값 60"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="완료된 split seed 결과도 다시 실행한다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_seeds = parse_integer_list(args.split_seeds)

    run_repeated_holdout(
        data_root=args.data_root,
        output_root=args.output_root,
        split_seeds=split_seeds,
        test_size=args.test_size,
        ga_seed=args.ga_seed,
        population_size=args.population_size,
        generations=args.generations,
        workers=args.workers,
        area_soft_limit_px=args.area_soft_limit_px,
        overwrite=args.overwrite,
    )
