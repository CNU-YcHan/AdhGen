from __future__ import annotations
import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from . import optimizer as ga

DEFAULT_SPLIT_SEEDS = list(range(52, 62))
DEFAULT_TEST_SIZE = 0.30
PRIMARY_METRIC = "mean_detected_fa_count_per_cell"


def _parts(path: str) -> list[str]:
    return [p for p in re.split(r"[\\/]+", str(path)) if p]


def condition_from_path(path: str) -> str:
    for p in _parts(path):
        if p.lower() == "soft": return "Soft"
        if p.lower() == "stiff": return "Stiff"
    return "Unknown"


def folder_from_path(path: str) -> str:
    parts = _parts(path)
    lower = [p.lower() for p in parts]
    for name in ("soft", "stiff"):
        if name in lower:
            i = lower.index(name)
            if i > 0: return parts[i - 1]
    return "Unknown"


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for token in value.split(","):
        token = token.strip()
        if token and int(token) not in seeds: seeds.append(int(token))
    if not seeds: raise ValueError("At least one split seed is required")
    return seeds


def select(mapping: Mapping[str, object], paths: Iterable[str]) -> dict[str, object]:
    return {p: mapping[p] for p in sorted(paths)}


def build_manifest(image_paths: Sequence[str], test_size: float, split_seed: int) -> pd.DataFrame:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rows = [{"path": str(p), "condition": condition_from_path(p), "folder_id": folder_from_path(p)} for p in sorted(image_paths)]
    m = pd.DataFrame(rows)
    if m.empty: raise ValueError("No images available for splitting")
    if (m["condition"] == "Unknown").any(): raise ValueError("Condition could not be inferred from one or more paths")
    if m["path"].duplicated().any(): raise ValueError("Duplicate image paths detected")
    m["stratum"] = m["condition"] + "_folder_" + m["folder_id"].astype(str)
    counts = m.groupby("stratum", observed=True).size().sort_index()
    if (counts < 2).any(): raise ValueError("Every condition+folder stratum must contain at least two images")
    target = int(np.rint(len(m) * test_size))
    lo, hi = len(counts), int((counts - 1).sum())
    if not lo <= target <= hi: raise ValueError("Requested test size is infeasible while preserving every stratum")
    alloc = pd.DataFrame({"n_total": counts.astype(int), "ideal": counts.astype(float) * test_size})
    alloc["n_test"] = np.floor(alloc["ideal"]).astype(int).clip(lower=1)
    alloc["n_test"] = np.minimum(alloc["n_test"], alloc["n_total"] - 1).astype(int)
    while int(alloc["n_test"].sum()) < target:
        c = alloc[alloc["n_test"] < alloc["n_total"] - 1].copy(); c["priority"] = c["ideal"] - c["n_test"]
        idx = c.sort_values(["priority", "ideal", "n_total"], ascending=[False, False, False]).index[0]
        alloc.loc[idx, "n_test"] += 1
    while int(alloc["n_test"].sum()) > target:
        c = alloc[alloc["n_test"] > 1].copy(); c["priority"] = c["n_test"] - c["ideal"]
        idx = c.sort_values(["priority", "ideal", "n_total"], ascending=[False, True, True]).index[0]
        alloc.loc[idx, "n_test"] -= 1
    m["split"] = "train"
    rng = np.random.default_rng(split_seed)
    for stratum, group in m.groupby("stratum", sort=True, observed=True):
        idxs = group.index.to_numpy().copy(); rng.shuffle(idxs)
        m.loc[idxs[:int(alloc.loc[stratum, "n_test"])], "split"] = "test"
    return m.sort_values(["split", "stratum", "path"]).reset_index(drop=True)


def add_fa_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cells = pd.to_numeric(out["n_cells"], errors="coerce").astype(float)
    objs = pd.to_numeric(out["n_detected_objects"], errors="coerce").astype(float)
    out[PRIMARY_METRIC] = np.where(cells > 0, objs / cells, np.nan)
    return out


def summarize_test(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = add_fa_per_cell(df)
    soft = df.loc[df.condition == "Soft", PRIMARY_METRIC].dropna().to_numpy(float)
    stiff = df.loc[df.condition == "Stiff", PRIMARY_METRIC].dropna().to_numpy(float)
    if not len(soft) or not len(stiff): raise ValueError("Both Soft and Stiff test observations are required")
    mw = mannwhitneyu(stiff, soft, alternative="two-sided", method="auto")
    def s(x):
        return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
                "q1": float(np.quantile(x, .25)), "q3": float(np.quantile(x, .75)),
                "min": float(np.min(x)), "max": float(np.max(x))}
    summary = {"Soft": s(soft), "Stiff": s(stiff), "mann_whitney_u": float(mw.statistic), "mann_whitney_p": float(mw.pvalue)}
    rows = []
    for cond, x in (("Soft", soft), ("Stiff", stiff)):
        q = s(x); q["condition"] = cond; rows.append(q)
    return summary, pd.DataFrame(rows)


def micro_f1(df: pd.DataFrame) -> float:
    denom = int(df["n_reference_points"].sum()) + int(df["n_detected_objects"].sum())
    return 0.0 if denom <= 0 else float(2.0 * float(df["matched_quality_sum"].sum()) / denom)


def annotate(df: pd.DataFrame, split_seed: int, split: str, chromosome: Sequence[int]) -> pd.DataFrame:
    out = add_fa_per_cell(df)
    out.insert(0, "split_seed", int(split_seed)); out.insert(1, "split", split)
    for name, value in zip(("r_min","r_max","g_min","g_max","b_min","b_max","area_min","area_max"), chromosome): out[name] = int(value)
    out["folder_id"] = out["path"].map(folder_from_path)
    return out


def run_repeated_holdout(data_root: str | Path, output_root: str | Path,
                         split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS, test_size: float = DEFAULT_TEST_SIZE,
                         ga_seed: int = 42, population_size: int = 100, generations: int = 100,
                         workers: int = 4, area_soft_limit_px: float = ga.AREA_SOFT_LIMIT_PX) -> None:
    data_root, output_root = Path(data_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = ga.collect_image_paths(data_root, ga.EXCLUDED_FOLDERS)
    images = ga.preload_images(paths)
    references, failures = ga.precompute_references(images, ga.POINT_CONFIG_FINAL, ga.NUCLEUS_CONFIG_FINAL,
        ga.CELL_CONFIG_FINAL, ga.REFERENCE_MASK_CONFIG, max_workers=workers)
    failures.to_csv(output_root / "reference_failures.csv", index=False)
    if not failures.empty: raise RuntimeError("Reference generation failed; see reference_failures.csv")
    ga.validate_reference_images(references)
    ga.reference_summary_dataframe(references).to_csv(output_root / "fixed_reference_summary.csv", index=False)
    config = replace(ga.GA_CONFIG, population_size=population_size, generations=generations, seed=ga_seed, max_workers=workers)
    all_test, seed_rows, stat_rows, chrom_rows = [], [], [], []
    for split_seed in split_seeds:
        seed_dir = output_root / f"split_seed_{split_seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(paths, test_size, split_seed); manifest.to_csv(seed_dir / "split_manifest.csv", index=False)
        train_paths = manifest.loc[manifest.split == "train", "path"].tolist(); test_paths = manifest.loc[manifest.split == "test", "path"].tolist()
        tr_i, te_i = select(images, train_paths), select(images, test_paths)
        tr_r, te_r = select(references, train_paths), select(references, test_paths)
        best, best_fit, history = ga.run_ga(tr_i, tr_r, ga.MATCH_RADIUS_PX, ga.REFERENCE_MASK_CONFIG, config, area_soft_limit_px)
        history.insert(0, "split_seed", split_seed); history.to_csv(seed_dir / "ga_history.csv", index=False)
        train_df = annotate(ga.evaluate_best_solution(tr_i, tr_r, best, ga.MATCH_RADIUS_PX, ga.REFERENCE_MASK_CONFIG, config, area_soft_limit_px), split_seed, "train", best)
        test_df = annotate(ga.evaluate_best_solution(te_i, te_r, best, ga.MATCH_RADIUS_PX, ga.REFERENCE_MASK_CONFIG, config, area_soft_limit_px), split_seed, "test", best)
        train_df.to_csv(seed_dir / "train_predictions.csv", index=False); test_df.to_csv(seed_dir / "test_predictions.csv", index=False)
        stats, cond_df = summarize_test(test_df); cond_df.to_csv(seed_dir / "test_condition_summary.csv", index=False)
        (seed_dir / "test_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        row = {"split_seed": split_seed, "n_train": len(train_paths), "n_test": len(test_paths),
               "train_fitness": float(best_fit), "train_micro_f1": micro_f1(train_df), "test_micro_f1": micro_f1(test_df),
               "soft_n": stats["Soft"]["n"], "soft_median": stats["Soft"]["median"], "soft_q1": stats["Soft"]["q1"], "soft_q3": stats["Soft"]["q3"],
               "stiff_n": stats["Stiff"]["n"], "stiff_median": stats["Stiff"]["median"], "stiff_q1": stats["Stiff"]["q1"], "stiff_q3": stats["Stiff"]["q3"],
               "mann_whitney_u": stats["mann_whitney_u"], "mann_whitney_p": stats["mann_whitney_p"]}
        seed_rows.append(row); stat_rows.append({"split_seed": split_seed, **stats["Soft"], **{f"stiff_{k}":v for k,v in stats["Stiff"].items()}, "mann_whitney_u":stats["mann_whitney_u"], "mann_whitney_p":stats["mann_whitney_p"]})
        chrom_rows.append({"split_seed": split_seed, "train_fitness": float(best_fit), **dict(zip(("r_min","r_max","g_min","g_max","b_min","b_max","area_min","area_max"), map(int,best)))})
        all_test.append(test_df)
    pd.concat(all_test, ignore_index=True).to_csv(output_root / "all_test_predictions_across_seeds.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output_root / "holdout_seed_summary.csv", index=False)
    pd.DataFrame(chrom_rows).to_csv(output_root / "chromosomes_across_split_seeds.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(output_root / "test_soft_vs_stiff_statistics_across_seeds.csv", index=False)
    metadata = {"version": "1.0.0", "split_seeds": list(map(int,split_seeds)), "test_size": float(test_size),
                "ga_seed": int(ga_seed), "population_size": int(population_size), "generations": int(generations),
                "workers": int(workers), "match_radius_px": float(ga.MATCH_RADIUS_PX), "area_soft_limit_px": float(area_soft_limit_px),
                "stratification": "condition + imaging folder", "fitness_uses_condition_labels": False,
                "warning": "Repeated test appearances of the same image must not be pooled as independent observations."}
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated stratified 70:30 image-level hold-out validation for AdhGen.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("holdout_results"))
    p.add_argument("--split-seeds", default=",".join(map(str, DEFAULT_SPLIT_SEEDS)))
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--ga-seed", type=int, default=42)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--area-soft-limit-px", type=float, default=ga.AREA_SOFT_LIMIT_PX)
    return p.parse_args()


def main() -> None:
    a = parse_args(); run_repeated_holdout(a.data_root, a.output_root, parse_seeds(a.split_seeds), a.test_size,
        a.ga_seed, a.population_size, a.generations, a.workers, a.area_soft_limit_px)


if __name__ == "__main__": main()
