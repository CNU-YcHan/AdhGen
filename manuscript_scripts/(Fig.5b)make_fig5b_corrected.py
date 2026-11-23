# -*- coding: utf-8 -*-

"""
Fig. 5b revision from the EXISTING `1. F-actin` folder
=======================================================

Purpose
-------
This script DOES NOT recalculate F-actin intensity from raw images.

It only:
1. reads the existing per-folder `Soft_intden_analysis.csv` /
   `Stiff_intden_analysis.csv`,
2. skips missing / empty / unreadable CSVs correctly,
3. prevents the stale-DataFrame duplication bug in the original notebook,
4. preserves the original Fig. 5b plotting style,
5. clarifies the x/y-axis labels requested by the reviewer,
6. recalculates the two-sided Mann-Whitney U test from the actually included ROIs,
7. writes a per-folder audit table so the final n can be traced.

Important correction from the original notebook
-----------------------------------------------
The original notebook used:

    try:
        f_actin_soft_csv = pd.read_csv(...)
    except:
        pass

    f_actin_soft_list.append(f_actin_soft_csv)

If a CSV was empty/unreadable, the previous loop's DataFrame could be
appended again. That produced duplicated rows.

This script appends a DataFrame ONLY when the current CSV was read successfully.

Expected structure
------------------
1. F-actin/
    1/
        Soft/Soft_intden_analysis.csv
        Stiff/Stiff_intden_analysis.csv
    2/
        ...
    ...
    12/

Example
-------
python "(Fig.5b)make_fig5b_corrected.py" ^
  --f-actin-folder "<ADHGEN_ROOT>\\1. F-actin" ^
  --save-dir "<ADHGEN_ROOT>\\Figure\\Fig5"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


# =============================================================================
# Settings
# =============================================================================

DEFAULT_SCOPES = [str(i) for i in range(1, 13)]
REQUIRED_COLUMNS = {"aspect_ratio", "intden"}
DPI = 300


# =============================================================================
# Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw corrected Fig. 5b from the existing 1. F-actin folder."
    )

    parser.add_argument(
        "--f-actin-folder",
        type=Path,
        default=Path("./1. F-actin"),
        help="Existing 1. F-actin folder.",
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("./figure"),
        help="Directory for Fig. 5b outputs.",
    )

    parser.add_argument(
        "--scopes",
        type=str,
        default=",".join(DEFAULT_SCOPES),
        help="Comma-separated folder IDs. Default: 1-12.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
    )

    return parser.parse_args()


# =============================================================================
# CSV loading
# =============================================================================

def read_intden_csv(
    csv_path: Path,
    *,
    condition: str,
    scope: str,
) -> tuple[pd.DataFrame | None, dict]:
    """
    Read one current CSV.

    Crucially, this function returns None if the CURRENT file cannot be used.
    It never reuses a DataFrame from the previous scope.
    """
    audit = {
        "scope": scope,
        "condition": condition,
        "csv_path": str(csv_path),
        "status": "",
        "reason": "",
        "rows_in_csv": 0,
        "valid_rows": 0,
    }

    if not csv_path.exists():
        audit["status"] = "skipped"
        audit["reason"] = "missing CSV"
        return None, audit

    if csv_path.stat().st_size == 0:
        audit["status"] = "skipped"
        audit["reason"] = "0-byte empty CSV"
        return None, audit

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        audit["status"] = "skipped"
        audit["reason"] = "empty CSV"
        return None, audit
    except Exception as exc:
        audit["status"] = "skipped"
        audit["reason"] = f"read error: {type(exc).__name__}: {exc}"
        return None, audit

    audit["rows_in_csv"] = int(len(df))

    if df.empty:
        audit["status"] = "skipped"
        audit["reason"] = "header-only / no rows"
        return None, audit

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        audit["status"] = "skipped"
        audit["reason"] = f"missing column(s): {sorted(missing)}"
        return None, audit

    # Preserve original notebook inclusion criterion:
    #     aspect_ratio > 0
    work = df.copy()

    work["aspect_ratio"] = pd.to_numeric(
        work["aspect_ratio"],
        errors="coerce",
    )

    work["intden"] = pd.to_numeric(
        work["intden"],
        errors="coerce",
    )

    valid_mask = (
        np.isfinite(work["aspect_ratio"])
        & np.isfinite(work["intden"])
        & (work["aspect_ratio"] > 0)
    )

    work = work.loc[valid_mask].copy()

    audit["valid_rows"] = int(len(work))

    if work.empty:
        audit["status"] = "skipped"
        audit["reason"] = "no rows after aspect_ratio > 0 / finite-value filtering"
        return None, audit

    work["scope"] = scope
    work["condition"] = condition

    audit["status"] = "included"
    audit["reason"] = ""

    return work, audit


def load_existing_results(
    f_actin_folder: Path,
    scopes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    soft_tables = []
    stiff_tables = []
    audit_rows = []

    for scope in scopes:
        for condition in ("Soft", "Stiff"):

            csv_path = (
                f_actin_folder
                / scope
                / condition
                / f"{condition}_intden_analysis.csv"
            )

            df, audit = read_intden_csv(
                csv_path,
                condition=condition,
                scope=scope,
            )

            audit_rows.append(audit)

            if df is None:
                print(
                    f"[SKIP] scope={scope:>2}  condition={condition:<5}  "
                    f"{audit['reason']}"
                )
                continue

            print(
                f"[LOAD] scope={scope:>2}  condition={condition:<5}  "
                f"rows={audit['valid_rows']}"
            )

            if condition == "Soft":
                soft_tables.append(df)
            else:
                stiff_tables.append(df)

    if not soft_tables:
        raise RuntimeError("No usable Soft data were found.")

    if not stiff_tables:
        raise RuntimeError("No usable Stiff data were found.")

    soft_df = pd.concat(
        soft_tables,
        ignore_index=True,
    )

    stiff_df = pd.concat(
        stiff_tables,
        ignore_index=True,
    )

    audit_df = pd.DataFrame(audit_rows)

    return soft_df, stiff_df, audit_df


# =============================================================================
# Statistics
# =============================================================================

def get_intden_values(df: pd.DataFrame) -> np.ndarray:
    return df["intden"].astype(float).to_numpy()


def mann_whitney_summary(
    soft: np.ndarray,
    stiff: np.ndarray,
) -> pd.DataFrame:
    """
    scipy returns U for the first sample supplied.

    To avoid ambiguity, save:
      - U_soft: U for Soft
      - U_stiff: complementary U for Stiff
      - U_min: the smaller of the two
      - p_value: two-sided p-value
    """
    result = mannwhitneyu(
        soft,
        stiff,
        alternative="two-sided",
        method="auto",
    )

    u_soft = float(result.statistic)
    u_stiff = float(len(soft) * len(stiff) - u_soft)
    u_min = min(u_soft, u_stiff)

    return pd.DataFrame(
        [
            {
                "test": "Mann-Whitney U",
                "alternative": "two-sided",
                "Soft_n": int(len(soft)),
                "Stiff_n": int(len(stiff)),
                "U_soft": u_soft,
                "U_stiff": u_stiff,
                "U_min": u_min,
                "p_value": float(result.pvalue),
            }
        ]
    )


# =============================================================================
# Fig. 5b
# =============================================================================

def make_fig5b(
    soft: np.ndarray,
    stiff: np.ndarray,
    save_dir: Path,
    dpi: int,
) -> None:

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Preserve the ORIGINAL notebook plotting style.
    # Do not force common bin edges; do not replace seaborn KDE.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    sns.histplot(
        soft,
        bins=15,
        color="blue",
        label="Soft",
        stat="density",
        element="step",
        linewidth=1.2,
        linestyle="--",
    )

    sns.kdeplot(
        soft,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        bw_adjust=1,
    )

    sns.histplot(
        stiff,
        bins=15,
        color="red",
        label="Stiff",
        stat="density",
        element="step",
        linewidth=1.2,
    )

    sns.kdeplot(
        stiff,
        color="red",
        linestyle="-",
        linewidth=1.5,
        bw_adjust=1,
    )

    # Reviewer-related clarification.
    plt.xlabel(
        "Total F-actin fluorescence intensity (a.u.)"
    )

    plt.ylabel(
        r"Probability density (a.u.$^{-1}$)"
    )

    plt.legend()
    plt.tight_layout()

    png_path = save_dir / "Fig5b_corrected.png"
    pdf_path = save_dir / "Fig5b_corrected.pdf"
    tiff_path = save_dir / "Fig5b_corrected.tiff"

    plt.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.savefig(
        tiff_path,
        dpi=dpi,
        bbox_inches="tight",
        format="tiff",
    )

    plt.close()

    print()
    print("[Figure saved]")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {tiff_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    f_actin_folder = (
        args.f_actin_folder
        .expanduser()
        .resolve()
    )

    save_dir = (
        args.save_dir
        .expanduser()
        .resolve()
    )

    scopes = [
        item.strip()
        for item in args.scopes.split(",")
        if item.strip()
    ]

    if not f_actin_folder.exists():
        raise FileNotFoundError(
            f"1. F-actin folder not found: {f_actin_folder}"
        )

    print("=" * 76)
    print("Corrected Fig. 5b aggregation")
    print("=" * 76)
    print(f"Source : {f_actin_folder}")
    print(f"Output : {save_dir}")
    print(
        "Rule   : append only the CURRENT successfully read CSV; "
        "empty CSVs are skipped"
    )
    print()

    soft_df, stiff_df, audit_df = load_existing_results(
        f_actin_folder,
        scopes,
    )

    soft = get_intden_values(
        soft_df
    )

    stiff = get_intden_values(
        stiff_df
    )

    print()
    print("=" * 76)
    print("Per-folder valid ROI counts")
    print("=" * 76)

    count_table = (
        audit_df
        .pivot(
            index="scope",
            columns="condition",
            values="valid_rows",
        )
        .fillna(0)
        .astype(int)
        .reindex(
            columns=["Soft", "Stiff"],
            fill_value=0,
        )
    )

    print(
        count_table.to_string()
    )

    print()
    print("=" * 76)
    print("Final Fig. 5b sample size")
    print("=" * 76)
    print(f"Soft  n = {len(soft)}")
    print(f"Stiff n = {len(stiff)}")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    stats_df = mann_whitney_summary(
        soft,
        stiff,
    )

    row = stats_df.iloc[0]

    print()
    print("Two-sided Mann-Whitney U test")
    print(f"U_soft  = {row['U_soft']:.6f}")
    print(f"U_stiff = {row['U_stiff']:.6f}")
    print(f"U_min   = {row['U_min']:.6f}")
    print(f"p       = {row['p_value']:.10g}")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    make_fig5b(
        soft,
        stiff,
        save_dir,
        args.dpi,
    )

    # -------------------------------------------------------------------------
    # Reproducibility outputs
    # -------------------------------------------------------------------------
    values_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Condition": "Soft",
                    "scope": soft_df["scope"].astype(str),
                    "intden": soft_df["intden"].astype(float),
                }
            ),
            pd.DataFrame(
                {
                    "Condition": "Stiff",
                    "scope": stiff_df["scope"].astype(str),
                    "intden": stiff_df["intden"].astype(float),
                }
            ),
        ],
        ignore_index=True,
    )

    values_path = (
        save_dir
        / "Fig5b_corrected_values.csv"
    )

    audit_path = (
        save_dir
        / "Fig5b_corrected_input_audit.csv"
    )

    counts_path = (
        save_dir
        / "Fig5b_corrected_counts_by_folder.csv"
    )

    stats_path = (
        save_dir
        / "Fig5b_corrected_Mann_Whitney.csv"
    )

    values_df.to_csv(
        values_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit_df.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    count_table.to_csv(
        counts_path,
        encoding="utf-8-sig",
    )

    stats_df.to_csv(
        stats_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("[Data saved]")
    print(f"  {values_path}")
    print(f"  {audit_path}")
    print(f"  {counts_path}")
    print(f"  {stats_path}")

    print()
    print("=" * 76)
    print("Done")
    print("=" * 76)


if __name__ == "__main__":
    main()
