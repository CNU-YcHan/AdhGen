"""Supplementary table for reviewer comment 1-3: reference-point sensitivity analysis (split seed 60).

Source: reference_sensitivity_scan_results/submitted/report/sensitivity_table.csv (no recomputation).
Output: CSV and a Markdown/plain-text version in the same "col | col" layout as Supplementary Tables S3/S4.
"""

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent
from pathlib import Path

import pandas as pd

SRC = Path(str(_SCRIPT_DIR / "reference_sensitivity_scan_results/submitted/report/sensitivity_table.csv"))
OUT = Path(str(_ADHGEN_ROOT / "Figure/Supp_reference_sensitivity"))

LABELS = {
    "baseline": "Submitted",
    "smooth_sigma_1.0": "Smoothing σ 1.0 px",
    "smooth_sigma_2.0": "Smoothing σ 2.0 px",
    "min_distance_6": "Separation 6 px",
    "min_distance_10": "Separation 10 px",
    "percentile_99.0": "Percentile 99.0",
    "percentile_99.75": "Percentile 99.75",
    "remove_10": "10% removed",
    "remove_20": "20% removed",
    "remove_30": "30% removed",
    "shift_1": "Shifted ≤ 1 px",
    "shift_2": "Shifted ≤ 2 px",
}

TITLE = "Supplementary Table S[ ]. Sensitivity of AdhGen results to the reference points (split seed 60)."
NOTE = ("Note. Detection settings were changed one at a time from the submitted values (smoothing σ 1.5 px, "
        "minimum separation 8 px, threshold percentile 99.5). For the perturbations, the submitted reference points "
        "were randomly removed or shifted by up to 1 or 2 px in x and y. For each setting, G min and the area bounds "
        "were re-optimized on the training images (n = 143) by grid search with the other parameters fixed, and "
        "applied to the held-out test images (n = 62; 29 soft, 33 stiff). Reference points, mean per image; "
        "FA count per cell, median over the test images; p, two-tailed Mann–Whitney U test.")


def round_half_up(x, digits="0.1"):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(round(float(x), 4))).quantize(Decimal(digits), rounding=ROUND_HALF_UP))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(SRC)
    rows = []
    for _, r in src.iterrows():
        rows.append({
            "Setting": LABELS[r.setting],
            "Reference points per image": f"{r.reference_points_per_image:.0f}",
            "FA count per cell (Soft / Stiff)": f"{round_half_up(r.test_soft_median_fa_per_cell)} / "
                                                f"{round_half_up(r.test_stiff_median_fa_per_cell)}",
            "p-value": f"{r.test_mann_whitney_p:.4f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Supp_Table_reference_sensitivity.csv", index=False, encoding="utf-8-sig")

    header = " | ".join(df.columns)
    lines = [TITLE, header] + [" | ".join(str(v) for v in row) for row in df.itertuples(index=False)] + ["", NOTE]
    (OUT / "Supp_Table_reference_sensitivity.txt").write_text("\n".join(lines), encoding="utf-8")

    md = [f"**{TITLE}**", "", "| " + header + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    md += ["", NOTE]
    (OUT / "Supp_Table_reference_sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
