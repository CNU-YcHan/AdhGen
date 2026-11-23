"""Wall-clock time of the AdhGen steps per image (single process), for a rough hands-on vs compute comparison.

Steps timed on the Soft/Stiff images (the submitted data set, excluded folders as in the paper):
  1. reference generation (nucleus / cell masks + green reference points)
  2. applying the final parameter set (split seed 60) = FA detection and per-cell count
GA optimization time per split is read from the submitted results (seed_summary.json).
"""

import os as _os
from pathlib import Path as _Path
_ADHGEN_ROOT = _Path(_os.environ.get("ADHGEN_ROOT", "."))  # project root, see manuscript_scripts/README.md
_SCRIPT_DIR = _Path(__file__).resolve().parent
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(_SCRIPT_DIR))
import numpy as np
import pandas as pd

import adhgen_ga_area_weighted_micro_f1_windows as ga

RESULT = Path(str(_ADHGEN_ROOT / "AdhGen result"))
OUT = Path(str(_SCRIPT_DIR / "timing_results"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(str(p) for p in ga.DATA_ROOT.glob("*/*/*_M.tif")
                   if p.parts[-3] not in ga.EXCLUDED_FOLDERS) if hasattr(ga, "EXCLUDED_FOLDERS") else None
    if not paths:
        manifest = pd.read_csv(RESULT / "split_seed_60" / "split_manifest.csv")
        paths = list(manifest.path)
    ind = json.loads((RESULT / "split_seed_60" / "seed_summary.json").read_text(encoding="utf-8"))["best_individual"]
    cfg = replace(ga.GA_CONFIG, max_workers=1)

    rows = []
    for p in paths:
        t0 = time.perf_counter()
        img = ga.preload_images([p])
        t1 = time.perf_counter()
        refs, _ = ga.precompute_references(img, ga.POINT_CONFIG_FINAL, ga.NUCLEUS_CONFIG_FINAL,
                                           ga.CELL_CONFIG_FINAL, ga.REFERENCE_MASK_CONFIG, max_workers=1)
        t2 = time.perf_counter()
        objs, _, _ = ga.detect_objects_in_union_mask(img[p], ind, refs[p], ga.REFERENCE_MASK_CONFIG, cfg)
        t3 = time.perf_counter()
        rows.append({"path": p, "load_s": t1 - t0, "reference_s": t2 - t1, "detect_s": t3 - t2,
                     "n_objects": len(objs), "n_cells": refs[p].n_cells})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_image_timing.csv", index=False)

    durations = [json.loads(f.read_text(encoding="utf-8"))["duration_seconds"]
                 for f in sorted(RESULT.glob("split_seed_*/seed_summary.json"))]
    print("images:", len(df), " cells:", int(df.n_cells.sum()))
    print(df[["load_s", "reference_s", "detect_s"]].describe().round(3).to_string())
    total = df[["load_s", "reference_s", "detect_s"]].sum()
    print("total for all images (s):", total.round(1).to_dict())
    print("GA optimization per split (s): median", round(float(np.median(durations))),
          "range", round(min(durations)), "-", round(max(durations)), "(143 training images, 4 workers)")


if __name__ == "__main__":
    main()
