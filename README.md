# AdhGen v1.1.0

This archive contains the core implementation and validation code used for the current AdhGen method. It is intentionally limited to the computational components required to reproduce the AdhGen optimization and repeated held-out evaluation; figure-only, diagnostic, duplicate, and superseded scripts are not included.

## Method summary

AdhGen operates on three-channel merged fluorescence images:

- Red: F-actin
- Green: Paxillin or Vinculin (FA marker)
- Blue: DAPI

The chromosome is `[Rmin, Rmax, Gmin, Gmax, Bmin, Bmax, Amin, Amax]`.

The workflow is:

1. DAPI-seeded cell segmentation.
2. Union of valid cell instances, excluding nuclei/nuclear margin and the lower-right scale-bar region.
3. Fixed reference points from local maxima of the green FA-marker channel.
4. Candidate FA objects from RGB bounds, 8-connected components, and chromosome-defined area bounds.
5. One-to-one Hungarian matching between fixed reference points and candidate objects.
6. Distance-weighted matching quality with a soft penalty for objects larger than 60 pixels.
7. Dataset-level micro-F1-based fitness: `2 * sum(matched quality) / (number of reference points + number of detected objects)`.

Soft/Stiff labels and between-group statistics are **not** used in the GA fitness.

## Final study settings

- Reference-point background sigma: 15
- Reference-point smoothing sigma: 1.5
- Reference threshold percentile: 99.5
- MAD multiplier: 5
- Minimum response: 8
- Minimum point distance: 8 px
- Matching radius: 4 px
- Area soft limit: 60 px
- GA population: 100
- GA generations: 100
- Crossover probability: 0.8
- Mutation probability: 0.5
- Elite size: 1
- Tournament size: 3
- GA seed: 42
- Repeated hold-out split seeds: 52–61
- Hold-out ratio: 70% train / 30% test
- Stratification: substrate condition + imaging folder
- AdhGen-excluded acquisition folders: 9 and 12 (488-nm channel is not Paxillin/Vinculin)

## Installation

```bash
python -m pip install -e .
```

## Data layout

Merged images are expected as:

```text
<data-root>/
  1/Soft/*_M.tif
  1/Stiff/*_M.tif
  2/Soft/*_M.tif
  ...
```

The microscopy dataset is available separately at:
https://huggingface.co/datasets/SCH-YcHan/AdhGen

## Prepare merged images

Raw wavelength images can be converted to the three-channel AdhGen input format with:

```bash
adhgen-prepare-images --data-root <raw-data-root> --output-root <merged-data-root>
```

The preprocessing script maps 405 nm to blue, 488 nm to green, and F-actin to red. Folders 3–6 use the 647-nm F-actin channel; other eligible folders use 546 nm. The script does not independently min-max normalize each image.

## Run AdhGen optimization

```bash
adhgen-optimize \
  --data-root <merged-data-root> \
  --output-root adhgen_results \
  --seed 42 \
  --population-size 100 \
  --generations 100 \
  --workers 4 \
  --area-soft-limit-px 60
```

Key outputs are `best_chromosome.json`, `ga_history.csv`, `fixed_reference_summary.csv`, and `best_solution_by_image.csv`.

## Repeated 70:30 held-out evaluation

```bash
adhgen-holdout \
  --data-root <merged-data-root> \
  --output-root holdout_results \
  --split-seeds 52,53,54,55,56,57,58,59,60,61 \
  --test-size 0.30 \
  --ga-seed 42 \
  --population-size 100 \
  --generations 100 \
  --workers 4 \
  --area-soft-limit-px 60
```

Soft/Stiff comparisons are calculated separately within each held-out split using a two-sided Mann–Whitney U test. The same image can appear in the test sets of multiple split seeds, so rows in `all_test_predictions_across_seeds.csv` must not be pooled as independent observations.

Aggregate outputs include:

- `chromosomes_across_split_seeds.csv`
- `holdout_seed_summary.csv`
- `all_test_predictions_across_seeds.csv`
- `test_soft_vs_stiff_statistics_across_seeds.csv`

## Included files

- `src/adhgen/cell_segmentation.py`: DAPI-seeded nucleus/cell segmentation.
- `src/adhgen/ga_utils.py`: reference-point generation and GA operators.
- `src/adhgen/optimizer.py`: RGB/area candidate detection, matching-based fitness, and GA optimization.
- `src/adhgen/holdout.py`: repeated stratified 70:30 held-out evaluation.
- `src/adhgen/preprocessing.py`: construction of standardized merged images from raw wavelength channels.

## Deliberately excluded from this archive

The supplied working directory also contained scripts used only to draw manuscript panels, annotate channels, visualize intermediate masks, number detected objects, or compare alternative/diagnostic representations. Those scripts do not define the final AdhGen optimization or held-out evaluation and were omitted to avoid redundant and potentially confusing code paths.

This archive also excludes the earlier group-separability/ANOVA optimization formulation. The version here implements only the condition-independent matching-based fitness used in the current manuscript.

## Manuscript scripts (added in v1.1.0)

The folder `manuscript_scripts/` contains the scripts that were run to produce the analyses, figures and tables of the revised manuscript, including the reference-point sensitivity analysis, the supplementary figures of reference points and matching distance, and the processing-time measurement. See `manuscript_scripts/README.md`.

## Code repository

Development repository: https://github.com/CNU-YcHan/AdhGen

Zenodo (all versions): https://doi.org/10.5281/zenodo.22119870

## Previous version

The notebooks of the first submission (including the earlier group-separability formulation) are kept in `legacy/` for reference. They are not used in the current manuscript.
