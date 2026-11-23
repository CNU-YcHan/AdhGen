# Manuscript scripts

This folder contains the scripts that were run to produce the analyses, figures and tables of the revised manuscript.
They are the working versions used in the study. The only changes are that local absolute paths were replaced and a
short path block was added at the top of each affected file. The packaged implementation in `src/adhgen` is unchanged
from v1.0.0.

## Setup

All scripts must stay in this folder, because they import each other by module name. Set the environment variable
`ADHGEN_ROOT` to a project folder with the following layout. Outputs are written to `Figure/` under that folder or
next to the scripts.

```text
<ADHGEN_ROOT>/
  merged_data_405_488_actin/<folder>/{Soft,Stiff}/*_M.tif   # merged three-channel images (01_build_merged_405_488_actin.py)
  AdhGen result/split_seed_<52-61>/                          # outputs of the repeated hold-out run
  1. F-actin/<folder>/{Soft,Stiff}/                           # outputs of f_actin_analyzer.py
  Figure/
```

The microscopy images are available at https://huggingface.co/datasets/SCH-YcHan/AdhGen.
In addition to `requirements.txt`, the scripts use `matplotlib`, `seaborn` and `Pillow`.

## Scripts

| Script | Used for |
|---|---|
| `01_build_merged_405_488_actin.py` | Merged three-channel input images |
| `cell_instance_utils_dapi_all_cells.py` | DAPI-seeded nucleus and cell segmentation |
| `adhgen_point_guided_agg_fix.py` | Reference-point generation and GA operators |
| `adhgen_ga_area_weighted_micro_f1_windows.py` | Candidate FA detection, matching-based fitness and GA optimization |
| `adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_windows.py` | Repeated stratified 70:30 hold-out evaluation (split seeds 52–61) |
| `(Fig.1c)analyze_fig1c_whole_image_f_actin_common_scale.py` | Fig. 1c |
| `(Fig.2b)02_make_fig2b_rgba_filtering.py`, `(Fig.2b)03_make_green_only_marker_comparison.py`, `(Fig.2b)export_rgb_channel_annotated.py`, `visualize_dapi_nucleus_segmentation.py`, `visualize_red_green_cell_area.py` | Fig. 2 panels |
| `visualize_adhgen_from_chromosome_multi_input_white_fa.py`, `visualize_adhgen_from_chromosome_multi_input_white_fa_numbered.py` | Visualization of AdhGen detections for a given parameter set |
| `make_fig3_revised.py` | Fig. 3 |
| `(Fig.4)make_fig4_revised_v4_save_individual_images.py` | Fig. 4 and the supplementary table of hold-out statistics |
| `f_actin_analyzer.py` | Per-image F-actin intensity profiles and integrated density (Fig. 5) |
| `(Fig.5b)make_fig5b_corrected.py` | Fig. 5b and the F-actin intensity statistics |
| `(Fig.6)make_fig6_main2_with_supp_FA_area.py` | Fig. 6 and the supplementary figure of FA count versus total FA area |
| `make_supp_reference_points.py` | Supplementary figure of reference points |
| `make_matching_distance_figure.py` | Supplementary figure of the matching distance |
| `adhgen_reference_sensitivity.py`, `adhgen_reference_sensitivity_scan.py`, `adhgen_reference_sensitivity_scan_report.py`, `make_supp_table_reference_sensitivity.py` | Sensitivity of the results to the reference points (supplementary table) |
| `adhgen_timing_per_image.py` | Processing time per image |
