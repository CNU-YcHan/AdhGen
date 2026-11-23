# Source map

The archive was curated from the current manuscript-revision scripts supplied by the authors.

## Incorporated into the Zenodo package

- `cell_instance_utils_dapi_all_cells(3).py` → `src/adhgen/cell_segmentation.py`
- `adhgen_point_guided_agg_fix(2).py` → only the final reference-point generator and GA operators were retained in `src/adhgen/ga_utils.py`
- `adhgen_ga_area_weighted_micro_f1_windows(1).py` → the final candidate detection, Hungarian matching, area-adjusted fitness, and GA optimization were retained in `src/adhgen/optimizer.py`
- `adhgen_ga_area_weighted_micro_f1_holdout_70_30_multiseed_cellmean_windows(5).py` → the repeated stratified held-out evaluation was retained and simplified in `src/adhgen/holdout.py`
- `01_build_merged_405_488_actin(4).py` → `src/adhgen/preprocessing.py`

## Not included

The following working scripts were intentionally omitted because they are manuscript-panel drawing, annotation, diagnostic, or duplicate visualization utilities rather than components of the final optimization/evaluation method:

- `(Fig.1c)analyze_fig1c_whole_image_f_actin_common_scale.py`
- `(Fig.2b)02_make_fig2b_rgba_filtering.py`
- `(Fig.2b)03_make_green_only_marker_comparison.py`
- `(Fig.2b)export_rgb_channel_annotated.py`
- `(Fig.3c)actual_same_format_v3.py`
- `(Fig.4)make_fig4_revised_v3_holdout(1).py`
- `(Fig.5b)make_fig5b_corrected(1).py`
- `(Fig.6)make_fig6_main2_with_supp_FA_area(3).py`
- `visualize_adhgen_from_chromosome_multi_input_white_fa(2).py`
- `visualize_adhgen_from_chromosome_multi_input_white_fa_numbered(2).py`
- `visualize_dapi_nucleus_segmentation(1).py`
- `visualize_red_green_cell_area(1).py`

The omission of these files does not change the AdhGen optimization, candidate detection, or repeated hold-out validation implemented in this archive.

## Added in v1.1.0

The working scripts listed above as not included, together with the working modules they import and the scripts of the revision analyses, are now provided unchanged in `manuscript_scripts/` (local paths replaced). `(Fig.3c)actual_same_format_v3.py` and `(Fig.4)make_fig4_revised_v3_holdout.py` were superseded by `make_fig3_revised.py` and `(Fig.4)make_fig4_revised_v4_save_individual_images.py`.
