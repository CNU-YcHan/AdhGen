# Source curation note

This Zenodo-ready package was curated from the current working scripts supplied for the manuscript revision.

Retained computational roles:
- standardized three-channel image preparation;
- DAPI-seeded cell segmentation;
- fixed green-marker reference-point generation;
- RGB/area chromosome candidate detection;
- distance/area-adjusted Hungarian matching fitness;
- genetic optimization;
- repeated stratified 70:30 image-level hold-out evaluation;
- per-split Mann–Whitney U comparison of image-level FA count per cell.

Removed as redundant or non-core:
- duplicate FA visualization scripts (numbered/non-numbered variants);
- DAPI and red/green intermediate visualization-only scripts;
- manuscript panel annotation utilities;
- green-only chromosome-based comparison script, because it is not the fixed local-maxima reference-point objective used by the final method;
- figure-layout scripts that do not change AdhGen optimization/evaluation;
- ANOVA and Welch tests from the working hold-out script; the current manuscript uses the two-sided Mann–Whitney U test for Soft/Stiff comparisons;
- hard-coded local Windows paths and revision/debug comments.

Method-preserving fixes made while curating:
- hold-out default split seeds are 52–61 to match the current analysis;
- GA defaults are population 100 and 100 generations;
- acquisition folders 9 and 12 are explicitly excluded in image preparation and AdhGen input collection;
- the final point-detection settings and area soft limit (60 px) are set directly in the packaged defaults.
