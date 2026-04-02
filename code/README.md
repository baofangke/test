# Code Overview

This directory contains the subset of scripts needed to support the real-data results discussed in the manuscript.

## MATLAB

- `matProj.m`: projection-based inference for low-rank matrix regression
- `tsrProj.m`: projection-based inference for low Tucker-rank tensor regression
- `fit_low_rank_trace_regression.m`: projected-gradient solver for fixed-rank matrix regression
- `fit_tucker_tensor_regression.m`: Tucker-regression fitting routine
- `predict_tucker_tensor_regression.m`: prediction helper for Tucker regression
- `run_roi10_matproj_log_contrast_rank1_ci_analysis.m`: matrix inference for the `log10(EC)-log10(EO)` covariate
- `run_roi10_tucker_tensor_regression_cv.m`: Tucker-rank selection by cross-validation
- `run_roi10_tsrproj_loading_ci_analysis.m`: tensor inference for selected fixed Tucker rank
- `resolve_release_paths.m`: repository-relative default path helper for the public release

## Python

- `run_hbn_trace_regression_5fold_cv_log_contrast.py`: 5-fold CV rank selection for the matrix log-contrast analysis
- `run_hbn_trace_regression_nested_cv.py`: shared helper functions imported by the 5-fold CV script
- `analyze_roi10_log_training_distributions.py`: fold-level inspection of the logarithmic covariate distributions

## Important note

The public-release entry-point scripts now default to repository-relative paths. The main expected input is the locally regenerated ROI10 MATLAB file at `data/hbn_bandpower_8band_roi10.mat`, with outputs written under `results/`.
