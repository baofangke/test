# Projection-Based Inference for Low-Rank Matrix and Tensor Regression

This repository is a curated public-release skeleton for the manuscript currently maintained as `ConstrainMat0330.tex`. It collects:

- the paper source used for submission preparation,
- the core MATLAB/Python scripts used to generate the real-data results reported in the paper,
- lightweight result files, figures, and algorithm notes needed to verify the written claims.

It does **not** redistribute the raw Healthy Brain Network (HBN) EEG data or heavy subject-level derivative files.

## Repository layout

```text
data/       Placeholder location for locally regenerated ROI10 MATLAB input
paper/      Manuscript source, tables, and figures used by the paper
code/       MATLAB and Python scripts used for rank selection and inference
results/    Lightweight CSV/PNG/PDF outputs that support the reported results
docs/       Reproducibility notes, data-access notes, and file-to-result mapping
external/   Optional third-party dependencies such as Tensor Toolbox
```

## What is included

- Final manuscript source in [`paper/main.tex`](paper/main.tex)
- Real-data figures used by the paper
- Matrix rank-selection and inference outputs for the `log10(EC) - log10(EO)` analysis
- Tensor rank-selection outputs and fixed-rank `(2,5,2)` inference outputs
- Standalone PDF notes describing the matrix/tensor rank-selection and inference procedures

## What is not included

- Raw HBN releases or participant-level EEG files
- Large `.mat` files that contain subject-level processed power arrays
- Legacy manuscript drafts and LaTeX build artifacts
- Intermediate chunk directories used only to complete long tensor runs

## Reproducibility scope

The public release is intended to support:

1. inspection of the manuscript source,
2. inspection of the main analysis scripts,
3. verification of the reported summary results from lightweight outputs,
4. reconstruction of the workflow once the user has separately obtained the required HBN data.

Some scripts in [`code/`](code) still contain local absolute paths from the original workstation. See [`docs/PATHS_AND_PORTABILITY.md`](docs/PATHS_AND_PORTABILITY.md) before attempting a clean rerun on a new machine.
The current release removes the original workstation drive paths from the copied entry-point scripts and instead expects the processed ROI10 MATLAB file at [`data/hbn_bandpower_8band_roi10.mat`](data/hbn_bandpower_8band_roi10.mat) unless an explicit path is provided.

## Entry points

- Matrix 5-fold CV rank selection:
  [`code/python/run_hbn_trace_regression_5fold_cv_log_contrast.py`](code/python/run_hbn_trace_regression_5fold_cv_log_contrast.py)
- Matrix rank-1 inference:
  [`code/matlab/run_roi10_matproj_log_contrast_rank1_ci_analysis.m`](code/matlab/run_roi10_matproj_log_contrast_rank1_ci_analysis.m)
- Tensor Tucker-rank CV:
  [`code/matlab/run_roi10_tucker_tensor_regression_cv.m`](code/matlab/run_roi10_tucker_tensor_regression_cv.m)
- Tensor fixed-rank `(2,5,2)` inference:
  [`code/matlab/run_roi10_tsrproj_loading_ci_analysis.m`](code/matlab/run_roi10_tsrproj_loading_ci_analysis.m)

## Before publishing to GitHub

- choose a license,
- remove any remaining private local paths if you want one-command reproducibility,
- decide whether to keep compiled PDFs under version control,
- verify that no subject-level restricted data are included.

See [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).
