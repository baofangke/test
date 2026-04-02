# Projection-Based Inference for Low-Rank Matrix and Tensor Regression

This repository accompanies the manuscript on projection-based inference for low-rank matrix and tensor regression. It contains the paper source, the main MATLAB/Python scripts used for the reported real-data analyses, and lightweight result files underlying the figures and tables discussed in the manuscript.

The real-data application uses resting-state EEG summaries derived from the Healthy Brain Network (HBN). The repository does not redistribute raw HBN files or subject-level processed data.

## Repository layout

```text
data/       Placeholder location for locally regenerated ROI10 MATLAB input
paper/      Manuscript source, tables, and figures used by the paper
code/       MATLAB and Python scripts used for rank selection and inference
results/    Lightweight CSV/PNG outputs supporting the reported results
docs/       Short notes on data access and software dependencies
external/   Optional third-party dependencies such as Tensor Toolbox
```

## Included

- manuscript source in [`paper/main.tex`](paper/main.tex)
- simulation and real-data figures used by the paper
- matrix rank-selection outputs for the `log10(EC)-log10(EO)` analysis
- matrix inference outputs for the selected rank-1 model
- tensor rank-selection outputs
- tensor inference outputs for the fixed-rank `(2,5,2)` analysis
- curated MATLAB and Python entry-point scripts

## Not included

- Raw HBN releases or participant-level EEG files
- Large `.mat` files that contain subject-level processed power arrays
- Legacy manuscript drafts and LaTeX build artifacts
- Detailed internal process notes and draft-stage writeups

## Main entry points

- Matrix 5-fold CV rank selection:
  [`code/python/run_hbn_trace_regression_5fold_cv_log_contrast.py`](code/python/run_hbn_trace_regression_5fold_cv_log_contrast.py)
- Matrix rank-1 inference:
  [`code/matlab/run_roi10_matproj_log_contrast_rank1_ci_analysis.m`](code/matlab/run_roi10_matproj_log_contrast_rank1_ci_analysis.m)
- Tensor Tucker-rank CV:
  [`code/matlab/run_roi10_tucker_tensor_regression_cv.m`](code/matlab/run_roi10_tucker_tensor_regression_cv.m)
- Tensor fixed-rank `(2,5,2)` inference:
  [`code/matlab/run_roi10_tsrproj_loading_ci_analysis.m`](code/matlab/run_roi10_tsrproj_loading_ci_analysis.m)

## Data and dependencies

- place the regenerated ROI10 MATLAB file at [`data/hbn_bandpower_8band_roi10.mat`](data/hbn_bandpower_8band_roi10.mat), or pass an explicit path to the scripts;
- for the tensor analysis, provide Tensor Toolbox either via `TENSOR_TOOLBOX_ROOT` or by placing it under [`external/`](external);
- see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) and [`docs/MATLAB_DEPENDENCIES.md`](docs/MATLAB_DEPENDENCIES.md).
