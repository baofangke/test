# Results Map

This file maps the manuscript's main real-data claims to the scripts and result files included in this repository.

## Manuscript source

- Main manuscript: [`paper/main.tex`](../paper/main.tex)

## Real-data matrix analysis

- Rank selection script:
  [`code/python/run_hbn_trace_regression_5fold_cv_log_contrast.py`](../code/python/run_hbn_trace_regression_5fold_cv_log_contrast.py)
- Rank-selection result directory:
  [`results/matrix/rank_selection_log_contrast_cv5`](../results/matrix/rank_selection_log_contrast_cv5)
- Inference script:
  [`code/matlab/run_roi10_matproj_log_contrast_rank1_ci_analysis.m`](../code/matlab/run_roi10_matproj_log_contrast_rank1_ci_analysis.m)
- Inference result directory:
  [`results/matrix/inference_log_contrast_rank1`](../results/matrix/inference_log_contrast_rank1)

Used in the paper as:

- Real-data matrix figure:
  [`paper/figures/realdata_matrix_structured_loadings.png`](../paper/figures/realdata_matrix_structured_loadings.png)
- Selected matrix intervals table:
  [`paper/tables/table_realdata_infer.tex`](../paper/tables/table_realdata_infer.tex)

## Real-data tensor analysis

- Tucker-rank selection script:
  [`code/matlab/run_roi10_tucker_tensor_regression_cv.m`](../code/matlab/run_roi10_tucker_tensor_regression_cv.m)
- Rank-selection result directory:
  [`results/tensor/rank_selection`](../results/tensor/rank_selection)
- Inference script:
  [`code/matlab/run_roi10_tsrproj_loading_ci_analysis.m`](../code/matlab/run_roi10_tsrproj_loading_ci_analysis.m)
- Inference result directory:
  [`results/tensor/inference_rank252_foldcentered`](../results/tensor/inference_rank252_foldcentered)

Used in the paper as:

- Real-data tensor figure:
  [`paper/figures/realdata_tensor_shared_contrast.png`](../paper/figures/realdata_tensor_shared_contrast.png)
- Selected tensor intervals table:
  [`paper/tables/table_realdata_infer.tex`](../paper/tables/table_realdata_infer.tex)

## Simulation tables and figures

The simulation section of the manuscript uses the following paper assets:

- [`paper/tables/table_five_methods_combined_ind.tex`](../paper/tables/table_five_methods_combined_ind.tex)
- [`paper/tables/table_five_methods_combined.tex`](../paper/tables/table_five_methods_combined.tex)
- [`paper/tables/table_cp_ci_sd1_combined.tex`](../paper/tables/table_cp_ci_sd1_combined.tex)
- [`paper/tables/table_tsr_methods_combined_ind.tex`](../paper/tables/table_tsr_methods_combined_ind.tex)
- [`paper/tables/table_tsr_methods_combined.tex`](../paper/tables/table_tsr_methods_combined.tex)
- [`paper/tables/table_tsr_cp_ci_sd1_combined.tex`](../paper/tables/table_tsr_cp_ci_sd1_combined.tex)
- [`paper/figures/density_compare_sd1_inf_ind.pdf`](../paper/figures/density_compare_sd1_inf_ind.pdf)
- [`paper/figures/density_compare_all_settings_inf.pdf`](../paper/figures/density_compare_all_settings_inf.pdf)
- [`paper/figures/density_tsr_compare_sd1_inf_ind.pdf`](../paper/figures/density_tsr_compare_sd1_inf_ind.pdf)
- [`paper/figures/density_tsr_compare_sd1_inf.pdf`](../paper/figures/density_tsr_compare_sd1_inf.pdf)
