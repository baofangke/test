# Analysis Pipeline

This note records the main workflow behind the real-data results reported in the manuscript.

## EEG preprocessing and summaries

1. Start from HBN resting-state EEG recordings under eyes-closed (EC) and eyes-open (EO) conditions.
2. Apply standard EEG preprocessing, including filtering, artifact annotation, bad-channel handling, rereferencing, and epoch screening.
3. Estimate band power from clean epochs using Welch-type spectral summaries.
4. Aggregate channel-level power into 10 prespecified ROI groups across 8 canonical frequency bands.

This produces ROI-by-band summaries for each subject and condition.

## Matrix analysis

1. Construct the matrix covariate
   \[
   X_i = \log_{10}(EC_i) - \log_{10}(EO_i),
   \]
   where each subject-level matrix is of size `10 x 8`.
2. Select the matrix rank by 5-fold cross-validation using mean validation MSE.
3. Fit the selected rank-1 model.
4. Run `matProj` on prespecified Frobenius-normalized loadings:
   - single-entry loadings,
   - ROI-slice loadings,
   - band-slice loadings,
   - the global loading.
5. Report point estimates and nominal 95% confidence intervals.

## Tensor analysis

1. Stack `log10(EC)` and `log10(EO)` to form a `10 x 8 x 2` tensor covariate for each subject.
2. Run Tucker-rank selection over a predefined rank grid.
3. Use a fixed Tucker rank `(2,5,2)` for the final inference run reported in the current manuscript version.
4. Run `tsrProj` on prespecified loadings:
   - single-entry loadings,
   - ROI-slice loadings,
   - band-slice loadings,
   - condition-slice loadings,
   - full-tensor loading,
   - shared and contrast loadings built from the condition mode.
5. Report point estimates and nominal 95% confidence intervals.

## Centering convention

The current release uses fold-specific centering inside `matProj` and `tsrProj`, so the projection direction and variance estimator are computed from centered covariates and centered responses within the cross-fitted folds.
