import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from run_hbn_trace_regression_nested_cv import (
    build_predictor_arrays,
    center_by_train,
    fit_low_rank_trace_regression,
    matlab_cellstr,
    matlab_cellstr_to_list,
    predict_centered,
    r2_score_centered,
    save_coef_csv,
    stratified_release_kfold,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAT_FILE = REPO_ROOT / "data" / "hbn_bandpower_8band_roi10.mat"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "matrix" / "rank_selection_log_contrast_cv5"


@dataclass(frozen=True)
class FiveFoldTraceRegressionConfig:
    response_name: str = "age"
    n_splits: int = 5
    random_state: int = 20260320
    max_rank: int = 8
    max_iter: int = 4000
    tol: float = 1e-7
    power_transform: str = "log10"
    condition: str = "EC_minus_EO"


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def plot_cv_curve(summary_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(
        summary_df["rank"],
        summary_df["mean_val_mse"],
        yerr=summary_df["std_val_mse"],
        fmt="-o",
        color="#2F5AA8",
        ecolor="#7A9BD1",
        capsize=4,
        linewidth=1.6,
        markersize=5,
    )
    best_row = summary_df.iloc[0]
    ax.scatter([best_row["rank"]], [best_row["mean_val_mse"]], color="#C03A2B", s=45, zorder=4)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Validation MSE")
    ax.set_title("5-fold CV for log10(EC) - log10(EO) trace regression")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    path: str,
    config: FiveFoldTraceRegressionConfig,
    best_rank: int,
    best_row: pd.Series,
    fold_rows: pd.DataFrame,
    final_train_mse: float,
    final_train_r2: float,
) -> None:
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("5-fold CV trace regression summary\n")
        fid.write("Predictor: log10(EC) - log10(EO)\n")
        fid.write("Condition: EC_minus_EO\n")
        fid.write("Transform: log10\n")
        fid.write(f"Number of folds: {config.n_splits}\n")
        fid.write(f"Split seed: {config.random_state}\n")
        fid.write(f"Max rank searched: {config.max_rank}\n")
        fid.write(f"Max iterations: {config.max_iter}\n")
        fid.write(f"Tolerance: {config.tol}\n")
        fid.write(f"Selected rank: {best_rank}\n")
        fid.write(f"Mean CV MSE: {best_row['mean_val_mse']:.6f}\n")
        fid.write(f"Std CV MSE: {best_row['std_val_mse']:.6f}\n")
        fid.write(f"Mean CV R2: {best_row['mean_val_r2']:.6f}\n")
        fid.write(f"Std CV R2: {best_row['std_val_r2']:.6f}\n")
        fid.write(f"Fold-wise validation MSE: {np.array2string(fold_rows['val_mse'].to_numpy(), precision=6)}\n")
        fid.write(f"Fold-wise validation R2: {np.array2string(fold_rows['val_r2'].to_numpy(), precision=6)}\n")
        fid.write(f"Full-sample centered training MSE at selected rank: {final_train_mse:.6f}\n")
        fid.write(f"Full-sample centered training R2 at selected rank: {final_train_r2:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple 5-fold CV rank selection for log10(EC)-log10(EO) trace regression."
    )
    parser.add_argument(
        "--mat-file",
        default=str(DEFAULT_MAT_FILE),
        help="Combined MATLAB file containing predictors and age.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory.",
    )
    parser.add_argument("--max-rank", type=int, default=None, help="Optional override for maximum searched rank.")
    parser.add_argument("--max-iter", type=int, default=None, help="Optional override for projected-gradient iterations.")
    parser.add_argument("--n-splits", type=int, default=None, help="Optional override for number of folds.")
    args = parser.parse_args()

    default_config = FiveFoldTraceRegressionConfig()
    config = FiveFoldTraceRegressionConfig(
        max_rank=args.max_rank if args.max_rank is not None else default_config.max_rank,
        max_iter=args.max_iter if args.max_iter is not None else default_config.max_iter,
        n_splits=args.n_splits if args.n_splits is not None else default_config.n_splits,
    )
    mat_file = Path(args.mat_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not mat_file.exists():
        raise FileNotFoundError(
            "MAT file not found. Place the ROI10 MATLAB file at "
            f"{mat_file} or pass --mat-file explicitly."
        )

    mat = loadmat(mat_file)
    participant_ids = matlab_cellstr_to_list(mat["participant_id"])
    releases = matlab_cellstr_to_list(mat["release"])
    band_names = matlab_cellstr_to_list(mat["band_names"])
    age = mat[config.response_name].reshape(-1).astype(float)

    feature_label, feature_names, arrays, transform_metadata = build_predictor_arrays(
        mat,
        config.power_transform,
    )
    X = arrays[config.condition].astype(np.float64)

    n_subjects = len(participant_ids)
    if not (len(releases) == len(age) == n_subjects == X.shape[0]):
        raise ValueError("Subject metadata length mismatch.")

    search_max_rank = min(config.max_rank, X.shape[1], X.shape[2])
    folds = stratified_release_kfold(releases, config.n_splits, config.random_state)

    fold_manifest_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        for idx in train_idx:
            fold_manifest_rows.append(
                {
                    "participant_id": participant_ids[idx],
                    "release": releases[idx],
                    "age": age[idx],
                    "fold": fold_idx,
                    "split": "train",
                }
            )
        for idx in val_idx:
            fold_manifest_rows.append(
                {
                    "participant_id": participant_ids[idx],
                    "release": releases[idx],
                    "age": age[idx],
                    "fold": fold_idx,
                    "split": "validation",
                }
            )

    for rank in range(1, search_max_rank + 1):
        fold_mse: list[float] = []
        fold_r2: list[float] = []
        fold_iters: list[int] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            X_train = X[train_idx]
            y_train = age[train_idx]
            X_val = X[val_idx]
            y_val = age[val_idx]

            X_train_c, y_train_c, X_val_c, y_val_c, _, _ = center_by_train(X_train, y_train, X_val, y_val)
            B_hat, fit_info = fit_low_rank_trace_regression(
                X_train_c,
                y_train_c,
                rank=rank,
                max_iter=config.max_iter,
                tol=config.tol,
            )
            y_val_pred = predict_centered(X_val_c, B_hat)
            val_mse = mse(y_val_c, y_val_pred)
            val_r2 = r2_score_centered(y_val_c, y_val_pred)
            fold_mse.append(val_mse)
            fold_r2.append(val_r2)
            fold_iters.append(int(fit_info["n_iter"]))

            detail_rows.append(
                {
                    "rank": rank,
                    "fold": fold_idx,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "val_mse": val_mse,
                    "val_r2": val_r2,
                    "fit_iterations": int(fit_info["n_iter"]),
                    "train_objective": float(fit_info["objective"]),
                    "effective_rank": int(fit_info["effective_rank"]),
                    "step_size": float(fit_info["step_size"]),
                    "lipschitz": float(fit_info["lipschitz"]),
                }
            )

        summary_rows.append(
            {
                "rank": rank,
                "mean_val_mse": float(np.mean(fold_mse)),
                "std_val_mse": float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0,
                "mean_val_r2": float(np.nanmean(fold_r2)),
                "std_val_r2": float(np.nanstd(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
                "mean_fit_iterations": float(np.mean(fold_iters)),
            }
        )

    cv_details = pd.DataFrame(detail_rows).sort_values(["rank", "fold"]).reset_index(drop=True)
    cv_summary = pd.DataFrame(summary_rows).sort_values(["mean_val_mse", "rank"]).reset_index(drop=True)
    best_rank = int(cv_summary.iloc[0]["rank"])

    X_full_c, y_full_c, _, _, X_mean_full, y_mean_full = center_by_train(X, age, X, age)
    B_full, fit_info_full = fit_low_rank_trace_regression(
        X_full_c,
        y_full_c,
        rank=best_rank,
        max_iter=config.max_iter,
        tol=config.tol,
    )
    yhat_full = predict_centered(X_full_c, B_full)
    full_train_mse = mse(y_full_c, yhat_full)
    full_train_r2 = r2_score_centered(y_full_c, yhat_full)

    manifest_csv = output_root / "cv5_fold_manifest.csv"
    details_csv = output_root / "cv5_rank_details.csv"
    summary_csv = output_root / "cv5_rank_summary.csv"
    coef_csv = output_root / f"selected_rank_{best_rank}_coefficients.csv"
    summary_txt = output_root / "cv5_summary.txt"
    plot_png = output_root / "cv5_mean_mse_by_rank.png"
    results_mat = output_root / "cv5_selected_model.mat"
    config_json = output_root / "cv5_config.json"

    pd.DataFrame(fold_manifest_rows).sort_values(["fold", "split", "release", "participant_id"]).to_csv(manifest_csv, index=False)
    cv_details.to_csv(details_csv, index=False)
    cv_summary.to_csv(summary_csv, index=False)
    save_coef_csv(str(coef_csv), B_full, feature_names, band_names, feature_label)
    write_summary_text(
        str(summary_txt),
        config,
        best_rank,
        cv_summary.iloc[0],
        cv_details[cv_details["rank"] == best_rank].copy(),
        full_train_mse,
        full_train_r2,
    )
    plot_cv_curve(cv_summary, str(plot_png))

    with open(config_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                **asdict(config),
                "mat_file": str(mat_file),
                "output_root": str(output_root),
                "feature_label": feature_label,
                "n_subjects": n_subjects,
                "band_names": band_names,
                "feature_names": feature_names,
                "transform_metadata": transform_metadata,
            },
            fh,
            indent=2,
        )

    savemat(
        str(results_mat),
        {
            "selected_rank": np.asarray([[best_rank]], dtype=np.int32),
            "B_full": B_full.astype(np.float32),
            "X_mean_full": X_mean_full.astype(np.float32),
            "y_mean_full": np.asarray([[y_mean_full]], dtype=np.float64),
            "yhat_full": yhat_full.astype(np.float32).reshape(-1, 1),
            "age": age.reshape(-1, 1).astype(np.float32),
            "participant_id": matlab_cellstr(participant_ids),
            "release": matlab_cellstr(releases),
            "feature_names": matlab_cellstr(feature_names),
            "feature_label": np.asarray([[feature_label]], dtype=object),
            "band_names": matlab_cellstr(band_names),
            "cv_rank_summary": cv_summary.to_records(index=False),
            "cv_rank_details": cv_details.to_records(index=False),
        },
    )

    print(f"Selected rank: {best_rank}")
    print(f"Mean CV MSE: {cv_summary.iloc[0]['mean_val_mse']:.6f}")
    print(f"Mean CV R2: {cv_summary.iloc[0]['mean_val_r2']:.6f}")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()
