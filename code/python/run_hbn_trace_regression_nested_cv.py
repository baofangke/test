import argparse
import gc
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


@dataclass(frozen=True)
class NestedTraceRegressionConfig:
    response_name: str = "age"
    outer_splits: int = 5
    inner_splits: int = 5
    random_state: int = 20260320
    max_rank: int = 8
    max_iter: int = 1000
    tol: float = 1e-7
    conditions: tuple[str, ...] = ("EO", "EC", "EC_minus_EO")
    power_transform: str = "raw"


def matlab_cellstr_to_list(arr: np.ndarray) -> list[str]:
    values: list[str] = []
    for item in arr.squeeze():
        if isinstance(item, np.ndarray):
            values.append(str(item.squeeze().item()))
        else:
            values.append(str(item))
    return values


def matlab_cellstr(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(-1, 1)


def build_predictor_arrays(
    mat: dict,
    power_transform: str,
) -> tuple[str, list[str], dict[str, np.ndarray], dict[str, float]]:
    if all(key in mat for key in ("EO_power_uV2", "EC_power_uV2", "EC_minus_EO_power_uV2")):
        feature_names = matlab_cellstr_to_list(mat["channel_names"])
        feature_label = "channel"
        raw_eo = mat["EO_power_uV2"].astype(np.float64)
        raw_ec = mat["EC_power_uV2"].astype(np.float64)
    elif all(key in mat for key in ("EO_roi_power_uV2", "EC_roi_power_uV2", "EC_minus_EO_roi_power_uV2")):
        feature_names = matlab_cellstr_to_list(mat["roi_names"])
        feature_label = "roi"
        raw_eo = mat["EO_roi_power_uV2"].astype(np.float64)
        raw_ec = mat["EC_roi_power_uV2"].astype(np.float64)
    else:
        raise ValueError("Unrecognized MAT structure.")

    metadata = {"power_transform": power_transform}
    if power_transform == "raw":
        arrays = {
            "EO": raw_eo,
            "EC": raw_ec,
            "EC_minus_EO": raw_ec - raw_eo,
        }
        metadata["log10_floor_uV2"] = np.nan
        metadata["relative_total_floor_uV2"] = np.nan
        return feature_label, feature_names, arrays, metadata

    total_floor = 1e-12
    eo_total = np.maximum(raw_eo.sum(axis=-1, keepdims=True), total_floor)
    ec_total = np.maximum(raw_ec.sum(axis=-1, keepdims=True), total_floor)
    eo_rel = raw_eo / eo_total
    ec_rel = raw_ec / ec_total

    if power_transform == "relative":
        arrays = {
            "EO": eo_rel,
            "EC": ec_rel,
            "EC_minus_EO": ec_rel - eo_rel,
        }
        metadata["log10_floor_uV2"] = np.nan
        metadata["relative_total_floor_uV2"] = float(total_floor)
        return feature_label, feature_names, arrays, metadata

    if power_transform not in {"log10", "log10_relative"}:
        raise ValueError(f"Unsupported power_transform: {power_transform}")

    base_eo = raw_eo if power_transform == "log10" else eo_rel
    base_ec = raw_ec if power_transform == "log10" else ec_rel

    positives = np.concatenate([base_eo[base_eo > 0], base_ec[base_ec > 0]])
    if positives.size == 0:
        raise ValueError("No positive power values available for log transform.")
    min_positive = float(np.min(positives))
    log_floor = max(min_positive * 0.5, 1e-12)
    eo_log = np.log10(np.maximum(base_eo, log_floor))
    ec_log = np.log10(np.maximum(base_ec, log_floor))
    arrays = {
        "EO": eo_log,
        "EC": ec_log,
        "EC_minus_EO": ec_log - eo_log,
    }
    metadata["log10_floor_uV2"] = float(log_floor)
    metadata["relative_total_floor_uV2"] = float(total_floor) if power_transform == "log10_relative" else np.nan
    return feature_label, feature_names, arrays, metadata


def stratified_release_kfold(
    releases: list[str],
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    releases_arr = np.asarray(releases)
    n_samples = len(releases_arr)
    if n_samples < n_splits:
        raise ValueError(f"n_samples={n_samples} smaller than n_splits={n_splits}")

    rng = np.random.default_rng(random_state)
    per_release_splits: dict[str, list[np.ndarray]] = {}
    for release in sorted(np.unique(releases_arr)):
        idx = np.flatnonzero(releases_arr == release)
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        per_release_splits[release] = [np.asarray(split, dtype=int) for split in np.array_split(shuffled, n_splits)]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n_samples, dtype=int)
    for fold_idx in range(n_splits):
        test_parts = [splits[fold_idx] for splits in per_release_splits.values()]
        test_idx = np.sort(np.concatenate(test_parts))
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        folds.append((train_idx, test_idx))
    return folds


def project_rank(B: np.ndarray, rank: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    keep = min(rank, len(s))
    if keep <= 0:
        return np.zeros_like(B)
    return (U[:, :keep] * s[:keep]) @ Vt[:keep, :]


def fit_low_rank_trace_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rank: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_samples, n_features, n_bands = X_train.shape
    Z = X_train.reshape(n_samples, n_features * n_bands)

    smax = float(np.linalg.svd(Z, compute_uv=False, full_matrices=False)[0]) if len(Z) else 0.0
    lipschitz = max((smax * smax) / max(n_samples, 1), 1e-8)
    step = 1.0 / lipschitz

    B = np.zeros((n_features, n_bands), dtype=np.float64)
    prev_obj = np.inf
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        residual = Z @ B.reshape(-1) - y_train
        grad = ((Z.T @ residual) / n_samples).reshape(n_features, n_bands)
        candidate = project_rank(B - step * grad, rank)
        residual_new = Z @ candidate.reshape(-1) - y_train
        obj = 0.5 * float(np.mean(residual_new ** 2))
        rel_change = np.linalg.norm(candidate - B) / max(np.linalg.norm(B), 1e-12)
        B = candidate
        n_iter = iteration
        obj_change = np.inf if not np.isfinite(prev_obj) else abs(prev_obj - obj) / max(abs(prev_obj), 1.0)
        if rel_change < tol or obj_change < tol:
            prev_obj = obj
            break
        prev_obj = obj

    singular_values = np.linalg.svd(B, compute_uv=False, full_matrices=False)
    info = {
        "step_size": step,
        "lipschitz": lipschitz,
        "n_iter": n_iter,
        "objective": prev_obj,
        "effective_rank": int(np.sum(singular_values > 1e-8)),
    }
    return B, info


def center_by_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_other: np.ndarray,
    y_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    X_mean = X_train.mean(axis=0)
    y_mean = float(y_train.mean())
    X_train_c = X_train - X_mean[None, :, :]
    X_other_c = X_other - X_mean[None, :, :]
    y_train_c = y_train - y_mean
    y_other_c = y_other - y_mean
    return X_train_c, y_train_c, X_other_c, y_other_c, X_mean, y_mean


def predict_centered(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.tensordot(X, B, axes=([1, 2], [0, 1]))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score_centered(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum(y_true ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def save_coef_csv(path: str, B: np.ndarray, feature_names: list[str], band_names: list[str], feature_label: str) -> None:
    df = pd.DataFrame(B, index=feature_names, columns=band_names)
    df.index.name = feature_label
    df.to_csv(path)


def choose_mode_rank(ranks: list[int]) -> int:
    counts = Counter(ranks)
    best_freq = max(counts.values())
    winners = sorted(rank for rank, freq in counts.items() if freq == best_freq)
    return winners[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested-CV low-rank trace regression on HBN power matrices.")
    parser.add_argument("--mat-file", required=True, help="Combined MATLAB file containing predictors and age.")
    parser.add_argument("--output-root", required=True, help="Output root for nested-CV results.")
    parser.add_argument("--max-rank", type=int, default=None, help="Optional override for maximum searched rank.")
    parser.add_argument("--max-iter", type=int, default=None, help="Optional override for projected-gradient iterations.")
    parser.add_argument("--outer-splits", type=int, default=None, help="Optional outer CV fold count.")
    parser.add_argument("--inner-splits", type=int, default=None, help="Optional inner CV fold count.")
    parser.add_argument(
        "--power-transform",
        choices=["raw", "log10", "relative", "log10_relative"],
        default="raw",
        help="Predictor transform: raw, log10, relative power, or log10 relative power.",
    )
    args = parser.parse_args()

    default_config = NestedTraceRegressionConfig()
    config = NestedTraceRegressionConfig(
        max_rank=args.max_rank if args.max_rank is not None else default_config.max_rank,
        max_iter=args.max_iter if args.max_iter is not None else default_config.max_iter,
        outer_splits=args.outer_splits if args.outer_splits is not None else default_config.outer_splits,
        inner_splits=args.inner_splits if args.inner_splits is not None else default_config.inner_splits,
        power_transform=args.power_transform,
    )
    os.makedirs(args.output_root, exist_ok=True)

    mat = loadmat(args.mat_file)
    participant_ids = matlab_cellstr_to_list(mat["participant_id"])
    releases = matlab_cellstr_to_list(mat["release"])
    band_names = matlab_cellstr_to_list(mat["band_names"])
    age = mat["age"].reshape(-1).astype(float)

    feature_label, feature_names, arrays, transform_metadata = build_predictor_arrays(mat, config.power_transform)

    n_subjects = len(participant_ids)
    if not (len(releases) == len(age) == n_subjects):
        raise ValueError("Subject metadata length mismatch.")

    outer_folds = stratified_release_kfold(releases, config.outer_splits, config.random_state)
    fold_assignments = np.full(n_subjects, -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(outer_folds, start=1):
        fold_assignments[test_idx] = fold_id
    fold_df = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "release": releases,
            "age": age,
            "outer_fold": fold_assignments,
            "outer_split_role": "test",
        }
    ).sort_values(["outer_fold", "release", "participant_id"])
    fold_csv = os.path.join(args.output_root, "outer_fold_assignments.csv")
    fold_df.to_csv(fold_csv, index=False)

    summary_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for condition in config.conditions:
        X = arrays[condition]
        max_rank = min(config.max_rank, X.shape[1], X.shape[2])
        ranks = list(range(1, max_rank + 1))

        condition_dir = os.path.join(args.output_root, condition.lower())
        os.makedirs(condition_dir, exist_ok=True)

        inner_rows: list[dict[str, object]] = []
        outer_grid_rows: list[dict[str, object]] = []
        outer_selected_rows: list[dict[str, object]] = []

        coef_by_outer_rank = np.full((config.outer_splits, max_rank, X.shape[1], X.shape[2]), np.nan, dtype=np.float32)
        singular_values_by_outer_rank = np.full(
            (config.outer_splits, max_rank, min(X.shape[1], X.shape[2])),
            np.nan,
            dtype=np.float32,
        )
        selected_rank_by_outer = np.full(config.outer_splits, np.nan, dtype=np.float32)

        for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_folds, start=1):
            outer_train_releases = [releases[idx] for idx in outer_train_idx]
            inner_folds = stratified_release_kfold(
                outer_train_releases,
                config.inner_splits,
                config.random_state + 1000 * outer_fold_idx,
            )

            inner_summary_by_rank: dict[int, list[float]] = {rank: [] for rank in ranks}

            for inner_fold_idx, (inner_train_rel_idx, inner_val_rel_idx) in enumerate(inner_folds, start=1):
                inner_train_idx = outer_train_idx[inner_train_rel_idx]
                inner_val_idx = outer_train_idx[inner_val_rel_idx]

                X_inner_train, y_inner_train = X[inner_train_idx], age[inner_train_idx]
                X_inner_val, y_inner_val = X[inner_val_idx], age[inner_val_idx]
                X_inner_train_c, y_inner_train_c, X_inner_val_c, y_inner_val_c, _, y_inner_mean = center_by_train(
                    X_inner_train, y_inner_train, X_inner_val, y_inner_val
                )
                baseline_val_mse = float(np.mean((y_inner_val - y_inner_mean) ** 2))

                for rank in ranks:
                    B_hat, fit_info = fit_low_rank_trace_regression(
                        X_inner_train_c,
                        y_inner_train_c,
                        rank=rank,
                        max_iter=config.max_iter,
                        tol=config.tol,
                    )
                    train_pred = predict_centered(X_inner_train_c, B_hat)
                    val_pred = predict_centered(X_inner_val_c, B_hat)
                    train_mse = mse(y_inner_train_c, train_pred)
                    val_mse = mse(y_inner_val_c, val_pred)
                    inner_summary_by_rank[rank].append(val_mse)
                    inner_rows.append(
                        {
                            "condition": condition,
                            "outer_fold": outer_fold_idx,
                            "inner_fold": inner_fold_idx,
                            "rank": rank,
                            "train_subjects": int(len(inner_train_idx)),
                            "val_subjects": int(len(inner_val_idx)),
                            "train_mse": train_mse,
                            "val_mse": val_mse,
                            "train_rmse": float(np.sqrt(train_mse)),
                            "val_rmse": float(np.sqrt(val_mse)),
                            "baseline_val_mse": baseline_val_mse,
                            "step_size": fit_info["step_size"],
                            "n_iter": fit_info["n_iter"],
                            "effective_rank": fit_info["effective_rank"],
                        }
                    )

            inner_mean_val_by_rank = {rank: float(np.mean(vals)) for rank, vals in inner_summary_by_rank.items()}
            selected_rank = min(ranks, key=lambda rank: (inner_mean_val_by_rank[rank], rank))
            selected_rank_by_outer[outer_fold_idx - 1] = selected_rank

            X_outer_train, y_outer_train = X[outer_train_idx], age[outer_train_idx]
            X_outer_test, y_outer_test = X[outer_test_idx], age[outer_test_idx]
            X_outer_train_c, y_outer_train_c, X_outer_test_c, y_outer_test_c, X_outer_mean, y_outer_mean = center_by_train(
                X_outer_train, y_outer_train, X_outer_test, y_outer_test
            )
            baseline_test_mse = float(np.mean((y_outer_test - y_outer_mean) ** 2))

            selected_outer_test_mse = np.nan
            selected_outer_test_rmse = np.nan
            selected_outer_test_r2 = np.nan

            for rank in ranks:
                B_hat, fit_info = fit_low_rank_trace_regression(
                    X_outer_train_c,
                    y_outer_train_c,
                    rank=rank,
                    max_iter=config.max_iter,
                    tol=config.tol,
                )
                train_pred = predict_centered(X_outer_train_c, B_hat)
                test_pred = predict_centered(X_outer_test_c, B_hat)
                train_mse = mse(y_outer_train_c, train_pred)
                test_mse = mse(y_outer_test_c, test_pred)
                train_rmse = rmse(y_outer_train_c, train_pred)
                test_rmse = rmse(y_outer_test_c, test_pred)
                train_r2 = r2_score_centered(y_outer_train_c, train_pred)
                test_r2 = r2_score_centered(y_outer_test_c, test_pred)

                coef_by_outer_rank[outer_fold_idx - 1, rank - 1] = B_hat.astype(np.float32)
                singular_values = np.linalg.svd(B_hat, compute_uv=False, full_matrices=False)
                singular_values_by_outer_rank[outer_fold_idx - 1, rank - 1, : len(singular_values)] = singular_values.astype(
                    np.float32
                )

                outer_grid_rows.append(
                    {
                        "condition": condition,
                        "outer_fold": outer_fold_idx,
                        "rank": rank,
                        "train_subjects": int(len(outer_train_idx)),
                        "test_subjects": int(len(outer_test_idx)),
                        "inner_mean_val_mse": inner_mean_val_by_rank[rank],
                        "inner_mean_val_rmse": float(np.sqrt(inner_mean_val_by_rank[rank])),
                        "outer_train_mse": train_mse,
                        "outer_test_mse": test_mse,
                        "outer_train_rmse": train_rmse,
                        "outer_test_rmse": test_rmse,
                        "outer_train_r2": train_r2,
                        "outer_test_r2": test_r2,
                        "baseline_test_mse": baseline_test_mse,
                        "selected_by_inner_cv": bool(rank == selected_rank),
                        "step_size": fit_info["step_size"],
                        "n_iter": fit_info["n_iter"],
                        "effective_rank": fit_info["effective_rank"],
                    }
                )

                coef_csv = os.path.join(condition_dir, f"{condition.lower()}_outerfold_{outer_fold_idx:02d}_coef_rank_{rank:02d}.csv")
                save_coef_csv(coef_csv, B_hat, feature_names, band_names, feature_label)

                if rank == selected_rank:
                    selected_outer_test_mse = test_mse
                    selected_outer_test_rmse = test_rmse
                    selected_outer_test_r2 = test_r2

            outer_selected_rows.append(
                {
                    "condition": condition,
                    "outer_fold": outer_fold_idx,
                    "selected_rank": selected_rank,
                    "outer_test_mse": selected_outer_test_mse,
                    "outer_test_rmse": selected_outer_test_rmse,
                    "outer_test_r2": selected_outer_test_r2,
                    "baseline_test_mse": baseline_test_mse,
                    "train_subjects": int(len(outer_train_idx)),
                    "test_subjects": int(len(outer_test_idx)),
                }
            )

            print(
                f"{condition} outer_fold={outer_fold_idx} selected_rank={selected_rank} "
                f"outer_test_mse={selected_outer_test_mse:.6f}"
            )
            gc.collect()

        inner_df = pd.DataFrame(inner_rows).sort_values(["outer_fold", "inner_fold", "rank"])
        inner_csv = os.path.join(condition_dir, f"{condition.lower()}_inner_cv_rank_metrics.csv")
        inner_df.to_csv(inner_csv, index=False)

        outer_grid_df = pd.DataFrame(outer_grid_rows).sort_values(["outer_fold", "rank"])
        outer_grid_csv = os.path.join(condition_dir, f"{condition.lower()}_outer_rank_grid_metrics.csv")
        outer_grid_df.to_csv(outer_grid_csv, index=False)

        outer_selected_df = pd.DataFrame(outer_selected_rows).sort_values("outer_fold")
        outer_selected_csv = os.path.join(condition_dir, f"{condition.lower()}_outer_selected_rank_by_fold.csv")
        outer_selected_df.to_csv(outer_selected_csv, index=False)

        recommended_rank = choose_mode_rank(outer_selected_df["selected_rank"].astype(int).tolist())
        mean_outer_test_mse = float(outer_selected_df["outer_test_mse"].mean())
        std_outer_test_mse = float(outer_selected_df["outer_test_mse"].std(ddof=1)) if len(outer_selected_df) > 1 else 0.0
        mean_outer_test_r2 = float(outer_selected_df["outer_test_r2"].mean())
        rank_freq = Counter(outer_selected_df["selected_rank"].astype(int).tolist())

        X_full = arrays[condition]
        X_full_mean = X_full.mean(axis=0)
        y_full_mean = float(age.mean())
        X_full_c = X_full - X_full_mean[None, :, :]
        y_full_c = age - y_full_mean

        full_coef_by_rank = np.full((max_rank, X_full.shape[1], X_full.shape[2]), np.nan, dtype=np.float32)
        full_train_mse_by_rank = np.full(max_rank, np.nan, dtype=np.float64)
        for rank in ranks:
            B_hat, _ = fit_low_rank_trace_regression(
                X_full_c,
                y_full_c,
                rank=rank,
                max_iter=config.max_iter,
                tol=config.tol,
            )
            full_coef_by_rank[rank - 1] = B_hat.astype(np.float32)
            full_pred = predict_centered(X_full_c, B_hat)
            full_train_mse_by_rank[rank - 1] = mse(y_full_c, full_pred)
            full_coef_csv = os.path.join(condition_dir, f"{condition.lower()}_full_data_coef_rank_{rank:02d}.csv")
            save_coef_csv(full_coef_csv, B_hat, feature_names, band_names, feature_label)

        selected_json = os.path.join(condition_dir, f"{condition.lower()}_nested_cv_selected_rank.json")
        selected_summary = {
            "condition": condition,
            "recommended_rank": int(recommended_rank),
            "rank_frequency": {str(k): int(v) for k, v in sorted(rank_freq.items())},
            "mean_outer_test_mse": mean_outer_test_mse,
            "std_outer_test_mse": std_outer_test_mse,
            "mean_outer_test_r2": mean_outer_test_r2,
            "outer_splits": config.outer_splits,
            "inner_splits": config.inner_splits,
            "feature_mode": feature_label,
            "power_transform": config.power_transform,
            "n_subjects": n_subjects,
        }
        with open(selected_json, "w", encoding="utf-8") as handle:
            json.dump(selected_summary, handle, indent=2)

        condition_mat = os.path.join(condition_dir, f"{condition.lower()}_nested_cv_trace_regression.mat")
        savemat(
            condition_mat,
            {
                "condition": matlab_cellstr([condition]),
                "participant_id": matlab_cellstr(participant_ids),
                "release": matlab_cellstr(releases),
                f"{feature_label}_names": matlab_cellstr(feature_names),
                "band_names": matlab_cellstr(band_names),
                "power_transform": matlab_cellstr([config.power_transform]),
                "outer_fold_assignments": fold_assignments.reshape(-1, 1),
                "ranks": np.asarray(ranks, dtype=np.int32).reshape(-1, 1),
                "selected_rank_by_outer_fold": selected_rank_by_outer.reshape(-1, 1),
                "coef_by_outer_fold_rank": coef_by_outer_rank,
                "singular_values_by_outer_fold_rank": singular_values_by_outer_rank,
                "mean_inner_val_mse_by_outer_rank": outer_grid_df.pivot(index="outer_fold", columns="rank", values="inner_mean_val_mse")
                .reindex(index=range(1, config.outer_splits + 1), columns=ranks)
                .to_numpy(dtype=np.float64),
                "outer_test_mse_by_rank": outer_grid_df.pivot(index="outer_fold", columns="rank", values="outer_test_mse")
                .reindex(index=range(1, config.outer_splits + 1), columns=ranks)
                .to_numpy(dtype=np.float64),
                "outer_test_r2_by_rank": outer_grid_df.pivot(index="outer_fold", columns="rank", values="outer_test_r2")
                .reindex(index=range(1, config.outer_splits + 1), columns=ranks)
                .to_numpy(dtype=np.float64),
                "full_data_coef_by_rank": full_coef_by_rank,
                "full_data_train_mse_by_rank": full_train_mse_by_rank.reshape(-1, 1),
                "recommended_rank": np.asarray([[recommended_rank]], dtype=np.int32),
                "log10_floor_uV2": np.asarray([[transform_metadata["log10_floor_uV2"]]], dtype=np.float64),
                "relative_total_floor_uV2": np.asarray(
                    [[transform_metadata["relative_total_floor_uV2"]]], dtype=np.float64
                ),
            },
            do_compression=True,
        )

        summary_rows.append(
            {
                "condition": condition,
                "power_transform": config.power_transform,
                "recommended_rank": int(recommended_rank),
                "mean_outer_test_mse": mean_outer_test_mse,
                "std_outer_test_mse": std_outer_test_mse,
                "mean_outer_test_r2": mean_outer_test_r2,
                "rank_frequency": json.dumps({str(k): int(v) for k, v in sorted(rank_freq.items())}),
            }
        )
        manifest_rows.append(
            {
                "condition": condition,
                "inner_metrics_csv": inner_csv,
                "outer_grid_csv": outer_grid_csv,
                "outer_selected_csv": outer_selected_csv,
                "selected_json": selected_json,
                "condition_mat": condition_mat,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("condition")
    summary_csv = os.path.join(args.output_root, "nested_cv_selected_rank_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    manifest_df = pd.DataFrame(manifest_rows).sort_values("condition")
    manifest_csv = os.path.join(args.output_root, "nested_cv_output_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)

    config_json = os.path.join(args.output_root, "nested_cv_trace_regression_config.json")
    with open(config_json, "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    summary_lines = [
        f"MAT file: {args.mat_file}",
        f"Output root: {args.output_root}",
        f"Response: {config.response_name}",
        f"Feature mode: {feature_label}",
        f"Power transform: {config.power_transform}",
        f"Subjects: {n_subjects}",
        f"Outer splits: {config.outer_splits}",
        f"Inner splits: {config.inner_splits}",
        f"Max rank searched: {min(config.max_rank, len(feature_names), len(band_names))}",
        f"log10 floor uV2: {transform_metadata['log10_floor_uV2']}",
        f"relative total floor uV2: {transform_metadata['relative_total_floor_uV2']}",
        f"Summary CSV: {summary_csv}",
        f"Manifest CSV: {manifest_csv}",
    ]
    summary_txt = os.path.join(args.output_root, "nested_cv_trace_regression_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print(f"Nested-CV summary: {summary_csv}")
    print(f"Manifest: {manifest_csv}")
    print(f"Config: {config_json}")


if __name__ == "__main__":
    main()
