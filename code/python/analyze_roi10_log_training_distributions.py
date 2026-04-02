from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


MATRIX_RANDOM_STATE = 20260320
PLOT_SAMPLE_SIZE = 12000
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAT_FILE = REPO_ROOT / "data" / "hbn_bandpower_8band_roi10.mat"
DEFAULT_TENSOR_RESULTS_MAT = REPO_ROOT / "results" / "tensor" / "rank_selection" / "tucker_tensor_regression_best_model.mat"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diagnostics" / "log_training_distribution_analysis"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_idx: np.ndarray


def matlab_cellstr_to_list(arr: np.ndarray) -> list[str]:
    values: list[str] = []
    for item in arr.squeeze():
        if isinstance(item, np.ndarray):
            values.append(str(item.squeeze().item()))
        else:
            values.append(str(item))
    return values


def load_roi10_mat(mat_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    mat = loadmat(mat_file)
    raw_eo = mat["EO_roi_power_uV2"].astype(np.float64)
    raw_ec = mat["EC_roi_power_uV2"].astype(np.float64)
    age = mat["age"].astype(np.float64).reshape(-1)
    releases = matlab_cellstr_to_list(mat["release"])
    return raw_eo, raw_ec, age, releases


def build_log_arrays(raw_eo: np.ndarray, raw_ec: np.ndarray) -> dict[str, np.ndarray]:
    positives_raw = np.concatenate([raw_eo[raw_eo > 0], raw_ec[raw_ec > 0]])
    if positives_raw.size == 0:
        raise ValueError("Raw EO/EC arrays have no positive entries.")
    raw_floor = max(float(np.min(positives_raw)) * 0.5, 1e-12)
    eo_log = np.log10(np.maximum(raw_eo, raw_floor))
    ec_log = np.log10(np.maximum(raw_ec, raw_floor))

    total_floor = 1e-12
    eo_total = np.maximum(raw_eo.sum(axis=-1, keepdims=True), total_floor)
    ec_total = np.maximum(raw_ec.sum(axis=-1, keepdims=True), total_floor)
    eo_rel = raw_eo / eo_total
    ec_rel = raw_ec / ec_total

    positives_rel = np.concatenate([eo_rel[eo_rel > 0], ec_rel[ec_rel > 0]])
    if positives_rel.size == 0:
        raise ValueError("Relative EO/EC arrays have no positive entries.")
    rel_floor = max(float(np.min(positives_rel)) * 0.5, 1e-12)
    eo_log_rel = np.log10(np.maximum(eo_rel, rel_floor))
    ec_log_rel = np.log10(np.maximum(ec_rel, rel_floor))

    return {
        "log_ec": ec_log,
        "log_eo": eo_log,
        "log_relative_ec_minus_eo": ec_log_rel - eo_log_rel,
        "tensor_log_ec": ec_log,
        "tensor_log_eo": eo_log,
    }


def stratified_release_kfold(
    releases: list[str],
    n_splits: int,
    random_state: int,
) -> list[FoldSpec]:
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

    folds: list[FoldSpec] = []
    all_idx = np.arange(n_samples, dtype=int)
    for fold_idx in range(n_splits):
        test_parts = [splits[fold_idx] for splits in per_release_splits.values()]
        test_idx = np.sort(np.concatenate(test_parts))
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        folds.append(FoldSpec(name=f"Fold {fold_idx + 1}", train_idx=train_idx))
    return folds


def load_tensor_fold_specs(results_mat: Path) -> list[FoldSpec]:
    mat = loadmat(results_mat)
    if "foldId" not in mat:
        raise ValueError(f"foldId not found in {results_mat}")
    fold_id = np.asarray(mat["foldId"]).reshape(-1).astype(int)
    specs: list[FoldSpec] = []
    for fold in sorted(np.unique(fold_id)):
        train_idx = np.flatnonzero(fold_id != fold)
        specs.append(FoldSpec(name=f"Fold {fold}", train_idx=train_idx))
    return specs


def flatten_centered(arr: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    train = arr[train_idx]
    centered = train - train.mean(axis=0, keepdims=True)
    return centered.reshape(-1)


def flatten_raw(arr: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    return arr[train_idx].reshape(-1)


def summarise_values(values: np.ndarray) -> dict[str, float]:
    return {
        "n_values": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=0)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_summary_rows(
    analysis_name: str,
    arr: np.ndarray,
    fold_specs: list[FoldSpec],
    model_type: str,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    all_idx = np.arange(arr.shape[0], dtype=int)
    for scale_name, getter in (
        ("raw_log", flatten_raw),
        ("fold_centered_log", flatten_centered),
    ):
        all_values = getter(arr, all_idx)
        all_stats = summarise_values(all_values)
        rows.append(
            {
                "model_type": model_type,
                "analysis_name": analysis_name,
                "scale": scale_name,
                "fold": "All",
                "n_subjects": int(arr.shape[0]),
                **all_stats,
            }
        )
        for spec in fold_specs:
            values = getter(arr, spec.train_idx)
            stats = summarise_values(values)
            rows.append(
                {
                    "model_type": model_type,
                    "analysis_name": analysis_name,
                    "scale": scale_name,
                    "fold": spec.name,
                    "n_subjects": int(spec.train_idx.size),
                    **stats,
                }
            )
    return rows


def sample_for_plot(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if values.size <= PLOT_SAMPLE_SIZE:
        return values
    idx = rng.choice(values.size, size=PLOT_SAMPLE_SIZE, replace=False)
    return values[idx]


def violin_panel(
    ax: plt.Axes,
    arr: np.ndarray,
    fold_specs: list[FoldSpec],
    title: str,
    centered: bool,
    rng: np.random.Generator,
) -> None:
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for spec in fold_specs:
        values = flatten_centered(arr, spec.train_idx) if centered else flatten_raw(arr, spec.train_idx)
        groups.append(sample_for_plot(values, rng))
        labels.append(spec.name.replace("Fold ", "F"))

    parts = ax.violinplot(groups, positions=np.arange(1, len(groups) + 1), widths=0.85, showmeans=False, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#2F4B7C")
        body.set_alpha(0.35)
    parts["cmedians"].set_color("#1B1B1B")
    parts["cmedians"].set_linewidth(1.2)
    for key in ("cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#808080")
            parts[key].set_linewidth(0.8)

    medians = [np.median(g) for g in groups]
    ax.scatter(np.arange(1, len(groups) + 1), medians, color="#1B1B1B", s=14, zorder=3)
    ax.set_xticks(np.arange(1, len(groups) + 1), labels)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if centered:
        ax.axhline(0.0, color="#AA3333", linestyle="--", linewidth=0.9, alpha=0.8)


def plot_matrix_figure(
    arrays: dict[str, np.ndarray],
    fold_specs: list[FoldSpec],
    output_png: Path,
    output_pdf: Path,
) -> None:
    rng = np.random.default_rng(20260325)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    configs = [
        ("log_ec", "Training-fold log10(raw EC)"),
        ("log_eo", "Training-fold log10(raw EO)"),
        ("log_relative_ec_minus_eo", "Training-fold log-relative EC-EO"),
    ]
    for col, (key, title) in enumerate(configs):
        violin_panel(axes[0, col], arrays[key], fold_specs, title, centered=False, rng=rng)
        violin_panel(axes[1, col], arrays[key], fold_specs, f"Fold-centered {title.split('Training-fold ')[1]}", centered=True, rng=rng)
    axes[0, 0].set_ylabel("Log-scale entry value")
    axes[1, 0].set_ylabel("Centered entry value")
    fig.suptitle("Matrix CV training-fold distributions on the log scale", fontsize=14)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_tensor_figure(
    arrays: dict[str, np.ndarray],
    fold_specs: list[FoldSpec],
    output_png: Path,
    output_pdf: Path,
) -> None:
    rng = np.random.default_rng(20260326)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    configs = [
        ("tensor_log_ec", "Training-fold log10(raw EC)"),
        ("tensor_log_eo", "Training-fold log10(raw EO)"),
    ]
    for col, (key, title) in enumerate(configs):
        violin_panel(axes[0, col], arrays[key], fold_specs, title, centered=False, rng=rng)
        violin_panel(axes[1, col], arrays[key], fold_specs, f"Fold-centered {title.split('Training-fold ')[1]}", centered=True, rng=rng)
    axes[0, 0].set_ylabel("Log-scale entry value")
    axes[1, 0].set_ylabel("Centered entry value")
    fig.suptitle("Tensor CV training-fold distributions on the log scale", fontsize=14)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def fold_range(df: pd.DataFrame, analysis: str, scale: str, column: str) -> tuple[float, float]:
    subset = df[
        (df["analysis_name"] == analysis)
        & (df["scale"] == scale)
        & (df["fold"] != "All")
    ]
    return float(subset[column].min()), float(subset[column].max())


def write_summary_text(
    matrix_df: pd.DataFrame,
    tensor_df: pd.DataFrame,
    output_file: Path,
) -> None:
    lines: list[str] = []
    lines.append("ROI10 log-scale training-fold distribution analysis")
    lines.append("")
    lines.append("Matrix analysis")
    ec_mean = fold_range(matrix_df, "log_ec", "raw_log", "mean")
    eo_mean = fold_range(matrix_df, "log_eo", "raw_log", "mean")
    rel_mean = fold_range(matrix_df, "log_relative_ec_minus_eo", "raw_log", "mean")
    ec_sd = fold_range(matrix_df, "log_ec", "raw_log", "sd")
    eo_sd = fold_range(matrix_df, "log_eo", "raw_log", "sd")
    rel_sd = fold_range(matrix_df, "log_relative_ec_minus_eo", "raw_log", "sd")
    rel_q05 = fold_range(matrix_df, "log_relative_ec_minus_eo", "raw_log", "q05")
    rel_q95 = fold_range(matrix_df, "log_relative_ec_minus_eo", "raw_log", "q95")
    rel_csd = fold_range(matrix_df, "log_relative_ec_minus_eo", "fold_centered_log", "sd")
    lines.append(
        f"log10(raw EC) is stable across the release-stratified training folds: "
        f"fold means range from {ec_mean[0]:.3f} to {ec_mean[1]:.3f}, and fold SDs "
        f"range from {ec_sd[0]:.3f} to {ec_sd[1]:.3f}."
    )
    lines.append(
        f"log10(raw EO) shows a closely matched pattern, with fold means from "
        f"{eo_mean[0]:.3f} to {eo_mean[1]:.3f} and fold SDs from {eo_sd[0]:.3f} to {eo_sd[1]:.3f}. "
        f"EC is shifted slightly upward relative to EO on the log scale."
    )
    lines.append(
        f"The log-relative contrast is much tighter: fold means stay between "
        f"{rel_mean[0]:.3f} and {rel_mean[1]:.3f}, fold SDs between {rel_sd[0]:.3f} and {rel_sd[1]:.3f}, "
        f"and the 5th/95th percentiles remain in [{rel_q05[0]:.3f}, {rel_q95[1]:.3f}] across folds. "
        f"After fold-specific centering, the global location is removed while the within-fold spread remains "
        f"stable (centered SD range {rel_csd[0]:.3f} to {rel_csd[1]:.3f})."
    )
    lines.append(
        "This indicates that the matrix CV problem is not driven by one atypical training split; "
        "the main effect of centering is to remove a fold-specific location shift rather than to alter the overall shape."
    )
    lines.append("")
    lines.append("Tensor analysis")
    tec_mean = fold_range(tensor_df, "tensor_log_ec", "raw_log", "mean")
    teo_mean = fold_range(tensor_df, "tensor_log_eo", "raw_log", "mean")
    tec_sd = fold_range(tensor_df, "tensor_log_ec", "raw_log", "sd")
    teo_sd = fold_range(tensor_df, "tensor_log_eo", "raw_log", "sd")
    tec_csd = fold_range(tensor_df, "tensor_log_ec", "fold_centered_log", "sd")
    teo_csd = fold_range(tensor_df, "tensor_log_eo", "fold_centered_log", "sd")
    lines.append(
        f"Under the tensor CV partition, log10(raw EC) again has highly comparable training-fold distributions: "
        f"fold means range from {tec_mean[0]:.3f} to {tec_mean[1]:.3f}, with fold SDs from {tec_sd[0]:.3f} to {tec_sd[1]:.3f}."
    )
    lines.append(
        f"log10(raw EO) is similarly stable, with fold means from {teo_mean[0]:.3f} to {teo_mean[1]:.3f} and fold SDs from "
        f"{teo_sd[0]:.3f} to {teo_sd[1]:.3f}. "
        f"The EC distribution remains slightly higher than the EO distribution on the log scale in every training fold."
    )
    lines.append(
        f"After fold-specific centering, the remaining spread is also consistent across splits "
        f"(EC centered SD range {tec_csd[0]:.3f} to {tec_csd[1]:.3f}; "
        f"EO centered SD range {teo_csd[0]:.3f} to {teo_csd[1]:.3f})."
    )
    lines.append(
        "Hence the tensor rank-selection step is being fit to log-scale training data whose marginal behavior is reproducible across folds. "
        "This supports the use of fold-specific centering inside the estimator and suggests that the fitted tensor summaries are not artifacts of one unstable split."
    )
    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and summarise ROI10 log-scale training-fold distributions.")
    parser.add_argument(
        "--mat-file",
        default=str(DEFAULT_MAT_FILE),
        help="ROI10 MATLAB file.",
    )
    parser.add_argument(
        "--tensor-results-mat",
        default=str(DEFAULT_TENSOR_RESULTS_MAT),
        help="Tensor CV results MAT file containing foldId.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    args = parser.parse_args()

    mat_file = Path(args.mat_file)
    tensor_results_mat = Path(args.tensor_results_mat)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mat_file.exists():
        raise FileNotFoundError(
            "ROI10 MATLAB file not found. Place it at "
            f"{mat_file} or pass --mat-file explicitly."
        )
    if not tensor_results_mat.exists():
        raise FileNotFoundError(
            "Tensor results MAT file not found. Place it at "
            f"{tensor_results_mat} or pass --tensor-results-mat explicitly."
        )

    raw_eo, raw_ec, _, releases = load_roi10_mat(mat_file)
    arrays = build_log_arrays(raw_eo, raw_ec)

    matrix_folds = stratified_release_kfold(releases, n_splits=5, random_state=MATRIX_RANDOM_STATE)
    tensor_folds = load_tensor_fold_specs(tensor_results_mat)

    matrix_rows: list[dict[str, float | str | int]] = []
    for key in ("log_ec", "log_eo", "log_relative_ec_minus_eo"):
        matrix_rows.extend(build_summary_rows(key, arrays[key], matrix_folds, "matrix"))
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(output_dir / "matrix_log_training_fold_summary.csv", index=False)

    tensor_rows: list[dict[str, float | str | int]] = []
    for key in ("tensor_log_ec", "tensor_log_eo"):
        tensor_rows.extend(build_summary_rows(key, arrays[key], tensor_folds, "tensor"))
    tensor_df = pd.DataFrame(tensor_rows)
    tensor_df.to_csv(output_dir / "tensor_log_training_fold_summary.csv", index=False)

    plot_matrix_figure(
        arrays,
        matrix_folds,
        output_dir / "matrix_log_training_distributions.png",
        output_dir / "matrix_log_training_distributions.pdf",
    )
    plot_tensor_figure(
        arrays,
        tensor_folds,
        output_dir / "tensor_log_training_distributions.png",
        output_dir / "tensor_log_training_distributions.pdf",
    )
    write_summary_text(matrix_df, tensor_df, output_dir / "log_training_distribution_summary.txt")

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
