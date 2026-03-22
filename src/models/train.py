"""Training entrypoint for the wallet classification pipeline.

Orchestrates the full workflow:
1. Load features (parquet or ClickHouse)
2. Load ground-truth labels
3. Stratified train / val / test split
4. Train XGBoost + MLP
5. Optimize ensemble weights
6. Evaluate on test set
7. Save model artefacts
8. (Optional) log metrics to Weights & Biases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from src.config import settings
from src.models.classifier import WalletClassifier
from src.models.evaluation import (
    compute_shap_values,
    cross_validate,
    evaluate_model,
    generate_evaluation_report,
    plot_calibration_curve,
    plot_confusion_matrix,
)

logger = structlog.get_logger(__name__)

# Default label names -- can be overridden via the labels file.
DEFAULT_LABEL_NAMES: list[str] = [
    "whale",
    "dex_trader",
    "nft_collector",
    "defi_farmer",
    "airdrop_hunter",
    "bot",
    "retail",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_features(features_path: str | None = None) -> pd.DataFrame:
    """Load feature matrix from a parquet file or ClickHouse.

    If *features_path* is provided and the file exists, load from parquet.
    Otherwise fall back to ClickHouse (all wallets).
    """
    if features_path and Path(features_path).exists():
        logger.info("loading_features_from_parquet", path=features_path)
        return pd.read_parquet(features_path)

    logger.info("loading_features_from_clickhouse")
    from src.data.clickhouse_sync import get_client

    client = get_client()
    db = settings.clickhouse_database
    result = client.query(f"SELECT * FROM {db}.wallet_features")
    return pd.DataFrame(result.result_rows, columns=result.column_names)


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load ground-truth labels from a CSV or parquet file.

    Expected columns: ``wallet_address``, ``label`` (integer-encoded) or
    ``label_name`` (string).
    """
    path = Path(labels_path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    if "label" not in df.columns and "label_name" in df.columns:
        unique_names = sorted(df["label_name"].unique())
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        df["label"] = df["label_name"].map(name_to_id)
        logger.info("label_encoding_created", mapping=name_to_id)

    logger.info("labels_loaded", n_samples=len(df))
    return df


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def run_training(
    features_path: str | None = None,
    labels_path: str = "data/labels.csv",
    output_dir: str | None = None,
    n_trials: int = 100,
    epochs: int = 100,
    run_cv: bool = False,
    device: str = "cpu",
) -> dict:
    """Execute the full training pipeline and return evaluation results."""

    output_dir = output_dir or settings.model_artifacts_path

    # ---- 1. Load data ----
    features_df = load_features(features_path)
    labels_df = load_labels(labels_path)

    # Merge on wallet_address
    df = features_df.merge(labels_df[["wallet_address", "label"]], on="wallet_address", how="inner")
    logger.info("data_merged", n_samples=len(df))

    if df.empty:
        logger.error("no_samples_after_merge")
        sys.exit(1)

    # Separate features / labels using the canonical 12-column schema
    from src.features.feature_engineering import FEATURE_COLUMNS

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Derive label names
    if "label_name" in labels_df.columns:
        label_id_to_name = labels_df[["label", "label_name"]].drop_duplicates().sort_values("label")
        label_names = label_id_to_name["label_name"].tolist()
    else:
        label_names = DEFAULT_LABEL_NAMES[: int(y.max()) + 1]

    num_classes = len(label_names)
    logger.info("dataset_ready", n_samples=len(x), n_features=x.shape[1], num_classes=num_classes)

    # ---- 2. Stratified split: 70 / 15 / 15 ----
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        stratify=y,
        random_state=settings.random_seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=settings.random_seed,
    )
    logger.info(
        "data_split",
        train=len(x_train),
        val=len(x_val),
        test=len(x_test),
    )

    # ---- 3. Optional cross-validation ----
    cv_results: dict | None = None
    if run_cv:
        logger.info("starting_cross_validation")
        cv_results = cross_validate(
            WalletClassifier,
            x,
            y,
            label_names,
            n_folds=5,
            num_classes=num_classes,
            device=device,
        )
        logger.info("cross_validation_done", mean_f1=cv_results["mean_f1"])

    # ---- 4. Train models ----
    clf = WalletClassifier(num_classes=num_classes, device=device)
    clf.train_xgboost(x_train, y_train, x_val, y_val, n_trials=n_trials)
    clf.train_mlp(x_train, y_train, x_val, y_val, epochs=epochs)

    # ---- 5. Optimize ensemble ----
    clf.optimize_ensemble_weights(x_val, y_val)

    # ---- 6. Evaluate on test set ----
    results = evaluate_model(clf, x_test, y_test, label_names)
    logger.info(
        "test_evaluation",
        macro_f1=results["macro_f1"],
        meets_target=results["meets_target"],
    )

    if cv_results is not None:
        results["cv"] = cv_results

    # ---- 7. Save artefacts ----
    clf.save(output_dir)

    # Confusion matrix plot
    fig_cm = plot_confusion_matrix(y_test, results["y_pred"], label_names)
    fig_cm.savefig(Path(output_dir) / "confusion_matrix.png", dpi=150)

    # Calibration curve plot
    fig_cal = plot_calibration_curve(y_test, results["y_prob"], label_names)
    fig_cal.savefig(Path(output_dir) / "calibration_curve.png", dpi=150)

    # SHAP
    try:
        shap_values = compute_shap_values(clf, x_test[:500], feature_cols)
        import shap as shap_lib

        shap_lib.summary_plot(shap_values, show=False)
        import matplotlib.pyplot as plt

        plt.savefig(Path(output_dir) / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        logger.warning("shap_plot_failed", exc_info=True)

    # Full report
    generate_evaluation_report(results, output_dir)

    # ---- 8. W&B logging (optional) ----
    _log_to_wandb(results, output_dir, feature_cols)

    return results


# ---------------------------------------------------------------------------
# Weights & Biases integration
# ---------------------------------------------------------------------------


def _log_to_wandb(results: dict, output_dir: str, feature_names: list[str]) -> None:
    """Log metrics and artefacts to W&B if the package is available."""
    try:
        import wandb
    except ImportError:
        logger.info("wandb_not_installed_skipping")
        return

    try:
        wandb.init(
            project=settings.wandb_project,
            config={
                "num_features": len(feature_names),
                "feature_names": feature_names,
            },
            reinit=True,
        )

        wandb.log(
            {
                "macro_f1": results["macro_f1"],
                "meets_target": int(results["meets_target"]),
                "mean_confidence": results.get("mean_confidence"),
            }
        )

        # Per-class metrics
        for name, metrics in results.get("per_class", {}).items():
            wandb.log(
                {
                    f"{name}/precision": metrics["precision"],
                    f"{name}/recall": metrics["recall"],
                    f"{name}/f1": metrics["f1"],
                }
            )

        # Upload artefacts
        artefact_dir = Path(output_dir)
        for img in artefact_dir.glob("*.png"):
            wandb.log({img.stem: wandb.Image(str(img))})

        wandb.finish()
        logger.info("wandb_logging_complete")

    except Exception:
        logger.warning("wandb_logging_failed", exc_info=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the wallet classification ensemble.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features parquet file. Falls back to ClickHouse if omitted.",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="data/labels.csv",
        help="Path to ground-truth labels (CSV or parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory for model artefacts (default: {settings.model_artifacts_path}).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for XGBoost tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of MLP training epochs.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run stratified 5-fold cross-validation before final training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="PyTorch device.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_training(
        features_path=args.features_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        epochs=args.epochs,
        run_cv=args.cv,
        device=args.device,
    )
    print(f"\nMacro F1: {results['macro_f1']:.4f}  |  Target met: {results['meets_target']}")
