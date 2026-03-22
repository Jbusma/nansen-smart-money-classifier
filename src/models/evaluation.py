"""Evaluation and interpretability utilities for the wallet classifier.

Provides stratified k-fold cross-validation, standard classification metrics,
calibration diagnostics, SHAP-based feature importance, and a report generator.
Target: >0.85 macro F1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
import structlog
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

from src.config import settings

if TYPE_CHECKING:
    from src.models.classifier import WalletClassifier

matplotlib.use("Agg")  # non-interactive backend for server-side rendering

logger = structlog.get_logger(__name__)

F1_TARGET = 0.85

# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    classifier: WalletClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Run full evaluation on a held-out test set.

    Returns a dict with:
    - ``macro_f1``: float
    - ``per_class``: dict mapping label name to precision / recall / f1
    - ``confusion_matrix``: list[list[int]]
    - ``classification_report``: str (sklearn formatted text)
    - ``meets_target``: bool (True if macro F1 >= 0.85)
    """
    y_pred, confidence = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )

    results = {
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": report_text,
        "meets_target": macro_f1 >= F1_TARGET,
        "mean_confidence": float(confidence.mean()),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    logger.info(
        "evaluation_complete",
        macro_f1=macro_f1,
        meets_target=results["meets_target"],
    )
    return results


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    classifier_factory: type[WalletClassifier],
    x: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    n_folds: int = 5,
    *,
    num_classes: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    """Stratified k-fold cross-validation.

    A fresh ``WalletClassifier`` is built in each fold.  Returns per-fold
    macro F1 scores plus the mean and std.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=settings.random_seed)
    fold_f1s: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y), 1):
        logger.info("cv_fold_start", fold=fold_idx, n_folds=n_folds)

        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = classifier_factory(num_classes=num_classes, device=device)
        clf.train_xgboost(x_train, y_train, x_val, y_val, n_trials=30)
        clf.train_mlp(x_train, y_train, x_val, y_val, epochs=50)
        clf.optimize_ensemble_weights(x_val, y_val)

        y_pred, _ = clf.predict(x_val)
        fold_f1 = f1_score(y_val, y_pred, average="macro")
        fold_f1s.append(float(fold_f1))
        logger.info("cv_fold_done", fold=fold_idx, macro_f1=fold_f1)

    result = {
        "fold_f1s": fold_f1s,
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
        "meets_target": float(np.mean(fold_f1s)) >= F1_TARGET,
    }
    logger.info("cv_complete", mean_f1=result["mean_f1"], std_f1=result["std_f1"])
    return result


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def compute_shap_values(
    classifier: WalletClassifier,
    x: np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    """Compute SHAP values using the XGBoost component of the ensemble.

    Returns a ``shap.Explanation`` object suitable for ``shap.summary_plot``
    and per-wallet waterfall plots.
    """
    if classifier.xgb_model is None:
        raise RuntimeError("XGBoost model is not trained; cannot compute SHAP values.")

    explainer = shap.TreeExplainer(classifier.xgb_model)
    shap_values = explainer(x)

    # Attach feature names for downstream plotting
    shap_values.feature_names = feature_names

    logger.info("shap_values_computed", n_samples=x.shape[0], n_features=x.shape[1])
    return shap_values


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> plt.Figure:
    """Return a matplotlib figure showing the confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(label_names)

    fig, ax = plt.subplots(figsize=(max(8, n_classes), max(6, n_classes * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> plt.Figure:
    """Return a matplotlib figure with per-class calibration (reliability) curves."""
    n_classes = len(label_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls_idx in range(n_classes):
        # Binarize: one-vs-rest
        binary_true = (y_true == cls_idx).astype(int)
        prob_cls = y_prob[:, cls_idx]

        if binary_true.sum() < 5:
            continue  # skip classes with too few positive samples

        fraction_pos, mean_predicted = calibration_curve(
            binary_true,
            prob_cls,
            n_bins=10,
            strategy="uniform",
        )
        ax.plot(mean_predicted, fraction_pos, marker="o", label=label_names[cls_idx])

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (per class)")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_evaluation_report(results: dict[str, Any], output_dir: str) -> None:
    """Persist evaluation artefacts to *output_dir*.

    Saves:
    - ``metrics.json`` -- scalar metrics and per-class breakdown
    - ``confusion_matrix.png``
    - ``calibration_curve.png``
    - ``classification_report.txt``
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Serializable metrics ----
    serializable = {k: v for k, v in results.items() if k not in {"y_prob", "y_pred"}}
    with open(out / "metrics.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # ---- Text report ----
    if "classification_report" in results:
        (out / "classification_report.txt").write_text(results["classification_report"])

    logger.info("evaluation_report_saved", output_dir=str(out))
