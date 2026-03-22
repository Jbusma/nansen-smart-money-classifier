"""Tests for ML models: classifier, clustering, and training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FEATURE_COLUMNS

N_FEATURES = len(FEATURE_COLUMNS)  # 12


@pytest.fixture
def synthetic_data() -> dict[str, Any]:
    """Generate synthetic classification data with 12 features."""
    np.random.seed(42)
    n_samples = 500
    n_classes = 7

    x = np.random.randn(n_samples, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)

    # Make the data slightly separable
    for cls in range(n_classes):
        mask = y == cls
        x[mask, cls % N_FEATURES] += 2.0

    split = int(0.8 * n_samples)
    return {
        "x_train": x[:split],
        "y_train": y[:split],
        "x_val": x[split:],
        "y_val": y[split:],
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestClusteringPipeline:
    def test_fit_produces_labels(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        df = pd.DataFrame(
            synthetic_data["x_train"],
            columns=FEATURE_COLUMNS,
        )
        pipeline.fit(df)

        assert pipeline.labels_ is not None
        assert len(pipeline.labels_) == len(df)

    def test_embedding_is_2d(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        df = pd.DataFrame(
            synthetic_data["x_train"],
            columns=FEATURE_COLUMNS,
        )
        pipeline.fit(df)

        assert pipeline.embedding_ is not None
        assert pipeline.embedding_.shape[1] == 2

    def test_cluster_stats_returns_dict(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        df = pd.DataFrame(
            synthetic_data["x_train"],
            columns=FEATURE_COLUMNS,
        )
        pipeline.fit(df)

        stats = pipeline.get_cluster_stats()
        assert isinstance(stats, dict)

    def test_feature_names_recorded(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        df = pd.DataFrame(
            synthetic_data["x_train"],
            columns=FEATURE_COLUMNS,
        )
        pipeline.fit(df)

        assert pipeline.feature_names_ == FEATURE_COLUMNS

    def test_save_and_load(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        df = pd.DataFrame(
            synthetic_data["x_train"],
            columns=FEATURE_COLUMNS,
        )
        pipeline.fit(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "clustering.joblib"
            pipeline.save(save_path)
            loaded = ClusteringPipeline.load(save_path)

            assert loaded.labels_ is not None
            assert loaded.feature_names_ == FEATURE_COLUMNS
            assert pipeline.labels_ is not None
            np.testing.assert_array_equal(loaded.labels_, pipeline.labels_)

    def test_predict_after_fit(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        df = pd.DataFrame(
            synthetic_data["x_train"],
            columns=FEATURE_COLUMNS,
        )
        pipeline.fit(df)

        new_df = pd.DataFrame(
            synthetic_data["x_val"],
            columns=FEATURE_COLUMNS,
        )
        labels = pipeline.predict(new_df)
        assert len(labels) == len(new_df)

    def test_evaluate_returns_metrics(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        df = pd.DataFrame(
            synthetic_data["x_train"],
            columns=FEATURE_COLUMNS,
        )
        pipeline.fit(df)

        metrics = pipeline.evaluate(df)
        assert "silhouette" in metrics
        assert "n_clusters" in metrics
        assert "noise_ratio" in metrics


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestWalletClassifier:
    def test_xgboost_trains(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=synthetic_data["n_classes"])
        clf.train_xgboost(
            synthetic_data["x_train"],
            synthetic_data["y_train"],
            synthetic_data["x_val"],
            synthetic_data["y_val"],
            n_trials=3,
        )
        assert clf.xgb_model is not None

    def test_predict_returns_labels_and_confidence(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=synthetic_data["n_classes"])
        clf.train_xgboost(
            synthetic_data["x_train"],
            synthetic_data["y_train"],
            synthetic_data["x_val"],
            synthetic_data["y_val"],
            n_trials=3,
        )

        labels, confidences = clf.predict(synthetic_data["x_val"])
        assert len(labels) == len(synthetic_data["x_val"])
        assert len(confidences) == len(synthetic_data["x_val"])
        assert all(0 <= c <= 1 for c in confidences)

    def test_predict_proba_sums_to_one(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=synthetic_data["n_classes"])
        clf.train_xgboost(
            synthetic_data["x_train"],
            synthetic_data["y_train"],
            synthetic_data["x_val"],
            synthetic_data["y_val"],
            n_trials=3,
        )

        proba = clf.predict_proba(synthetic_data["x_val"])
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_save_and_load(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=synthetic_data["n_classes"])
        clf.train_xgboost(
            synthetic_data["x_train"],
            synthetic_data["y_train"],
            synthetic_data["x_val"],
            synthetic_data["y_val"],
            n_trials=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clf.save(tmpdir)
            loaded = WalletClassifier.load(tmpdir)

            assert loaded.xgb_model is not None
            assert loaded.num_classes == clf.num_classes

            # Predictions should match
            orig_labels, _ = clf.predict(synthetic_data["x_val"])
            loaded_labels, _ = loaded.predict(synthetic_data["x_val"])
            np.testing.assert_array_equal(orig_labels, loaded_labels)

    def test_input_dim_is_12(self, synthetic_data: dict[str, Any]) -> None:
        """Classifier should work with 12-dimensional input."""
        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=synthetic_data["n_classes"])
        clf.train_xgboost(
            synthetic_data["x_train"],
            synthetic_data["y_train"],
            synthetic_data["x_val"],
            synthetic_data["y_val"],
            n_trials=3,
        )

        assert synthetic_data["x_train"].shape[1] == N_FEATURES


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_evaluate_returns_metrics(self, synthetic_data: dict[str, Any]) -> None:
        from src.models.classifier import WalletClassifier
        from src.models.evaluation import evaluate_model

        clf = WalletClassifier(num_classes=synthetic_data["n_classes"])
        clf.train_xgboost(
            synthetic_data["x_train"],
            synthetic_data["y_train"],
            synthetic_data["x_val"],
            synthetic_data["y_val"],
            n_trials=3,
        )

        label_names = [f"class_{i}" for i in range(synthetic_data["n_classes"])]
        metrics = evaluate_model(clf, synthetic_data["x_val"], synthetic_data["y_val"], label_names)

        assert "macro_f1" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["macro_f1"] <= 1


# ---------------------------------------------------------------------------
# Training pipeline (load features)
# ---------------------------------------------------------------------------


class TestTrainingPipeline:
    def test_load_features_from_parquet(self) -> None:
        """Features loaded from parquet should include the 12 canonical columns."""
        from src.models.train import load_features

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            df = pd.DataFrame(
                np.random.randn(50, N_FEATURES),
                columns=FEATURE_COLUMNS,
            )
            df.insert(0, "wallet_address", [f"0x{i:040x}" for i in range(50)])
            df.to_parquet(f.name, index=False)

            loaded = load_features(f.name)
            assert "wallet_address" in loaded.columns
            for col in FEATURE_COLUMNS:
                assert col in loaded.columns
