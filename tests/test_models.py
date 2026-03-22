"""Tests for ML models."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    """Generate synthetic classification data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_classes = 7

    x = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    # Make the data slightly separable
    for cls in range(n_classes):
        mask = y == cls
        x[mask, cls % n_features] += 2.0

    split = int(0.8 * n_samples)
    return {
        "x_train": x[:split],
        "y_train": y[:split],
        "x_val": x[split:],
        "y_val": y[split:],
        "n_classes": n_classes,
    }


class TestClusteringPipeline:
    def test_fit_produces_labels(self, synthetic_data):
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        import pandas as pd

        df = pd.DataFrame(synthetic_data["x_train"])
        pipeline.fit(df)

        assert pipeline.labels_ is not None
        assert len(pipeline.labels_) == len(df)

    def test_embedding_is_2d(self, synthetic_data):
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        import pandas as pd

        df = pd.DataFrame(synthetic_data["x_train"])
        pipeline.fit(df)

        assert pipeline.embedding_ is not None
        assert pipeline.embedding_.shape[1] == 2

    def test_cluster_stats_returns_dict(self, synthetic_data):
        from src.models.clustering import ClusteringPipeline

        pipeline = ClusteringPipeline(min_cluster_size=20, min_samples=5)
        import pandas as pd

        df = pd.DataFrame(synthetic_data["x_train"])
        pipeline.fit(df)

        stats = pipeline.get_cluster_stats()
        assert isinstance(stats, dict)


class TestWalletClassifier:
    def test_xgboost_trains(self, synthetic_data):
        from src.models.classifier import WalletClassifier

        clf = WalletClassifier(num_classes=synthetic_data["n_classes"])
        clf.train_xgboost(
            synthetic_data["x_train"],
            synthetic_data["y_train"],
            synthetic_data["x_val"],
            synthetic_data["y_val"],
            n_trials=3,  # Few trials for speed
        )
        assert clf.xgb_model is not None

    def test_predict_returns_labels_and_confidence(self, synthetic_data):
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

    def test_predict_proba_sums_to_one(self, synthetic_data):
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


class TestEvaluation:
    def test_evaluate_returns_metrics(self, synthetic_data):
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
