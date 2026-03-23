"""Tests for the Streamlit dashboard and supporting modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FEATURE_COLUMNS
from src.models.cluster_analysis import build_cluster_profiles, save_profiles

N_FEATURES = len(FEATURE_COLUMNS)  # 12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_clustered_data() -> dict[str, Any]:
    """Synthetic features + fake HDBSCAN labels + 2D embedding."""
    np.random.seed(42)
    n_samples = 200

    features = pd.DataFrame(
        np.random.randn(n_samples, N_FEATURES).astype(np.float32),
        columns=FEATURE_COLUMNS,
    )
    # 3 clusters + some noise
    labels = np.array(
        [0] * 80 + [1] * 60 + [2] * 40 + [-1] * 20,
    )
    # Make clusters slightly different
    for c in range(3):
        mask = labels == c
        features.loc[mask, FEATURE_COLUMNS[c % N_FEATURES]] += 3.0

    embedding = np.random.randn(n_samples, 2).astype(np.float32)
    wallet_addresses = pd.Series([f"0x{i:040x}" for i in range(n_samples)])

    return {
        "features": features,
        "labels": labels,
        "embedding": embedding,
        "wallet_addresses": wallet_addresses,
    }


# ---------------------------------------------------------------------------
# Cluster Analysis
# ---------------------------------------------------------------------------


class TestBuildClusterProfiles:
    def test_returns_dataframe(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        assert isinstance(profiles, pd.DataFrame)

    def test_has_required_columns(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        required = {"cluster", "feature", "mean", "std", "global_mean", "ratio_to_global", "z_score", "size", "pct"}
        assert required.issubset(set(profiles.columns))

    def test_covers_all_clusters(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        expected_clusters = {-1, 0, 1, 2}
        assert set(profiles["cluster"].unique()) == expected_clusters

    def test_covers_all_features(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        assert set(profiles["feature"].unique()) == set(FEATURE_COLUMNS)

    def test_row_count(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        n_clusters = 4  # -1, 0, 1, 2
        assert len(profiles) == n_clusters * N_FEATURES

    def test_ratios_positive(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        # Ratios can be negative if means are negative, but for boosted features
        # check that at least the boosted cluster has high ratio
        cluster_0_boosted = profiles[(profiles["cluster"] == 0) & (profiles["feature"] == FEATURE_COLUMNS[0])]
        assert cluster_0_boosted["ratio_to_global"].values[0] > 1.0

    def test_sizes_sum_to_total(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        # Each cluster has one size value per feature; pick one feature
        sizes = profiles[profiles["feature"] == FEATURE_COLUMNS[0]]["size"]
        assert sizes.sum() == len(synthetic_clustered_data["labels"])

    def test_pct_sums_to_100(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        pcts = profiles[profiles["feature"] == FEATURE_COLUMNS[0]]["pct"]
        assert pcts.sum() == pytest.approx(100.0, abs=0.1)


class TestSaveProfiles:
    def test_saves_valid_json(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            save_profiles(profiles, path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_json_has_cluster_keys(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            save_profiles(profiles, path)

            with open(path) as f:
                data = json.load(f)
            assert "noise" in data
            assert "cluster_0" in data
            assert "cluster_1" in data
            assert "cluster_2" in data

    def test_json_has_features_and_suggested_label(self, synthetic_clustered_data: dict[str, Any]) -> None:
        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            save_profiles(profiles, path)

            with open(path) as f:
                data = json.load(f)
            for key, cluster_data in data.items():
                assert "size" in cluster_data, f"Missing 'size' in {key}"
                assert "features" in cluster_data, f"Missing 'features' in {key}"
                assert "suggested_label" in cluster_data, f"Missing 'suggested_label' in {key}"
                assert len(cluster_data["features"]) == N_FEATURES


# ---------------------------------------------------------------------------
# Cluster Viz
# ---------------------------------------------------------------------------


class TestClusterViz:
    def test_umap_scatter_returns_figure(self, synthetic_clustered_data: dict[str, Any]) -> None:
        from src.models.cluster_viz import plot_umap_scatter

        fig = plot_umap_scatter(
            synthetic_clustered_data["embedding"],
            synthetic_clustered_data["labels"],
        )
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_umap_interactive_returns_plotly_fig(self, synthetic_clustered_data: dict[str, Any]) -> None:
        from src.models.cluster_viz import plot_umap_interactive

        fig = plot_umap_interactive(
            synthetic_clustered_data["embedding"],
            synthetic_clustered_data["labels"],
            synthetic_clustered_data["features"],
            synthetic_clustered_data["wallet_addresses"],
        )
        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

    def test_feature_heatmap_returns_figure(self, synthetic_clustered_data: dict[str, Any]) -> None:
        from src.models.cluster_viz import plot_feature_heatmap

        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        fig = plot_feature_heatmap(profiles)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_radar_charts_returns_figure(self, synthetic_clustered_data: dict[str, Any]) -> None:
        from src.models.cluster_viz import plot_radar_charts

        profiles = build_cluster_profiles(
            synthetic_clustered_data["features"],
            synthetic_clustered_data["labels"],
        )
        fig = plot_radar_charts(profiles)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dashboard data helpers
# ---------------------------------------------------------------------------


class TestDashboardDataLoading:
    def test_load_cluster_labels_empty(self) -> None:
        """Returns empty dict when no labels file exists."""
        with patch("src.serving.dashboard.LABELS_PATH", Path("/nonexistent/path.json")):
            from src.serving.dashboard import load_cluster_labels

            # Clear cache so it re-executes
            load_cluster_labels.clear()
            result = load_cluster_labels()
            assert result == {}

    def test_save_and_load_cluster_labels(self) -> None:
        from src.serving.dashboard import load_cluster_labels, save_cluster_labels

        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "cluster_labels.json"
            with patch("src.serving.dashboard.LABELS_PATH", labels_path):
                test_labels = {"0": "Smart Money", "1": "MEV Bot", "2": "DeFi Farmer"}
                save_cluster_labels(test_labels)

                load_cluster_labels.clear()
                loaded = load_cluster_labels()
                assert loaded == test_labels

    def test_load_ground_truth_missing(self) -> None:
        """Returns None when no ground truth file exists."""
        with patch("src.serving.dashboard.GROUND_TRUTH_PATH", Path("/nonexistent/gt.parquet")):
            from src.serving.dashboard import load_ground_truth

            load_ground_truth.clear()
            result = load_ground_truth()
            assert result is None

    def test_load_ground_truth_exists(self) -> None:
        from src.serving.dashboard import load_ground_truth

        gt_df = pd.DataFrame(
            {
                "address": [f"0x{i:040x}" for i in range(10)],
                "label": ["mev_bot"] * 5 + ["whale"] * 5,
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            gt_df.to_parquet(f.name, index=False)
            with patch("src.serving.dashboard.GROUND_TRUTH_PATH", Path(f.name)):
                load_ground_truth.clear()
                result = load_ground_truth()
                assert result is not None
                assert len(result) == 10


# ---------------------------------------------------------------------------
# Silhouette computation (used in dashboard)
# ---------------------------------------------------------------------------


class TestSilhouetteComputation:
    def test_silhouette_samples_on_clusters(self, synthetic_clustered_data: dict[str, Any]) -> None:
        """Verify silhouette_samples runs on our data shape."""
        from sklearn.metrics import silhouette_samples

        embedding = synthetic_clustered_data["embedding"]
        labels = synthetic_clustered_data["labels"]
        valid_mask = labels != -1

        sil = silhouette_samples(embedding[valid_mask], labels[valid_mask])
        assert len(sil) == valid_mask.sum()
        assert all(-1 <= s <= 1 for s in sil)

    def test_silhouette_per_cluster_averages(self, synthetic_clustered_data: dict[str, Any]) -> None:
        from sklearn.metrics import silhouette_samples

        embedding = synthetic_clustered_data["embedding"]
        labels = synthetic_clustered_data["labels"]
        valid_mask = labels != -1

        sil = silhouette_samples(embedding[valid_mask], labels[valid_mask])
        valid_labels = labels[valid_mask]

        for c in [0, 1, 2]:
            cluster_sil = sil[valid_labels == c]
            assert len(cluster_sil) > 0
            avg = float(np.mean(cluster_sil))
            assert -1 <= avg <= 1
