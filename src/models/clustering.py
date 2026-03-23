"""Unsupervised clustering pipeline for wallet behaviour segmentation.

Uses UMAP for dimensionality reduction and HDBSCAN for density-based clustering
to identify wallet archetypes such as smart_money, mev_bot, defi_farmer, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import hdbscan
import joblib
import numpy as np
import pandas as pd
import structlog
import umap
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from src.config import settings

logger = structlog.get_logger(__name__)

EXPECTED_CLUSTERS: list[str] = [
    "smart_money",
    "mev_bot",
    "defi_farmer",
    "airdrop_hunter",
    "retail_trader",
    "hodler",
    "nft_trader",
]


class ClusteringPipeline:
    """End-to-end clustering pipeline: UMAP reduction -> HDBSCAN clustering.

    Attributes
    ----------
    reducer : umap.UMAP
        Fitted UMAP dimensionality-reduction model.
    clusterer : hdbscan.HDBSCAN
        Fitted HDBSCAN clustering model.
    embedding_ : np.ndarray | None
        2-D UMAP embedding produced during ``fit``, useful for visualisation.
    labels_ : np.ndarray | None
        Cluster labels assigned during ``fit``.
    feature_names_ : list[str] | None
        Column names from the training DataFrame.
    """

    _ARTIFACT_NAME: ClassVar[str] = "clustering_pipeline.joblib"

    def __init__(
        self,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        metric: str = "cosine",
        min_cluster_size: int = 100,
        min_samples: int = 10,
    ) -> None:
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=2,
            n_jobs=-1,
            random_state=settings.random_seed,
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=-1,
            prediction_data=True,
        )

        # Populated after fit
        self.embedding_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

        logger.info(
            "clustering_pipeline.init",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            random_seed=settings.random_seed,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> ClusteringPipeline:
        """Fit UMAP and HDBSCAN on *features_df*.

        Parameters
        ----------
        features_df:
            Numeric feature matrix (rows = wallets, columns = features).

        Returns
        -------
        self
        """
        self.feature_names_ = list(features_df.columns)
        logger.info(
            "clustering_pipeline.fit.start",
            n_samples=len(features_df),
            n_features=len(self.feature_names_),
        )

        # Step 1 – dimensionality reduction
        self.embedding_ = self.reducer.fit_transform(features_df.values)
        logger.info("clustering_pipeline.umap.done", shape=self.embedding_.shape)

        # Step 2 – density-based clustering
        self.labels_ = self.clusterer.fit_predict(self.embedding_)
        n_clusters = len(set(self.labels_) - {-1})
        n_noise = int((self.labels_ == -1).sum())
        logger.info(
            "clustering_pipeline.hdbscan.done",
            n_clusters=n_clusters,
            n_noise=n_noise,
        )

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Project new wallets into the existing embedding and predict clusters.

        Uses UMAP ``transform`` followed by HDBSCAN ``approximate_predict``.

        Parameters
        ----------
        features_df:
            Numeric feature matrix with the same columns used during ``fit``.

        Returns
        -------
        np.ndarray
            Cluster label for each row.
        """
        if self.embedding_ is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call fit() first.")

        embedding = self.reducer.transform(features_df.values)
        labels, _ = hdbscan.approximate_predict(self.clusterer, embedding)
        logger.info("clustering_pipeline.predict", n_samples=len(labels))
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def get_cluster_stats(self) -> dict:
        """Return per-cluster statistics: size, centroid, and top features.

        Returns
        -------
        dict
            Mapping ``cluster_id -> {size, centroid, top_features}``.
        """
        if self.labels_ is None or self.embedding_ is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call fit() first.")

        stats: dict = {}
        unique_labels = sorted(set(self.labels_) - {-1})

        for label in unique_labels:
            mask = self.labels_ == label
            cluster_embedding = self.embedding_[mask]
            centroid = cluster_embedding.mean(axis=0).tolist()
            stats[int(label)] = {
                "size": int(mask.sum()),
                "centroid": centroid,
                "top_features": self._top_features_for_mask(mask),
            }

        # Include noise cluster stats
        noise_mask = self.labels_ == -1
        if noise_mask.any():
            stats[-1] = {
                "size": int(noise_mask.sum()),
                "centroid": self.embedding_[noise_mask].mean(axis=0).tolist(),
                "top_features": [],
            }

        return stats

    def get_cluster_exemplars(self, features_df: pd.DataFrame, n: int = 20) -> dict[int, pd.DataFrame]:
        """Return the *n* most representative wallets per cluster.

        Exemplars are those closest to the cluster centroid in UMAP space.

        Parameters
        ----------
        features_df:
            The same DataFrame used during ``fit`` (index = wallet address).
        n:
            Number of exemplars per cluster.

        Returns
        -------
        dict[int, pd.DataFrame]
        """
        if self.labels_ is None or self.embedding_ is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call fit() first.")

        exemplars: dict[int, pd.DataFrame] = {}
        unique_labels = sorted(set(self.labels_) - {-1})

        for label in unique_labels:
            mask = self.labels_ == label
            cluster_embedding = self.embedding_[mask]
            centroid = cluster_embedding.mean(axis=0)

            # Euclidean distance to centroid in 2-D UMAP space
            dists = np.linalg.norm(cluster_embedding - centroid, axis=1)
            top_idx = np.argsort(dists)[:n]

            # Map back to original DataFrame rows
            original_indices = np.where(mask)[0][top_idx]
            exemplars[int(label)] = features_df.iloc[original_indices].copy()

        return exemplars

    def evaluate(self, features_df: pd.DataFrame) -> dict:
        """Compute clustering quality metrics.

        Parameters
        ----------
        features_df:
            The same DataFrame used during ``fit``.

        Returns
        -------
        dict
            ``{silhouette, calinski_harabasz, n_clusters, noise_ratio}``.
        """
        if self.labels_ is None or self.embedding_ is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call fit() first.")

        # Exclude noise points for metric computation
        valid_mask = self.labels_ != -1
        valid_labels = self.labels_[valid_mask]
        valid_embedding = self.embedding_[valid_mask]

        n_unique = len(set(valid_labels))
        if n_unique < 2:
            logger.warning(
                "clustering_pipeline.evaluate.insufficient_clusters",
                n_clusters=n_unique,
            )
            return {
                "silhouette": float("nan"),
                "calinski_harabasz": float("nan"),
                "n_clusters": n_unique,
                "noise_ratio": float((self.labels_ == -1).mean()),
            }

        sil = float(silhouette_score(valid_embedding, valid_labels))
        ch = float(calinski_harabasz_score(valid_embedding, valid_labels))

        metrics = {
            "silhouette": sil,
            "calinski_harabasz": ch,
            "n_clusters": n_unique,
            "noise_ratio": float((self.labels_ == -1).mean()),
        }
        logger.info("clustering_pipeline.evaluate", **metrics)
        return metrics

    def stability_analysis(
        self,
        features_df: pd.DataFrame,
        n_runs: int = 5,
        subsample_ratio: float = 0.8,
    ) -> dict:
        """Assess cluster stability via repeated sub-sampling.

        For each run an 80 % subsample is drawn, UMAP + HDBSCAN are re-run,
        and evaluation metrics are collected.  The returned dict contains
        per-run metrics as well as mean / std summaries.

        Parameters
        ----------
        features_df:
            The same DataFrame used during ``fit``.
        n_runs:
            Number of bootstrap iterations.
        subsample_ratio:
            Fraction of rows to sample each iteration.

        Returns
        -------
        dict
            ``{runs: [...], mean_silhouette, std_silhouette,
            mean_calinski_harabasz, std_calinski_harabasz,
            mean_n_clusters, std_n_clusters}``.
        """
        logger.info(
            "clustering_pipeline.stability.start",
            n_runs=n_runs,
            subsample_ratio=subsample_ratio,
        )

        rng = np.random.RandomState(settings.random_seed)
        run_metrics: list[dict] = []

        for i in range(n_runs):
            idx = rng.choice(
                len(features_df),
                size=int(len(features_df) * subsample_ratio),
                replace=False,
            )
            sub_df = features_df.iloc[idx]

            sub_pipeline = ClusteringPipeline(
                n_neighbors=self.reducer.n_neighbors,
                min_dist=self.reducer.min_dist,
                metric=self.reducer.metric,
                min_cluster_size=self.clusterer.min_cluster_size,
                min_samples=self.clusterer.min_samples,
            )
            sub_pipeline.fit(sub_df)
            metrics = sub_pipeline.evaluate(sub_df)
            metrics["run"] = i
            run_metrics.append(metrics)
            logger.info("clustering_pipeline.stability.run", **metrics)

        sils = [m["silhouette"] for m in run_metrics]
        chs = [m["calinski_harabasz"] for m in run_metrics]
        ncs = [m["n_clusters"] for m in run_metrics]

        result = {
            "runs": run_metrics,
            "mean_silhouette": float(np.nanmean(sils)),
            "std_silhouette": float(np.nanstd(sils)),
            "mean_calinski_harabasz": float(np.nanmean(chs)),
            "std_calinski_harabasz": float(np.nanstd(chs)),
            "mean_n_clusters": float(np.mean(ncs)),
            "std_n_clusters": float(np.std(ncs)),
        }
        logger.info(
            "clustering_pipeline.stability.done",
            mean_silhouette=result["mean_silhouette"],
            std_silhouette=result["std_silhouette"],
            mean_n_clusters=result["mean_n_clusters"],
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Serialise the fitted pipeline to disk.

        Parameters
        ----------
        path:
            Destination file.  Defaults to
            ``{settings.model_artifacts_path}/clustering_pipeline.joblib``.

        Returns
        -------
        Path
            The path the artifact was written to.
        """
        if path is None:
            path = Path(settings.model_artifacts_path) / self._ARTIFACT_NAME
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "reducer": self.reducer,
            "clusterer": self.clusterer,
            "embedding_": self.embedding_,
            "labels_": self.labels_,
            "feature_names_": self.feature_names_,
        }
        joblib.dump(state, path)
        logger.info("clustering_pipeline.save", path=str(path))
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> ClusteringPipeline:
        """Load a previously saved pipeline from disk.

        Parameters
        ----------
        path:
            Source file.  Defaults to
            ``{settings.model_artifacts_path}/clustering_pipeline.joblib``.

        Returns
        -------
        ClusteringPipeline
        """
        if path is None:
            path = Path(settings.model_artifacts_path) / cls._ARTIFACT_NAME
        path = Path(path)

        state = joblib.load(path)
        pipeline = cls.__new__(cls)
        pipeline.reducer = state["reducer"]
        pipeline.clusterer = state["clusterer"]
        pipeline.embedding_ = state["embedding_"]
        pipeline.labels_ = state["labels_"]
        pipeline.feature_names_ = state["feature_names_"]
        logger.info("clustering_pipeline.load", path=str(path))
        return pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _top_features_for_mask(self, mask: np.ndarray, n: int = 5) -> list[dict[str, float]]:
        """Identify the features whose mean is highest for the given cluster.

        This is a simple heuristic: features are ranked by the ratio of
        within-cluster mean to global mean.  Only works when *feature_names_*
        were recorded during ``fit``.
        """
        if self.feature_names_ is None or self.embedding_ is None:
            return []

        # We do not have the raw features stored; return empty.
        # In practice the caller would pass them in.  This placeholder
        # exists so get_cluster_stats can return a well-typed structure.
        return []


# ======================================================================
# CLI entry-point
# ======================================================================

if __name__ == "__main__":
    import argparse
    import json

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    parser = argparse.ArgumentParser(description="Run the wallet clustering pipeline.")
    parser.add_argument(
        "--features",
        type=str,
        default="data/features.parquet",
        help="Path to the input features parquet file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the fitted pipeline artifact.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=5,
        help="Number of stability analysis sub-sample runs.",
    )
    args = parser.parse_args()

    log = structlog.get_logger("clustering.__main__")

    # 1. Load features
    log.info("loading_features", path=args.features)
    features_df = pd.read_parquet(args.features)
    log.info("features_loaded", shape=features_df.shape)

    # 2. Separate wallet addresses from features
    wallet_addresses = features_df["wallet_address"] if "wallet_address" in features_df.columns else None
    numeric_features = features_df.select_dtypes(include=[np.number])
    log.info("numeric_features", columns=list(numeric_features.columns))

    # 3. Fit pipeline
    pipeline = ClusteringPipeline()
    pipeline.fit(numeric_features)

    # 4. Evaluate
    metrics = pipeline.evaluate(numeric_features)
    log.info("evaluation_metrics", **metrics)

    # 5. Cluster statistics
    stats = pipeline.get_cluster_stats()
    log.info("cluster_stats", n_clusters=len(stats))

    # 6. Stability analysis
    stability = pipeline.stability_analysis(numeric_features, n_runs=args.stability_runs)
    log.info(
        "stability_summary",
        mean_silhouette=stability["mean_silhouette"],
        std_silhouette=stability["std_silhouette"],
        mean_n_clusters=stability["mean_n_clusters"],
    )

    # 7. Save
    artifact_path = pipeline.save(args.output)
    log.info("pipeline_saved", path=str(artifact_path))

    # 8. Print summary
    summary = {
        "metrics": metrics,
        "stability": {k: v for k, v in stability.items() if k != "runs"},
        "cluster_sizes": {k: v["size"] for k, v in stats.items()},
        "expected_clusters": EXPECTED_CLUSTERS,
        "artifact": str(artifact_path),
    }
    print(json.dumps(summary, indent=2, default=str))
