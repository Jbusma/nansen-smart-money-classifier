"""FastAPI prediction endpoint backing the MCP server and dashboard."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Global state populated at startup
# ---------------------------------------------------------------------------
_classifier = None
_clustering = None
_feature_store = None
_insight_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and feature store on startup."""
    global _classifier, _clustering, _feature_store, _insight_generator

    from src.features.feature_store import FeatureStore
    from src.llm.insight_generator import InsightGenerator
    from src.models.classifier import WalletClassifier
    from src.models.clustering import ClusteringPipeline

    artifacts = Path(settings.model_artifacts_path)

    logger.info("loading_models", path=str(artifacts))
    try:
        _classifier = WalletClassifier.load(artifacts)
        logger.info("classifier_loaded")
    except Exception:
        logger.warning("classifier_load_failed", exc_info=True)

    try:
        _clustering = ClusteringPipeline.load()
        logger.info("clustering_loaded")
    except Exception:
        logger.warning("clustering_load_failed", exc_info=True)

    try:
        _feature_store = FeatureStore()
        logger.info("feature_store_initialized")
    except Exception:
        logger.warning("feature_store_init_failed", exc_info=True)

    try:
        _insight_generator = InsightGenerator()
        logger.info("insight_generator_initialized")
    except Exception:
        logger.warning("insight_generator_init_failed", exc_info=True)

    yield

    logger.info("shutting_down")


app = FastAPI(
    title="Nansen Smart Money Classifier",
    version="0.1.0",
    description="Ethereum wallet behavioral classification API",
    lifespan=lifespan,
)

LABEL_NAMES = [
    "smart_money",
    "mev_bot",
    "defi_farmer",
    "airdrop_hunter",
    "retail_trader",
    "hodler",
    "nft_trader",
]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    wallet_address: str = Field(..., pattern=r"^0x[a-fA-F0-9]{40}$")


class ClassifyResponse(BaseModel):
    wallet_address: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, object]
    latency_ms: float


class ExplainResponse(BaseModel):
    wallet_address: str
    label: str
    confidence: float
    narrative: str
    features: dict[str, object]


class SimilarWalletsRequest(BaseModel):
    wallet_address: str = Field(..., pattern=r"^0x[a-fA-F0-9]{40}$")
    top_k: int = Field(default=10, ge=1, le=100)


class SimilarWallet(BaseModel):
    wallet_address: str
    similarity: float
    label: str


class SimilarWalletsResponse(BaseModel):
    query_wallet: str
    similar_wallets: list[SimilarWallet]


class ClusterProfileResponse(BaseModel):
    cluster_id: int
    size: int
    profile: str
    top_features: dict[str, float]
    exemplar_wallets: list[str]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    feature_store_connected: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_features(wallet_address: str) -> dict[str, object]:
    if _feature_store is None:
        raise HTTPException(503, "Feature store not initialized")
    features = _feature_store.get_features(wallet_address)
    if not features:
        raise HTTPException(404, f"Wallet {wallet_address} not found in feature store")
    return dict(features)


def _features_to_array(features: dict[str, object]) -> np.ndarray:
    """Convert feature dict to array in the correct column order."""
    if _feature_store is None:
        raise HTTPException(503, "Feature store not initialized")
    names = _feature_store.get_feature_names()
    vals = [features.get(n, 0.0) for n in names]
    return np.array([[v if isinstance(v, (int, float)) else float(str(v)) for v in vals]])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    fs_ok = False
    if _feature_store is not None:
        with suppress(Exception):
            fs_ok = _feature_store.health_check()
    return HealthResponse(
        status="ok",
        models_loaded=_classifier is not None,
        feature_store_connected=fs_ok,
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_wallet(req: ClassifyRequest) -> ClassifyResponse:
    if _classifier is None:
        raise HTTPException(503, "Classifier not loaded")

    t0 = time.perf_counter()
    features = _get_features(req.wallet_address)
    x = _features_to_array(features)

    labels, confidences = _classifier.predict(x)
    proba = _classifier.predict_proba(x)[0]
    latency = (time.perf_counter() - t0) * 1000

    label_idx = int(labels[0])
    return ClassifyResponse(
        wallet_address=req.wallet_address,
        label=LABEL_NAMES[label_idx],
        confidence=float(confidences[0]),
        probabilities={name: float(p) for name, p in zip(LABEL_NAMES, proba, strict=False)},
        features=features,
        latency_ms=round(latency, 2),
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain_wallet(req: ClassifyRequest) -> ExplainResponse:
    if _classifier is None or _insight_generator is None:
        raise HTTPException(503, "Models not loaded")

    features = _get_features(req.wallet_address)
    x = _features_to_array(features)
    labels, confidences = _classifier.predict(x)
    label = LABEL_NAMES[int(labels[0])]
    confidence = float(confidences[0])

    narrative = _insight_generator.generate_wallet_narrative(
        wallet_address=req.wallet_address,
        features=features,
        label=label,
        confidence=confidence,
    )

    return ExplainResponse(
        wallet_address=req.wallet_address,
        label=label,
        confidence=confidence,
        narrative=narrative,
        features=features,
    )


@app.post("/similar", response_model=SimilarWalletsResponse)
async def find_similar_wallets(req: SimilarWalletsRequest) -> SimilarWalletsResponse:
    if _classifier is None or _feature_store is None:
        raise HTTPException(503, "Models not loaded")

    features = _get_features(req.wallet_address)
    x_query = _features_to_array(features)

    # Get all features and compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity

    all_features_df = _feature_store.get_all_features()
    if all_features_df is None or all_features_df.empty:
        raise HTTPException(503, "Feature store empty")

    feature_names = _feature_store.get_feature_names()
    x_all = all_features_df[feature_names].values
    addresses = all_features_df["wallet_address"].values

    similarities = cosine_similarity(x_query, x_all)[0]

    # Exclude the query wallet itself
    mask = addresses != req.wallet_address
    similarities = similarities[mask]
    addresses = addresses[mask]

    top_indices = np.argsort(similarities)[-req.top_k :][::-1]

    # Classify the similar wallets
    similar = []
    for idx in top_indices:
        addr = addresses[idx]
        x_i = x_all[mask][idx : idx + 1]
        label_idx, _ = _classifier.predict(x_i)
        similar.append(
            SimilarWallet(
                wallet_address=str(addr),
                similarity=float(similarities[idx]),
                label=LABEL_NAMES[int(label_idx[0])],
            )
        )

    return SimilarWalletsResponse(
        query_wallet=req.wallet_address,
        similar_wallets=similar,
    )


@app.get("/cluster/{cluster_id}", response_model=ClusterProfileResponse)
async def get_cluster_profile(cluster_id: int) -> ClusterProfileResponse:
    if _clustering is None or _insight_generator is None:
        raise HTTPException(503, "Models not loaded")

    stats = _clustering.get_cluster_stats()
    if cluster_id not in stats:
        raise HTTPException(404, f"Cluster {cluster_id} not found")

    cluster = stats[cluster_id]
    profile = _insight_generator.generate_cluster_profile(
        cluster_id=cluster_id,
        cluster_stats=cluster,
        exemplar_wallets=cluster.get("exemplar_addresses", []),
    )

    return ClusterProfileResponse(
        cluster_id=cluster_id,
        size=cluster.get("size", 0),
        profile=profile,
        top_features=cluster.get("top_features", {}),
        exemplar_wallets=cluster.get("exemplar_addresses", [])[:5],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.api:app", host=settings.api_host, port=settings.api_port, reload=True)
