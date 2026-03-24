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


class ContractInteraction(BaseModel):
    address: str
    protocol_label: str | None
    category: str
    interaction_count: int
    total_eth: float


class TokenSummary(BaseModel):
    token_address: str
    transfer_count: int
    erc20_count: int
    erc721_count: int


class TransactionSummary(BaseModel):
    total_transactions: int
    total_eth_volume: float
    avg_tx_value_eth: float
    first_seen: str | None
    last_seen: str | None


class TokenActivity(BaseModel):
    unique_tokens: int
    top_tokens: list[TokenSummary]


class TimingPatterns(BaseModel):
    most_active_hours: list[int]
    weekday_ratio: float
    hourly_distribution: list[float]


class WalletContextResponse(BaseModel):
    wallet_address: str
    transaction_summary: TransactionSummary | None
    top_contracts: list[ContractInteraction] | None
    token_activity: TokenActivity | None
    timing_patterns: TimingPatterns | None


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


@app.get("/wallet/{address}/context", response_model=WalletContextResponse)
async def wallet_context(address: str) -> WalletContextResponse:
    """Return rich on-chain context for a wallet address.

    Queries ClickHouse for transaction summaries, top contract interactions
    (with protocol labels), token activity, and timing patterns.
    """
    from src.data.wallet_context import get_wallet_context

    try:
        ctx = get_wallet_context(address)
    except Exception as exc:
        logger.error("wallet_context_failed", wallet=address, error=str(exc))
        raise HTTPException(503, f"Context query failed: {exc}") from exc

    return WalletContextResponse(
        wallet_address=ctx["wallet_address"],
        transaction_summary=ctx.get("transaction_summary"),
        top_contracts=ctx.get("top_contracts"),
        token_activity=ctx.get("token_activity"),
        timing_patterns=ctx.get("timing_patterns"),
    )


# ---------------------------------------------------------------------------
# Protocol enrichment
# ---------------------------------------------------------------------------


class EnrichRequest(BaseModel):
    etherscan: bool = Field(default=False, description="Also resolve via Etherscan API")
    top_n: int = Field(default=500, ge=1, le=5000, description="Top-N unknown contracts for Etherscan")


class EnrichResponse(BaseModel):
    hardcoded: int = 0
    token_list: int = 0
    defillama: int = 0
    etherscan: int = 0
    total_registry_size: int = 0


@app.post("/enrich", response_model=EnrichResponse)
async def enrich_registry(req: EnrichRequest | None = None) -> EnrichResponse:
    if req is None:
        req = EnrichRequest()
    """Run protocol registry enrichment from free sources."""
    from src.data.protocol_enrichment import enrich_all

    try:
        results = enrich_all(etherscan=req.etherscan, top_n=req.top_n)
    except Exception as exc:
        logger.error("enrichment_failed", error=str(exc))
        raise HTTPException(503, f"Enrichment failed: {exc}") from exc

    return EnrichResponse(**{k: v for k, v in results.items() if k in EnrichResponse.model_fields})


# ---------------------------------------------------------------------------
# Wallet labeling
# ---------------------------------------------------------------------------


class LabelWalletRequest(BaseModel):
    wallet_address: str = Field(..., pattern=r"^0x[a-fA-F0-9]{40}$")
    label: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = Field(default="", max_length=2000)
    source: str = Field(default="agent_verified")


class LabelClusterRequest(BaseModel):
    cluster_id: int = Field(..., ge=-1)
    label: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = Field(default="", max_length=2000)
    source: str = Field(default="agent_cluster_label")


class LabelResponse(BaseModel):
    labeled: int
    label: str


@app.post("/label/wallet", response_model=LabelResponse)
async def label_wallet(req: LabelWalletRequest) -> LabelResponse:
    """Assign a behavioral label to a single wallet."""
    from src.data.clickhouse_sync import get_client

    client = get_client()
    db = settings.clickhouse_database

    client.insert(
        f"{db}.ground_truth",
        [
            [
                req.wallet_address.lower(),
                req.label,
                req.source,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # metrics filled later
                req.wallet_address.lower(),
            ]
        ],
        column_names=[
            "address",
            "label",
            "source",
            "total_tx",
            "dex_tx",
            "dex_ratio",
            "total_eth",
            "tx_per_day",
            "wallet_address",
        ],
    )
    return LabelResponse(labeled=1, label=req.label)


@app.post("/label/cluster", response_model=LabelResponse)
async def label_cluster(req: LabelClusterRequest) -> LabelResponse:
    """Bulk-label all wallets in a cluster using clustering pipeline assignments."""
    from pathlib import Path

    import joblib
    import pandas as pd

    from src.data.clickhouse_sync import get_client

    pipeline_path = Path(settings.model_artifacts_path) / "clustering_pipeline.joblib"
    features_path = Path("data/features.parquet")

    if not pipeline_path.exists() or not features_path.exists():
        raise HTTPException(404, "clustering_pipeline.joblib or features.parquet not found")

    pipeline = joblib.load(pipeline_path)
    labels = pipeline["labels_"]
    df = pd.read_parquet(features_path, columns=["wallet_address"])

    # -1 = noise, 0/1/2 = clusters
    mask = labels == req.cluster_id
    addresses = df.loc[mask, "wallet_address"].tolist()

    if not addresses:
        raise HTTPException(404, f"No wallets found for cluster {req.cluster_id}")

    client = get_client()
    db = settings.clickhouse_database

    rows = [
        [
            addr.lower(),
            req.label,
            req.source,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            addr.lower(),
        ]
        for addr in addresses
    ]

    client.insert(
        f"{db}.ground_truth",
        rows,
        column_names=[
            "address",
            "label",
            "source",
            "total_tx",
            "dex_tx",
            "dex_ratio",
            "total_eth",
            "tx_per_day",
            "wallet_address",
        ],
    )
    return LabelResponse(labeled=len(rows), label=req.label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.api:app", host=settings.api_host, port=settings.api_port, reload=True)
