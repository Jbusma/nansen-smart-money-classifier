"""Claude API integration for generating wallet narratives and cluster profiles."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.llm.cache import NarrativeCache
from src.llm.prompts import (
    CLUSTER_PROFILE_PROMPT,
    WALLET_COMPARISON_PROMPT,
    WALLET_NARRATIVE_PROMPT,
)

logger = structlog.get_logger(__name__)


def _format_features(features: dict[str, Any]) -> str:
    """Pretty-print a feature dict for inclusion in a prompt."""
    return "\n".join(f"  {k}: {v}" for k, v in features.items())


class InsightGenerator:
    """Generate natural-language insights for wallets and clusters via Claude.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to ``settings.anthropic_api_key``.
    model:
        Claude model identifier.
    cache_enabled:
        When *True* (default), narratives are cached in ClickHouse for 24 h.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        cache_enabled: bool = True,
    ) -> None:
        resolved_key = api_key or settings.anthropic_api_key
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._cache_enabled = cache_enabled
        self._cache: NarrativeCache | None = None

        if self._cache_enabled:
            try:
                self._cache = NarrativeCache()
            except Exception:
                logger.warning(
                    "narrative_cache_unavailable, proceeding without cache"
                )
                self._cache_enabled = False

    # ------------------------------------------------------------------
    # Low-level API call with retry
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIConnectionError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_claude(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Send a single prompt to Claude and return the text response."""
        logger.debug("calling_claude", model=self._model, prompt_len=len(prompt))
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------

    def generate_wallet_narrative(
        self,
        wallet_address: str,
        features: dict[str, Any],
        label: str,
        confidence: float,
    ) -> str:
        """Return a 2-3 sentence intelligence briefing for *wallet_address*.

        Results are served from cache when available.
        """
        # Check cache first
        if self._cache_enabled and self._cache is not None:
            cached = self._cache.get(wallet_address)
            if cached is not None:
                return cached

        prompt = WALLET_NARRATIVE_PROMPT.format(
            wallet_address=wallet_address,
            label=label,
            confidence=confidence,
            features=_format_features(features),
        )
        narrative = self._call_claude(prompt)

        # Store in cache
        if self._cache_enabled and self._cache is not None:
            cluster_id = features.get("cluster_id", -1)
            self._cache.set(wallet_address, narrative, cluster_id=cluster_id)

        logger.info("wallet_narrative_generated", wallet_address=wallet_address)
        return narrative

    def generate_cluster_profile(
        self,
        cluster_id: int,
        cluster_stats: dict[str, Any],
        exemplar_wallets: list[str],
    ) -> str:
        """Return a JSON string with cluster name, description, and trading implications."""
        prompt = CLUSTER_PROFILE_PROMPT.format(
            cluster_id=cluster_id,
            cluster_stats=json.dumps(cluster_stats, indent=2, default=str),
            exemplar_wallets="\n".join(
                f"  - {w}" for w in exemplar_wallets
            ),
        )
        profile = self._call_claude(prompt, max_tokens=768)
        logger.info("cluster_profile_generated", cluster_id=cluster_id)
        return profile

    def generate_wallet_comparison(
        self,
        wallet_a_features: dict[str, Any],
        wallet_b_features: dict[str, Any],
    ) -> str:
        """Return a comparison analysis of two wallet profiles."""
        prompt = WALLET_COMPARISON_PROMPT.format(
            wallet_a_features=_format_features(wallet_a_features),
            wallet_b_features=_format_features(wallet_b_features),
        )
        comparison = self._call_claude(prompt)
        logger.info("wallet_comparison_generated")
        return comparison

    def batch_generate_narratives(
        self,
        wallets: list[dict[str, Any]],
    ) -> list[str]:
        """Generate narratives for multiple wallets sequentially.

        Each element of *wallets* must contain keys:
        ``wallet_address``, ``features``, ``label``, ``confidence``.

        Returns a list of narrative strings in the same order.
        """
        results: list[str] = []
        for i, wallet in enumerate(wallets):
            logger.debug(
                "batch_progress",
                current=i + 1,
                total=len(wallets),
                wallet_address=wallet["wallet_address"],
            )
            narrative = self.generate_wallet_narrative(
                wallet_address=wallet["wallet_address"],
                features=wallet["features"],
                label=wallet["label"],
                confidence=wallet["confidence"],
            )
            results.append(narrative)
        logger.info("batch_narratives_complete", count=len(results))
        return results
