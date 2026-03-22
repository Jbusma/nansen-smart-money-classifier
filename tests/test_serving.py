"""Tests for the FastAPI serving layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Create a test client (models won't be loaded in test)."""
    from src.serving.api import app

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_reports_model_status(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "feature_store_connected" in data


class TestClassifyEndpoint:
    def test_invalid_address_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify", json={"wallet_address": "not-an-address"})
        assert response.status_code == 422

    def test_missing_address_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify", json={})
        assert response.status_code == 422

    def test_valid_address_format_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/classify",
            json={"wallet_address": "0x" + "a" * 40},
        )
        # Either 503 (no models) or 404 (not in feature store) -- not 422
        assert response.status_code in (503, 404)

    def test_classify_with_mocked_models(self, client: TestClient) -> None:
        """Test full classify flow with mocked models."""
        import src.serving.api as api_module

        mock_feature_store = MagicMock()
        mock_feature_store.get_features.return_value = {
            "wallet_address": "0x" + "a" * 40,
            "tx_frequency_per_day": 1.5,
            "activity_regularity": 0.3,
            "hour_of_day_entropy": 3.0,
            "weekend_vs_weekday_ratio": 0.2,
            "avg_holding_duration_estimate": 48.0,
            "gas_price_sensitivity": -0.1,
            "is_contract": 0.0,
            "dex_to_total_ratio": 0.4,
            "lending_to_total_ratio": 0.1,
            "counterparty_concentration": 0.05,
            "value_velocity": 2.0,
            "burst_score": 1.5,
        }
        mock_feature_store.get_feature_names.return_value = [
            "tx_frequency_per_day",
            "activity_regularity",
            "hour_of_day_entropy",
            "weekend_vs_weekday_ratio",
            "avg_holding_duration_estimate",
            "gas_price_sensitivity",
            "is_contract",
            "dex_to_total_ratio",
            "lending_to_total_ratio",
            "counterparty_concentration",
            "value_velocity",
            "burst_score",
        ]

        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = (
            np.array([0]),
            np.array([0.95]),
        )
        mock_classifier.predict_proba.return_value = np.array([[0.95, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]])

        orig_fs = api_module._feature_store
        orig_clf = api_module._classifier
        try:
            api_module._feature_store = mock_feature_store
            api_module._classifier = mock_classifier

            response = client.post(
                "/classify",
                json={"wallet_address": "0x" + "a" * 40},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["label"] == "smart_money"
            assert data["confidence"] == pytest.approx(0.95)
            assert "probabilities" in data
            assert "features" in data
            assert "latency_ms" in data
        finally:
            api_module._feature_store = orig_fs
            api_module._classifier = orig_clf


class TestExplainEndpoint:
    def test_invalid_address_returns_422(self, client: TestClient) -> None:
        response = client.post("/explain", json={"wallet_address": "invalid"})
        assert response.status_code == 422


class TestSimilarEndpoint:
    def test_invalid_address_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/similar",
            json={"wallet_address": "invalid", "top_k": 5},
        )
        assert response.status_code == 422

    def test_top_k_bounds(self, client: TestClient) -> None:
        response = client.post(
            "/similar",
            json={"wallet_address": "0x" + "a" * 40, "top_k": 200},
        )
        assert response.status_code == 422
