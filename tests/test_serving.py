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


class TestEnrichEndpoint:
    def test_enrich_default_params(self, client: TestClient) -> None:
        """Enrich with default params calls enrich_all(etherscan=False)."""
        mock_results = {
            "hardcoded": 17,
            "token_list": 4954,
            "defillama": 270,
            "etherscan": 0,
            "total_registry_size": 5241,
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.data.protocol_enrichment.enrich_all",
                lambda **kwargs: mock_results,
            )
            response = client.post("/enrich", json={})
            assert response.status_code == 200
            data = response.json()
            assert data["hardcoded"] == 17
            assert data["token_list"] == 4954
            assert data["total_registry_size"] == 5241
            assert data["etherscan"] == 0

    def test_enrich_with_etherscan(self, client: TestClient) -> None:
        mock_results = {
            "hardcoded": 17,
            "token_list": 4954,
            "defillama": 270,
            "etherscan": 510,
            "total_registry_size": 5751,
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.data.protocol_enrichment.enrich_all",
                lambda **kwargs: mock_results,
            )
            response = client.post("/enrich", json={"etherscan": True, "top_n": 100})
            assert response.status_code == 200
            data = response.json()
            assert data["etherscan"] == 510

    def test_enrich_failure_returns_503(self, client: TestClient) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.data.protocol_enrichment.enrich_all",
                lambda **kwargs: (_ for _ in ()).throw(RuntimeError("CH down")),
            )
            response = client.post("/enrich", json={})
            assert response.status_code == 503


class TestLabelWalletEndpoint:
    def test_label_wallet_success(self, client: TestClient) -> None:
        mock_client = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.data.clickhouse_sync.get_client", lambda: mock_client)
            response = client.post(
                "/label/wallet",
                json={
                    "wallet_address": "0x" + "ab" * 20,
                    "label": "defi_power_user",
                    "confidence": 0.85,
                    "evidence": "High dex ratio, many protocols",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["labeled"] == 1
            assert data["label"] == "defi_power_user"
            mock_client.insert.assert_called_once()

    def test_label_wallet_invalid_address(self, client: TestClient) -> None:
        response = client.post(
            "/label/wallet",
            json={
                "wallet_address": "not-valid",
                "label": "defi_power_user",
                "confidence": 0.5,
            },
        )
        assert response.status_code == 422

    def test_label_wallet_missing_label(self, client: TestClient) -> None:
        response = client.post(
            "/label/wallet",
            json={
                "wallet_address": "0x" + "ab" * 20,
                "confidence": 0.5,
            },
        )
        assert response.status_code == 422

    def test_label_wallet_confidence_bounds(self, client: TestClient) -> None:
        response = client.post(
            "/label/wallet",
            json={
                "wallet_address": "0x" + "ab" * 20,
                "label": "test",
                "confidence": 1.5,
            },
        )
        assert response.status_code == 422

        response = client.post(
            "/label/wallet",
            json={
                "wallet_address": "0x" + "ab" * 20,
                "label": "test",
                "confidence": -0.1,
            },
        )
        assert response.status_code == 422


class TestLabelClusterEndpoint:
    def test_label_cluster_success(self, client: TestClient) -> None:
        mock_client = MagicMock()
        mock_labels = np.array([0, 0, 1, 1, 1, -1])

        mock_pipeline = {"labels_": mock_labels}
        mock_df = MagicMock()
        mock_df.loc.__getitem__ = MagicMock(
            return_value=MagicMock(tolist=MagicMock(return_value=["0x" + "aa" * 20, "0x" + "bb" * 20]))
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.data.clickhouse_sync.get_client", lambda: mock_client)
            mp.setattr("joblib.load", lambda _path: mock_pipeline)
            mp.setattr("pandas.read_parquet", lambda _path, **kw: mock_df)
            mp.setattr("pathlib.Path.exists", lambda _self: True)

            response = client.post(
                "/label/cluster",
                json={
                    "cluster_id": 0,
                    "label": "institutional_otc",
                    "confidence": 0.9,
                    "evidence": "High volume, low frequency",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["labeled"] == 2
            assert data["label"] == "institutional_otc"
            mock_client.insert.assert_called_once()

    def test_label_cluster_invalid_id(self, client: TestClient) -> None:
        response = client.post(
            "/label/cluster",
            json={
                "cluster_id": -2,
                "label": "test",
                "confidence": 0.5,
            },
        )
        assert response.status_code == 422

    def test_label_cluster_missing_label(self, client: TestClient) -> None:
        response = client.post(
            "/label/cluster",
            json={
                "cluster_id": 0,
                "confidence": 0.5,
            },
        )
        assert response.status_code == 422
