"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client (models won't be loaded in test)."""
    from src.serving.api import app

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_reports_model_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "feature_store_connected" in data


class TestClassifyEndpoint:
    def test_invalid_address_returns_422(self, client):
        response = client.post("/classify", json={"wallet_address": "not-an-address"})
        assert response.status_code == 422

    def test_missing_address_returns_422(self, client):
        response = client.post("/classify", json={})
        assert response.status_code == 422

    def test_valid_address_format_accepted(self, client):
        # Will return 503 since models aren't loaded, but the address format is valid
        response = client.post(
            "/classify",
            json={"wallet_address": "0x" + "a" * 40},
        )
        # Either 503 (no models) or 404 (not in feature store) — not 422
        assert response.status_code in (503, 404)


class TestExplainEndpoint:
    def test_invalid_address_returns_422(self, client):
        response = client.post("/explain", json={"wallet_address": "invalid"})
        assert response.status_code == 422


class TestSimilarEndpoint:
    def test_invalid_address_returns_422(self, client):
        response = client.post(
            "/similar",
            json={"wallet_address": "invalid", "top_k": 5},
        )
        assert response.status_code == 422

    def test_top_k_bounds(self, client):
        response = client.post(
            "/similar",
            json={"wallet_address": "0x" + "a" * 40, "top_k": 200},
        )
        assert response.status_code == 422
