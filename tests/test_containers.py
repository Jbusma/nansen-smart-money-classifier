"""Container smoke tests — verify dockerized services work end-to-end.

Run with:  pytest tests/test_containers.py --container -v

Requires:  docker compose up -d  (clickhouse, api, mcp-server)
"""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest
import requests

pytestmark = pytest.mark.container

API_URL = "http://localhost:8000"
MCP_URL = "http://localhost:3001/mcp"
CLICKHOUSE_HTTP = "http://localhost:8123"
DUMMY_WALLET = "0x" + "ab" * 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _mcp_request(method: str, params: dict[str, Any] | None = None, req_id: int = 1) -> requests.Response:
    """Send a JSON-RPC request to the MCP server."""
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params:
        payload["params"] = params
    return requests.post(
        MCP_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        timeout=10,
        stream=True,
    )


def _parse_sse_response(resp: requests.Response) -> dict[str, Any]:
    """Parse the first SSE event from a streaming response."""
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise ValueError("No SSE data event received")


# ---------------------------------------------------------------------------
# Port / connectivity checks
# ---------------------------------------------------------------------------


class TestPortsOpen:
    """Verify that all expected container ports are reachable."""

    def test_clickhouse_http_port(self) -> None:
        assert _port_open("localhost", 8123), "ClickHouse HTTP port 8123 not reachable"

    def test_clickhouse_native_port(self) -> None:
        assert _port_open("localhost", 9000), "ClickHouse native port 9000 not reachable"

    def test_api_port(self) -> None:
        assert _port_open("localhost", 8000), "FastAPI port 8000 not reachable"

    def test_mcp_port(self) -> None:
        assert _port_open("localhost", 3001), "MCP server port 3001 not reachable"


# ---------------------------------------------------------------------------
# ClickHouse direct
# ---------------------------------------------------------------------------


class TestClickHouse:
    """Verify ClickHouse is responsive and the nansen database exists."""

    def test_clickhouse_ping(self) -> None:
        resp = requests.get(f"{CLICKHOUSE_HTTP}/ping", timeout=5)
        assert resp.status_code == 200
        assert resp.text.strip() == "Ok."

    def test_nansen_database_exists(self) -> None:
        resp = requests.get(
            CLICKHOUSE_HTTP,
            params={"query": "SHOW DATABASES FORMAT JSON"},
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        db_names = [row["name"] for row in data["data"]]
        assert "nansen" in db_names, f"nansen DB not found, got: {db_names}"

    def test_wallet_features_table_exists(self) -> None:
        resp = requests.get(
            CLICKHOUSE_HTTP,
            params={"query": "SHOW TABLES FROM nansen FORMAT JSON"},
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        table_names = [row["name"] for row in data["data"]]
        assert "wallet_features" in table_names, f"wallet_features not found, got: {table_names}"


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


class TestAPIHealth:
    """Verify the FastAPI backend is healthy."""

    def test_health_endpoint(self) -> None:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "feature_store_connected" in data

    def test_feature_store_connected(self) -> None:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        data = resp.json()
        assert data["feature_store_connected"] is True, "API cannot reach ClickHouse — feature_store_connected is False"


class TestAPIEndpoints:
    """Verify API endpoints accept/reject requests correctly."""

    def test_classify_invalid_address_returns_422(self) -> None:
        resp = requests.post(
            f"{API_URL}/classify",
            json={"wallet_address": "not-an-address"},
            timeout=5,
        )
        assert resp.status_code == 422

    def test_classify_unknown_wallet_returns_404_or_503(self) -> None:
        """A valid but unknown wallet should return 404 (not in store) or 503 (no model)."""
        resp = requests.post(
            f"{API_URL}/classify",
            json={"wallet_address": DUMMY_WALLET},
            timeout=5,
        )
        assert resp.status_code in (404, 503)

    def test_wallet_context_returns_200(self) -> None:
        """Context endpoint should succeed even for unknown wallets (empty data)."""
        resp = requests.get(
            f"{API_URL}/wallet/{DUMMY_WALLET}/context",
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["wallet_address"] == DUMMY_WALLET
        assert "transaction_summary" in data

    def test_explain_no_model_returns_503(self) -> None:
        resp = requests.post(
            f"{API_URL}/explain",
            json={"wallet_address": DUMMY_WALLET},
            timeout=5,
        )
        assert resp.status_code == 503

    def test_similar_invalid_returns_422(self) -> None:
        resp = requests.post(
            f"{API_URL}/similar",
            json={"wallet_address": "bad", "top_k": 5},
            timeout=5,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


class TestMCPServer:
    """Verify MCP server handles JSON-RPC correctly."""

    def test_non_mcp_path_returns_404(self) -> None:
        resp = requests.get("http://localhost:3001/", timeout=5)
        assert resp.status_code == 404

    def test_missing_accept_header_rejected(self) -> None:
        resp = requests.post(
            MCP_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        # SDK rejects without the right Accept header
        assert resp.status_code in (400, 406, 500)

    def test_initialize_handshake(self) -> None:
        resp = _mcp_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        )
        assert resp.status_code == 200
        data = _parse_sse_response(resp)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        result = data["result"]
        assert result["serverInfo"]["name"] == "nansen-smart-money-classifier"
        assert "tools" in result["capabilities"]

    def test_tools_list(self) -> None:
        # Initialize first
        init_resp = _mcp_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        )
        assert init_resp.status_code == 200

        # List tools
        resp = _mcp_request("tools/list", req_id=2)
        assert resp.status_code == 200
        data = _parse_sse_response(resp)
        tools = data["result"]["tools"]
        tool_names = {t["name"] for t in tools}
        assert "classify_wallet" in tool_names
        assert "get_wallet_context" in tool_names
        assert "explain_wallet" in tool_names
        assert "find_similar_wallets" in tool_names
        assert "get_cluster_profile" in tool_names


# ---------------------------------------------------------------------------
# Cross-service integration: MCP -> API -> ClickHouse
# ---------------------------------------------------------------------------


class TestCrossService:
    """Verify the MCP -> FastAPI -> ClickHouse chain works."""

    def test_mcp_get_wallet_context_calls_api(self) -> None:
        """MCP get_wallet_context tool should call API which queries ClickHouse."""
        # Initialize
        _mcp_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        )

        # Call the tool
        resp = _mcp_request(
            "tools/call",
            {
                "name": "get_wallet_context",
                "arguments": {"wallet_address": DUMMY_WALLET},
            },
            req_id=3,
        )
        assert resp.status_code == 200
        data = _parse_sse_response(resp)
        result = data["result"]

        # Should return content (even if wallet has no data)
        assert "content" in result
        assert len(result["content"]) > 0
        text = result["content"][0]["text"]
        assert DUMMY_WALLET in text

    def test_api_context_queries_clickhouse(self) -> None:
        """API /wallet/.../context should query ClickHouse and return structured data."""
        resp = requests.get(f"{API_URL}/wallet/{DUMMY_WALLET}/context", timeout=10)
        assert resp.status_code == 200
        data = resp.json()

        # Transaction summary should be present (possibly zeroed)
        ts = data["transaction_summary"]
        assert ts is not None
        assert isinstance(ts["total_transactions"], int)
        assert isinstance(ts["total_eth_volume"], (int, float))

        # Other sections should be present (possibly empty)
        assert "top_contracts" in data
        assert "token_activity" in data
        assert "timing_patterns" in data
