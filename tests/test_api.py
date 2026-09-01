"""
Tests for api/server.py
FastAPI endpoint tests using TestClient.
"""

import pytest

# Mark all tests in this module as integration (require httpx and server)
pytestmark = pytest.mark.integration

try:
    from fastapi.testclient import TestClient
    from api.server import app
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False


@pytest.fixture
def client():
    if not HAS_TESTCLIENT:
        pytest.skip("httpx or fastapi not available for TestClient")
    return TestClient(app)


# ── Health Check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


# ── Dashboard Data ───────────────────────────────────────────────────────────

class TestDashboardEndpoint:
    @pytest.mark.slow
    def test_dashboard_returns_200(self, client):
        response = client.get("/dashboard?ticker=SPY&period=5d")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "signal" in data


# ── Symbol Resolution ────────────────────────────────────────────────────────

class TestSymbolEndpoint:
    @pytest.mark.slow
    def test_resolve_symbol(self, client):
        response = client.get("/resolve-symbol?query=AAPL")
        assert response.status_code == 200
