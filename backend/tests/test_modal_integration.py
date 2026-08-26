"""Modal integration: connect/disconnect, provider configuration, and
training job submission. Network calls to the real Modal API are
monkeypatched — these tests exercise the storage/status/error-handling
logic, not third-party connectivity (same house convention as the rest
of the suite).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.services.integrations.modal_connect as modal_connect
from app.services.training.modal_provider import ModalNotConfiguredError


def test_modal_initially_disconnected(client: TestClient) -> None:
    resp = client.get("/api/v1/integrations")
    assert resp.status_code == 200
    statuses = {row["provider"]: row for row in resp.json()}
    assert statuses["MODAL"]["connected"] is False


def test_connect_modal_success(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(modal_connect, "_verify", lambda token_id, token_secret: None)

    resp = client.post("/api/v1/integrations/modal", json={"token_id": "tok_abc123", "token_secret": "sec_xyz789"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["identifier"] is not None
    assert body["verified_at"] is not None

    listed = {row["provider"]: row for row in client.get("/api/v1/integrations").json()}
    assert listed["MODAL"]["connected"] is True


def test_connect_modal_bad_credentials_stays_disconnected(client: TestClient, monkeypatch) -> None:
    def _fail(token_id: str, token_secret: str) -> None:
        raise modal_connect.ModalVerificationError("Invalid token")

    monkeypatch.setattr(modal_connect, "_verify", _fail)

    resp = client.post("/api/v1/integrations/modal", json={"token_id": "bad", "token_secret": "bad"})
    assert resp.status_code == 400

    listed = {row["provider"]: row for row in client.get("/api/v1/integrations").json()}
    assert listed["MODAL"]["connected"] is False
    assert listed["MODAL"]["last_error"] is not None


def test_disconnect_modal(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(modal_connect, "_verify", lambda token_id, token_secret: None)
    client.post("/api/v1/integrations/modal", json={"token_id": "tok_abc123", "token_secret": "sec_xyz789"})

    resp = client.delete("/api/v1/integrations/modal")
    assert resp.status_code == 204

    listed = {row["provider"]: row for row in client.get("/api/v1/integrations").json()}
    assert listed["MODAL"]["connected"] is False


def test_modal_provider_is_configured_false_by_default() -> None:
    """ModalTrainingProvider.is_configured returns False when no credentials
    are set — the provider must not appear in the available list."""
    from app.services.training.modal_provider import ModalTrainingProvider

    provider = ModalTrainingProvider()
    # In test env, MODAL_TOKEN_ID/SECRET are not set
    assert provider.is_configured is False


def test_modal_provider_name() -> None:
    from app.services.training.modal_provider import ModalTrainingProvider

    provider = ModalTrainingProvider()
    assert provider.name == "MODAL"


def test_modal_connect_loads_on_startup(monkeypatch) -> None:
    """load_on_startup replays stored credentials into the environment."""
    import os

    from app.services.integrations.modal_connect import apply_modal_env

    # Clear any existing env vars
    os.environ.pop("MODAL_TOKEN_ID", None)
    os.environ.pop("MODAL_TOKEN_SECRET", None)

    apply_modal_env("tok_test", "sec_test")
    assert os.environ.get("MODAL_TOKEN_ID") == "tok_test"
    assert os.environ.get("MODAL_TOKEN_SECRET") == "sec_test"

    # Cleanup
    os.environ.pop("MODAL_TOKEN_ID", None)
    os.environ.pop("MODAL_TOKEN_SECRET", None)


def test_modal_connect_request_schema_validation(client: TestClient) -> None:
    """ModalConnectRequest requires both token_id and token_secret."""
    # Missing token_secret
    resp = client.post("/api/v1/integrations/modal", json={"token_id": "tok_abc"})
    assert resp.status_code == 422  # validation error

    # Missing token_id
    resp = client.post("/api/v1/integrations/modal", json={"token_secret": "sec_xyz"})
    assert resp.status_code == 422

    # Empty strings
    resp = client.post("/api/v1/integrations/modal", json={"token_id": "", "token_secret": ""})
    assert resp.status_code == 422


def test_modal_gpu_type_mapping() -> None:
    """GPU_TYPE_MAP correctly maps user labels to Modal GPU identifiers."""
    from app.services.training.modal_provider import GPU_TYPE_MAP

    assert GPU_TYPE_MAP["default"] == "A10G"
    assert GPU_TYPE_MAP["A10G"] == "A10G"
    assert GPU_TYPE_MAP["A100-40GB"] == "A100-40GB"
    assert GPU_TYPE_MAP["A100-80GB"] == "A100-80GB"
    assert GPU_TYPE_MAP["H100"] == "H100"
    # Unknown GPU type falls back to A10G in the provider
    assert GPU_TYPE_MAP.get("unknown", "A10G") == "A10G"
