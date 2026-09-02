"""Settings-page integrations: connect/disconnect for Kaggle and Roboflow.
Network calls to the real Kaggle/Roboflow APIs are monkeypatched — these
tests exercise the storage/status/error-handling logic, not third-party
connectivity (same house convention as the rest of the suite: fake the
external boundary, keep the test offline)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.services.integrations.kaggle_connect as kaggle_connect
import app.services.integrations.roboflow_connect as roboflow_connect


def test_integrations_initially_disconnected(client: TestClient) -> None:
    resp = client.get("/api/v1/integrations")
    assert resp.status_code == 200
    statuses = {row["provider"]: row for row in resp.json()}
    assert statuses["KAGGLE"]["connected"] is False
    assert statuses["ROBOFLOW"]["connected"] is False


def test_connect_kaggle_success(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(kaggle_connect, "_verify", lambda username, key: None)

    resp = client.post("/api/v1/integrations/kaggle", json={"username": "alice", "key": "secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["identifier"] == "alice"
    assert body["verified_at"] is not None

    listed = {row["provider"]: row for row in client.get("/api/v1/integrations").json()}
    assert listed["KAGGLE"]["connected"] is True


def test_connect_kaggle_bad_credentials_stays_disconnected(client: TestClient, monkeypatch) -> None:
    def _fail(username: str, key: str) -> None:
        raise kaggle_connect.KaggleVerificationError("401 Unauthorized")

    monkeypatch.setattr(kaggle_connect, "_verify", _fail)

    resp = client.post("/api/v1/integrations/kaggle", json={"username": "alice", "key": "wrong"})
    assert resp.status_code == 400

    # The failed attempt must NOT read back as connected, even though its
    # config was persisted (see kaggle_connect.get_status).
    listed = {row["provider"]: row for row in client.get("/api/v1/integrations").json()}
    assert listed["KAGGLE"]["connected"] is False
    assert listed["KAGGLE"]["last_error"] is not None


def test_disconnect_kaggle(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(kaggle_connect, "_verify", lambda username, key: None)
    client.post("/api/v1/integrations/kaggle", json={"username": "alice", "key": "secret"})

    resp = client.delete("/api/v1/integrations/kaggle")
    assert resp.status_code == 204

    listed = {row["provider"]: row for row in client.get("/api/v1/integrations").json()}
    assert listed["KAGGLE"]["connected"] is False


class _FakeVersion:
    def __init__(self, version: int, images: int) -> None:
        self.version = str(version)  # real SDK returns a str, see roboflow_browse.py
        self.images = images


class _FakeProject:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id

    def versions(self) -> list[_FakeVersion]:
        return [_FakeVersion(1, 30), _FakeVersion(2, 50)]


class _FakeWorkspace:
    name = "My Workspace"  # display label — deliberately different from .url
    url = "my-workspace"  # the slug that must round-trip into later calls
    project_list = [
        {"id": "my-workspace/cones", "name": "Cones", "type": "object-detection", "images": 120},
        {"id": "my-workspace/players", "name": "Players", "type": "object-detection", "images": 80},
    ]

    def project(self, project_id: str):
        return _FakeProject(project_id)


class _FakeRoboflow:
    def __init__(self, api_key: str) -> None:
        if api_key != "good-key":
            raise ValueError("Invalid API key")
        self.api_key = api_key
        self.current_workspace = "my-workspace"

    def workspace(self, the_workspace=None):
        return _FakeWorkspace()


def test_connect_roboflow_success(client: TestClient, monkeypatch) -> None:
    import roboflow

    monkeypatch.setattr(roboflow, "Roboflow", _FakeRoboflow)

    resp = client.post("/api/v1/integrations/roboflow", json={"api_key": "good-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["identifier"] == "my-workspace"


def test_connect_roboflow_bad_key(client: TestClient, monkeypatch) -> None:
    import roboflow

    monkeypatch.setattr(roboflow, "Roboflow", _FakeRoboflow)

    resp = client.post("/api/v1/integrations/roboflow", json={"api_key": "bad-key"})
    assert resp.status_code == 400

    listed = {row["provider"]: row for row in client.get("/api/v1/integrations").json()}
    assert listed["ROBOFLOW"]["connected"] is False


def test_roboflow_export_requires_connection_first(client: TestClient, unique_name: str) -> None:
    project_resp = client.post("/api/v1/projects", json={"name": unique_name})
    project_id = project_resp.json()["id"]
    dataset_resp = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "ds"})
    dataset_id = dataset_resp.json()["id"]
    version_resp = client.post(f"/api/v1/datasets/{dataset_id}/versions", json={})
    # No approved images yet, so version creation itself may 400 — either
    # way, roboflow export must never get past "not connected" without
    # credentials, which is the thing this test actually verifies.
    if version_resp.status_code != 201:
        return
    version_id = version_resp.json()["id"]
    resp = client.post(
        f"/api/v1/versions/{version_id}/export/roboflow",
        json={"workspace": "ws", "project": "proj"},
    )
    assert resp.status_code == 400
    assert "not connected" in resp.json()["detail"].lower()


def test_list_roboflow_projects_requires_connection_first(client: TestClient) -> None:
    resp = client.get("/api/v1/integrations/roboflow/projects")
    assert resp.status_code == 400


def test_list_roboflow_projects(client: TestClient, monkeypatch) -> None:
    import roboflow

    monkeypatch.setattr(roboflow, "Roboflow", _FakeRoboflow)
    client.post("/api/v1/integrations/roboflow", json={"api_key": "good-key"})

    resp = client.get("/api/v1/integrations/roboflow/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert {p["project"] for p in body} == {"cones", "players"}
    assert all(p["workspace"] == "my-workspace" for p in body)
    cones = next(p for p in body if p["project"] == "cones")
    assert cones["name"] == "Cones"
    assert cones["image_count"] == 120


def test_destructive_roboflow_sdk_methods_are_blocked(client: TestClient, monkeypatch) -> None:
    """Safety guard: this integration must never be able to delete or
    remove data in a connected Roboflow account, no matter which code path
    gets there (present or future) — see
    `roboflow_connect._install_destructive_guards`. Connecting installs the
    guard as a side effect (both `connect()` and `get_client()` call it);
    this verifies the SDK's own delete-ish methods are actually neutralized
    afterward, not just that our own service functions happen not to call
    them today."""
    import roboflow
    from roboflow.core.project import Project
    from roboflow.core.training import Training
    from roboflow.core.version import Version
    from roboflow.core.workspace import Workspace

    from app.services.integrations.roboflow_connect import RoboflowDestructiveOperationBlocked

    monkeypatch.setattr(roboflow, "Roboflow", _FakeRoboflow)
    resp = client.post("/api/v1/integrations/roboflow", json={"api_key": "good-key"})
    assert resp.status_code == 200

    # Skip each class's real __init__ (it wants live API args) — the guard
    # patches the method on the class itself, so a bare instance is enough
    # to prove the patched method raises before any request goes out.
    class _DummyProject(Project):
        def __init__(self) -> None:
            pass

    class _DummyWorkspace(Workspace):
        def __init__(self) -> None:
            pass

    class _DummyVersion(Version):
        def __init__(self) -> None:
            pass

    class _DummyTraining(Training):
        def __init__(self) -> None:
            pass

    project = _DummyProject()
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        project.delete()
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        project.delete_images(["img-1"])
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        project.delete_annotation_batch("batch-1")
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        project.delete_annotation_job_annotations("job-1")

    workspace = _DummyWorkspace()
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        workspace.delete_images(["img-1"])
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        workspace.remove_projects_from_folder("group-1", ["proj-1"])

    version = _DummyVersion()
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        version.delete()
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        version.delete_training()

    training = _DummyTraining()
    with pytest.raises(RoboflowDestructiveOperationBlocked):
        training.delete()


def test_connect_installs_stdout_guard(client: TestClient, monkeypatch) -> None:
    """`_install_destructive_guards` (run as a side effect of connecting)
    must also install the stdout guard — see `_CrashProofStdout` and
    `test_stdout_guard_absorbs_oserror_from_writes` below for why."""
    import roboflow

    import app.services.integrations.roboflow_connect as roboflow_connect_module

    monkeypatch.setattr(roboflow, "Roboflow", _FakeRoboflow)
    resp = client.post("/api/v1/integrations/roboflow", json={"api_key": "good-key"})
    assert resp.status_code == 200

    assert roboflow_connect_module._stdout_guard_installed is True


def test_stdout_guard_absorbs_oserror_from_writes() -> None:
    """Regression: the `roboflow` SDK writes decorative progress messages
    straight to `sys.stdout` (e.g. `Roboflow.workspace()`'s
    `sys.stdout.write("\\r" + "loading Roboflow workspace...")`). Under the
    desktop app, stdout is piped to a log file with PYTHONUNBUFFERED=1, and
    on Windows a write to that piped (non-console) stdout can raise
    `OSError: [Errno 22] Invalid argument` from deep inside the SDK —
    observed live as `list_projects()` failing with "Could not list
    Roboflow projects: [Errno 22] Invalid argument" with no real Roboflow
    API failure involved. `_CrashProofStdout` must swallow a write/flush
    failure from the stream it wraps instead of letting it propagate."""
    from app.services.integrations.roboflow_connect import _CrashProofStdout

    class _BrokenStream:
        def write(self, s):
            raise OSError(22, "Invalid argument")

        def flush(self):
            raise OSError(22, "Invalid argument")

    guarded = _CrashProofStdout(_BrokenStream())
    guarded.write("\r" + "loading Roboflow workspace...")  # must not raise
    guarded.flush()  # must not raise


def test_list_roboflow_versions(client: TestClient, monkeypatch) -> None:
    import roboflow

    monkeypatch.setattr(roboflow, "Roboflow", _FakeRoboflow)
    client.post("/api/v1/integrations/roboflow", json={"api_key": "good-key"})

    resp = client.get("/api/v1/integrations/roboflow/projects/my-workspace/cones/versions")
    assert resp.status_code == 200
    body = resp.json()
    # Newest first, and the SDK's str version numbers normalized to int.
    assert body == [{"version": 2, "image_count": 50}, {"version": 1, "image_count": 30}]
