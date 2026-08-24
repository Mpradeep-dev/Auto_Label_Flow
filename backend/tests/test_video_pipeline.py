"""Video upload + frame extraction, exercised through the real API with a
Celery task run in eager mode (see conftest's `real_client`/`real_db_session`
— a genuinely-committing session, since the task opens its own DB
connection and needs to see committed data)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def _make_test_video(path: Path, num_frames: int = 40, fps: float = 20.0, size=(64, 48)) -> None:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=(i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture()
def project_and_dataset(real_client: TestClient, unique_name: str) -> tuple[str, str]:
    project = real_client.post("/api/v1/projects", json={"name": unique_name}).json()
    dataset = real_client.post(f"/api/v1/projects/{project['id']}/datasets", json={"name": "d"}).json()
    return project["id"], dataset["id"]


def test_video_upload_probes_metadata(real_client: TestClient, project_and_dataset, tmp_path: Path) -> None:
    _, dataset_id = project_and_dataset
    video_path = tmp_path / "clip.mp4"
    _make_test_video(video_path, num_frames=40, fps=20.0, size=(64, 48))

    with open(video_path, "rb") as f:
        resp = real_client.post(
            f"/api/v1/datasets/{dataset_id}/videos", files={"file": ("clip.mp4", f, "video/mp4")}
        )
    assert resp.status_code == 201, resp.text
    video = resp.json()
    assert video["width"] == 64
    assert video["height"] == 48
    assert video["status"] == "UPLOADED"
    assert video["total_frames"] and video["total_frames"] >= 30  # allow for container/codec rounding


def test_video_upload_rejects_bad_extension(real_client: TestClient, project_and_dataset) -> None:
    _, dataset_id = project_and_dataset
    resp = real_client.post(
        f"/api/v1/datasets/{dataset_id}/videos", files={"file": ("clip.txt", b"not a video", "text/plain")}
    )
    assert resp.status_code == 400


def test_extract_frames_creates_video_frame_images(
    real_client: TestClient, project_and_dataset, tmp_path: Path
) -> None:
    _, dataset_id = project_and_dataset
    video_path = tmp_path / "clip.mp4"
    _make_test_video(video_path, num_frames=40, fps=20.0, size=(64, 48))

    with open(video_path, "rb") as f:
        video = real_client.post(
            f"/api/v1/datasets/{dataset_id}/videos", files={"file": ("clip.mp4", f, "video/mp4")}
        ).json()

    resp = real_client.post(f"/api/v1/videos/{video['id']}/extract-frames", json={"interval": 5})
    assert resp.status_code == 202, resp.text
    extracted = resp.json()
    assert extracted["status"] == "EXTRACTED"
    assert extracted["extracted_frame_count"] >= 6  # ~40 frames / interval 5

    images = real_client.get(f"/api/v1/datasets/{dataset_id}/images").json()
    assert images["total"] == extracted["extracted_frame_count"]
    frame = images["items"][0]
    assert frame["source_type"] == "VIDEO_FRAME"
    assert frame["video_id"] == video["id"]
    assert frame["frame_index"] is not None
    assert frame["frame_timestamp_s"] is not None
    assert frame["width"] == 64 and frame["height"] == 48


def test_extract_frames_missing_video_404(real_client: TestClient) -> None:
    resp = real_client.post(
        "/api/v1/videos/00000000-0000-0000-0000-000000000000/extract-frames", json={"interval": 5}
    )
    assert resp.status_code == 404
