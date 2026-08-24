"""Kaggle training provider — dataset push, kernel push, status poll,
artifact pull. Kaggle credentials (`KAGGLE_USERNAME`/`KAGGLE_KEY`) are
optional application-wide (PLAN "Do not make Kaggle mandatory" /
"the unconfigured path is the default path"): `is_configured` gates
everything else, and the registry (`registry.py`) omits this provider
entirely from `GET /api/training/providers` when it's False rather than
letting a caller reach a half-working provider.

Not live-tested in this environment — no `~/.kaggle` credentials exist on
this machine (confirmed during planning). Written against the documented
`kaggle` package API (`kaggle.api.kaggle_api_extended.KaggleApi`); the
`is_configured=False` path IS exercised by the test suite, since that is
the path every install without Kaggle credentials actually takes.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset_version import DatasetVersion
from app.models.ml_model import MLModel
from app.models.training_job import TrainingJob, TrainingJobStatus
from app.services.dataset.export_yolo import write_yolo_dataset
from app.services.training.provider import TrainingProvider


class KaggleNotConfiguredError(RuntimeError):
    pass


_KERNEL_TEMPLATE = """\
import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics=={ultralytics_version}"], check=True)
from ultralytics import YOLO

model = YOLO("{base_weights_filename}")
model.train(
    data="{data_yaml_filename}",
    epochs={epochs},
    imgsz={image_size},
    batch={batch_size},
    lr0={learning_rate},
)
"""


class KaggleTrainingProvider(TrainingProvider):
    @property
    def name(self) -> str:
        return "KAGGLE"

    @property
    def is_configured(self) -> bool:
        return settings.kaggle_configured

    def _client(self):
        if not self.is_configured:
            raise KaggleNotConfiguredError(
                "Kaggle training is not configured — set KAGGLE_USERNAME and KAGGLE_KEY to enable it."
            )
        # Imported lazily: the `kaggle` package reads credentials from the
        # environment / ~/.kaggle/kaggle.json at IMPORT time and raises if
        # neither is present, so importing it eagerly at module load would
        # break every deployment that doesn't use Kaggle at all.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api

    def start_training(self, db: Session, job: TrainingJob) -> None:
        api = self._client()

        base_model = db.get(MLModel, job.base_model_id) if job.base_model_id else None
        if base_model is None:
            raise ValueError("A base model is required to fine-tune from")
        version = db.get(DatasetVersion, job.dataset_version_id)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            data_yaml_path = write_yolo_dataset(db, version_id=version.id, root=root)

            dataset_slug = f"annotate-v{version.version_number}-{str(job.id)[:8]}"
            (root / "dataset-metadata.json").write_text(
                json.dumps({"title": dataset_slug, "id": f"{settings.KAGGLE_USERNAME}/{dataset_slug}", "licenses": [{"name": "CC0-1.0"}]}),
                encoding="utf-8",
            )
            api.dataset_create_new(folder=str(root), dir_mode="zip", quiet=True)

            kernel_dir = Path(tmp) / "kernel"
            kernel_dir.mkdir()
            import shutil

            shutil.copyfile(base_model.weights_path, kernel_dir / Path(base_model.weights_path).name)
            shutil.copyfile(data_yaml_path, kernel_dir / "data.yaml")

            import ultralytics

            script = _KERNEL_TEMPLATE.format(
                ultralytics_version=ultralytics.__version__,
                base_weights_filename=Path(base_model.weights_path).name,
                data_yaml_filename="data.yaml",
                epochs=job.epochs,
                image_size=job.image_size,
                batch_size=job.batch_size,
                learning_rate=job.learning_rate or 0.01,
            )
            (kernel_dir / "train.py").write_text(script, encoding="utf-8")

            kernel_slug = f"annotate-train-{str(job.id)[:8]}"
            (kernel_dir / "kernel-metadata.json").write_text(
                json.dumps(
                    {
                        "id": f"{settings.KAGGLE_USERNAME}/{kernel_slug}",
                        "title": kernel_slug,
                        "code_file": "train.py",
                        "language": "python",
                        "kernel_type": "script",
                        "is_private": True,
                        "enable_gpu": True,
                        "dataset_sources": [f"{settings.KAGGLE_USERNAME}/{dataset_slug}"],
                    }
                ),
                encoding="utf-8",
            )
            api.kernels_push(str(kernel_dir))

        job.kaggle_kernel_ref = f"{settings.KAGGLE_USERNAME}/{kernel_slug}"
        job.status = TrainingJobStatus.RUNNING
        db.commit()

    def get_status(self, db: Session, job: TrainingJob) -> str:
        if not job.kaggle_kernel_ref:
            return job.status.value
        api = self._client()
        status_response = api.kernels_status(job.kaggle_kernel_ref)
        kaggle_status = getattr(status_response, "status", None) or status_response.get("status", "unknown")
        mapped = {
            "complete": TrainingJobStatus.COMPLETED,
            "error": TrainingJobStatus.FAILED,
            "running": TrainingJobStatus.RUNNING,
            "queued": TrainingJobStatus.QUEUED,
        }.get(str(kaggle_status).lower(), TrainingJobStatus.RUNNING)
        if mapped != job.status:
            job.status = mapped
            db.commit()
        return job.status.value

    def get_logs(self, db: Session, job: TrainingJob) -> str:
        if not job.kaggle_kernel_ref:
            return "kernel not yet pushed"
        api = self._client()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            api.kernels_output(job.kaggle_kernel_ref, path=tmp, quiet=True)
            log_path = Path(tmp) / f"{job.kaggle_kernel_ref.split('/')[-1]}.log"
            return log_path.read_text(encoding="utf-8") if log_path.exists() else "log not available yet"

    def download_artifacts(self, db: Session, job: TrainingJob) -> Path | None:
        if self.get_status(db, job) != TrainingJobStatus.COMPLETED.value:
            return None
        api = self._client()
        dest = settings.TRAINING_OUTPUT_DIR / str(job.id) / "kaggle_output"
        dest.mkdir(parents=True, exist_ok=True)
        api.kernels_output(job.kaggle_kernel_ref, path=str(dest), quiet=True)
        candidates = list(dest.rglob("best.pt"))
        return candidates[0] if candidates else None

    def cancel_training(self, db: Session, job: TrainingJob) -> None:
        # The Kaggle API has no kernel-cancel endpoint as of this writing;
        # recording intent is the honest behaviour here rather than
        # claiming a cancel that can't actually be issued remotely.
        job.error = (job.error or "") + " [cancel requested by user; Kaggle API has no remote-stop endpoint]"
        db.commit()
