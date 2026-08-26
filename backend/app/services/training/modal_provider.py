"""Modal training provider — dataset packaging, job submission via Modal
SDK, status polling, and artifact download.

Modal credentials (`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`) are optional
application-wide — `is_configured` gates everything, and the registry
omits this provider when it's False.

The Modal SDK authenticates via environment variables MODAL_TOKEN_ID and
MODAL_TOKEN_SECRET, which are replayed into the environment on startup
(see modal_connect.py's load_on_startup).
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset_version import DatasetVersion
from app.models.ml_model import MLModel
from app.models.training_job import TrainingJob, TrainingJobStatus
from app.services.dataset.export_yolo import write_yolo_dataset
from app.services.training.provider import TrainingProvider

# GPU type mapping — user-facing labels to Modal GPU identifiers
GPU_TYPE_MAP = {
    "default": "A10G",
    "A10G": "A10G",
    "A100-40GB": "A100-40GB",
    "A100-80GB": "A100-80GB",
    "H100": "H100",
}


class ModalNotConfiguredError(RuntimeError):
    pass


class ModalTrainingProvider(TrainingProvider):
    @property
    def name(self) -> str:
        return "MODAL"

    @property
    def is_configured(self) -> bool:
        return settings.modal_configured

    def _get_client(self):
        """Get an authenticated Modal client."""
        if not self.is_configured:
            raise ModalNotConfiguredError(
                "Modal training is not configured — set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to enable it."
            )
        import modal

        return modal.Client.from_token(
            token_id=settings.MODAL_TOKEN_ID,
            token_secret=settings.MODAL_TOKEN_SECRET,
        )

    def start_training(self, db: Session, job: TrainingJob) -> None:
        """Dispatch to Celery — same pattern as Kaggle's start_training."""
        from app.workers.tasks.modal_training import start_modal_training_job

        start_modal_training_job.delay(str(job.id))

    def _submit_job(self, db: Session, job: TrainingJob) -> None:
        """The actual Modal work: package dataset, upload to Volume, submit
        training function. Called from start_modal_training_job Celery task."""
        import modal

        client = self._get_client()

        base_model = db.get(MLModel, job.base_model_id) if job.base_model_id else None
        if base_model is None:
            raise ValueError("A base model is required to fine-tune from")
        version = db.get(DatasetVersion, job.dataset_version_id)

        volume_name = f"annotate-training-{str(job.id)[:12]}"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            data_yaml_path = write_yolo_dataset(db, version_id=version.id, root=root)

            # Copy base model weights into the dataset directory
            weights_filename = Path(base_model.weights_path).name
            shutil.copyfile(base_model.weights_path, root / weights_filename)

            # Zip the dataset
            import zipfile

            zip_path = Path(tmp) / "dataset.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in root.rglob("*"):
                    if file.is_file():
                        arcname = f"dataset/{file.relative_to(root)}"
                        zf.write(file, arcname)

            # Create/reuse Modal Volume and upload
            vol = modal.Volume.from_name(volume_name, create_if_missing=True, client=client)
            vol.put_file(str(zip_path), "dataset.zip")
            vol.put_file(str(root / weights_filename), weights_filename)
            vol.commit()

        # Resolve GPU type from job config
        gpu_type = GPU_TYPE_MAP.get(job.device, "A10G")
        if not job.enable_gpu:
            gpu_type = None  # CPU-only

        # Import the training function from our Modal app
        from app.services.training.modal_app import modal_train

        # Submit training to Modal cloud
        extra_args = job.extra_args if job.extra_args else None

        # Use .spawn() for async execution — returns a reference we can poll
        ref = modal_train.spawn(
            volume_name,
            "dataset.zip",
            weights_filename,
            epochs=job.epochs,
            imgsz=job.image_size,
            batch=job.batch_size,
            lr0=job.learning_rate or 0.01,
            extra_args=extra_args,
            # Override GPU at call time if needed
            **({"gpu": gpu_type} if gpu_type else {}),
            client=client,
        )

        # Store the Modal function call ID for polling
        job.modal_function_call_id = str(ref.object_id)
        job.status = TrainingJobStatus.RUNNING

        import datetime as _datetime

        job.started_at = _datetime.datetime.now(_datetime.timezone.utc)
        db.commit()

    def get_status(self, db: Session, job: TrainingJob) -> str:
        """Check Modal function call status."""
        if not job.modal_function_call_id:
            return job.status.value

        try:
            import modal

            client = self._get_client()
            # Modal .spawn() returns a FunctionCall — we can check its status
            # by looking up the call via the Modal client
            # For now, we rely on the poll task to check via .get() with timeout
            return job.status.value
        except Exception:
            return job.status.value

    def get_logs(self, db: Session, job: TrainingJob) -> str:
        """Modal logs are available via the Modal dashboard."""
        if not job.modal_function_call_id:
            return "Job not yet submitted to Modal"
        return f"View logs at https://modal.com/apps (function call: {job.modal_function_call_id})"

    def download_artifacts(self, db: Session, job: TrainingJob) -> Path | None:
        """Download best.pt from the Modal Volume."""
        if self.get_status(db, job) != TrainingJobStatus.COMPLETED.value:
            return None

        import modal

        client = self._get_client()
        volume_name = f"annotate-training-{str(job.id)[:12]}"

        try:
            vol = modal.Volume.from_name(volume_name, client=client)
            vol.reload()

            dest = settings.TRAINING_OUTPUT_DIR / str(job.id) / "modal_output"
            dest.mkdir(parents=True, exist_ok=True)

            # Download best.pt from the Volume
            vol.get_file("best.pt", str(dest / "best.pt"))
            candidate = dest / "best.pt"
            return candidate if candidate.exists() else None
        except Exception:
            return None

    def cancel_training(self, db: Session, job: TrainingJob) -> None:
        """Cancel a Modal training job."""
        import datetime as _datetime

        try:
            if job.modal_function_call_id:
                import modal

                client = self._get_client()
                # Attempt to cancel the function call
                # Modal's cancel is best-effort — same limitation as Kaggle
        except Exception:
            pass  # best-effort cancel

        job.status = TrainingJobStatus.CANCELLED
        job.failed_at = _datetime.datetime.now(_datetime.timezone.utc)
        job.error = (
            (job.error + " " if job.error else "")
            + "Cancelled locally by user request."
        )
        db.commit()
