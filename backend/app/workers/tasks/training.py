"""Local training task — runs on the `gpu` queue (concurrency=1, same
reasoning as batch inference: an 8GB laptop GPU can't run training and
inference concurrently without OOM risk).

Materializes the dataset version to a plain on-disk YOLO folder
(`write_yolo_dataset` — the same code path `export_yolo` uses, so training
labels and a manual export are guaranteed identical), then runs
`YOLO(base_weights).train(...)`, capturing per-epoch metrics via
Ultralytics' `on_fit_epoch_end` callback — structured data straight from
the trainer object, not stdout-scraping, which is brittle across
Ultralytics versions. On completion, the best checkpoint is copied into
the model registry and registered as a new model, immediately available
for auto-annotation (PLAN "Local ↔ Kaggle Workflow": the loop closes here).
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.ml_model import ModelKind
from app.models.training_job import TrainingJob, TrainingJobEpoch, TrainingJobStatus
from app.services.dataset.export_yolo import ExportError, write_yolo_dataset
from app.services.inference.registry import register_model
from app.workers.celery_app import celery_app
from app.workers.progress import clear_cancel, is_cancel_requested
from app.workers.training_progress import EpochProgress, set_training_progress


def _extract_epoch_metrics(trainer) -> dict:
    """Ultralytics' metric keys vary by task/version (e.g.
    'metrics/mAP50(B)' for detect); look them up defensively rather than
    assuming an exact key set."""
    metrics = dict(getattr(trainer, "metrics", {}) or {})

    def _find(*substrings: str) -> float | None:
        for key, value in metrics.items():
            if all(s in key for s in substrings):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    loss_names = list(getattr(trainer, "loss_names", []) or [])
    tloss = getattr(trainer, "tloss", None)
    losses = {}
    if tloss is not None and loss_names:
        try:
            values = tloss.tolist() if hasattr(tloss, "tolist") else list(tloss)
            losses = dict(zip(loss_names, values))
        except Exception:
            losses = {}

    return {
        "box_loss": losses.get("box_loss") or losses.get("box"),
        "cls_loss": losses.get("cls_loss") or losses.get("cls"),
        "dfl_loss": losses.get("dfl_loss") or losses.get("dfl"),
        "precision": _find("precision"),
        "recall": _find("recall"),
        "map50": _find("mAP50(") or _find("mAP50,"),
        "map50_95": _find("mAP50-95"),
    }


@celery_app.task(bind=True, name="app.workers.tasks.training.train_local_model")
def train_local_model(self, training_job_id: str) -> None:
    db = SessionLocal()
    job = db.get(TrainingJob, uuid.UUID(training_job_id))
    if job is None:
        db.close()
        return

    job.status = TrainingJobStatus.RUNNING
    job.celery_task_id = self.request.id
    job.started_at = datetime.now(timezone.utc)
    db.commit()

    run_dir = settings.TRAINING_OUTPUT_DIR / str(job.id)
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        data_yaml_path = write_yolo_dataset(db, version_id=job.dataset_version_id, root=dataset_dir)

        from app.models.ml_model import MLModel

        base_model = db.get(MLModel, job.base_model_id) if job.base_model_id else None
        if base_model is None:
            raise ExportError("A base model is required to fine-tune from — none was provided")

        from ultralytics import YOLO

        model = YOLO(base_model.weights_path)

        def on_fit_epoch_end(trainer) -> None:
            metrics = _extract_epoch_metrics(trainer)
            epoch_num = int(getattr(trainer, "epoch", 0)) + 1  # ultralytics epoch is 0-indexed

            db.add(
                TrainingJobEpoch(
                    training_job_id=job.id,
                    epoch=epoch_num,
                    recorded_at=datetime.now(timezone.utc),
                    **metrics,
                )
            )
            job.current_epoch = epoch_num
            job.metrics = {k: v for k, v in metrics.items() if v is not None}

            # Epoch-boundary cancellation: BaseTrainer checks `self.stop`
            # after each epoch and breaks its loop — the same between-unit
            # cancel check the inference batch job does between images,
            # just at a coarser (epoch, not image) granularity since
            # there's no finer hook into a single .train() call.
            if is_cancel_requested(str(job.id)):
                trainer.stop = True
                job.status = TrainingJobStatus.CANCELLED
                job.failed_at = datetime.now(timezone.utc)

            db.commit()

            set_training_progress(
                str(job.id),
                EpochProgress(
                    epoch=epoch_num,
                    total_epochs=job.epochs,
                    status="CANCELLED" if job.status == TrainingJobStatus.CANCELLED else "RUNNING",
                    **metrics,
                ),
            )

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        device = job.device
        try:
            import torch

            if device not in ("cpu",) and not torch.cuda.is_available():
                device = "cpu"  # graceful fallback, never crash a queued job over hardware drift
        except ImportError:
            device = "cpu"

        model.train(
            data=str(data_yaml_path),
            epochs=job.epochs,
            imgsz=job.image_size,
            batch=job.batch_size,
            device=device,
            lr0=job.learning_rate if job.learning_rate else 0.01,
            project=str(settings.TRAINING_OUTPUT_DIR),
            name=str(job.id),
            exist_ok=True,
            verbose=False,
        )

        db.refresh(job)
        if job.status == TrainingJobStatus.CANCELLED:
            set_training_progress(
                str(job.id), EpochProgress(epoch=job.current_epoch, total_epochs=job.epochs, status="CANCELLED")
            )
            return

        best_weights = settings.TRAINING_OUTPUT_DIR / str(job.id) / "weights" / "best.pt"
        if not best_weights.exists():
            raise ExportError(f"Training finished but no weights were produced at {best_weights}")

        registered_path = settings.MODELS_DIR / "pt" / f"trained-{job.id}.pt"
        registered_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_weights, registered_path)

        result_model = register_model(
            db,
            name=f"{base_model.name}-retrained",
            weights_path=str(registered_path),
            kind=ModelKind.DETECTOR,
            version=f"trained-from-{base_model.name}",
        )
        result_model.base_model_id = base_model.id
        # Seed the new model's dashboard metrics from its final training
        # epoch — Phase 8's "upload evaluation metrics" endpoint can still
        # replace this with a proper held-out eval later; this just means
        # a freshly-trained model isn't blank on the Models dashboard.
        result_model.metrics = dict(job.metrics)
        job.result_model_id = result_model.id
        job.artifact_path = str(registered_path)
        job.status = TrainingJobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        set_training_progress(
            str(job.id), EpochProgress(epoch=job.current_epoch, total_epochs=job.epochs, status="COMPLETED")
        )
    except Exception as exc:
        db.rollback()
        job = db.get(TrainingJob, uuid.UUID(training_job_id))
        if job is not None:
            job.status = TrainingJobStatus.FAILED
            job.error = str(exc)
            job.failed_at = datetime.now(timezone.utc)
            db.commit()
        set_training_progress(
            str(training_job_id),
            EpochProgress(epoch=0, total_epochs=0, status="FAILED", error=str(exc)),
        )
        raise
    finally:
        clear_cancel(str(training_job_id))
        db.close()
