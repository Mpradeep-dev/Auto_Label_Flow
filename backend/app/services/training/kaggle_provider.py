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


def _decode_kaggle_log(raw_text: str) -> str:
    """Kaggle's `kernels_output` `.log` file is a JSON array of
    `{"stream_name", "time", "data"}` records — confirmed live against a
    real completed kernel's actual output, not documented anywhere obvious
    up front. Every prior version of `get_logs` returned that raw JSON
    directly, which meant: the FAILED-job error snippet shown to users was
    an unreadable JSON blob instead of the real error text buried inside
    one of its `data` fields, and — the thing that sent us looking in the
    first place — `ultralytics_log_parser.py`'s regexes never matched
    anything, because there's no real Ultralytics progress line at the top
    level of the raw file; it's trapped inside escaped `"data"` string
    values. This concatenates every record's `data` field, in original
    order, into the plain stdout/stderr text those regexes (and a human
    reading `job.error`) actually expect.

    Falls back to the raw text unchanged if it isn't that JSON-array shape
    — an older/plain log format, or a future Kaggle API version that
    changes this again — so a decode miss degrades to "unparsed", never a
    crash.
    """
    try:
        records = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return raw_text
    if not isinstance(records, list):
        return raw_text
    return "".join(r.get("data", "") for r in records if isinstance(r, dict))


_KERNEL_TEMPLATE = """\
import subprocess

ENABLE_GPU = {enable_gpu}

if ENABLE_GPU:
    # Confirmed live, twice: Kaggle's own pre-installed torch is NOT
    # necessarily matched to whatever GPU it hands out. A run assigned a
    # Tesla P100 (Pascal, compute capability 6.0) failed both times with
    # "no kernel image is available for execution on the device" — the
    # stock image's own torch was already 2.10.0+cu128 before any pip
    # install ran, and PyPI's cu128 wheels dropped Pascal/Maxwell
    # (sm_60/sm_50) support entirely starting with the 2.8 series. There's
    # no API to request a specific GPU generation from Kaggle, so instead
    # of trusting whatever's pre-installed (this app's previous fix —
    # confirmed live NOT to help, since the baseline itself was already
    # incompatible), pin explicitly to the last well-established
    # torch/torchvision/CUDA combo confirmed to still build sm_60 into its
    # gencode list. Deliberately not the newest such combo (2.6.0/0.21.0):
    # that pairing has a known, still-open pip metadata bug where the
    # torchvision 0.21.0 wheel declares Requires-Dist: torch>=2.8.0, which
    # makes `pip install torch==2.6.0 torchvision==0.21.0` fail outright on
    # a dependency conflict — 2.5.1/0.20.1 has no such report.
    #
    # `torch`/`torchvision` are deliberately not imported anywhere above
    # this point: they're native-extension packages, and Python doesn't
    # cleanly unload/reload a compiled .so a process already has mapped —
    # importing the stock version first and reinstalling over it on disk
    # would leave the process running inconsistent, already-loaded native
    # code even though `torch.__version__` might report the new number.
    # This subprocess is the ONLY thing that touches torch before the
    # process's one and only `import torch` below.
    subprocess.run(
        [
            "pip", "install", "-q",
            "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    with open("/tmp/torch_constraints.txt", "w") as f:
        f.write("torch==2.5.1\\n")
        f.write("torchvision==0.20.1\\n")
        f.write("torchaudio==2.5.1\\n")
    constraint_args = ["-c", "/tmp/torch_constraints.txt"]
else:
    # CPU-only: no CUDA arch to match, so just pin to whatever's already
    # installed and let pip leave torch/torchvision alone entirely.
    import torch
    import torchvision

    with open("/tmp/torch_constraints.txt", "w") as f:
        f.write("torch==" + torch.__version__ + "\\n")
        f.write("torchvision==" + torchvision.__version__ + "\\n")
    constraint_args = ["-c", "/tmp/torch_constraints.txt"]

subprocess.run(
    ["pip", "install", "-q", "ultralytics=={ultralytics_version}"] + constraint_args,
    check=True,
)

import torch
from ultralytics import YOLO

print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))

model = YOLO("{base_weights_path}")
model.train(
    data="{data_yaml_path}",
    epochs={epochs},
    imgsz={image_size},
    batch={batch_size},
    lr0={learning_rate},
{extra_args_lines}
)
"""

# The typed fields above always win — any of these names appearing in
# `job.extra_args` too is dropped rather than emitted, since a duplicate
# keyword argument in the generated script would be a Python SyntaxError
# on Kaggle's side, not a silent override.
_TYPED_TRAIN_KWARGS = {"data", "epochs", "imgsz", "batch", "lr0"}


def _render_extra_args(extra_args: dict) -> str:
    filtered = {k: v for k, v in (extra_args or {}).items() if k not in _TYPED_TRAIN_KWARGS}
    return "\n".join(f"    {key}={value!r}," for key, value in filtered.items())


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
        # Dispatched, not run inline — mirrors LocalTrainingProvider's own
        # start_training (train_local_model.delay(...)), which is what lets
        # POST /training/jobs return immediately and the frontend's existing
        # QUEUED/RUNNING polling (TrainingRunsPage.tsx's refetchInterval)
        # show real progress. Before this, this method DID the actual work
        # (dataset export/zip, upload, a readiness poll that alone can take
        # up to 5 minutes, kernel push) synchronously inside the request —
        # confirmed live to read as "just hangs, no update, then eventually
        # fails" from the UI: the Start button's only state during all of
        # that is a static "Starting…", and depending on how long the
        # request took relative to any intermediate proxy/browser limits, a
        # job that Kaggle-side ultimately succeeded could still show as a
        # failure to the user who was watching the button, not the job list.
        from app.workers.tasks.kaggle_training import start_kaggle_training_job

        start_kaggle_training_job.delay(str(job.id))

    def _push_kernel(self, db: Session, job: TrainingJob) -> None:
        """The actual synchronous Kaggle work `start_training` used to do
        inline — now run from `start_kaggle_training_job`'s own Celery task
        instead, off the request thread. Left as a plain method (not the
        `start_training` the `TrainingProvider` ABC expects) so nothing
        outside `kaggle_training.py`'s task calls it directly by accident."""
        api = self._client()

        base_model = db.get(MLModel, job.base_model_id) if job.base_model_id else None
        if base_model is None:
            raise ValueError("A base model is required to fine-tune from")
        version = db.get(DatasetVersion, job.dataset_version_id)

        import shutil
        import tempfile
        import time

        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            data_yaml_path = write_yolo_dataset(db, version_id=version.id, root=root)

            dataset_slug = f"annotate-v{version.version_number}-{str(job.id)[:8]}"
            # Where Kaggle actually mounts this dataset on the kernel's own
            # filesystem, read-only, once attached via `dataset_sources`
            # below. NOT `/kaggle/input/{slug}` — that's the older/simpler
            # path documented in most Kaggle API examples, but a real
            # kernel's own `find /kaggle/input` (added as a one-off
            # diagnostic, then confirmed against 5 consecutive live runs)
            # showed it actually nested under datasets/<username>/<slug>.
            kaggle_input_dir = f"/kaggle/input/datasets/{settings.KAGGLE_USERNAME}/{dataset_slug}"

            # write_yolo_dataset's own `path:` is the LOCAL build directory
            # (`root`, here a throwaway tempdir) — correct for LOCAL
            # training, which trains against that same filesystem, but
            # meaningless once this folder is re-uploaded as a Kaggle
            # dataset: Ultralytics would try to resolve images/train,
            # images/val, etc. against a path that only ever existed on
            # this machine. Rewritten to where it will actually live at
            # train time.
            data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
            data_yaml["path"] = kaggle_input_dir
            data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

            # The base model's weights file has no other legitimate way to
            # reach the kernel: `kernels_push` only uploads the kernel's
            # own code_file, not arbitrary sibling files sitting in the
            # local push directory — confirmed live, the previous approach
            # (copying the weights next to train.py and referencing it by
            # a bare relative filename) raised FileNotFoundError on
            # Kaggle's side, since that file was never actually there.
            # Bundling it into the SAME dataset already being attached via
            # `dataset_sources` is the one mechanism Kaggle actually
            # supports for getting an arbitrary file onto a kernel.
            weights_filename = Path(base_model.weights_path).name
            shutil.copyfile(base_model.weights_path, root / weights_filename)

            (root / "dataset-metadata.json").write_text(
                json.dumps({"title": dataset_slug, "id": f"{settings.KAGGLE_USERNAME}/{dataset_slug}", "licenses": [{"name": "CC0-1.0"}]}),
                encoding="utf-8",
            )
            api.dataset_create_new(folder=str(root), dir_mode="zip", quiet=True)

            # dataset_create_new() returns as soon as Kaggle *accepts* the
            # upload — it doesn't wait for Kaggle's own backend processing
            # (indexing/virus-scan/etc.) to finish making the dataset's
            # files actually mountable. Confirmed live: pushing the kernel
            # immediately after this raised FileNotFoundError for a file
            # that genuinely was in the uploaded folder, just not yet
            # visible to a kernel attaching to it seconds later. Poll for
            # "ready"; give up and push anyway after a bounded wait rather
            # than hanging a job forever on a slow Kaggle-side process.
            #
            # The status check itself is best-effort, not a hard dependency:
            # also confirmed live, `dataset_status` can 403 with "Permission
            # datasets.get was denied" in the first moments after creation —
            # its own propagation lag, not a real permissions problem (same
            # account that just created it). Treat that identically to
            # "not ready yet" rather than letting it abort the whole job.
            # Confirmed live: `dataset_status` genuinely returns "ready" once
            # Kaggle's processing finishes, and this app's own datasets
            # (~20MB / ~200 images) took well over a minute to get there —
            # a short poll budget just exhausts itself on the early
            # datasets.get-403 window (see above) without ever seeing it.
            # 5 minutes, 5s apart: enough headroom for a real dataset this
            # size, still bounded so a genuinely stuck Kaggle-side process
            # doesn't hang job creation forever.
            dataset_ref = f"{settings.KAGGLE_USERNAME}/{dataset_slug}"
            for _ in range(60):
                try:
                    if str(api.dataset_status(dataset_ref)).strip().lower() == "ready":
                        break
                except Exception:
                    pass
                time.sleep(5)

            kernel_dir = Path(tmp) / "kernel"
            kernel_dir.mkdir()

            import ultralytics

            script = _KERNEL_TEMPLATE.format(
                enable_gpu=job.enable_gpu,
                ultralytics_version=ultralytics.__version__,
                base_weights_path=f"{kaggle_input_dir}/{weights_filename}",
                data_yaml_path=f"{kaggle_input_dir}/data.yaml",
                epochs=job.epochs,
                image_size=job.image_size,
                batch_size=job.batch_size,
                learning_rate=job.learning_rate or 0.01,
                extra_args_lines=_render_extra_args(job.extra_args),
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
                        "enable_gpu": job.enable_gpu,
                        # Kaggle kernels have NO internet access by default —
                        # without this, the kernel's own `pip install
                        # ultralytics` (see _KERNEL_TEMPLATE) fails immediately
                        # with a DNS resolution error before training ever
                        # starts. Confirmed live: exactly this failure, in a
                        # real kernel's own logs, on a job that otherwise
                        # looked like it was just "stuck."
                        "enable_internet": True,
                        "dataset_sources": [f"{settings.KAGGLE_USERNAME}/{dataset_slug}"],
                    }
                ),
                encoding="utf-8",
            )
            api.kernels_push(str(kernel_dir))

        import datetime as _datetime

        job.kaggle_kernel_ref = f"{settings.KAGGLE_USERNAME}/{kernel_slug}"
        job.status = TrainingJobStatus.RUNNING
        # Set here, not left to get_status()'s own RUNNING-transition check
        # below — the job is already RUNNING the moment the kernel push
        # succeeds, so that check would never see a QUEUED->RUNNING edge to
        # catch it on, and started_at would stay null forever.
        job.started_at = _datetime.datetime.now(_datetime.timezone.utc)
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
            import datetime as _datetime

            now = _datetime.datetime.now(_datetime.timezone.utc)
            if mapped == TrainingJobStatus.RUNNING and job.started_at is None:
                job.started_at = now
            elif mapped == TrainingJobStatus.COMPLETED:
                job.completed_at = now
            elif mapped == TrainingJobStatus.FAILED:
                job.failed_at = now
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
            if not log_path.exists():
                return "log not available yet"
            return _decode_kaggle_log(log_path.read_text(encoding="utf-8"))

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
        # The Kaggle API has no kernel-cancel endpoint as of this writing, so
        # the remote kernel keeps running until Kaggle's own timeout — but
        # this job's local status must still move off RUNNING/QUEUED, or
        # the Cancel button visibly does nothing (it stayed RUNNING forever,
        # the poll loop kept polling it, and a stuck kernel looked
        # indistinguishable from a healthy one). CANCELLED here means "this
        # app has given up tracking it," recorded honestly, not "the remote
        # job was stopped."
        import datetime as _datetime

        job.status = TrainingJobStatus.CANCELLED
        job.failed_at = _datetime.datetime.now(_datetime.timezone.utc)
        job.error = (
            (job.error + " " if job.error else "")
            + "Cancelled locally by user request. Kaggle has no remote-stop API, so the kernel "
            + "may keep running on Kaggle's side until it finishes or times out on its own — "
            + "check kaggle.com/code if you need to confirm it actually stopped."
        )
        db.commit()
