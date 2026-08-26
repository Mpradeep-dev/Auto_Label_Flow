"""Modal App definition for remote YOLO training. This module defines the
Modal application and its training function that runs on Modal's GPU cloud.

The training function:
1. Mounts a Modal Volume containing the YOLO dataset + base model weights
2. Extracts the dataset zip
3. Installs ultralytics + torch with CUDA support
4. Runs YOLO training with the user-specified parameters
5. Returns the path to best.pt in the Volume

Usage from modal_provider.py:
    ref = modal_train.spawn(volume_name, data_yaml_path, weights_path, ...)
    # poll ref.get_status() until COMPLETED
    # ref.get() returns the path to best.pt
"""
from __future__ import annotations

import modal

# Container image: Debian slim + CUDA torch + ultralytics.
# The torch pin matches Kaggle's approach (2.5.1/0.20.1 with cu121) for
# broad GPU compatibility (Pascal through Hopper).
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    )
    .pip_install("ultralytics==8.4.102", "PyYAML==6.0.3")
)

app = modal.App("annotate-training", image=IMAGE)


@app.function(
    gpu="A10G",
    timeout=21600,  # 6 hours max
    volumes={"/data": modal.Volume.from_name("annotate-training", create_if_missing=True)},
    memory=16384,  # 16 GB RAM
)
def modal_train(
    volume_name: str,
    dataset_zip_path: str,
    weights_filename: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.01,
    extra_args: dict | None = None,
) -> str:
    """Train a YOLO model on Modal's GPU cloud.

    Args:
        volume_name: Name of the Modal Volume containing the dataset.
        dataset_zip_path: Path to the dataset zip within the Volume.
        weights_filename: Filename of the base model weights in the Volume root.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        lr0: Initial learning rate.
        extra_args: Additional ultralytics YOLO.train() kwargs.

    Returns:
        Path to best.pt within the Volume.
    """
    import os
    import subprocess
    import zipfile
    from pathlib import Path

    import yaml

    # Re-mount to pick up the uploaded files
    vol = modal.Volume.from_name(volume_name)
    vol.reload()

    data_root = Path("/data")
    dataset_dir = data_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Extract dataset zip
    zip_path = data_root / dataset_zip_path
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found at {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    # Fix data.yaml paths — the zip contents are nested under a "dataset" folder
    # from write_yolo_dataset, so find data.yaml wherever it ended up
    data_yaml_candidates = list(dataset_dir.rglob("data.yaml"))
    if not data_yaml_candidates:
        raise FileNotFoundError("data.yaml not found in extracted dataset")
    data_yaml_path = data_yaml_candidates[0]

    # Rewrite path: to point to the extracted directory
    data_yaml = yaml.safe_load(data_yaml_path.read_text())
    data_yaml["path"] = str(data_yaml_path.parent)
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # Find base model weights
    weights_path = data_root / weights_filename
    if not weights_path.exists():
        raise FileNotFoundError(f"Base model weights not found at {weights_path}")

    print(f"Dataset: {data_yaml_path}")
    print(f"Weights: {weights_path}")
    print(f"Classes: {data_yaml.get('nc', '?')} — {data_yaml.get('names', [])}")

    # Train
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "lr0": lr0,
    }
    if extra_args:
        # Filter out typed kwargs to avoid duplicates
        typed = {"data", "epochs", "imgsz", "batch", "lr0"}
        train_kwargs.update({k: v for k, v in extra_args.items() if k not in typed})

    results = model.train(**train_kwargs)

    # Find best.pt
    best_pt = data_yaml_path.parent.parent / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        # Ultralytics sometimes puts it elsewhere
        candidates = list(data_root.rglob("best.pt"))
        if candidates:
            best_pt = candidates[0]
        else:
            raise FileNotFoundError("best.pt not found after training")

    # Copy best.pt to Volume root for easy retrieval
    import shutil

    dest = data_root / "best.pt"
    shutil.copyfile(best_pt, dest)
    vol.commit()

    print(f"Training complete. best.pt at: {dest}")
    return str(dest)
