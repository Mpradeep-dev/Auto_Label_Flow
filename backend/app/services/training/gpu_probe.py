"""Real GPU probe (PLAN spec section 29: "Detect available GPU and
display: GPU name, VRAM, CUDA availability, PyTorch version"). A live
`torch` call, not a config toggle — if CUDA genuinely isn't available this
correctly reports that and the UI falls back to CPU."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUInfo:
    torch_version: str
    cuda_available: bool
    device_name: str | None
    vram_total_mb: float | None
    cuda_version: str | None


def probe_gpu() -> GPUInfo:
    import torch

    cuda_available = torch.cuda.is_available()
    device_name = None
    vram_total_mb = None
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    return GPUInfo(
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        device_name=device_name,
        vram_total_mb=vram_total_mb,
        cuda_version=torch.version.cuda if cuda_available else None,
    )
