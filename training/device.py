"""TrustLM Device Management Module.

Handles device discovery, compute acceleration (CUDA/CPU),
GPU memory optimizations, and device telemetry.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional

import torch


def get_device_info(preferred_device: Optional[str] = None) -> Dict[str, Any]:
    """Inspect and return system compute capabilities.

    Args:
        preferred_device: Optional override ('cuda', 'cpu', 'cuda:0', etc.).

    Returns:
        Dictionary detailing resolved device, GPU hardware specs, and software versions.
    """
    cuda_available = torch.cuda.is_available()

    if preferred_device:
        target_device = torch.device(preferred_device)
    elif cuda_available:
        target_device = torch.device("cuda")
    else:
        target_device = torch.device("cpu")

    info: Dict[str, Any] = {
        "device": target_device,
        "device_type": target_device.type,
        "is_cuda": target_device.type == "cuda",
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if cuda_available:
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        # Capability
        props = torch.cuda.get_device_properties(0)
        info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    else:
        info["gpu_count"] = 0
        info["gpu_name"] = None
        info["cuda_version"] = None
        info["vram_total_gb"] = 0.0
        info["compute_capability"] = None

    return info


def get_training_device(preferred_device: Optional[str] = None) -> torch.device:
    """Resolve and return the optimal PyTorch compute device."""
    info = get_device_info(preferred_device)
    return info["device"]


def print_device_summary(info: Optional[Dict[str, Any]] = None) -> None:
    """Print formatted device and compute environment summary."""
    if info is None:
        info = get_device_info()

    line = "=" * 70
    print(f"\n{line}")
    print("                 COMPUTE DEVICE TELEMETRY")
    print(line)
    print(f" Target Device        : {info['device']}")
    print(f" PyTorch Version      : {info['pytorch_version']}")
    print(f" Python Version       : {info['python_version']}")
    print(f" CUDA Available       : {info['is_cuda']}")
    if info["is_cuda"]:
        print(f" GPU Model            : {info['gpu_name']} ({info['gpu_count']} device(s))")
        print(f" CUDA Toolkit Version : {info['cuda_version']}")
        print(f" Total VRAM           : {info['vram_total_gb']} GB")
        print(f" Compute Capability   : {info['compute_capability']}")
    else:
        print(" GPU Acceleration     : None detected (running on CPU engine)")
    print(f"{line}\n")
