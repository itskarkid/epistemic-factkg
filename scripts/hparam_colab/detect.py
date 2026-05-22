"""Environment detection for hparam_colab launcher."""
from __future__ import annotations

import sys
from enum import Enum, auto


class Env(Enum):
    COLAB = auto()       # running inside Google Colab
    LOCAL_GPU = auto()   # local machine with CUDA GPU
    LOCAL_CPU = auto()   # local machine, CPU-only (use Colab)


def detect() -> Env:
    """Return the current execution environment."""
    if _is_colab():
        return Env.COLAB
    if _has_cuda():
        return Env.LOCAL_GPU
    return Env.LOCAL_CPU


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def describe(env: Env) -> str:
    return {
        Env.COLAB:     "Google Colab (GPU available)",
        Env.LOCAL_GPU: f"Local GPU — {_gpu_name()}",
        Env.LOCAL_CPU: "Local CPU-only (no CUDA)",
    }[env]


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—"
    except Exception:
        return "—"
