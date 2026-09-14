"""Machine-specific environment: where the data lives, what device to train on.

Everything platform-dependent is decided here so the rest of the codebase is
portable. Override the data location with the TRAJPRED_DATA env var.
"""

import os
import platform
from pathlib import Path


def data_root() -> Path:
    override = os.environ.get("TRAJPRED_DATA")
    if override:
        return Path(override)
    if platform.system() == "Windows":
        return Path(r"C:\data\av2")
    return Path.home() / "data" / "av2"


def raw_dir(split: str) -> Path:
    return data_root() / "raw" / split


def processed_dir(split: str) -> Path:
    return data_root() / "processed" / split


def pick_device():
    """cuda (NVIDIA) > mps (Apple Silicon) > cpu."""
    import torch  # local import: path helpers shouldn't force a torch import

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
