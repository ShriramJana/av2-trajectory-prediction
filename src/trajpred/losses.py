"""Training losses. Each takes (model output, batch) and returns a scalar."""

import torch.nn.functional as F
from torch import Tensor


def unimodal_loss(output: tuple[Tensor, Tensor], batch: dict[str, Tensor]) -> Tensor:
    """S1: smooth-L1 between the single predicted trajectory and the GT future."""
    traj, _ = output
    return F.smooth_l1_loss(traj[:, 0], batch["future"])


def build_loss(cfg: dict):
    if cfg["stage"] == "s1":
        return unimodal_loss
    raise ValueError(f"no loss registered for stage {cfg['stage']!r}")
