"""S0: constant-velocity baseline. No parameters, no training."""

import torch
from torch import Tensor


def constant_velocity(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Extrapolate the focal agent's last observed velocity for 60 steps.

    Returns traj (B, 1, 60, 2) and probs (B, 1).
    """
    hist = batch["agent_hist"][:, 0]  # focal agent is row 0
    last = hist[:, -1]
    velocity = last - hist[:, -2]  # meters per step
    steps = torch.arange(1, 61, device=hist.device, dtype=hist.dtype)
    traj = last[:, None, :] + velocity[:, None, :] * steps[None, :, None]
    return traj[:, None], torch.ones(len(hist), 1, device=hist.device, dtype=hist.dtype)
