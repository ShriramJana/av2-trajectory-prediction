"""Training losses. Each takes (model output, batch) and returns a scalar."""

import torch
import torch.nn.functional as F
from torch import Tensor


def unimodal_loss(output: tuple[Tensor, Tensor], batch: dict[str, Tensor]) -> Tensor:
    """S1: smooth-L1 between the single predicted trajectory and the GT future."""
    traj, _ = output
    return F.smooth_l1_loss(traj[:, 0], batch["future"])


def wta_loss(traj: Tensor, logits: Tensor, gt: Tensor, alpha: float = 1.0) -> Tensor:
    """Winner-take-all: regress only the mode closest to the GT, and teach the
    logits to predict which mode that is.

    traj (B, K, T, 2), logits (B, K), gt (B, T, 2). If every mode were regressed
    toward the GT they would all converge to the mean future (mode collapse);
    training only the winner lets each mode specialize on its own kind of future.
    """
    with torch.no_grad():  # choosing the winner is a discrete decision, not differentiable
        ade = torch.linalg.norm(traj - gt[:, None], dim=-1).mean(dim=-1)  # (B, K)
        winner = ade.argmin(dim=-1)  # (B,)
    best = traj[torch.arange(len(traj), device=traj.device), winner]  # (B, T, 2)
    return F.smooth_l1_loss(best, gt) + alpha * F.cross_entropy(logits, winner)


def multimodal_loss(output: tuple[Tensor, Tensor], batch: dict[str, Tensor]) -> Tensor:
    traj, logits = output
    return wta_loss(traj, logits, batch["future"])


def build_loss(cfg: dict):
    if cfg["stage"] == "s1":
        return unimodal_loss
    if cfg["stage"] in ("s2", "s3"):
        return multimodal_loss
    raise ValueError(f"no loss registered for stage {cfg['stage']!r}")
