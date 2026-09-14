"""Official Argoverse 2 forecasting metrics.

Shapes everywhere: pred (B, K, T, 2), probs (B, K) summing to 1, gt (B, T, 2).
Unimodal models are the K=1 special case. All distances in meters. Each
function returns a per-scenario (B,) tensor; averaging over the dataset is the
caller's job so subsets can be sliced for error analysis.
"""

import torch
from torch import Tensor


def _l2_per_step(pred: Tensor, gt: Tensor) -> Tensor:
    """(B, K, T) L2 distance between each mode and the ground truth per step."""
    return torch.linalg.norm(pred - gt[:, None], dim=-1)


def min_ade(pred: Tensor, gt: Tensor) -> Tensor:
    return _l2_per_step(pred, gt).mean(dim=-1).min(dim=-1).values


def min_fde(pred: Tensor, gt: Tensor) -> Tensor:
    return _l2_per_step(pred, gt)[..., -1].min(dim=-1).values


def miss_rate(pred: Tensor, gt: Tensor, thresh: float = 2.0) -> Tensor:
    return (min_fde(pred, gt) > thresh).float()


def brier_min_fde(pred: Tensor, probs: Tensor, gt: Tensor) -> Tensor:
    fde = _l2_per_step(pred, gt)[..., -1]  # (B, K)
    best_fde, best_idx = fde.min(dim=-1)
    p_best = probs.gather(-1, best_idx[:, None]).squeeze(-1)
    return best_fde + (1.0 - p_best) ** 2


def evaluate_batch(pred: Tensor, probs: Tensor, gt: Tensor) -> dict[str, Tensor]:
    return {
        "minADE": min_ade(pred, gt),
        "minFDE": min_fde(pred, gt),
        "MR": miss_rate(pred, gt),
        "brier_minFDE": brier_min_fde(pred, probs, gt),
    }
