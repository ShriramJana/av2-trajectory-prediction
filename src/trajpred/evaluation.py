"""Run any stage over a dataset and collect per-scenario metrics."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from trajpred.dataset import collate
from trajpred.metrics import evaluate_batch
from trajpred.models import build_model

# batch -> (traj (B, K, 60, 2), probs (B, K))
PredictFn = Callable[[dict[str, Tensor]], tuple[Tensor, Tensor]]


def to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def model_predict_fn(model: torch.nn.Module) -> PredictFn:
    """Wrap a trained model: logits -> probabilities."""

    def predict(batch):
        traj, logits = model(batch)
        return traj, logits.softmax(dim=-1)

    return predict


def load_predictor(cfg: dict, device: torch.device, ckpt: Path | None = None) -> PredictFn:
    """Build a config's model, load its best checkpoint, return it as a PredictFn."""
    model = build_model(cfg["stage"], cfg["model"]).to(device)
    ckpt = ckpt or Path(cfg["ckpt_dir"]) / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    return model_predict_fn(model.eval())


@torch.no_grad()
def evaluate_dataset(
    predict: PredictFn, dataset: Dataset, device: torch.device, batch_size: int = 128
) -> dict[str, np.ndarray]:
    """Per-scenario metrics in dataset order, plus `best_mode` (argmin-FDE mode index)."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    out: dict[str, list[Tensor]] = {}
    for batch in loader:
        batch = to_device(batch, device)
        traj, probs = predict(batch)
        metrics = evaluate_batch(traj, probs, batch["future"])
        fde = torch.linalg.norm(traj[:, :, -1] - batch["future"][:, None, -1], dim=-1)
        metrics["best_mode"] = fde.argmin(dim=-1)
        for k, v in metrics.items():
            out.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}
