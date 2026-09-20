"""Model ladder. Every model maps a batch to (traj (B, K, 60, 2), logits (B, K))."""

from torch import nn


def build_model(stage: str, model_cfg: dict) -> nn.Module:
    """Construct the trainable model for a config's `stage:` key."""
    if stage == "s1":
        from trajpred.models.gru import GRUPredictor

        return GRUPredictor(**model_cfg)
    if stage == "s2":
        from trajpred.models.multimodal import MultiModalGRU

        return MultiModalGRU(**model_cfg)
    if stage == "s3":
        from trajpred.models.polyline import PolylineNet

        return PolylineNet(**model_cfg)
    raise ValueError(f"unknown trainable stage: {stage!r}")
