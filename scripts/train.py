"""Train one stage from its yaml config.

    python scripts/train.py --config configs/s1_gru.yaml

Runs the overfit-one-batch check first and refuses to start a real run if the
model can't memorize 16 scenes.
"""

import argparse
from pathlib import Path

from trajpred.config import load_config
from trajpred.losses import build_loss
from trajpred.models import build_model
from trajpred.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--skip-overfit-check", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg["stage"], cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['name']}: stage {cfg['stage']}, {n_params / 1e6:.2f}M parameters")

    trainer = Trainer(model, build_loss(cfg), cfg)
    print(f"device: {trainer.device}")

    if not args.skip_overfit_check:
        first, last = trainer.overfit_one_batch()
        print(f"overfit-one-batch: loss {first:.4f} -> {last:.4f}")
        assert last < 0.05 * first, "model failed to memorize one small batch — fix before training"

    best = trainer.fit()
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
