"""How much does the model lean on a *perfect* map?

    python scripts/map_robustness.py --config configs/s3_full.yaml

HD-map lanes are centimeter-accurate; lanes perceived online are not. Without
retraining anything, degrade the map at test time in two ways and re-score on
the frozen val subset:

  shift    every lane is rigidly displaced by its own N(0, sigma^2) offset
  dropout  each lane is removed with probability p

Writes results/map_robustness.csv (rows are replaced per stage) and
results/figures/map_robustness.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from trajpred.config import REPO_ROOT, load_config
from trajpred.dataset import AV2Dataset
from trajpred.env import pick_device, processed_dir
from trajpred.evaluation import PredictFn, evaluate_dataset, load_predictor

RESULTS = REPO_ROOT / "results"
SHIFT_SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]  # meters
DROPOUT_PS = [0.0, 0.25, 0.5, 0.75, 1.0]
METRICS = ["minADE", "minFDE", "MR", "brier_minFDE"]


def degraded(predict: PredictFn, kind: str, level: float, seed: int = 0) -> PredictFn:
    """Wrap a predictor so it sees a degraded map. Seeded, so runs are repeatable."""
    gen = torch.Generator().manual_seed(seed)

    def wrapped(batch):
        batch = dict(batch)
        lanes, mask = batch["lanes"], batch["lane_mask"]
        if kind == "shift":
            offset = torch.randn(*lanes.shape[:2], 1, 2, generator=gen) * level
            batch["lanes"] = lanes + offset.to(lanes.device)
        else:
            keep = torch.rand(mask.shape, generator=gen) >= level
            batch["lane_mask"] = mask & keep.to(mask.device)
        return predict(batch)

    return wrapped


def plot(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, kind, xlabel in zip(
        axes, ("shift", "dropout"), ("lane position error, sigma (m)", "share of lanes removed")
    ):
        for stage, sub in table[table["kind"] == kind].groupby("stage"):
            ax.plot(sub["level"], sub["minFDE"], marker="o", label=stage)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("minFDE$_6$ (m)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "map_robustness.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    dataset = AV2Dataset(processed_dir("val"))
    predict = load_predictor(cfg, device)

    rows = []
    for kind, levels in (("shift", SHIFT_SIGMAS), ("dropout", DROPOUT_PS)):
        for level in levels:
            per = evaluate_dataset(degraded(predict, kind, level), dataset, device)
            rows.append(
                {"stage": cfg["name"], "kind": kind, "level": level}
                | {m: round(float(per[m].mean()), 4) for m in METRICS}
            )
            print(rows[-1])

    path = RESULTS / "map_robustness.csv"
    table = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[old["stage"] != cfg["name"]], table], ignore_index=True)
    table.to_csv(path, index=False)
    plot(table)


if __name__ == "__main__":
    main()
