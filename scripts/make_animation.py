"""Animated demo: history plays, then the six predictions unfold against reality.

    python scripts/make_animation.py --config configs/s3_full.yaml

Scenes are chosen by rule, not by eye: the turning scenarios whose minFDE sits
at the model's median for turns, so the GIF shows typical behaviour. Needs
results/per_scenario/<stage>.csv from evaluate.py.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from trajpred.config import REPO_ROOT, load_config
from trajpred.dataset import AV2Dataset, collate
from trajpred.env import pick_device, processed_dir
from trajpred.evaluation import load_predictor, to_device
from trajpred.viz import plot_scene

RESULTS = REPO_ROOT / "results"
TURN_THRESHOLD = np.deg2rad(30)


def median_turning_scenes(name: str, dataset: AV2Dataset, n: int) -> list[int]:
    per = pd.read_csv(RESULTS / "per_scenario" / f"{name}.csv").set_index("scenario_id")["minFDE"]
    turning = [
        i for i in range(len(dataset)) if abs(dataset[i]["final_heading"]) > TURN_THRESHOLD
    ]
    turning.sort(key=lambda i: per[dataset.scenario_id(i)])
    mid = len(turning) // 2
    return turning[mid - n // 2 : mid - n // 2 + n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    dataset = AV2Dataset(processed_dir("val"))
    idx = median_turning_scenes(cfg["name"], dataset, args.n)
    with torch.no_grad():
        traj, probs = load_predictor(cfg, device)(to_device(collate([dataset[i] for i in idx]), device))
    traj, probs = traj.cpu().numpy(), probs.cpu().numpy()

    # (history steps shown, future steps shown): 5 s of history, then 6 s of future, then hold
    frames = [(h, 0) for h in range(2, 51, 3)] + [(50, f) for f in range(3, 61, 3)] + [(50, 60)] * 8
    fig, axes = plt.subplots(1, args.n, figsize=(4 * args.n, 4.3))

    def draw(frame: tuple[int, int]):
        hist_upto, fut_upto = frame
        for ax, i, t, pr in zip(axes, idx, traj, probs):
            ax.clear()
            plot_scene(dataset[i], t, pr, ax=ax, radius=45, hist_upto=hist_upto, fut_upto=fut_upto)
            clock = (hist_upto - 50) / 10 if fut_upto == 0 else fut_upto / 10
            ax.set_title(f"t = {clock:+.1f} s", fontsize=10)

    fig.tight_layout()
    out = RESULTS / "figures" / f"{cfg['name']}_demo.gif"
    FuncAnimation(fig, draw, frames=frames).save(out, writer=PillowWriter(fps=8), dpi=70)
    plt.close(fig)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB), scenes: {[dataset.scenario_id(i)[:8] for i in idx]}")


if __name__ == "__main__":
    main()
