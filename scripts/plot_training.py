"""Collect per-epoch training logs into results/ and plot the learning curves.

    python scripts/plot_training.py

Each training run writes checkpoints/<name>/log.csv (gitignored with the
weights). This copies them to results/training_logs/<name>.csv and draws val
minFDE per epoch: the 15k-subset runs on the left, the full-data runs on the right.
"""

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from trajpred.config import REPO_ROOT

RESULTS = REPO_ROOT / "results"


def main() -> None:
    out = RESULTS / "training_logs"
    out.mkdir(parents=True, exist_ok=True)
    logs = {}
    for path in sorted((REPO_ROOT / "checkpoints").glob("*/log.csv")):
        shutil.copyfile(path, out / f"{path.parent.name}.csv")
        logs[path.parent.name] = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for name, log in logs.items():
        ax = axes[1] if name.startswith("s3_full") else axes[0]
        ax.plot(log["epoch"], log["val_minFDE"], marker=".", label=f"{name} ({log['val_minFDE'].min():.2f})")
    for ax, title in zip(axes, ("trained on the 15k subset", "trained on all 199,908 scenarios")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(frameon=False, fontsize=8, title="run (best val minFDE, m)", title_fontsize=8)
    axes[0].set_ylabel("val minFDE (m), frozen 5k val")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "training_curves.png", dpi=110)
    print(f"{len(logs)} logs -> {out}")


if __name__ == "__main__":
    main()
