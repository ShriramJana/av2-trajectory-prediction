"""Evaluate any stage on the frozen val subset and record one results row.

    python scripts/evaluate.py --stage s0
    python scripts/evaluate.py --config configs/s1_gru.yaml
    python scripts/evaluate.py --config configs/s3_polyline.yaml --save-figs 12
    python scripts/evaluate.py --config configs/s3_full.yaml --val-split val_full

Every stage goes through this one script on the same scenarios, so the rows of
results/val_metrics.csv are directly comparable. Re-evaluating a stage replaces
its row. Per-scenario metrics land in results/per_scenario/ for error analysis.
Rows evaluated on the whole official val split are named <stage>@val_full.
"""

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
import torch

from trajpred.config import REPO_ROOT, load_config
from trajpred.dataset import AV2Dataset, collate
from trajpred.env import pick_device, processed_dir
from trajpred.evaluation import PredictFn, evaluate_dataset, load_predictor, to_device
from trajpred.models.constant_velocity import constant_velocity

RESULTS = REPO_ROOT / "results"
COLUMNS = ["stage", "K", "minADE", "minFDE", "MR", "brier_minFDE", "n_scenarios", "date"]
METRICS = ["minADE", "minFDE", "MR", "brier_minFDE"]


def write_row(row: dict) -> None:
    path = RESULTS / "val_metrics.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    # re-evaluating a stage replaces its row in place; new stages go at the end
    stages = [r["stage"] for r in rows]
    if row["stage"] in stages:
        rows[stages.index(row["stage"])] = {k: str(v) for k, v in row.items()}
    else:
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_per_scenario(name: str, dataset: AV2Dataset, per: dict[str, np.ndarray]) -> None:
    out = RESULTS / "per_scenario" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", *METRICS, "best_mode"])
        for i in range(len(dataset)):
            writer.writerow(
                [dataset.scenario_id(i), *(f"{per[m][i]:.4f}" for m in METRICS), per["best_mode"][i]]
            )


@torch.no_grad()
def save_worst_figs(
    name: str, n: int, predict: PredictFn, dataset: AV2Dataset, per: dict, device: torch.device
) -> None:
    """Plot the n scenarios with the worst minFDE: failures teach more than successes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trajpred.viz import plot_scene

    out = RESULTS / "figures" / name
    out.mkdir(parents=True, exist_ok=True)
    worst = np.argsort(per["minFDE"])[::-1][:n]
    traj, probs = predict(to_device(collate([dataset[i] for i in worst]), device))
    for rank, (i, t, pr) in enumerate(zip(worst, traj.cpu().numpy(), probs.cpu().numpy())):
        ax = plot_scene(dataset[i], t, pr)
        ax.set_title(f"{name} | {dataset.scenario_id(i)[:8]} | minFDE {per['minFDE'][i]:.1f} m")
        ax.figure.savefig(out / f"worst_{rank:02d}_{dataset.scenario_id(i)[:8]}.png", dpi=80, bbox_inches="tight")
        plt.close(ax.figure)
    print(f"saved {len(worst)} figures to {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["s0"], help="parameter-free stages")
    p.add_argument("--config", type=Path, help="yaml of a trained stage")
    p.add_argument("--ckpt", type=Path, help="defaults to <ckpt_dir>/best.pt")
    p.add_argument("--val-split", default="val", choices=["val", "val_full"])
    p.add_argument("--save-figs", type=int, default=0, metavar="N", help="plot the N worst scenarios")
    args = p.parse_args()
    if (args.stage is None) == (args.config is None):
        p.error("give exactly one of --stage or --config")

    device = pick_device()
    dataset = AV2Dataset(processed_dir(args.val_split), cache=False)

    if args.stage == "s0":
        name, k, predict = "s0", 1, constant_velocity
    else:
        cfg = load_config(args.config)
        name, k = cfg["name"], cfg["model"].get("k", 1)
        predict = load_predictor(cfg, device, args.ckpt)

    if args.val_split != "val":
        name = f"{name}@{args.val_split}"
    per = evaluate_dataset(predict, dataset, device)

    row = {
        "stage": name,
        "K": k,
        **{m: f"{per[m].mean():.4f}" for m in METRICS},
        "n_scenarios": len(dataset),
        "date": date.today().isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    write_row(row)
    write_per_scenario(name, dataset, per)
    print(row)
    if k > 1:
        wins = np.bincount(per["best_mode"], minlength=k) / len(dataset)
        print("share of scenarios won by each mode:", np.round(wins, 3))
    if args.save_figs:
        save_worst_figs(name, args.save_figs, predict, dataset, per, device)


if __name__ == "__main__":
    main()
