"""Evaluate any stage on the frozen val subset and record one results row.

    python scripts/evaluate.py --stage s0
    python scripts/evaluate.py --config configs/s1_gru.yaml

Every stage goes through this one script on the same scenarios, so the rows of
results/val_metrics.csv are directly comparable. Re-evaluating a stage replaces
its row. Per-scenario metrics land in results/per_scenario/ for error analysis.
"""

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
import torch

from trajpred.config import REPO_ROOT, load_config
from trajpred.dataset import AV2Dataset
from trajpred.env import pick_device, processed_dir
from trajpred.evaluation import evaluate_dataset, model_predict_fn
from trajpred.models import build_model
from trajpred.models.constant_velocity import constant_velocity

RESULTS = REPO_ROOT / "results"
COLUMNS = ["stage", "K", "minADE", "minFDE", "MR", "brier_minFDE", "n_scenarios", "date"]
METRICS = ["minADE", "minFDE", "MR", "brier_minFDE"]


def write_row(row: dict) -> None:
    path = RESULTS / "val_metrics.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["stage"] != row["stage"]]
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["s0"], help="parameter-free stages")
    p.add_argument("--config", type=Path, help="yaml of a trained stage")
    p.add_argument("--ckpt", type=Path, help="defaults to <ckpt_dir>/best.pt")
    args = p.parse_args()
    if (args.stage is None) == (args.config is None):
        p.error("give exactly one of --stage or --config")

    device = pick_device()
    dataset = AV2Dataset(processed_dir("val"))

    if args.stage == "s0":
        name, k, predict = "s0", 1, constant_velocity
    else:
        cfg = load_config(args.config)
        name, k = cfg["name"], cfg["model"].get("k", 1)
        model = build_model(cfg["stage"], cfg["model"]).to(device)
        ckpt = args.ckpt or Path(cfg["ckpt_dir"]) / "best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
        model.eval()
        predict = model_predict_fn(model)

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


if __name__ == "__main__":
    main()
