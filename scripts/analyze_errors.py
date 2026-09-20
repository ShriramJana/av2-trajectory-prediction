"""Where do the errors live? Break per-scenario metrics down by maneuver,
speed and scene density, and draw side-by-side comparisons between stages.

    python scripts/analyze_errors.py
    python scripts/analyze_errors.py --compare configs/s2_multimodal.yaml configs/s3_polyline.yaml

Consumes results/per_scenario/<stage>.csv (written by evaluate.py). Writes
results/breakdown.csv, results/figures/breakdown.png and, with --compare,
results/figures/<a>_vs_<b>.png showing the scenarios where b improves most on a.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from trajpred.config import REPO_ROOT, load_config
from trajpred.dataset import AV2Dataset, collate
from trajpred.env import pick_device, processed_dir
from trajpred.evaluation import load_predictor, to_device
from trajpred.viz import plot_scene

RESULTS = REPO_ROOT / "results"
METRICS = ["minADE", "minFDE", "MR", "brier_minFDE"]
TURN_THRESHOLD = np.deg2rad(30)


def scenario_tags(dataset: AV2Dataset) -> pd.DataFrame:
    """Tag each val scenario from ground truth: maneuver, speed band, density."""
    rows = []
    for i in range(len(dataset)):
        b = dataset[i]
        speed = np.linalg.norm(b["agent_hist"][0, 49] - b["agent_hist"][0, 48]) * 10  # m/s
        rows.append(
            {
                "scenario_id": dataset.scenario_id(i),
                "maneuver": "turning" if abs(b["final_heading"]) > TURN_THRESHOLD else "straight",
                "speed": "0-2 m/s" if speed < 2 else "2-8 m/s" if speed < 8 else "8+ m/s",
                "agents": "1-10" if len(b["agent_hist"]) <= 10
                else "11-30" if len(b["agent_hist"]) <= 30 else "31+",
            }
        )
    return pd.DataFrame(rows)


def breakdown(tags: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for path in sorted((RESULTS / "per_scenario").glob("*.csv")):
        per = pd.read_csv(path).merge(tags, on="scenario_id")
        for group in ("maneuver", "speed", "agents"):
            t = per.groupby(group)[METRICS].mean().round(3)
            t.insert(0, "n", per.groupby(group).size())
            t = t.reset_index().rename(columns={group: "bucket"})
            t.insert(0, "group", group)
            t.insert(0, "stage", path.stem)
            tables.append(t)
    return pd.concat(tables, ignore_index=True)


def plot_breakdown(table: pd.DataFrame, stages: list[str]) -> None:
    groups = ["maneuver", "speed", "agents"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(stages)))
    for ax, group in zip(axes, groups):
        sub = table[table["group"] == group]
        buckets = list(dict.fromkeys(sub["bucket"]))
        width = 0.8 / len(stages)
        for j, (stage, color) in enumerate(zip(stages, colors)):
            vals = sub[sub["stage"] == stage].set_index("bucket")["minFDE"].reindex(buckets)
            ax.bar(np.arange(len(buckets)) + j * width, vals, width, label=stage, color=color)
        counts = sub[sub["stage"] == stages[0]].set_index("bucket")["n"].reindex(buckets)
        ax.set_xticks(np.arange(len(buckets)) + 0.4 - width / 2)
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(buckets, counts)])
        ax.set_title(f"by {group}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("minFDE$_6$ (m)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "breakdown.png", dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_comparison(cfg_a: dict, cfg_b: dict, dataset: AV2Dataset, tags: pd.DataFrame, n: int = 4) -> None:
    """Top row: stage a. Bottom row: stage b, on the n turning scenarios where b gains most."""
    device = pick_device()
    per = {
        c["name"]: pd.read_csv(RESULTS / "per_scenario" / f"{c['name']}.csv").set_index("scenario_id")
        for c in (cfg_a, cfg_b)
    }
    a, b = cfg_a["name"], cfg_b["name"]
    gain = (per[a]["minFDE"] - per[b]["minFDE"])[tags.set_index("scenario_id")["maneuver"] == "turning"]
    chosen = gain.sort_values(ascending=False).index[:n]
    index_of = {dataset.scenario_id(i): i for i in range(len(dataset))}
    idx = [index_of[sid] for sid in chosen]
    batch = to_device(collate([dataset[i] for i in idx]), device)

    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.6))
    for row, cfg in enumerate((cfg_a, cfg_b)):
        traj, probs = load_predictor(cfg, device)(batch)
        for col, (i, sid) in enumerate(zip(idx, chosen)):
            ax = plot_scene(dataset[i], traj[col].cpu().numpy(), probs[col].cpu().numpy(), ax=axes[row, col])
            ax.set_title(f"{cfg['name']} | {sid[:8]} | minFDE {per[cfg['name']].loc[sid, 'minFDE']:.1f} m", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / f"{a}_vs_{b}.png", dpi=90)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compare", nargs=2, type=Path, metavar=("CONFIG_A", "CONFIG_B"))
    args = p.parse_args()

    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    dataset = AV2Dataset(processed_dir("val"))
    tags = scenario_tags(dataset)

    table = breakdown(tags)
    table.to_csv(RESULTS / "breakdown.csv", index=False)
    stages = [s for s in dict.fromkeys(table["stage"]) if s not in ("s0", "s1")]  # K=6 stages only
    plot_breakdown(table, stages)
    print(table[table["group"] == "maneuver"].to_string(index=False))

    if args.compare:
        plot_comparison(load_config(args.compare[0]), load_config(args.compare[1]), dataset, tags)


if __name__ == "__main__":
    main()
