"""Two things a planner cares about that minFDE does not measure.

    python scripts/deployment_metrics.py --config configs/s3_full.yaml

off-lane rate   share of predicted endpoints more than 3 m from every lane
                centerline in the scene. Reported for all modes, for the most
                probable mode, and for the ground truth itself: real cars also
                end up off the lane graph (driveways, parking lots, lanes beyond
                the 128 we keep), so the GT rate is the floor to compare against.
latency         milliseconds per scene, batch 1 (online use) and batch 64.

Writes one row per stage to results/deployment_metrics.csv.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trajpred.config import REPO_ROOT, load_config
from trajpred.dataset import AV2Dataset, collate
from trajpred.env import pick_device, processed_dir
from trajpred.evaluation import load_predictor, to_device

RESULTS = REPO_ROOT / "results"
OFF_LANE_THRESHOLD = 3.0  # meters; a lane is ~3.5 m wide, so this is "outside the adjacent lane"


def distance_to_lanes(points: Tensor, lanes: Tensor, lane_mask: Tensor) -> Tensor:
    """Distance from each point (B, P, 2) to the nearest valid centerline (B, L, 20, 2),
    measured to the line segments, not just the vertices. Returns (B, P)."""
    a, b = lanes[:, :, :-1], lanes[:, :, 1:]  # segment ends (B, L, 19, 2)
    ab = b - a
    ap = points[:, :, None, None, :] - a[:, None]  # (B, P, L, 19, 2)
    t = (ap * ab[:, None]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)[:, None]
    closest = a[:, None] + t.clamp(0, 1)[..., None] * ab[:, None]
    dist = torch.linalg.norm(points[:, :, None, None, :] - closest, dim=-1)  # (B, P, L, 19)
    dist = dist.masked_fill(~lane_mask[:, None, :, None], float("inf"))
    return dist.flatten(2).min(dim=-1).values


@torch.no_grad()
def off_lane_rates(predict, dataset: AV2Dataset, device: torch.device) -> dict[str, float]:
    total = {"all_modes": 0.0, "top_mode": 0.0, "ground_truth": 0.0}
    for batch in DataLoader(dataset, batch_size=64, collate_fn=collate):
        batch = to_device(batch, device)
        traj, probs = predict(batch)
        end = traj[:, :, -1]  # (B, K, 2)
        top = end[torch.arange(len(end)), probs.argmax(-1)][:, None]
        gt = batch["future"][:, None, -1]
        for key, pts in (("all_modes", end), ("top_mode", top), ("ground_truth", gt)):
            off = distance_to_lanes(pts, batch["lanes"], batch["lane_mask"]) > OFF_LANE_THRESHOLD
            total[key] += off.float().mean(dim=1).sum().item()
    return {f"off_lane_{k}": round(v / len(dataset), 4) for k, v in total.items()}


@torch.no_grad()
def latency_ms(predict, dataset: AV2Dataset, device: torch.device, batch_size: int) -> float:
    """Median wall-clock per scene over 100 batches, after warm-up, excluding data loading."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    batches = [to_device(b, device) for b, _ in zip(loader, range(110))]
    times = []
    for i, batch in enumerate(batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 10:  # first calls pay for kernel selection and allocation
            times.append((time.perf_counter() - start) / batch_size)
    return round(1000 * sorted(times)[len(times) // 2], 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--skip-latency", action="store_true", help="when the GPU is busy")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    dataset = AV2Dataset(processed_dir("val"))
    predict = load_predictor(cfg, device)

    row = {"stage": cfg["name"]} | off_lane_rates(predict, dataset, device)
    if not args.skip_latency:
        row |= {
            "device": torch.cuda.get_device_name() if device.type == "cuda" else device.type,
            "ms_per_scene_batch1": latency_ms(predict, dataset, device, 1),
            "ms_per_scene_batch64": latency_ms(predict, dataset, device, 64),
        }
    print(row)

    path = RESULTS / "deployment_metrics.csv"
    table = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[old["stage"] != cfg["name"]], table], ignore_index=True)
    table.to_csv(path, index=False)


if __name__ == "__main__":
    main()
