"""Build an Argoverse 2 motion-forecasting leaderboard submission.

    python scripts/download_subset.py --split test --all
    python scripts/preprocess.py --split test_full
    python scripts/make_submission.py --config configs/s3_full.yaml

Predictions are made in the focal frame, rotated back to city coordinates, and
written in the devkit's ChallengeSubmission parquet layout: one row per
(scenario, track, mode) with columns scenario_id, track_id, probability,
predicted_trajectory_x, predicted_trajectory_y. Upload the file at
https://eval.ai/web/challenges/challenge-page/1719 (needs an EvalAI account).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajpred.config import load_config
from trajpred.dataset import AV2Dataset, collate
from trajpred.env import data_root, pick_device, processed_dir, raw_dir
from trajpred.evaluation import load_predictor, to_device
from trajpred.geometry import from_frame


def focal_track_id(scenario_dir: Path) -> str:
    path = next(scenario_dir.glob("scenario_*.parquet"))
    return pq.read_table(path, columns=["focal_track_id"]).column(0)[0].as_py()


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", default="test_full")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    dataset = AV2Dataset(processed_dir(args.split), cache=False)
    predict = load_predictor(cfg, device)

    rows = []
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate)
    for b, batch in enumerate(tqdm(loader, desc=args.split)):
        traj, probs = predict(to_device(batch, device))
        traj, probs = traj.cpu().numpy().astype(np.float64), probs.cpu().numpy().astype(np.float64)
        for j in range(len(traj)):
            i = b * loader.batch_size + j
            sid = dataset.scenario_id(i)
            with np.load(dataset.paths[i]) as f:
                world = from_frame(traj[j], f["origin"], float(f["heading"]))  # (K, 60, 2)
            p_norm = probs[j] / probs[j].sum()  # float32 softmax can miss 1.0 by ~1e-7
            track = focal_track_id(raw_dir(args.split) / sid)
            for k in range(len(world)):
                rows.append((sid, track, p_norm[k], world[k, :, 0], world[k, :, 1]))

    out = data_root() / "submissions" / f"{cfg['name']}_{args.split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["scenario_id", "track_id", "probability", "predicted_trajectory_x", "predicted_trajectory_y"],
    ).to_parquet(out)
    print(f"{len(dataset)} scenarios x {len(world)} modes -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
