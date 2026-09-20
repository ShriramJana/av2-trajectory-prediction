"""Convert every raw scenario of a split into an npz bundle."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from trajpred.env import processed_dir, raw_dir
from trajpred.preprocess import CONFIG, config_hash, preprocess_scenario


def _work(args: tuple[Path, Path]) -> bool:
    scenario_dir, out_dir = args
    bundle = preprocess_scenario(scenario_dir)
    if bundle is None:
        return False
    # uncompressed: bundles are small and load time matters more than disk
    np.savez(out_dir / f"{scenario_dir.name}.npz", **bundle)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = p.parse_args()

    out_dir = processed_dir(args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_dirs = sorted(d for d in raw_dir(args.split).iterdir() if d.is_dir())

    jobs = [(d, out_dir) for d in scenario_dirs]
    with ProcessPoolExecutor(args.workers) as pool:
        ok = list(tqdm(pool.map(_work, jobs, chunksize=32), total=len(jobs)))

    (out_dir / "_meta.json").write_text(
        json.dumps({"config": CONFIG, "hash": config_hash(), "n": sum(ok)}, indent=2)
    )
    print(
        f"{args.split}: wrote {sum(ok)} bundles, skipped {len(ok) - sum(ok)} -> {out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
