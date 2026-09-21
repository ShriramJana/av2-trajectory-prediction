"""Sample a fixed-seed subset of AV2 motion forecasting scenarios and download them.

The bucket is public; we use unsigned (anonymous) requests. Scenario IDs are
listed first, shuffled with a fixed seed, and the chosen subset is written to a
manifest file so the subset is frozen and reproducible.

With --all, every scenario of the split is downloaded to raw/<split>_full
instead, leaving the frozen subset and its manifest untouched.
"""

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from trajpred.env import data_root

BUCKET = "argoverse"
PREFIX = "datasets/av2/motion-forecasting"
MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


def make_client():
    return boto3.client(
        "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=64)
    )


def list_scenario_ids(s3, split: str) -> list[str]:
    """List all scenario directory names under a split via paginated prefix listing."""
    ids = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix=f"{PREFIX}/{split}/", Delimiter="/"
    ):
        for p in page.get("CommonPrefixes", []):
            ids.append(p["Prefix"].rstrip("/").rsplit("/", 1)[-1])
    assert len(ids) > 1000, f"listing looks wrong: {len(ids)} ids"
    return sorted(ids)


def download_scenario(s3, split: str, sid: str, out_root: Path) -> str:
    """Download every file in one scenario directory. Skips if already complete."""
    dest = out_root / sid
    keys = []
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/{split}/{sid}/")
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    if dest.is_dir() and len(list(dest.iterdir())) == len(keys):
        return sid
    dest.mkdir(parents=True, exist_ok=True)
    for key in keys:
        s3.download_file(BUCKET, key, str(dest / key.rsplit("/", 1)[-1]))
    return sid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--n", type=int, help="subset size (required unless --all)")
    p.add_argument("--all", action="store_true", help="whole split -> raw/<split>_full")
    p.add_argument("--out", default=str(data_root() / "raw"))
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    if args.all == (args.n is not None):
        p.error("give exactly one of --n or --all")

    s3 = make_client()
    out_root = Path(args.out) / (f"{args.split}_full" if args.all else args.split)
    # manifests live in the repo so every machine downloads the identical subset
    manifest = MANIFEST_DIR / f"{args.split}_ids.txt"

    if args.all:
        chosen = list_scenario_ids(s3, args.split)
        print(f"{len(chosen)} scenarios in {args.split}", file=sys.stderr)
    elif manifest.exists():
        chosen = manifest.read_text().split()
        print(f"using existing manifest ({len(chosen)} ids)", file=sys.stderr)
        assert len(chosen) == args.n, "manifest size != --n; delete it to resample"
    else:
        ids = list_scenario_ids(s3, args.split)
        print(f"{len(ids)} scenarios in {args.split}", file=sys.stderr)
        random.Random(42).shuffle(ids)
        chosen = sorted(ids[: args.n])
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(chosen))

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_scenario, s3, args.split, sid, out_root): sid
            for sid in chosen
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.split):
            sid = futures[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - record and continue
                failed.append((sid, repr(e)))

    if failed:
        print(f"{len(failed)} scenarios failed; rerun to retry:", file=sys.stderr)
        for sid, err in failed[:10]:
            print(f"  {sid}: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"downloaded {len(chosen)} scenarios to {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
