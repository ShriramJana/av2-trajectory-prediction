"""Stage configs: one yaml per trained model, shared by train.py and evaluate.py."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 30,
    "grad_accum": 1,
    "seed": 42,
    "train_split": "train",  # "train_full" = every official training scenario
    "model": {},
}


def load_config(path: Path) -> dict:
    """Read a stage yaml, fill defaults. `name` (results row, checkpoint folder)
    defaults to the file stem; `stage` picks the model class."""
    path = Path(path)
    cfg = {**DEFAULTS, **yaml.safe_load(path.read_text())}
    cfg.setdefault("name", path.stem)
    cfg.setdefault("ckpt_dir", str(REPO_ROOT / "checkpoints" / cfg["name"]))
    return cfg
