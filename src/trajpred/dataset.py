"""Dataset over preprocessed npz bundles, plus the padding collate.

Scenes are ragged (different numbers of agents and lanes), so a batch pads
every scene up to the batch maximum and carries boolean masks saying which
rows are real. Every model op that mixes rows must consume those masks.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from trajpred.preprocess import config_hash

# per-scene arrays whose first dim is ragged and needs padding
_AGENT_KEYS = ("agent_hist", "agent_mask")
_LANE_KEYS = ("lanes", "lane_mask", "lane_is_intersection")
_FIXED_KEYS = ("future", "final_heading")


class AV2Dataset(Dataset):
    def __init__(self, processed_dir: Path, cache: bool = True):
        processed_dir = Path(processed_dir)
        self.paths = sorted(processed_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no bundles in {processed_dir}; run scripts/preprocess.py")
        meta = json.loads((processed_dir / "_meta.json").read_text())
        if meta["hash"] != config_hash():
            raise RuntimeError(
                f"{processed_dir} was built with preprocessing config {meta['hash']}, "
                f"code is now {config_hash()}; re-run scripts/preprocess.py"
            )
        # the whole subset fits in RAM, so after the first epoch no disk reads
        self._cache: dict[int, dict[str, np.ndarray]] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def scenario_id(self, i: int) -> str:
        return self.paths[i].stem

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        with np.load(self.paths[i]) as f:
            bundle = {k: f[k] for k in _AGENT_KEYS + _LANE_KEYS + _FIXED_KEYS}
        if self._cache is not None:
            self._cache[i] = bundle
        return bundle


def _pad_stack(arrays: list[np.ndarray]) -> torch.Tensor:
    """Stack arrays that differ in dim 0, zero-padding (False for bool) to the max."""
    n_max = max(len(a) for a in arrays)
    out = np.zeros((len(arrays), n_max) + arrays[0].shape[1:], dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return torch.from_numpy(out)


def collate(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    out = {k: _pad_stack([b[k] for b in batch]) for k in _AGENT_KEYS + _LANE_KEYS}
    for k in _FIXED_KEYS:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    return out
