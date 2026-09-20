import json

import numpy as np
import pytest
import torch

from trajpred.dataset import AV2Dataset, collate
from trajpred.preprocess import config_hash


def _bundle(n_agents: int, n_lanes: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "agent_hist": rng.normal(size=(n_agents, 50, 2)).astype(np.float32) + 5.0,
        "agent_mask": np.ones((n_agents, 50), dtype=bool),
        "lanes": rng.normal(size=(n_lanes, 20, 2)).astype(np.float32) + 5.0,
        "lane_mask": np.ones(n_lanes, dtype=bool),
        "lane_is_intersection": np.zeros(n_lanes, dtype=bool),
        "future": rng.normal(size=(60, 2)).astype(np.float32),
        "future_mask": np.ones(60, dtype=bool),
        "final_heading": np.float32(0.1),
        "origin": np.zeros(2),
        "heading": np.float64(0.0),
    }


@pytest.fixture
def processed(tmp_path):
    np.savez(tmp_path / "a.npz", **_bundle(3, 4, seed=0))
    np.savez(tmp_path / "b.npz", **_bundle(5, 2, seed=1))
    (tmp_path / "_meta.json").write_text(json.dumps({"hash": config_hash()}))
    return tmp_path


def test_collate_pads_to_batch_max(processed):
    ds = AV2Dataset(processed)
    batch = collate([ds[0], ds[1]])
    assert batch["agent_hist"].shape == (2, 5, 50, 2)
    assert batch["agent_mask"].shape == (2, 5, 50)
    assert batch["lanes"].shape == (2, 4, 20, 2)
    assert batch["lane_mask"].shape == (2, 4)
    assert batch["future"].shape == (2, 60, 2)
    assert batch["final_heading"].shape == (2,)


def test_masks_false_exactly_on_padding(processed):
    ds = AV2Dataset(processed)
    batch = collate([ds[0], ds[1]])
    # scene a has 3 agents / 4 lanes, scene b has 5 agents / 2 lanes
    assert batch["agent_mask"][0, :3].all() and not batch["agent_mask"][0, 3:].any()
    assert batch["agent_mask"][1].all()
    assert batch["lane_mask"][0].all()
    assert batch["lane_mask"][1, :2].all() and not batch["lane_mask"][1, 2:].any()
    # padded values are exactly zero
    assert (batch["agent_hist"][0, 3:] == 0).all()
    assert (batch["lanes"][1, 2:] == 0).all()


def test_dtypes(processed):
    ds = AV2Dataset(processed)
    batch = collate([ds[0], ds[1]])
    for k in ("agent_hist", "lanes", "future", "final_heading"):
        assert batch[k].dtype == torch.float32, k
    for k in ("agent_mask", "lane_mask", "lane_is_intersection"):
        assert batch[k].dtype == torch.bool, k


def test_stale_cache_is_rejected(processed):
    (processed / "_meta.json").write_text(json.dumps({"hash": "not-the-current-hash"}))
    with pytest.raises(RuntimeError, match="re-run"):
        AV2Dataset(processed)
