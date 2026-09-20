import numpy as np

from trajpred.env import raw_dir
from trajpred.preprocess import preprocess_scenario, resample_polyline


def test_bundle_contract():
    d = preprocess_scenario(sorted(raw_dir("val").iterdir())[0])
    assert d["agent_hist"].shape[1:] == (50, 2) and d["agent_hist"].shape[0] <= 64
    assert d["agent_mask"].shape == d["agent_hist"].shape[:2]
    assert d["lanes"].shape[1:] == (20, 2) and d["lanes"].shape[0] <= 128
    assert d["lane_mask"].shape == d["lane_is_intersection"].shape == d["lanes"].shape[:1]
    assert d["future"].shape == (60, 2)
    assert d["agent_hist"].dtype == np.float32 and d["agent_mask"].dtype == bool
    # focal agent is row 0, fully observed, and at origin facing +x at t=49
    assert d["agent_mask"][0].all()
    assert np.allclose(d["agent_hist"][0, 49], [0.0, 0.0], atol=1e-6)
    step = d["agent_hist"][0, 49] - d["agent_hist"][0, 48]
    assert abs(np.arctan2(step[1], step[0])) < 0.35 or np.linalg.norm(step) < 0.1
    # every other kept agent is visible at t=49 and sorted nearest-first
    assert d["agent_mask"][:, 49].all()
    dist = np.linalg.norm(d["agent_hist"][1:, 49], axis=-1)
    assert (np.diff(dist) >= 0).all()
    # no NaNs anywhere
    assert not np.isnan(d["agent_hist"]).any()
    assert not np.isnan(d["lanes"]).any()
    assert not np.isnan(d["future"]).any()


def test_resample_is_uniform_in_arc_length():
    # an L-shaped line with very uneven input spacing
    line = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.0, 10.0]])
    out = resample_polyline(line, 21)
    assert out.shape == (21, 2)
    assert np.allclose(out[0], line[0]) and np.allclose(out[-1], line[-1])
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(seg, 1.0, atol=1e-9)  # total length 20 over 20 segments


def test_resample_degenerate_line():
    out = resample_polyline(np.array([[3.0, 4.0], [3.0, 4.0]]), 20)
    assert np.allclose(out, [3.0, 4.0])
