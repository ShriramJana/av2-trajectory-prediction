import json

import numpy as np
import pandas as pd

from trajpred.env import raw_dir
from trajpred.scenario import load_scenario

RAW_VAL = raw_dir("val")


def test_load_scenario_shapes():
    scenario_dir = sorted(RAW_VAL.iterdir())[0]
    s = load_scenario(scenario_dir)
    assert s.positions.shape[1] == 110 and s.positions.shape[2] == 2
    assert s.headings.shape == s.positions.shape[:2]
    assert len(s.track_ids) == s.positions.shape[0]
    assert len(s.object_types) == s.positions.shape[0]
    assert 0 <= s.focal_idx < len(s.track_ids)
    # focal track is fully observed over the 11 s window
    assert not np.isnan(s.positions[s.focal_idx]).any()
    assert len(s.lane_centerlines) > 0
    assert all(cl.ndim == 2 and cl.shape[1] == 2 for cl in s.lane_centerlines)


def test_unobserved_steps_are_nan():
    # across the first few scenarios, at least one non-focal track should be
    # partially observed (cars enter/leave sensor range)
    found_partial = False
    for scenario_dir in sorted(RAW_VAL.iterdir())[:5]:
        s = load_scenario(scenario_dir)
        nan_frac = np.isnan(s.positions[:, :, 0]).mean()
        if 0 < nan_frac < 1:
            found_partial = True
    assert found_partial


def test_float_typed_timesteps_are_accepted(tmp_path):
    # seen in the official test split: timestep stored as float64
    pd.DataFrame(
        {
            "track_id": ["a", "a", "b"],
            "object_type": ["vehicle", "vehicle", "pedestrian"],
            "timestep": [0.0, 1.0, 1.0],
            "position_x": [1.0, 2.0, 5.0],
            "position_y": [0.0, 0.0, 5.0],
            "heading": [0.0, 0.0, 1.0],
            "focal_track_id": ["a"] * 3,
            "scenario_id": ["s"] * 3,
        }
    ).to_parquet(tmp_path / "scenario_s.parquet")
    lane = {"centerline": [{"x": 0, "y": 0}, {"x": 9, "y": 0}], "is_intersection": False}
    (tmp_path / "log_map_archive_s.json").write_text(json.dumps({"lane_segments": {"1": lane}}))

    s = load_scenario(tmp_path)
    assert s.focal_idx == 0
    assert np.allclose(s.positions[0, :2], [[1.0, 0.0], [2.0, 0.0]])
    assert np.isnan(s.positions[1, 0]).all() and np.allclose(s.positions[1, 1], [5.0, 5.0])
