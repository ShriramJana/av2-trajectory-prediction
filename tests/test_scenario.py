import numpy as np

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
