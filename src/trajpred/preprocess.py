"""Raw AV2 scenario -> canonical agent-centric npz bundle.

Done once, offline: parsing parquet + map JSON per batch would starve the GPU.
Everything is expressed in the focal frame (focal agent at the origin at the
last observed step, heading along +x), so models never see city coordinates.

Bundle keys (N <= MAX_AGENTS, L <= MAX_LANES):
    agent_hist            (N, 50, 2) f32   focal frame; row 0 = focal agent
    agent_mask            (N, 50)    bool  True where observed
    lanes                 (L, 20, 2) f32   centerlines, uniform arc-length
    lane_mask             (L,)       bool  True for real lanes
    lane_is_intersection  (L,)       bool
    future                (60, 2)    f32   focal GT future, focal frame
    future_mask           (60,)      bool
    final_heading         ()         f32   focal heading at t=109 minus t=49, wrapped
    origin                (2,)       f64   world position of the frame origin
    heading               ()         f64   world heading of the frame +x axis
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from trajpred.geometry import to_frame
from trajpred.scenario import load_scenario

OBS_LEN = 50
FUT_LEN = 60
MAX_AGENTS = 64
MAX_LANES = 128
LANE_POINTS = 20

# Anything that changes the bundle contents belongs in here: the hash is
# stored next to the bundles and checked at load time, so a stale cache
# can't silently poison a comparison.
CONFIG = {
    "version": 1,
    "obs_len": OBS_LEN,
    "fut_len": FUT_LEN,
    "max_agents": MAX_AGENTS,
    "max_lanes": MAX_LANES,
    "lane_points": LANE_POINTS,
}


def config_hash() -> str:
    return hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def resample_polyline(line: np.ndarray, n: int) -> np.ndarray:
    """Resample a (P, 2) polyline to n points spaced uniformly by arc length."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0.0:
        return np.repeat(line[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])], axis=-1)


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def preprocess_scenario(scenario_dir: Path) -> dict[str, np.ndarray] | None:
    """Returns the bundle, or None if the focal track is not fully observed."""
    raw = load_scenario(scenario_dir)
    focal = raw.focal_idx
    if np.isnan(raw.positions[focal]).any() or np.isnan(raw.headings[focal]).any():
        return None

    origin = raw.positions[focal, OBS_LEN - 1].copy()
    heading = float(raw.headings[focal, OBS_LEN - 1])
    pos = to_frame(raw.positions, origin, heading)  # (T_tracks, 110, 2)

    # agents: focal first, then the nearest others that are visible at t=49
    hist = pos[:, :OBS_LEN]
    dist = np.linalg.norm(hist[:, OBS_LEN - 1], axis=-1)  # NaN if unobserved at t=49
    others = [i for i in np.argsort(dist) if i != focal and not np.isnan(dist[i])]
    order = [focal] + others[: MAX_AGENTS - 1]
    agent_hist = hist[order]
    agent_mask = ~np.isnan(agent_hist[..., 0])
    agent_hist = np.nan_to_num(agent_hist, nan=0.0)

    # lanes: fixed 20 points each, nearest MAX_LANES by closest centerline point
    if raw.lane_centerlines:
        lanes = np.stack(
            [resample_polyline(cl, LANE_POINTS) for cl in raw.lane_centerlines]
        )
        lanes = to_frame(lanes, origin, heading)
        lane_dist = np.linalg.norm(lanes, axis=-1).min(axis=-1)
        keep = np.argsort(lane_dist)[:MAX_LANES]
        lanes = lanes[keep]
        is_intersection = np.array(raw.lane_is_intersection, dtype=bool)[keep]
    else:
        lanes = np.zeros((0, LANE_POINTS, 2))
        is_intersection = np.zeros((0,), dtype=bool)

    return {
        "agent_hist": agent_hist.astype(np.float32),
        "agent_mask": agent_mask,
        "lanes": lanes.astype(np.float32),
        "lane_mask": np.ones(len(lanes), dtype=bool),
        "lane_is_intersection": is_intersection,
        "future": pos[focal, OBS_LEN:].astype(np.float32),
        "future_mask": np.ones(FUT_LEN, dtype=bool),
        "final_heading": np.float32(_wrap(raw.headings[focal, -1] - heading)),
        "origin": origin,
        "heading": np.float64(heading),
    }
