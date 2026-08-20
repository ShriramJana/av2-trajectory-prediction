"""Load one raw AV2 motion forecasting scenario into dense numpy arrays.

The parquet is long-format (one row per track per timestep). We pivot it into
dense (num_tracks, 110, ...) arrays with NaN wherever a track was not observed,
because models consume fixed-shape tensors plus validity masks, not tables.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

NUM_TIMESTEPS = 110


@dataclass
class RawScenario:
    scenario_id: str
    track_ids: list[str]
    positions: np.ndarray  # (T_tracks, 110, 2) f64, NaN where unobserved
    headings: np.ndarray  # (T_tracks, 110) f64, NaN where unobserved
    object_types: list[str]  # per track (vehicle, pedestrian, ...)
    focal_idx: int  # row index of the focal track
    lane_centerlines: list[np.ndarray]  # each (P_i, 2) f64
    lane_is_intersection: list[bool]  # per lane


def load_scenario(scenario_dir: Path) -> RawScenario:
    scenario_dir = Path(scenario_dir)
    parquet_path = next(scenario_dir.glob("scenario_*.parquet"))
    map_path = next(scenario_dir.glob("log_map_archive_*.json"))

    df = pq.read_table(
        parquet_path,
        columns=[
            "track_id",
            "object_type",
            "timestep",
            "position_x",
            "position_y",
            "heading",
            "focal_track_id",
            "scenario_id",
        ],
    ).to_pandas()

    track_ids = df["track_id"].unique().tolist()
    row_of = {tid: i for i, tid in enumerate(track_ids)}

    positions = np.full((len(track_ids), NUM_TIMESTEPS, 2), np.nan)
    headings = np.full((len(track_ids), NUM_TIMESTEPS), np.nan)
    rows = df["track_id"].map(row_of).to_numpy()
    cols = df["timestep"].to_numpy()
    positions[rows, cols, 0] = df["position_x"].to_numpy()
    positions[rows, cols, 1] = df["position_y"].to_numpy()
    headings[rows, cols] = df["heading"].to_numpy()

    object_types = (
        df.drop_duplicates("track_id").set_index("track_id")["object_type"]
        .loc[track_ids]
        .tolist()
    )
    focal_idx = row_of[df["focal_track_id"].iloc[0]]

    lane_map = json.load(open(map_path))["lane_segments"]
    lane_centerlines = [
        np.array([[pt["x"], pt["y"]] for pt in seg["centerline"]])
        for seg in lane_map.values()
    ]
    lane_is_intersection = [bool(seg["is_intersection"]) for seg in lane_map.values()]

    return RawScenario(
        scenario_id=df["scenario_id"].iloc[0],
        track_ids=track_ids,
        positions=positions,
        headings=headings,
        object_types=object_types,
        focal_idx=focal_idx,
        lane_centerlines=lane_centerlines,
        lane_is_intersection=lane_is_intersection,
    )
