# AV2 motion forecasting schema (verified against real data, 2026-08-19)

## Scenario parquet (long format, one row per track per timestep)

Columns: `observed, track_id, object_type, object_category, timestep,
position_x, position_y, heading, velocity_x, velocity_y, scenario_id,
start_timestamp, end_timestamp, num_timestamps, focal_track_id, city`

- 110 timesteps at 10 Hz (0–49 observed, 50–109 future).
- Positions are meters in a **city frame** (values can be hundreds of meters).
- `focal_track_id` names the scored track; it is fully observed in train/val.
- `object_type`: vehicle, pedestrian, bus, background, static, riderless_bicycle…
- `object_category` 0–3 = ascending track quality (3 = focal).
- Non-focal tracks are partially observed → NaN after pivoting to dense arrays.

## Map JSON (`log_map_archive_*.json`)

Top keys: `drivable_areas`, `lane_segments`, `pedestrian_crossings`.

`lane_segments` is a dict id → {`centerline` (list of {x,y,z} points, variable
count), `is_intersection`, `lane_type` (VEHICLE/BIKE/BUS), `left/right_lane_boundary`,
`left/right_neighbor_id`, `predecessors`, `successors`, mark types}.

We currently use only centerlines + `is_intersection`; connectivity
(predecessors/successors) is available if a future model wants a lane graph.
