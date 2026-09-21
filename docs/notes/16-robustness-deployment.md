# Beyond minFDE: map robustness and deployment metrics — notes

minFDE says how close the best guess was. It does not say whether the model
survives an imperfect map, whether its guesses are drivable, or whether it is
fast enough to run in a car. Three cheap, evaluation-only experiments
(no retraining), all on the frozen 5k val subset.

## 1. What if the map is imperfect? (`scripts/map_robustness.py`)

HD-map lanes are centimeter-accurate. Lanes perceived online from cameras are
noisy and incomplete. Degrade the map *at test time only*:

**Lane position error** — every lane rigidly shifted by its own N(0, σ²) offset:

| σ (m) | 0 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| s3_full minFDE₆ | 1.88 | 1.88 | 1.89 | 1.92 | 2.05 |

**Missing lanes** — each lane removed with probability p:

| p | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| s3_full minFDE₆ | 1.88 | 2.12 | 2.63 | 3.75 | 7.63 |

Reading:
- **Position noise barely matters.** A full meter of lane misplacement costs
  0.05 m. The model uses the map for *topology* (is there a turn lane here,
  where does it lead), not for centimeter geometry.
- **Missing lanes hurt a lot, and total loss is catastrophic:** 7.63 m with no
  map, far worse than the model that was *trained* without a map (3.24 m, and
  that one only saw 15k scenes). A model trained on perfect maps learns to lean
  on them completely and has no fallback.
- The standard remedy, not done here, is to train with the degradation you expect
  at test time (random lane dropout / jitter as augmentation). The experiment
  shows it would be needed before pairing this model with perceived lanes.

## 2. Do the predictions stay on the road? (`scripts/deployment_metrics.py`)

Off-lane rate = share of predicted endpoints more than 3 m from every lane
centerline (distance to the line segments, not just vertices). The ground truth
itself is off-lane 4.4% of the time (driveways, parking lots, lanes beyond the
128 nearest that we keep), so 4.4% is the floor, not 0.

| model | all 6 modes | most probable mode |
|---|---|---|
| ground truth | 4.4% | — |
| s2 (no map) | 21.1% | 10.1% |
| s3 without map | 11.0% | 8.7% |
| s3 | 5.7% | 5.7% |
| s3_full | 4.7% | 4.4% |

With a map and enough data the predictions are as lane-compliant as reality.
Without a map, one in five of S2's modes ends somewhere no car could be — which
minFDE never punishes, because it only scores the best mode.

Caveat: this checks endpoints against centerlines only. It is not a full
kinematic-feasibility or drivable-area check.

## 3. Latency

See `results/deployment_metrics.csv` (measured on an idle RTX 4060, median of
100 batches after warm-up, model forward only — excludes preprocessing).
