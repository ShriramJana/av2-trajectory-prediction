# Trajectory Prediction on Argoverse 2

Predicting where a vehicle will be over the next 6 seconds from 5 seconds of
history, on the [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html)
dataset. Built as a ladder: each stage adds one idea, and every stage is scored
by the same script on the same frozen validation scenarios, so the table shows
what each idea is worth.

![S2 vs S3 on turning scenarios](results/figures/s2_vs_s3.png)

*Top: six predicted futures (red, labelled with probability) from a model that
sees only the car's own history. Bottom: the same scenes after adding the lane
graph. Black = observed history, green = what actually happened.*

## Results

5,000 validation scenarios. Official AV2 metrics in meters, lower is better.

| Stage | What it adds | K | minADE | minFDE | Miss rate | brier-minFDE |
|---|---|---|---|---|---|---|
| S0 | Constant velocity | 1 | 4.41 | 11.69 | 0.85 | 11.69 |
| S1 | GRU encoder–decoder | 1 | 3.22 | 8.50 | 0.83 | 8.50 |
| S2 | 6 modes, winner-take-all loss | 6 | 1.53 | 3.55 | 0.60 | 4.14 |
| **S3** | **Polyline transformer: map + other agents** | 6 | **1.18** | **2.45** | **0.41** | **3.09** |
| | S3 without the map | 6 | 1.44 | 3.24 | 0.55 | 3.86 |
| | S3 without other agents | 6 | 1.19 | 2.49 | 0.43 | 3.13 |

minFDE = endpoint error of the best of K predictions; miss rate = share of
scenarios where no prediction ends within 2 m; brier-minFDE also penalizes
giving the best prediction a low probability.

**Read this before quoting the numbers:**
- Trained on a **15,000-scenario subset (7.5% of the training set)** and
  evaluated on a 5,000-scenario subset of official val, both drawn with seed 42
  and frozen in [manifests/](manifests/). These are not leaderboard numbers.
- One training run per row. Differences below ~0.1 m are not evidence.
- Checkpoints were picked on the same validation set they are reported on.

## What each stage taught

**S0 → S1: learning helps, a little.** A GRU learns speed profiles that constant
velocity can't (cars that are braking keep braking). But a single prediction
trained by regression converges to the *average* future, and the average of
"turn left" and "go straight" is a path nobody drives. Miss rate barely moves.

**S1 → S2: the biggest win is the output, not the network.** Same encoder, but six
predictions, with only the closest one trained on each example
(winner-take-all), so the modes specialize instead of collapsing onto the mean.
minFDE drops from 8.50 to 3.55 for 40 seconds of training. All six modes stay in
use (each wins 7–26% of scenarios).

**S2 → S3: context.** Agent histories and lane centerlines are both polylines.
Each becomes one token via a small PointNet-style encoder; self-attention over
the tokens mixes them; the focal agent's token feeds the same 6-mode head as S2.
0.92 M parameters.

**Ablations: the map does the work.** Removing the map costs 0.79 m; removing the
other agents costs 0.04 m — indistinguishable from noise here. The map's gain is
concentrated where you'd expect:

![Error breakdown](results/figures/breakdown.png)

On turning scenarios the map is worth 2.4 m; without it, turning error is the
same as S2's. A model can't guess where the road goes. The near-zero effect of
other agents was a surprise; likely explanations (the car's own history already
reflects the traffic around it; 15k scenes are too few to learn rare
interactions) are discussed, not proven, in
[docs/notes/14-analysis.md](docs/notes/14-analysis.md).

**What still fails.** Median endpoint error is 1.7 m, but the worst 10% of
scenarios carry 38% of the total error: turns taken at speed where all six modes
went straight, unexpected stops, and wrong-branch choices at intersections.
Galleries of the 12 worst scenarios per model are in
[results/figures/](results/figures/).

## Engineering notes

- **Agent-centric frame:** every scene is translated and rotated so the focal car
  sits at the origin facing +x. The model never sees city coordinates.
- **Offline preprocessing** to fixed-format `.npz` bundles, versioned by a config
  hash so a stale cache can't be evaluated by accident.
- **Padding is provably inert:** tests fill padded agents, padded lanes and
  unobserved timesteps with garbage and require identical outputs.
- **Metrics cross-checked** against the official `av2` devkit on random inputs
  (max difference 1.4e-14).
- **Overfit-one-batch gate:** every training run must first memorize 16 scenes,
  or it refuses to start.
- 31 tests; one learning note per stage in [docs/notes/](docs/notes/), including
  the predictions that turned out wrong.

## Reproduce

```bash
# Python 3.12, any OS
python3.12 -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch             # Windows+NVIDIA: --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"

# 1. data: ~20k scenarios, a few GB (needs s5cmd; see scripts/download_subset.py)
python scripts/download_subset.py --split val --n 5000
python scripts/download_subset.py --split train --n 15000

# 2. preprocess (~30 s)
python scripts/preprocess.py --split val
python scripts/preprocess.py --split train

# 3. train (S1/S2: under a minute each; S3 variants: ~15 min each on an RTX 4060)
python scripts/train.py --config configs/s1_gru.yaml
python scripts/train.py --config configs/s2_multimodal.yaml
python scripts/train.py --config configs/s3_polyline.yaml
python scripts/train.py --config configs/s3_no_map.yaml
python scripts/train.py --config configs/s3_no_social.yaml

# 4. evaluate -> results/val_metrics.csv
python scripts/evaluate.py --stage s0
python scripts/evaluate.py --config configs/s3_polyline.yaml --save-figs 12   # etc. per config

# 5. error analysis -> results/breakdown.csv + figures
python scripts/analyze_errors.py --compare configs/s2_multimodal.yaml configs/s3_polyline.yaml
```

Data location defaults to `C:\data\av2` (Windows) or `~/data/av2` (macOS/Linux);
override with the `TRAJPRED_DATA` environment variable. Device is auto-selected:
CUDA → Apple MPS → CPU. Design and plan are in [docs/](docs/).

Windows note: if Smart App Control blocks a freshly released `pyarrow` DLL
("An Application Control policy has blocked this file"), install the previous
release (`python -m pip install "pyarrow<25"`).
