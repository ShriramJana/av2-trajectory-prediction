# Trajectory Prediction on Argoverse 2

Predicting 6-second vehicle futures from 5 seconds of history on the Argoverse 2
Motion Forecasting dataset, built as a staged ladder — each stage adds one idea
and one measurable improvement on the official metrics (minADE₆, minFDE₆,
Miss Rate, brier-minFDE₆):

1. Constant-velocity baseline
2. GRU encoder–decoder (unimodal)
3. K=6 modes with winner-take-all training
4. VectorNet-style polyline transformer (map + social context)
5. Ablations and error analysis

**Status: work in progress** — results table lands when the ladder is complete.
Design and plan live in [docs/](docs/).

## Setup

```bash
# any machine (Windows / macOS / Linux), Python 3.12
python3.12 -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch             # Windows+NVIDIA: --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"

# data: ~20k scenarios (a few GB), frozen subset defined by manifests/
python scripts/download_subset.py --split val --n 5000
python scripts/download_subset.py --split train --n 15000

pytest
```

Data location defaults to `C:\data\av2` (Windows) or `~/data/av2` (macOS/Linux);
override with the `TRAJPRED_DATA` environment variable. Training device is
auto-selected: CUDA → Apple MPS → CPU.
