# Argoverse 2 Trajectory Prediction Implementation Plan

**Goal:** A staged ladder of trajectory-prediction models (constant velocity → GRU → K-mode → polyline transformer) on an Argoverse 2 subset, evaluated on official metrics, with ablations and error analysis.

**Architecture:** Offline preprocessing converts each AV2 scenario (parquet + map JSON) into an agent-centric `.npz` bundle. A single Dataset/collate feeds all models. Every stage is evaluated by the same script on the same frozen val subset, appending one row to a shared results table.

**Tech Stack:** Python 3.12, PyTorch (CUDA), pyarrow, numpy, pyyaml, matplotlib, pytest. `s5cmd` for S3 download. `av2` devkit optional (metrics cross-check only).

**Spec:** `docs/specs/2026-08-19-trajectory-prediction-design.md`

## Global Constraints

- Python 3.12 via `py -3.12`; project venv at `.venv`; run everything with `.venv\Scripts\python`.
- Data lives at `C:\data\av2` (outside OneDrive). Raw: `C:\data\av2\raw\{train,val}`. Processed: `C:\data\av2\processed\{train,val}`. The repo's `data/` stays empty/gitignored.
- Subset sizes: 15,000 train scenarios, 5,000 val scenarios, sampled with seed 42. The val subset is frozen after Task 2 — never resampled.
- Task protocol: 50 observed steps (5 s), 60 predicted steps (6 s), 10 Hz, focal agent only. K = 6 modes.
- Agent-centric frame: focal agent's position at the last observed step (index 49) is the origin; its heading at that step points +x.
- Metrics: minADE_K, minFDE_K, MR@2m, brier-minFDE (all meters, focal agent only).
- Commit messages: plain conventional style, one commit per task.
- Teaching workflow: each task begins with a concept explainer and ends with a check-your-understanding exercise before the commit.

---

## File Structure

```
pyproject.toml                      package metadata + deps
configs/
  s1_gru.yaml  s2_multimodal.yaml  s3_polyline.yaml
  s3_no_map.yaml  s3_no_social.yaml
scripts/
  download_subset.py                sample + download scenario dirs from S3
  preprocess.py                     raw scenarios -> npz bundles
  train.py                          config-driven training entry point
  evaluate.py                       eval any stage on frozen val -> results row
  analyze_errors.py                 breakdowns by maneuver/speed + figures
src/trajpred/
  __init__.py
  geometry.py                       focal-frame transforms
  metrics.py                        minADE/minFDE/MR/brier
  scenario.py                       parquet + map JSON -> raw numpy arrays
  preprocess.py                     raw arrays -> canonical npz bundle
  dataset.py                        AV2Dataset + collate (padding + masks)
  losses.py                         WTA loss for K-mode training
  trainer.py                        training loop, checkpointing, overfit check
  viz.py                            plot scene: lanes, history, GT, predicted modes
  models/
    __init__.py
    constant_velocity.py            S0
    gru.py                          S1
    multimodal.py                   S2 (K-mode head, shared by S3)
    polyline.py                     S3 (VectorNet-style)
tests/
  test_geometry.py  test_metrics.py  test_preprocess.py
  test_dataset.py   test_losses.py   test_models.py
notebooks/
  01_explore_scenario.ipynb
results/
  val_metrics.csv                   one row per evaluated stage
  figures/
docs/notes/                         learning notes, one per task
```

### npz bundle contract (produced by Task 6, consumed by everything after)

| key | shape | dtype | meaning |
|---|---|---|---|
| `agent_hist` | (N, 50, 2) | f32 | positions in focal frame; index 0 = focal agent |
| `agent_mask` | (N, 50) | bool | True where the position is observed |
| `lanes` | (L, 20, 2) | f32 | lane centerlines resampled to 20 points, focal frame |
| `lane_mask` | (L,) | bool | True for real lanes (padding rows False) |
| `future` | (60, 2) | f32 | focal agent GT future, focal frame |
| `future_mask` | (60,) | bool | True where GT exists (focal is fully observed in train/val) |
| `origin` | (2,) | f64 | world position of focal frame origin |
| `heading` | () | f64 | world heading (rad) of focal frame +x axis |

N ≤ 64 agents (nearest to focal at t=49), L ≤ 128 lanes (nearest centerline point). Files are `C:\data\av2\processed\{split}\<scenario_id>.npz`.

### Batch contract (produced by `collate`, consumed by all models/losses/metrics)

Dict of torch tensors: `agent_hist` (B, N_max, 50, 2), `agent_mask` (B, N_max, 50), `lanes` (B, L_max, 20, 2), `lane_mask` (B, L_max), `future` (B, 60, 2). Models output `traj` (B, K, 60, 2) and `logits` (B, K); for S1, K = 1 and logits are zeros.

---

### Task 1: Environment + package scaffold

**Files:**
- Create: `pyproject.toml`, `src/trajpred/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: importable `trajpred` package; `.venv` with CUDA-enabled torch; `pytest` runs.

**Teaching checkpoint:** why Python 3.12 (torch wheel availability), what a src-layout package buys us (imports work identically in scripts/tests/notebooks).

- [ ] **Step 1: Install Python 3.12 and create venv**

```powershell
winget install -e --id Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "trajpred"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "numpy", "pyarrow", "pyyaml", "matplotlib", "tqdm", "pandas",
]

[project.optional-dependencies]
dev = ["pytest", "jupyter"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Install torch (CUDA) + package**

```powershell
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -e ".[dev]"
```

- [ ] **Step 4: Write smoke test `tests/test_smoke.py`**

```python
import torch
import trajpred  # noqa: F401

def test_cuda_available():
    assert torch.cuda.is_available()
```

`src/trajpred/__init__.py` is empty.

- [ ] **Step 5: Run `pytest -q`, expect 1 pass** (if CUDA is False, fix the torch install before proceeding — do not skip)

- [ ] **Step 6: Commit** — `chore: project scaffold with CUDA-enabled environment`

---

### Task 2: Download the AV2 subset

**Files:**
- Create: `scripts/download_subset.py`

**Interfaces:**
- Produces: `C:\data\av2\raw\train\<id>\` × 15,000 and `...\raw\val\<id>\` × 5,000, each containing `scenario_<id>.parquet` and `log_map_archive_<id>.json`; manifest files `C:\data\av2\raw\{split}_ids.txt` (the frozen subsets).

**Teaching checkpoint:** AV2 layout (one dir per scenario, public S3 bucket `s3://argoverse/datasets/av2/motion-forecasting/`), why we freeze the subset with a seed, why `s5cmd` (parallel, no AWS account via `--no-sign-request`).

- [ ] **Step 1: Install s5cmd** — download the Windows release binary from GitHub (`peak/s5cmd`) into `C:\data\av2\bin\s5cmd.exe` (keep it out of the repo).

- [ ] **Step 2: Write `scripts/download_subset.py`**

```python
"""Sample a fixed-seed subset of AV2 motion forecasting scenarios and download them."""
import argparse, random, subprocess, sys
from pathlib import Path

BUCKET = "s3://argoverse/datasets/av2/motion-forecasting"
S5CMD = r"C:\data\av2\bin\s5cmd.exe"

def list_scenario_ids(split: str) -> list[str]:
    out = subprocess.run(
        [S5CMD, "--no-sign-request", "ls", f"{BUCKET}/{split}/"],
        capture_output=True, text=True, check=True,
    ).stdout
    ids = [line.split()[-1].strip("/") for line in out.splitlines() if line.strip().endswith("/")]
    assert len(ids) > 1000, f"listing looks wrong: {len(ids)} ids"
    return sorted(ids)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", default=r"C:\data\av2\raw")
    args = p.parse_args()

    ids = list_scenario_ids(args.split)
    random.Random(42).shuffle(ids)
    chosen = sorted(ids[: args.n])

    out_root = Path(args.out) / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.out) / f"{args.split}_ids.txt"
    manifest.write_text("\n".join(chosen))

    cmds = "\n".join(
        f'cp "{BUCKET}/{args.split}/{sid}/*" "{out_root / sid}/"' for sid in chosen
    )
    subprocess.run([S5CMD, "--no-sign-request", "run"], input=cmds, text=True, check=True)
    print(f"downloaded {len(chosen)} scenarios to {out_root}", file=sys.stderr)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run for both splits** (val first — smaller, verifies the pipe)

```powershell
.venv\Scripts\python scripts/download_subset.py --split val --n 5000
.venv\Scripts\python scripts/download_subset.py --split train --n 15000
```

- [ ] **Step 4: Verify** — count dirs (`(Get-ChildItem C:\data\av2\raw\val).Count` = 5000; train = 15000); spot-check one dir contains a parquet + map JSON.

- [ ] **Step 5: Commit** — `feat: subset download script with frozen seed-42 manifests`

---

### Task 3: Scenario exploration notebook

**Files:**
- Create: `notebooks/01_explore_scenario.ipynb`, `src/trajpred/scenario.py`, `tests/test_scenario.py`

**Interfaces:**
- Produces (in `scenario.py`, used by preprocessing in Task 6):

```python
@dataclass
class RawScenario:
    scenario_id: str
    timestamps: np.ndarray          # (110,) int
    track_ids: list[str]            # length T_tracks
    positions: np.ndarray           # (T_tracks, 110, 2) f64, NaN where unobserved
    headings: np.ndarray            # (T_tracks, 110) f64, NaN where unobserved
    object_types: list[str]         # per track
    focal_idx: int                  # row of the focal track
    lane_centerlines: list[np.ndarray]  # each (P_i, 2) f64, variable length

def load_scenario(scenario_dir: Path) -> RawScenario: ...
```

**Teaching checkpoint:** the parquet schema (long format: one row per (track, timestep)), what the focal agent is (`focal_track_id` / category), what the map JSON holds, why we pivot long→dense arrays with NaN for missing.

- [ ] **Step 1: Write `tests/test_scenario.py`** (uses a real downloaded val scenario as fixture)

```python
from pathlib import Path
import numpy as np
from trajpred.scenario import load_scenario

RAW_VAL = Path(r"C:\data\av2\raw\val")

def test_load_scenario_shapes():
    scenario_dir = sorted(RAW_VAL.iterdir())[0]
    s = load_scenario(scenario_dir)
    assert s.positions.shape[1] == 110 and s.positions.shape[2] == 2
    assert 0 <= s.focal_idx < len(s.track_ids)
    # focal track fully observed over the 11 s window
    assert not np.isnan(s.positions[s.focal_idx]).any()
    assert len(s.lane_centerlines) > 0
    assert all(cl.ndim == 2 and cl.shape[1] == 2 for cl in s.lane_centerlines)
```

- [ ] **Step 2: Run it, expect FAIL** (`ModuleNotFoundError` / missing function)

- [ ] **Step 3: Implement `load_scenario` with pyarrow**

Read `scenario_*.parquet` (columns include `track_id`, `timestep`, `position_x`, `position_y`, `heading`, `object_type`, `object_category`, `focal_track_id`). Pivot to dense `(T_tracks, 110, ...)` arrays indexed by `timestep`. Focal = track whose `track_id == focal_track_id`. Parse `log_map_archive_*.json`: lane centerlines from `lane_segments[*]["centerline"]` (list of {x, y} dicts) → `(P, 2)` arrays. If the actual column names differ, adapt in this task and record the real schema in `docs/notes/03-av2-schema.md` — later tasks depend only on `RawScenario`.

- [ ] **Step 4: Run test, expect PASS**

- [ ] **Step 5: Build the notebook** — load one scenario, plot lane centerlines (grey), all agent tracks (blue), focal history (black) and future (green). This is the teaching artifact for the data's shape; keep outputs stripped before commit.

- [ ] **Step 6: Commit** — `feat: scenario loader and exploration notebook`

---

### Task 4: Geometry — focal frame transforms

**Files:**
- Create: `src/trajpred/geometry.py`, `tests/test_geometry.py`

**Interfaces:**
- Produces:

```python
def to_frame(points: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray
def from_frame(points: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray
```

`points` is (..., 2); both functions broadcast over leading dims and preserve NaN.

**Teaching checkpoint:** why agent-centric normalization is the single highest-leverage trick in trajectory prediction (the model never has to learn that "turning left at (5000, 3000) heading NE" equals "turning left at origin heading +x").

- [ ] **Step 1: Write `tests/test_geometry.py`**

```python
import numpy as np
from trajpred.geometry import to_frame, from_frame

def test_origin_maps_to_zero():
    origin = np.array([100.0, -50.0]); heading = 1.3
    assert np.allclose(to_frame(origin, origin, heading), [0.0, 0.0])

def test_point_ahead_maps_to_plus_x():
    origin = np.array([10.0, 20.0]); heading = np.pi / 2  # facing +y in world
    ahead = origin + np.array([0.0, 5.0])
    assert np.allclose(to_frame(ahead, origin, heading), [5.0, 0.0], atol=1e-9)

def test_round_trip():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(7, 50, 2)) * 100
    origin = np.array([3.0, 4.0]); heading = -2.1
    assert np.allclose(from_frame(to_frame(pts, origin, heading), origin, heading), pts, atol=1e-9)

def test_nan_preserved():
    pts = np.array([[np.nan, np.nan], [1.0, 2.0]])
    out = to_frame(pts, np.zeros(2), 0.5)
    assert np.isnan(out[0]).all() and not np.isnan(out[1]).any()
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import numpy as np

def _rot(heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s], [s, c]])

def to_frame(points, origin, heading):
    return (np.asarray(points) - origin) @ _rot(heading)   # equals R^T applied to rows

def from_frame(points, origin, heading):
    return np.asarray(points) @ _rot(heading).T + origin
```

- [ ] **Step 4: Run tests, expect PASS** (walkthrough: why `@ R` on row-vectors is the inverse rotation)

- [ ] **Step 5: Commit** — `feat: focal-frame coordinate transforms`

---

### Task 5: Metrics

**Files:**
- Create: `src/trajpred/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces (all torch, no grad needed; `pred` (B, K, 60, 2), `probs` (B, K) summing to 1, `gt` (B, 60, 2)):

```python
def min_ade(pred, gt) -> Tensor        # (B,) ADE of best mode (best = lowest ADE)
def min_fde(pred, gt) -> Tensor        # (B,) FDE of best mode (best = lowest FDE)
def miss_rate(pred, gt, thresh=2.0) -> Tensor   # (B,) 1.0 if no endpoint within thresh
def brier_min_fde(pred, probs, gt) -> Tensor    # (B,) minFDE + (1 - p_best_fde)^2
def evaluate_batch(pred, probs, gt) -> dict[str, Tensor]  # all four, each (B,)
```

**Teaching checkpoint:** why displacement error not accuracy; why min-over-K rewards diversity and what MR/brier add back; unimodal metrics are the K=1 special case of the same functions.

- [ ] **Step 1: Write `tests/test_metrics.py`** — hand-computed toy cases

```python
import torch
from trajpred.metrics import min_ade, min_fde, miss_rate, brier_min_fde

def make_toy():
    gt = torch.zeros(1, 60, 2); gt[0, :, 0] = torch.arange(60) + 1.0  # straight +x, 1 m/step
    pred = torch.zeros(1, 3, 60, 2)
    pred[0, 0] = gt[0]                          # exact
    pred[0, 1] = gt[0] + torch.tensor([0.0, 3.0])  # 3 m lateral offset everywhere
    pred[0, 2, :, 0] = torch.arange(60) + 1.0; pred[0, 2, :, 1] = 0.5  # 0.5 m offset
    return pred, gt

def test_min_ade_picks_exact_mode():
    pred, gt = make_toy()
    assert torch.allclose(min_ade(pred, gt), torch.tensor([0.0]))

def test_min_fde_zero_for_exact():
    pred, gt = make_toy()
    assert torch.allclose(min_fde(pred, gt), torch.tensor([0.0]))

def test_ade_without_exact_mode():
    pred, gt = make_toy()
    assert torch.allclose(min_ade(pred[:, 1:], gt), torch.tensor([0.5]))  # constant 0.5 m offset

def test_miss_rate_thresholds():
    pred, gt = make_toy()
    assert miss_rate(pred[:, 1:2], gt).item() == 1.0   # 3 m endpoint error > 2 m
    assert miss_rate(pred[:, 2:3], gt).item() == 0.0   # 0.5 m < 2 m

def test_brier_adds_confidence_penalty():
    pred, gt = make_toy()
    probs = torch.tensor([[0.5, 0.25, 0.25]])
    assert torch.allclose(brier_min_fde(pred, probs, gt), torch.tensor([0.25]))  # 0 + (1-0.5)^2
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** — L2 per step `torch.linalg.norm(pred - gt[:, None], dim=-1)` → (B, K, 60); ADE = mean over time, FDE = last step; min/argmin over K; brier uses argmin-by-FDE to index `probs`. `evaluate_batch` composes them.

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Cross-check against the av2 devkit** — `pip install av2` in a *throwaway* venv (it may not build on Windows; that's fine, it never enters the main env). Feed 100 random-tensor cases through both implementations; assert agreement to 1e-6. If the devkit won't install, record that in `docs/notes/05-metrics.md` and rely on the hand-computed tests.

- [ ] **Step 6: Commit** — `feat: official forecasting metrics with toy-case tests`

---

### Task 6: Preprocessing pipeline

**Files:**
- Create: `src/trajpred/preprocess.py`, `scripts/preprocess.py`, `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `load_scenario` (Task 3), `to_frame` (Task 4).
- Produces: `preprocess_scenario(scenario_dir: Path) -> dict[str, np.ndarray] | None` returning the **npz bundle contract** (see File Structure); `None` if the focal track is not fully observed. `scripts/preprocess.py --split {train,val}` writes one npz per scenario with a tqdm progress bar and prints a count of skipped scenarios.

**Teaching checkpoint:** why offline preprocessing (GPU starvation), agent selection (nearest 64 at t=49), lane resampling to fixed 20 points (uniform arc-length), and the config-hash versioning idea.

- [ ] **Step 1: Write `tests/test_preprocess.py`** (real val scenario as fixture)

```python
from pathlib import Path
import numpy as np
from trajpred.preprocess import preprocess_scenario

def test_bundle_contract():
    d = preprocess_scenario(sorted(Path(r"C:\data\av2\raw\val").iterdir())[0])
    assert d["agent_hist"].shape[1:] == (50, 2) and d["agent_hist"].shape[0] <= 64
    assert d["agent_mask"].shape == d["agent_hist"].shape[:2]
    assert d["lanes"].shape[1:] == (20, 2) and d["lanes"].shape[0] <= 128
    assert d["future"].shape == (60, 2)
    # focal agent is row 0, fully observed, and at origin facing +x at t=49
    assert d["agent_mask"][0].all()
    assert np.allclose(d["agent_hist"][0, 49], [0.0, 0.0], atol=1e-6)
    step = d["agent_hist"][0, 49] - d["agent_hist"][0, 48]
    assert abs(np.arctan2(step[1], step[0])) < 0.35 or np.linalg.norm(step) < 0.1
    # no NaNs anywhere masked-valid
    assert not np.isnan(d["agent_hist"][d["agent_mask"]]).any()
    assert not np.isnan(d["future"]).any()
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `preprocess_scenario`** — load raw; origin/heading from focal track at timestep 49 (heading from the parquet's heading column, not finite differences); transform all positions and centerlines with `to_frame`; NaN→0 with masks recording validity; rank agents by distance to origin at t=49 (drop unobserved-at-t49 tracks), keep ≤64 with focal forced to row 0; resample each centerline to 20 uniform-arc-length points, rank lanes by min distance to origin, keep ≤128; split focal into hist (0–49) and future (50–109).

- [ ] **Step 4: Run test, expect PASS**

- [ ] **Step 5: Write and run `scripts/preprocess.py` over both splits.** Verify counts roughly match manifests (a few % skipped is normal); spot-load 3 npz files and plot one with lanes + history + future to eyeball that geometry is sane (future should extend roughly +x for a moving agent).

- [ ] **Step 6: Commit** — `feat: scenario preprocessing to canonical npz bundles`

---

### Task 7: Dataset + collate

**Files:**
- Create: `src/trajpred/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: npz bundles (Task 6).
- Produces:

```python
class AV2Dataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir: Path): ...
    def __getitem__(self, i) -> dict[str, np.ndarray]   # one bundle
def collate(batch: list[dict]) -> dict[str, torch.Tensor]  # batch contract (File Structure)
```

**Teaching checkpoint:** ragged scenes → padding + masks; why every downstream attention/pooling op must consume the mask (padding must be *provably* inert).

- [ ] **Step 1: Write `tests/test_dataset.py`** — build 2 synthetic bundles in `tmp_path` with different N (3 vs 5 agents) and L (4 vs 2 lanes), collate, then assert: shapes are (2, 5, 50, 2) and (2, 4, 20, 2); `agent_mask` is False exactly on padded rows; padded positions are 0; `future` shape (2, 60, 2); all tensors float32/bool as per contract.

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** — `__getitem__` = `np.load`; `collate` pads N and L to the batch max with zeros and extends masks accordingly.

- [ ] **Step 4: Run tests, expect PASS; also time one full pass over val** with `DataLoader(batch_size=64, num_workers=0)` — should be seconds, not minutes (this validates the offline-preprocessing decision).

- [ ] **Step 5: Commit** — `feat: dataset and padding collate with mask tests`

---

### Task 8: S0 — constant velocity baseline + shared eval script

**Files:**
- Create: `src/trajpred/models/constant_velocity.py`, `scripts/evaluate.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: batch contract, `evaluate_batch` (Task 5).
- Produces:

```python
def constant_velocity(batch: dict) -> tuple[Tensor, Tensor]  # traj (B,1,60,2), probs (B,1)
```

`scripts/evaluate.py --stage s0 [--ckpt path] [--config path]` runs the named stage over the full processed val set and appends a row `stage, K, minADE, minFDE, MR, brier_minFDE, n_scenarios, date` to `results/val_metrics.csv`. All later stages reuse this script — it is the single source of comparison truth.

**Teaching checkpoint:** why CV is strong (most driving is straight, constant speed), and the prediction: CV minFDE₁ lands somewhere around 4–5 m. Before running, write your guess in `docs/notes/08-baseline.md`.

- [ ] **Step 1: Write test in `tests/test_models.py`**

```python
import torch
from trajpred.models.constant_velocity import constant_velocity

def test_cv_extrapolates_last_velocity():
    batch = {"agent_hist": torch.zeros(1, 1, 50, 2), "agent_mask": torch.ones(1, 1, 50, dtype=torch.bool)}
    batch["agent_hist"][0, 0, :, 0] = torch.arange(50) * 2.0   # 2 m/step in +x
    traj, probs = constant_velocity(batch)
    assert traj.shape == (1, 1, 60, 2) and probs.shape == (1, 1)
    assert torch.allclose(traj[0, 0, 0], torch.tensor([100.0, 0.0]))   # 98 + 2
    assert torch.allclose(traj[0, 0, -1], torch.tensor([218.0, 0.0]))  # 98 + 60*2
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** — velocity = `hist[:, 0, 49] - hist[:, 0, 48]` (focal row 0); positions = last + v·(1…60); probs = ones.

- [ ] **Step 4: Run test, expect PASS**

- [ ] **Step 5: Write `scripts/evaluate.py`** — loads val `AV2Dataset`, iterates batches, dispatches on `--stage` (s0 needs no checkpoint; later stages load model + ckpt via their config), accumulates `evaluate_batch` means, appends the CSV row, prints the row.

- [ ] **Step 6: Run `evaluate.py --stage s0`** — compare against your written prediction; record the actual number in the note.

- [ ] **Step 7: Commit** — `feat: constant velocity baseline and shared eval script`

---

### Task 9: Training infrastructure

**Files:**
- Create: `src/trajpred/trainer.py`, `scripts/train.py`, `configs/s1_gru.yaml`

**Interfaces:**
- Consumes: `AV2Dataset`, `collate`.
- Produces:

```python
class Trainer:
    def __init__(self, model, loss_fn, cfg: dict): ...   # builds AdamW, loaders, device
    def overfit_one_batch(self, steps=300) -> float       # returns final loss
    def fit(self) -> Path                                  # trains, saves best-val ckpt, returns path
```

`loss_fn(model_output, batch) -> Tensor` scalar. `scripts/train.py --config configs/<stage>.yaml` builds the stage's model + loss from the config's `stage:` key and runs `overfit_one_batch` (assert < 0.05) before `fit`. Config keys: `stage, lr, weight_decay, batch_size, epochs, processed_train, processed_val, ckpt_dir, model: {…}`.

`configs/s1_gru.yaml`: `stage: s1, lr: 1e-3, weight_decay: 1e-4, batch_size: 64, epochs: 30, model: {hidden: 128}`.

**Teaching checkpoint:** overfit-one-batch as the cheapest possible correctness proof; what "best val checkpoint" protects against.

- [ ] **Step 1: Implement `Trainer`** (no unit test — it's exercised end-to-end in Task 10; keep it < 150 lines: epoch loop, val loss each epoch, `torch.save` on improvement, tqdm, seed everything with 42)

- [ ] **Step 2: Verify** — construct a tiny linear dummy model on real data and check `overfit_one_batch` drives loss to ~0

- [ ] **Step 3: Commit** — `feat: config-driven trainer with overfit-one-batch check`

---

### Task 10: S1 — GRU encoder–decoder (unimodal)

**Files:**
- Create: `src/trajpred/models/gru.py`
- Modify: `tests/test_models.py`, `scripts/evaluate.py` (register stage s1), `scripts/train.py` (register stage s1)

**Interfaces:**
- Consumes: batch contract, `Trainer`.
- Produces:

```python
class GRUPredictor(nn.Module):
    def __init__(self, hidden: int = 128): ...
    def forward(self, batch) -> tuple[Tensor, Tensor]   # traj (B,1,60,2), logits (B,1)
```

Loss for s1: `smooth_l1_loss(traj[:, 0], batch["future"])`.

**Teaching checkpoint:** displacements vs. absolute positions as input (scale, stationarity); one-shot decoding (predict 60 displacement steps, cumsum) vs. autoregressive, and why one-shot avoids exposure bias here.

- [ ] **Step 1: Write shape/grad test** — random batch through `GRUPredictor`; assert output shapes, no NaNs, and `loss.backward()` populates grads on all parameters.

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** — focal displacements `hist[:, 0, 1:] - hist[:, 0, :-1]` (B, 49, 2) → GRU(2→hidden) → last hidden → MLP(hidden→hidden→120) → reshape (60, 2) displacements → cumsum → positions (add nothing: focal frame origin is the last observed point).

- [ ] **Step 4: Run test, expect PASS**

- [ ] **Step 5: Train** — `scripts/train.py --config configs/s1_gru.yaml` (overfit check runs automatically; a full run is ~30 epochs × 15k, minutes-per-epoch on the 4060)

- [ ] **Step 6: Evaluate** — register s1 in `evaluate.py`, run it. **Prediction first** (write in `docs/notes/10-s1.md`): where does minFDE₁ land relative to CV? Expected ≈ 3–4 m — better than CV, and *why it can't be dramatically better* (unimodal averaging).

- [ ] **Step 7: Commit** — `feat: unimodal GRU encoder-decoder (stage 1)`

---

### Task 11: S2 — K-mode decoder + WTA loss

**Files:**
- Create: `src/trajpred/models/multimodal.py`, `src/trajpred/losses.py`, `tests/test_losses.py`, `configs/s2_multimodal.yaml`
- Modify: `tests/test_models.py`, `scripts/train.py`, `scripts/evaluate.py` (register s2)

**Interfaces:**
- Consumes: batch contract, `Trainer`.
- Produces:

```python
class MultiModalGRU(nn.Module):
    def __init__(self, hidden: int = 128, k: int = 6): ...
    def forward(self, batch) -> tuple[Tensor, Tensor]     # traj (B,K,60,2), logits (B,K)

class KModeHead(nn.Module):                                # reused verbatim by S3
    def __init__(self, hidden: int, k: int = 6): ...
    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]  # (B,hidden) -> traj, logits

def wta_loss(traj, logits, gt, alpha: float = 1.0) -> Tensor
# winner = argmin over K of ADE(traj_k, gt); loss = smooth_l1(traj_winner, gt)
#          + alpha * cross_entropy(logits, winner)
```

`configs/s2_multimodal.yaml`: same as s1 plus `model: {hidden: 128, k: 6}`.

**Teaching checkpoint (the centerpiece):** mode collapse — if you backprop L2 through *all* heads they all learn the mean; WTA gives each head its own basin. Also the honest caveat: min-over-K metrics partially reward spraying, which is what MR and brier are for.

- [ ] **Step 1: Write `tests/test_losses.py`**

```python
import torch
from trajpred.losses import wta_loss

def test_wta_zero_regression_when_a_mode_is_exact():
    gt = torch.randn(2, 60, 2)
    traj = torch.randn(2, 4, 60, 2); traj[:, 2] = gt
    logits = torch.zeros(2, 4)
    # winner is mode 2 -> regression term 0; CE of uniform logits over 4 = log(4)
    assert torch.allclose(wta_loss(traj, logits, gt), torch.log(torch.tensor(4.0)), atol=1e-5)

def test_wta_gradient_only_flows_to_winner():
    gt = torch.zeros(1, 60, 2)
    traj = torch.zeros(1, 2, 60, 2, requires_grad=True)
    with torch.no_grad():
        traj[0, 1] += 100.0          # mode 1 is far; mode 0 wins
    wta_loss(traj, torch.zeros(1, 2), gt, alpha=0.0).backward()
    assert traj.grad[0, 1].abs().sum() == 0
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `wta_loss`** (detach the argmin selection; index with `gather` or arange) **and `KModeHead`/`MultiModalGRU`** (same GRU encoder as S1; head = MLP(hidden→hidden→K·121) split into K·(60·2) displacements + K logits, cumsum as before). Add a shape/grad test for the model mirroring Task 10's.

- [ ] **Step 4: Run all tests, expect PASS**

- [ ] **Step 5: Train + evaluate.** Prediction first (`docs/notes/11-s2.md`): minADE₆/minFDE₆ vs. S1's unimodal numbers — expect a large drop (~2.5 m minFDE₆). Then inspect: plot the 6 modes for 5 turning scenarios with `viz` from Task 13 pending — for now a quick matplotlib cell is fine. Do the modes actually diverge (left/straight/right) or did they collapse?

- [ ] **Step 6: Commit** — `feat: K-mode decoder with winner-take-all loss (stage 2)`

---

### Task 12: S3 — polyline transformer (VectorNet-style)

**Files:**
- Create: `src/trajpred/models/polyline.py`, `configs/s3_polyline.yaml`
- Modify: `tests/test_models.py`, `scripts/train.py`, `scripts/evaluate.py` (register s3)

**Interfaces:**
- Consumes: batch contract, `KModeHead` (Task 11), `Trainer`.
- Produces:

```python
class PolylineNet(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 4, n_heads: int = 8,
                 k: int = 6, use_map: bool = True, use_social: bool = True): ...
    def forward(self, batch) -> tuple[Tensor, Tensor]   # traj (B,K,60,2), logits (B,K)
```

`use_map=False` drops lane tokens; `use_social=False` keeps only the focal agent token + lanes. These flags are the Task 14 ablations — build them now, not later.

`configs/s3_polyline.yaml`: `stage: s3, lr: 5e-4, weight_decay: 1e-4, batch_size: 32, epochs: 40, model: {d_model: 128, n_layers: 4, n_heads: 8, k: 6}`.

**Teaching checkpoint:** the polyline abstraction — agents and lanes become the *same kind of token*, so social + map context is one attention mechanism; how `key_padding_mask` makes padding provably inert; parameter budget vs. 8 GB VRAM.

- [ ] **Step 1: Write tests** — (a) shape/grad test as before; (b) padding-invariance test: run a batch, then re-run with 3 extra padded (mask-False) agents and lanes appended — outputs must be identical to 1e-5; (c) flags test: `use_map=False` forward works and differs from the full model's output.

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

Per-point features: agents → (Δx, Δy, valid) per step in a (50, 3) sequence per agent; lanes → (x, y, Δx, Δy) per point (20, 4). Each polyline through its own small subgraph encoder (two blocks of Linear→LayerNorm→ReLU with max-pool concat, VectorNet-style; separate encoders for agents and lanes since features differ) → one d_model token each. Concatenate `[agent tokens | lane tokens]`, build `key_padding_mask` from `agent_mask.any(-1)` and `lane_mask`, run `nn.TransformerEncoder` (batch_first, 4 layers). Take the focal token (position 0) → `KModeHead`.

- [ ] **Step 4: Run tests, expect PASS** (the padding-invariance test is the one that catches real bugs — if it fails, the mask isn't reaching every op)

- [ ] **Step 5: Train + evaluate.** Prediction first (`docs/notes/12-s3.md`): expect minFDE₆ ≈ 1.5–2 m, with the biggest gains on turning scenarios. Watch VRAM (`nvidia-smi`); drop batch to 16 + `grad_accum: 2` if needed.

- [ ] **Step 6: Commit** — `feat: polyline transformer with map and social context (stage 3)`

---

### Task 13: Visualization

**Files:**
- Create: `src/trajpred/viz.py`
- Modify: `scripts/evaluate.py` (add `--save-figs n` to dump the n worst scenarios)

**Interfaces:**
- Consumes: npz bundles, model outputs.
- Produces:

```python
def plot_scene(bundle: dict, traj=None, probs=None, ax=None) -> Axes
# lanes grey, agent histories light blue, focal history black,
# GT future green, predicted modes red with alpha ∝ probability
```

**Teaching checkpoint:** qualitative eval catches what tables hide — a good minADE with kinematically absurd modes is a red flag recruiters *will* probe.

- [ ] **Step 1: Implement `plot_scene`** (no unit test; verify by eye on 5 scenarios)
- [ ] **Step 2: Wire `--save-figs` into evaluate.py** — after metric accumulation, re-run the n scenarios with the worst minFDE and save PNGs to `results/figures/<stage>/`
- [ ] **Step 3: Generate galleries for s2 and s3; look at them together** — this is a teaching session: find one mode-collapse case, one map-fixes-it case
- [ ] **Step 4: Commit** — `feat: scene visualization and worst-case galleries`

---### Task 14: Ablations, error analysis, results

**Files:**
- Create: `configs/s3_no_map.yaml`, `configs/s3_no_social.yaml`, `scripts/analyze_errors.py`, `results/README.md`
- Modify: repo `README.md` (created here)

**Interfaces:**
- Consumes: everything.
- Produces: final `results/val_metrics.csv` with rows s0, s1, s2, s3, s3_no_map, s3_no_social; breakdown table + figures; top-level README with the results table, a scene figure, and honest interpretation.

**Teaching checkpoint:** ablations as the difference between "I trained a model" and "I understand what each part buys"; maneuver bucketing (turning = |heading change over future| > 30°, from GT; speed bands 0–2, 2–8, 8+ m/s at t=49).

- [ ] **Step 1: Write the two ablation configs** (identical to s3 but `use_map: false` / `use_social: false`), train both, evaluate both
- [ ] **Step 2: Write `scripts/analyze_errors.py`** — per-scenario metrics for s2/s3 tagged with maneuver + speed band → grouped table (`results/breakdown.csv`) + bar chart (`results/figures/breakdown.png`)
- [ ] **Step 3: Teaching session on the numbers** — where does the map help most? (expected: turning scenarios, low-speed intersections) Does no-social hurt on dense scenes? Write conclusions in `docs/notes/14-analysis.md`
- [ ] **Step 4: Write the top-level `README.md`** — task, data subset caveat (15k/5k, seed 42 — say so explicitly; honesty beats inflated scope), results table, 2–3 figures, what each stage taught, how to reproduce (5 commands)
- [ ] **Step 5: Commit** — `feat: ablations, error analysis, and results`

---

### Task 15 (optional): Full-data headline run

- [ ] Pull full 250k train split (`s5cmd` sync, ~100 GB — check disk first), re-preprocess, retrain s3 with `epochs: 20`, evaluate on the *same frozen 5k val* and also full official val; add both rows to the results table marked accordingly.
- [ ] Commit — `feat: full-dataset training run`

---

## Self-review notes

- **Spec coverage:** env (T1), subset download (T2), preprocessing + frame (T3–6), dataset (T7), metrics + devkit cross-check (T5), S0–S3 (T8–12), overfit-check (T9), viz (T13), ablations/breakdowns/README (T14), optional full run (T15). Frozen val, seed 42, OneDrive-safe data path all in Global Constraints.
- **Type consistency:** all models return `(traj (B,K,60,2), logits/probs (B,K))`; CV returns probs directly (uniform ones), trained stages return logits and `evaluate.py` softmaxes before `evaluate_batch`. `KModeHead` defined in T11, reused in T12.
- **Known adaptation point:** exact AV2 parquet column names are verified in Task 3 against real data; only `RawScenario` leaks past that boundary.
