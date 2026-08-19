# Trajectory Prediction on Argoverse 2 — Design

**Date:** 2026-08-19
**Status:** Approved

## 1. Goal

Predict 6-second vehicle futures on the Argoverse 2 Motion Forecasting dataset, built as a staged ladder of models where each stage adds one concept and one measurable improvement. Evaluated on the official metrics (minADE₆, minFDE₆, Miss Rate @ 2m, brier-minFDE₆) on the official validation split.

**Success criteria:**

- Final model (stage 3) comfortably beats the constant-velocity baseline and the unimodal model on val.
- One honest results table comparing all stages on the same val subset with the same metrics.
- Ablations (no-map, no-social) and error breakdowns by maneuver type and speed.
- Failure-case visualizations: predicted modes drawn over the lane graph.
- Repo history and docs read as a coherent progression suitable for a portfolio.

**Non-goals:** leaderboard-competitive SOTA numbers, test-server submission, ensembling, pedestrian datasets (ETH/UCY deferred indefinitely).

## 2. Task definition

Standard Argoverse 2 motion forecasting protocol:

- Each scenario is 11 s at 10 Hz: 50 observed steps (5 s) → predict 60 steps (6 s).
- One designated **focal agent** per scenario is scored.
- Scenario data includes all other agent tracks and a local HD map (lane segments as polylines with connectivity, crosswalks, drivable area).
- Predictions: K = 6 trajectories of shape 60×2, each with a probability.

**Metrics** (all in meters, computed on the focal agent):

- **minADE₆** — mean L2 error over all 60 steps, for the best of 6 modes.
- **minFDE₆** — L2 error at step 60, best of 6 modes.
- **MR₂ₘ** — fraction of scenarios where no mode's endpoint lands within 2 m of the ground-truth endpoint.
- **brier-minFDE₆** — minFDE₆ + (1 − p̂)², where p̂ is the predicted probability of the best mode; penalizes low confidence in the correct mode.
- minADE₁/minFDE₁ reported for unimodal stages.

Our metric implementations are validated once against the `av2` devkit's official evaluation code.

## 3. Environment

- Windows 11, RTX 4060 (8 GB VRAM), Python 3.12 (installed alongside the existing 3.14, selected via the `py` launcher), project-local venv.
- PyTorch with CUDA 12.x wheels.
- `av2` devkit for scenario parquet and map JSON parsing. **Fallback** if the devkit is rough on Windows: read parquets directly with `pyarrow` (the schema is documented), losing only convenience functions.
- **Data lives outside OneDrive** at `C:\data\av2` to avoid sync-thrash on thousands of small files. Code stays in the OneDrive project directory. The repo's data directory is a gitignored pointer/symlink or config path.

## 4. Data plan — subset first

AV2 motion forecasting is stored one-directory-per-scenario in a public S3 bucket (no account required; pulled via AWS CLI or `s5cmd`).

- **Phase A (main work):** ~15k train scenarios + ~5k val scenarios (a few GB). All stages are developed and compared on this fixed subset. The val subset is frozen once drawn so comparisons stay honest.
- **Phase B (optional, end):** pull the full 250k train set for one long training run of the final architecture to get the headline number, still evaluated on official val.

## 5. Preprocessing

One offline pass converts each scenario (parquet + map JSON) into a training-ready `.npz` bundle, because per-batch parquet/JSON parsing would starve the GPU.

Per scenario:

- **Agent-centric canonical frame:** translate so the focal agent's position at the last observed step is the origin; rotate so its heading at that step points +x. Makes the model translation- and rotation-invariant.
- Focal history stored as **displacements** (Δx, Δy per step), not absolute positions.
- Up to ~64 nearest other agents: histories in the focal frame, with validity masks (tracks appear/disappear mid-scenario).
- Up to ~128 nearest lane segments, each resampled to a fixed number of points per polyline, in the focal frame, with lane attributes (e.g., is-intersection, lane type).
- Ground-truth future (60×2) in the focal frame.

Deterministic, versioned by a config hash so stale caches can't silently poison comparisons.

## 6. Model ladder

Each stage is a tagged, runnable checkpoint evaluated with the same script on the same val subset.

- **S0 — Constant velocity.** Extrapolate the last observed velocity for 60 steps. ~30 lines. First row of the results table.
- **S1 — GRU encoder–decoder.** Focal agent history only. Encoder GRU over 50 displacement steps; decoder emits 60 steps. Unimodal, smooth-L1 loss. Exists to prove the pipeline and eval are correct.
- **S2 — K = 6 mode decoder.** Same encoder; decoder emits 6 trajectories + 6 logits. Loss = winner-take-all smooth-L1 on the closest mode + cross-entropy on which mode won. The multimodality lesson lives here.
- **S3 — Polyline transformer (VectorNet-style).** Every agent history and lane polyline → shared PointNet-style subgraph encoder (per-point MLP + max-pool) → one token per polyline → stacked self-attention over the token set → focal agent's token feeds the S2 decoder unchanged. Social and map context via one mechanism. Target ~2–4 M params; batch 32–64 on 8 GB VRAM, gradient accumulation as fallback.
- **S4 — Ablations & analysis.** Results table across all stages; S3-minus-map and S3-minus-social ablations; error breakdown by maneuver (straight vs. turning) and speed band; failure-case gallery with predicted modes over the lane graph.

Expected val trajectory (rough): CV minFDE₁ ≈ 4–5 m → S1 minFDE₁ ≈ 3–4 m → S2 minFDE₆ ≈ 2.5 m → S3 minFDE₆ ≈ 1.5–2 m.

## 7. Repository layout

```
trajpred/
  data/            gitignored (points at C:\data\av2)
  src/trajpred/    preprocessing, datasets, models/, metrics, viz
  configs/         one yaml per stage
  scripts/         download, preprocess, train, eval
  notebooks/       exploration + teaching artifacts
  docs/            specs/, notes/ (learning notes)
  results/         metrics tables, figures
```

Git from day one; commit per milestone so the history shows the progression. Commit messages contain no AI attribution; no AI-tooling files are committed.

## 8. Workflow

Taught build. Per module: concept explainer (why it exists, what breaks without it) → code written and walked through, non-obvious parts explained → a short check-your-understanding exercise (e.g., predict the eval result before running it). Learning notes accumulate in `docs/notes/`.

## 9. Testing & verification

- Unit tests for silent-failure-prone pieces: coordinate transform round-trips, metrics vs. hand-computed toy cases, mask correctness in collate.
- Metrics cross-checked once against the `av2` devkit evaluation.
- Overfit-one-batch sanity check before every real training run.
- Fixed random seeds for subset selection and training so results are reproducible.

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `av2` devkit rough on Windows (Rust components) | Medium, contained to week 1 | pyarrow fallback reading parquets directly |
| 8 GB VRAM limits batch size for S3 | Low | 2–4 M param model, masking, gradient accumulation |
| OneDrive sync-thrash on many small files | Low | data at `C:\data\av2`, outside OneDrive |
| PyTorch unavailable on Python 3.14 | Resolved | install Python 3.12 alongside |
