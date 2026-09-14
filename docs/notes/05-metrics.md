# Metrics notes

Four metrics, each closing a loophole in the previous one:

- **minADE_K** — mean L2 over all 60 steps, best of K modes. Loophole: can nail
  the path shape but blow the endpoint.
- **minFDE_K** — L2 at step 60, best of K. The headline number (the horizon is
  where planning decisions live). Loophole: min-over-K rewards spraying.
- **MR@2m** — fraction of scenarios where no mode endpoint lands within 2 m.
  Punishes diversity that is noise rather than hypotheses.
- **brier-minFDE_K** — minFDE + (1 − p̂)² with p̂ the probability assigned to the
  best-FDE mode. Confidence must be honest; hedging costs. Official AV2 ranking
  metric.

Conventions locked in `trajpred.metrics`:
- shapes pred (B, K, 60, 2), probs (B, K), gt (B, 60, 2); unimodal = K=1.
- functions return per-scenario (B,) tensors; callers aggregate (needed for
  error breakdowns by maneuver/speed later).
- winner for brier is chosen by FDE, matching `compute_brier_fde` + argmin-FDE
  in the devkit.

## Devkit cross-check (2026-09-14)

100 random (B=100, K=6, T=60) cases through the official
`av2.datasets.motion_forecasting.eval.metrics` (in a throwaway venv at
C:\data\av2\check_venv — av2 pins deps, keep it out of the main env):
max abs diff 1.4e-14 (minADE) and exactly 0 for minFDE / MR / brier.
Our implementations are the official semantics.

Windows note: Smart App Control blocks pip.exe shims in fresh venvs; use
`python -m pip` instead.
