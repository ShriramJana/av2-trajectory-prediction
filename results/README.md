# Results

Everything here is produced by `scripts/evaluate.py` and
`scripts/analyze_errors.py` on the frozen 5,000-scenario val subset
(`manifests/val_ids.txt`).

| file | what |
|---|---|
| `val_metrics.csv` | one row per stage: minADE, minFDE, MR, brier-minFDE. Rows named `<stage>@val_full` are scored on the whole official val split (24,988 scenarios) instead of the frozen 5,000 |
| `breakdown.csv` | the same metrics per stage, bucketed by maneuver (turning = GT heading change > 30°), speed at t=49, and number of agents in the scene |
| `figures/breakdown.png` | minFDE₆ per bucket for the K=6 stages |
| `figures/s2_vs_s3.png` | the turning scenarios where S3 improves most on S2 |
| `figures/<stage>/worst_*.png` | the 12 worst scenarios (by minFDE) for that stage |
| `per_scenario/` | per-scenario metrics (gitignored; regenerate with `evaluate.py`) |

Interpretation and caveats: [docs/notes/14-analysis.md](../docs/notes/14-analysis.md).
