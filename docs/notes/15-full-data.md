# Full-data run — notes

Same S3 model, same recipe, one change: train on all 199,908 official training
scenarios instead of the 15k subset (20 epochs instead of 40; ~4 min 20 s per
epoch on the RTX 4060, ~1.5 h total).

| model | trained on | frozen 5k val minFDE₆ | full official val (24,988) minFDE₆ |
|---|---|---|---|
| s3 | 15k subset | 2.45 | 2.44 |
| s3_full | 199,908 | **1.88** | **1.86** |

Full rows (frozen val): minADE 0.95, minFDE 1.88, MR 0.31, brier-minFDE 2.51.

## What this shows
- **Data was the bottleneck, as the S3 notes guessed.** 13× more scenarios:
  minFDE −0.57 m, miss rate 0.41 → 0.31, with zero architecture changes. After
  only 3 epochs on full data the model was at 2.86 m; by epoch 8 it had passed
  the subset model's final score.
- **Biggest gain where the subset model was weakest:** turning scenarios
  5.13 → 3.33 m (straight: 1.93 → 1.60). Turns are 16% of scenarios, so the
  subset held only ~2,400 of them; the full set has ~32,000.
- **Still not saturated:** train loss 1.96 vs val 1.98 at the end — no
  overfitting gap at all. More epochs or a bigger model would likely still help.
- **The frozen 5k val subset is representative:** both models score within
  0.02 m on the subset and on the full official val split. Conclusions drawn on
  the subset (ablations, breakdowns) can be trusted at that resolution.

## Housekeeping
- Full data lives in `raw/train_full`, `raw/val_full`, `processed/…_full`
  (`download_subset.py --all`); the seed-42 subset and its manifests are
  untouched, and all 5,000 subset bundles are byte-identical in `val_full`.
- Checkpoint selection still uses the frozen 5k val, so `s3_full@val_full` is
  the cleaner number: 80% of those scenarios played no part in selection.
- Not done: ablations on full data. The "social context adds nothing" finding
  was measured at 15k scenes only — it may not hold at 200k, and that is the
  obvious next experiment.
