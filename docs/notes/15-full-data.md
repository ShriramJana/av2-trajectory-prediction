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

## Ablations and a second seed at full scale (added 2026-09-21)

| model (all 199,908 scenes) | minADE | minFDE | MR | brier-minFDE |
|---|---|---|---|---|
| s3_full (seed 42) | 0.95 | 1.88 | 0.31 | 2.51 |
| s3_full (seed 43) | 0.92 | 1.83 | 0.29 | 2.46 |
| s3_full − map | 1.15 | 2.52 | 0.43 | 3.13 |
| s3_full − social | 0.96 | 1.94 | 0.32 | 2.57 |

- **Seed spread is 0.05 m.** That is the ruler for every other difference here.
  Two seeds is a thin estimate of variance, but it is no longer zero seeds.
- **Map: 0.64–0.69 m** (0.79 at 15k). Turning scenes 3.3 → 5.4 m without it.
  Still the dominant context signal; slightly less dominant, because with 13×
  more data the model infers more road structure from motion alone.
- **Social: 0.06–0.11 m overall** — barely above seed noise. But broken down by
  scene density (minFDE, no-social vs the two full seeds):

  | agents in scene | n | seed 42 | seed 43 | − social |
  |---|---|---|---|---|
  | 1–10 | 412 | 1.85 | 1.91 | 1.86 |
  | 11–30 | 2864 | 1.89 | 1.81 | 1.92 |
  | 31+ | 1724 | 1.85 | 1.84 | 1.98 |

  No effect in sparse scenes, +0.13 m in dense ones, against both seeds. That is
  the dose-response you would expect if the effect is real. At 15k scenes the
  same table showed +0.10 m in dense scenes but with no seed to compare against.
  Conclusion: social context helps in dense traffic, by roughly a tenth of a
  meter of endpoint error — real, small, and an order of magnitude less than
  the map. The 15k-scene result ("no measurable effect") was under-powered, not
  wrong in direction.
- Why so small? Untested hypotheses: the focal car's own history already encodes
  its reaction to traffic; minFDE over 6 modes forgives yield-or-go ambiguity;
  and single-agent open-loop metrics do not reward scene consistency, which is
  where interaction modeling matters most.
