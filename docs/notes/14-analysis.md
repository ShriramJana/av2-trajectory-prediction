# Ablations and error analysis — notes

All numbers: frozen 5k val subset, K=6 unless noted.

| stage | minADE | minFDE | MR | brier-minFDE |
|---|---|---|---|---|
| s0 constant velocity (K=1) | 4.41 | 11.69 | 0.85 | 11.69 |
| s1 GRU (K=1) | 3.22 | 8.50 | 0.83 | 8.50 |
| s2 GRU + 6 modes | 1.53 | 3.55 | 0.60 | 4.14 |
| s3 polyline transformer | 1.18 | 2.45 | 0.41 | 3.09 |
| s3 − map | 1.44 | 3.24 | 0.55 | 3.86 |
| s3 − social | 1.19 | 2.49 | 0.43 | 3.13 |

## What each part buys (minFDE)
- Multimodality (s1 → s2): −4.95 m. By far the biggest step, and nearly free.
- Map (s3−map → s3): −0.79 m.
- Social context (s3−social → s3): −0.04 m. Within what a different seed could
  plausibly produce — we ran one seed per config, so treat it as "no measurable
  effect", not as a small positive one.
- Transformer + social without map (s2 → s3−map): −0.31 m.

## Where the map helps
| bucket | n | s2 | s3−map | s3 | map gain |
|---|---|---|---|---|---|
| straight | 4196 | 2.75 | 2.42 | 1.93 | 0.49 |
| turning (>30°) | 804 | 7.70 | 7.51 | 5.13 | **2.38** |
| 0–2 m/s | 1238 | 2.53 | 2.31 | 1.74 | 0.57 |
| 2–8 m/s | 2007 | 4.46 | 3.93 | 2.91 | **1.02** |
| 8+ m/s | 1755 | 3.22 | 3.11 | 2.41 | 0.70 |

As predicted: the map matters most on turns (5× the gain of straight scenes)
and in the 2–8 m/s band — intersection speeds. Without a map, turning error is
essentially unchanged from s2 (7.51 vs 7.70): social context cannot tell you
where the road goes. `results/figures/s2_vs_s3.png` shows the mechanism — s2's
modes are a generic fan, s3's modes bend along the actual turn lanes.

## The social surprise
Expected: no-social hurts in dense scenes. Observed: 1–10 agents +0.04 m,
11–30 +0.01 m, 31+ agents +0.10 m. The direction is right, the size is tiny.
Plausible reasons, none of them tested here:
1. The focal agent's own 5 s history already encodes most interactions that
   matter over the next seconds (it is already braking for the car ahead).
2. Interaction effects are rare events; 15k scenes may be too few to learn them.
3. minFDE over 6 modes forgives: "yield or go" just occupies two modes.
This is a finding about *this setup*, not a claim that social context is useless.

## What still fails (`results/figures/s3/`)
Median minFDE is 1.70 m but the worst 10% of scenarios hold 38% of the total
error (p90 4.7 m, p99 15.6 m). The worst cases are almost all one of:
- **Missed turn at speed:** all 6 modes go straight (differing only in speed)
  while the car turns. Along-track uncertainty eats the mode budget.
- **Unexpected stop / go:** the car stops mid-block or pulls away from a stop,
  with nothing in the history to say so.
- **Wrong branch:** modes cover left and straight; the car went right.
MR 0.41 is the honest summary: in 41% of scenarios no mode ends within 2 m.

## Caveats to state whenever quoting these numbers
- Everything above is at 15k training scenes. The full-data run
  ([15-full-data.md](15-full-data.md)) did not repeat the ablations.
- 15k/5k seed-42 subset, not the full benchmark; not comparable to leaderboard
  entries.
- One seed per config. Differences under ~0.1 m are not evidence.
- Checkpoints were selected on the same val set they are reported on.
