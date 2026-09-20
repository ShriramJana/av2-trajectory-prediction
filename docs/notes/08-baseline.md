# S0 constant velocity — notes

**Prediction written in the plan:** minFDE₁ ≈ 4–5 m.
**Actual:** minADE 4.41, **minFDE 11.69 m**, MR 0.85.

The guess was off by >2×, which prompted a bug hunt before trusting it. Checks:
- Smoother velocity (mean of last 5/10/20 steps) is *worse*: 12.3 / 12.9 / 14.3 m.
  Recent velocity is the best predictor; the last-step estimate isn't too noisy.
- "Stay put" scores 38.4 m, so CV is doing real work.
- Error decomposition in the focal frame: mean |along-track| error 9.6 m vs
  |cross-track| 3.7 m. **CV mostly fails on speed changes, not on turning.**
- By speed at t=49: 8.6 m (0–2 m/s), 14.7 m (2–8 m/s), 10.4 m (8+ m/s). The
  mid band is cars accelerating away from / braking into intersections.

Conclusion: no bug. The 4–5 m intuition fits shorter horizons (under constant
acceleration, error grows quadratically with the horizon) and datasets without
AV2's deliberate selection of "interesting" focal agents. Lesson: calibrate
expectations on the actual benchmark, and when a number surprises you,
decompose the error before believing or dismissing it.

Why CV is still a mandatory row: it costs 10 lines, has no training bugs, and
every learned model must beat it. If a network can't, the pipeline is broken.
