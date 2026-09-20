# Preprocessing notes

**Why offline:** parsing a parquet + a map JSON takes milliseconds per scenario;
a GPU consumes a batch of 64 in a few ms. Parsing inside the training loop would
leave the GPU idle most of the time. So: parse once, save training-ready arrays.
(Measured: all 20k scenarios preprocess in ~30 s on 8 processes; a full pass over
val from the bundles takes 2 s cold, 0.1 s from RAM.)

**What a bundle is** (`trajpred.preprocess`, one `.npz` per scenario):
everything already in the focal frame, fixed dtypes, NaNs replaced by 0 + masks.

Decisions and their reasons:
- **Frame from the heading column, not finite differences.** A stopped car has
  no displacement to take a direction from; the dataset's heading is still valid.
- **Agents: focal forced to row 0, then nearest ≤63 that are visible at t=49.**
  An agent that vanished before the present can't influence the future much, and
  "row 0 = focal" is a convention every model relies on.
- **Lanes: resample to 20 points by uniform arc length, keep nearest ≤128.**
  Raw centerlines have 2–50+ unevenly spaced points. Fixed count → one tensor;
  uniform spacing → point index means the same thing on every lane.
- **Masks instead of NaN.** NaN poisons every op it touches (0 × NaN = NaN, so
  even a zero attention weight leaks it). After preprocessing there are no NaNs
  anywhere, and `agent_mask` records what was real.
- **Extras beyond the plan's contract:** `lane_is_intersection` (a model feature)
  and `final_heading` (GT heading change over the future — used only to bucket
  scenarios into straight/turning for error analysis, never as a model input).
- **Config hash.** `_meta.json` next to the bundles stores a hash of the
  preprocessing constants; `AV2Dataset` refuses to load bundles built with
  different constants. Prevents evaluating a new model on a stale cache.

Subset facts (1000 random val scenes): ~27 agents and ~66 lanes per scene on
average; 99% of futures end at x>0 (forward); 17.5% are turns (>30° heading
change); median speed at t=49 is 5.8 m/s.
