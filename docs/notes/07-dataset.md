# Dataset + collate notes

Scenes are **ragged**: 3 agents here, 64 there. Tensors are rectangular. The
bridge is **pad + mask**: `collate` pads every scene to the batch max with zeros
and the boolean masks say which rows are real.

The contract that matters downstream: *padding must be provably inert*. A zero
row is not "nothing" to a network — a Linear layer's bias turns it into a
non-zero feature, and attention would happily attend to it. So every op that
mixes rows (max-pool over points, attention over tokens) must consume the mask.
S3's tests check this directly: appending garbage-filled padded rows must not
change the output.

Practical detail: the whole subset (~500 MB) fits in RAM, so `AV2Dataset` caches
bundles after first read — epoch 2+ never touches the disk.
