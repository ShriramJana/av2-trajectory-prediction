# Training infrastructure notes

One `Trainer` for every stage; a stage is (model, loss, yaml). The recipe is
fixed across stages so differences in the results table come from the *model*:
AdamW, one-cycle LR (10% warm-up, then cosine decay), grad-norm clip 5.0,
seed 42, best checkpoint by val minFDE.

**Overfit-one-batch** runs before every real run: a throwaway copy of the model
must memorize 16 scenes (loss to <5% of its initial value in 1000 steps).
It is the cheapest end-to-end proof that shapes line up, gradients reach every
layer, and the loss actually measures what the model outputs.

It earned its keep immediately: with 300 steps S2 stalled at 11% and the run was
refused. Splitting the loss showed regression 9.07 → 0.40 but classification
1.79 → 0.75. Not a bug: under winner-take-all the classification *target* (which
mode wins) keeps changing while the modes sort themselves out, so it converges
later. At 1000 steps the total reaches 0.01. The check needed more steps, not
the model a fix — but that was a conclusion from evidence, not an assumption.

**Caveat on "best val checkpoint":** the checkpoint is selected on the same 5k
val scenarios we report on (the test split has no public labels). The selection
effect is small — curves are flat near the end — but it is optimistic, and the
README says so.
