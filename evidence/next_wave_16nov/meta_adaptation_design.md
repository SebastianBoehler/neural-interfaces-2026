# Train for causal adaptation: proposal, not yet registered

The proposed experiment learns how to update a source-trained linear core, using
only labels from source-person query windows during meta-training. It does not
train on labels from the held-out test person. It is distinct from previous
adaptation-aware training:

- `experiments/ts_core2.py` flags `ct` and `cb` train the logistic head on causal
  features, or a union of causal and whole-session features. They do not
  meta-train through a head-gradient update. The completed `ctA.out` and `cbA.out`
  tests used a smaller 12,382-feature model in recording order: baseline 68.53%,
  ct 67.73%, cb 68.31% person-equal. These were standalone. The
  44,239-feature `Fcar_ov17_fs2_t0.1_le_k2_ct.out` job was canceled.
- §220 trained fbraw on source-session episodes using differentiable fixed-point
  class-bias balancing. The deployed network was frozen; no learned head-gradient
  adaptation was tested. §255/256 instead trained an auxiliary FBLight head and
  adapted its SessionBN parameters with an auxiliary loss.

For each outer held person, take one of the other nine as the inner source
person. Fit the tangent prior, StandardScaler and C=.3 population logistic head
using only the remaining eight. Build each inner person's causal features in
three pinned source orders. At a cut t≥40, the support is at most the previous
64 windows from the same session; the later query is at most the next 20 windows.
No held person's labels, later query windows, or another session's state enter
the update. Query labels are source labels and train the outer objective only.

Compare identical nine-person anchors with (1) no update, (2) fixed +1 update
multipliers, (3) one learned signed multiplier and (4) 51 learned signed
band/time-block multipliers. Each update takes three steps on mean prediction
entropy minus 0.1 times entropy of mean predictions, plus 0.1 normalized squared
anchor distance. Normalize by the full head-gradient norm and scale each step by
0.01 of anchor-head norm; bound total displacement to 0.03 of that norm.
Learnable multipliers use tanh, initially +0.5. All adapted arms leave the final
328 slow features and intercept unchanged. Unroll all three actual updates and
backpropagate source-query cross entropy to the multiplier. The proposed learning
schedule is Adam lr=.01, 12 meta-epochs, two seeds, with deterministic matching
episodes across scalar and block arms. Each epoch rotates one eligible cut per
source session; log counts and fail if a required source person has none.

For target delivery, derive one anchor from the nine training people, reset each
session, update at t=40,60,… using only strictly previous windows, and classify
the current window. Evaluate all eight pinned orders through the complete current
chain with the same fixed gate, each arm's own stages and all six CP bases. Zero
step must exactly match frozen anchor; gradients, learned parameters, displacement,
source exclusion, prefix invariance and session reset require explicit guards.

Proposed new-person deltas: learned scalar minus fixed +0.10 pp [−0.35,+0.45];
blocks minus scalar +0.05 [−0.35,+0.45]; learned versus unchanged core +0.05
[−0.5,+0.5]. An interval crossing zero remains undecided. A result ≤−0.3 pp
would disfavor that precise form, not this research direction.

Nine by ten source-only fitted heads and causal source streams are required;
existing target nine-person streams do not provide these double-held-out priors.
Runtime is not measured. The first real-data no-score smoke must measure cache,
inner-head fitting and meta-step time before selecting the production allocation.
The native §301 feature parity passed, but its converged optimizer did not match
the original finite-iteration head. Use one explicitly pinned solver across all
compared arms and report any changed anchor as a separate contrast.
