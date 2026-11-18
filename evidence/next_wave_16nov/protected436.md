# Protected baseline at recognition

The package computes its original shared inner networks, recognition gates and fine-tuning labels once per window. Two separate outer branches consume that output. Each has its own mixture, personal routing head weight, routed tail, count filter and causal personalization state. Both branches start from an empty session. On a real gate latch the solver emits the protected cap250 branch immediately. Until then it emits JS. The solver receives one EEG window at a time. Its decision uses no participant ID, query label or future recognition time.

This differs from [§350](policy350_results.md), which changed the mixture after recognition but continued the JS-influenced tail. §350 lost 0.119 pp against conditional cap250. [§363](branch_selector363.md) tested a learned terminal selector without the actual recognition policy.

## What can be protected

The first recognized window and all later windows can match cap250 even when JS changed predictions earlier. Earlier predictions have already been returned. When the two branches disagree before the latch, a gate-only policy cannot promise both the JS label for an unrecognized person and the cap250 label for an eventually recognized person with the same observed prefix.

The [historical returning attribution](js321_recognition_attribution.json) predicted a residual −0.08658 pp from the unrecognized prefix alone. A later score equal to that number tests implementation fidelity on those fixed streams. It supplies no independent evidence for selecting the policy. The B2_pool historical JS gain of +0.2724 pp used a different held-new emulation. A previously unseen person can trigger a false recognition in the production package.

## Reproduction

The private Galvani snapshots are `ni26_protected436_20261118_v1` (returning) and `ni26_protected436_ext_v1` (Scherer). Both use the same registered policy source. Neither snapshot edits an uploaded archive.

- Returning CPU source smoke 3125128 compares ordinary JS and the protected package on a forced-unknown, 100-window replay. Eight FT refits and ten CP fusions occurred, with exact output agreement. The source replay repeats 17 real windows.
- Returning CPU pilot 3125129 checks both held P04 sessions on within0. Their natural latches occurred at delivery indices 60 and 10. All 377 package labels and selected scores matched the independently computed endpoint for the current latch bitwise. The 15-cell continuation is 3125143, with collector 3125149.
- External A100 source smoke 3125140 checks 96 Scherer windows, the literal §385 cap250 labels and a fresh 17-window replay from reset. The nine-person continuation is 3125144, with collector 3125150. Each full stream checks the protected baseline output on every window against the independently run archive.

## Results

**Retain cap250 over this protected JS variant.** On the two §385 orders, the nine external Scherer people averaged 48.456% with exact cap250 and 47.900% with the protected solver. The paired change was −0.556 percentage points, with an 80% participant/session/order interval of [−1.028, −0.069]. The order differences were −0.813 and −0.298 points. Seven people declined and two improved. The branches disagreed on 174 of 4,260 decisions. The gate never recognized any of the 36 session/order streams. All 4,260 protected-baseline labels matched independent cap250. With no latch, the emitted JS branch itself produced the external decline. The source estimate favored cap250 in all nine participant-held choices, giving zero held gain over always using cap250. [External result](protected436_external.json) · [Selection sensitivity](protected436_selection.json)

All 16 conditional returning cells passed endpoint identity. Across 32 session/order streams the package emitted JS on 773 windows and protected cap250 on 5,323. Selected endpoint scores and labels matched bitwise throughout. The original person weights give −0.08658 pp [descriptive 95% −0.16493, +0.03203], with P04 −0.03360 and P10 −0.09983. This recovers all the historical post-recognition loss and retains the prefix loss. P04/P10 have direct supervised exposure through the historical classical, rest and eye controls. [Returning result](protected436_returning.json)

Both collectors independently passed the official replay check. The external comparison uses nine participants from a separate clinical population, reconstructed EEG channels and a repeatedly assessed cue alignment. Neither the historical public gain nor the external decline directly estimates a competition score. The study did not rebuild all ten public held-new solver packages. It also did not establish full-worker timing for an upload. Source revision `a455c00`.
