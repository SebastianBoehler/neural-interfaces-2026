# §363 Source-nested terminal selection between cap250 and JS

The §362 label-assisted bound is +1.291 pp over always-JS across six bases.
Test one standard-scaled logistic selector, C=.1, with no model or threshold
search. When the two archived complete branches disagree, the selector gets
only their current top-two margins and margins between the contested labels.
The original branches evolve separately. Terminal selection has no feedback
to either branch. Each current score uses only that window and the causal
state of its own branch. The selector receives no clock, subject, interface,
order, sampling metadata, or future values.

For each held-out person, the other nine supply training labels. Fit only
disagreements on which one branch is correct, using the inverse of the source
cell/class count and scheme weights. Fit scaling on that source only. The
learned policy selects JS at a fixed logit above zero. To match the expected
rate without seeing the held stream, fit a threshold on JS-minus-base current
top-two margin from source-only leave-one-source-person-out selector choices.
Report actual realized held rates along with the rate-calibrated control,
always-base and always-JS. One learned arm and one fixed margin control use the
same complete branch states and disagreement opportunities.

Require all ten held folds, all 8 orders and 6 bases, frozen endpoints, exact
no-select/all-select checks and active causal decisions before original official
scoring. Predict learned−always-JS +0.15 pp [−0.35,+0.60], and learned−fixed
+0.10 pp [−0.30,+0.50]. An upper interval below zero refutes the fixed contrast;
an interval including zero remains inconclusive. Report paired people/order
80% intervals and B2_pool separately. The shared people and branch selection
history limit independence. The held folds are new-person emulation.

One CPU2/8 GiB/5min allocation uses verified cached banks and no worker inference.
No predictor is applied to the two-person conditional returning proxy. Deployment
would require two causal branch outputs per call, or measured exact shared-state
execution. The §362 independent warm pair timings estimate serial work only.
No promotion, extra selector, dose search or upload.

## Cached feature lineage

This is a **selector-only held comparison on exposed DEV**. The selector did
exclude the outer person's labels, rows and features from its fitting and
threshold calibration. The cached source feature producers were originally
fit separately for each person's leave-one-person-out bank. A source person's
frozen JS inputs can therefore depend on the selector-held person's labels.

An archive trace verified all 48 actual CC30 stream hashes consumed by JS
against the §304 assembler manifest and all 60 producer files against their
recorded nine-person training row hashes. For example, selector-held P01 is
among the 2,466 training rows of P02's seed-0 source CC30 model, whereas P02 is
excluded from that model. The lineage passes through
`conditional30_s0_P02.npz`, the §304 stitched stream,
`ttt18within0_fb_cc30_base-all_loso-s0.npz`, and the actual §318 JS input
manifest. The same cross-fold exposure occurs across the 60 checked producers.

Consequently, held-selector intervals describe this fixed archive. They cannot
establish generalization of the full two-branch policy. A clean confirmation
would rebuild the source branch streams inside each outer-person exclusion.

## Result

Job 3122400 completed in 2m31s on two CPUs. The project replay, all ten
selector-held folds, both forced endpoints and the complete causal matrix
passed. The learned gate chose JS on 3,645 of 4,021 disagreements. The
source-rate-calibrated fixed gate chose JS on 3,588. For B2_pool those counts
were 625 and 615 of 682, respectively.

| Selector-only held, exposed DEV | Always base | Always JS | Fixed margin | Learned |
| --- | ---: | ---: | ---: | ---: |
| Six-base official cell BA | 76.2177% | 76.4410% | 76.4378% | 76.4318% |
| B2_pool official cell BA | 76.2793% | 76.5517% | 76.5396% | 76.5396% |

Compared with always-JS, the learned result changed the six-base metric by
**−0.0092 pp** [80% paired people/order official interval −0.0354, +0.0223].
B2_pool changed by **−0.0121 pp** [−0.0561, +0.0320]. Compared with the
source-rate-calibrated margin control, the changes were −0.0060 and
+0.00004 pp, with both intervals spanning zero. The observed oracle room
did not produce a useful gain in this fixed selector. No returning selector,
shared-state worker timing or candidate change follows. The selector used
302 allocated CPU core-seconds and zero GPU.
[Before-label proof](branch_selector363_before_labels.json) · [Full results](branch_selector363_result.json)
