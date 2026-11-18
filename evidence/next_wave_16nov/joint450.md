# §450 · Linked public role and delivery sensitivity

At the assumed 56.6% new and 20% P04 / 80% P10 returning mixture, historical three-seed cap250 scored **80.648%** against **79.538%** for historical v5.1b. Its +1.110 percentage-point gain stayed positive in every tested scheme and order. All registered point predictions held.

| Delivery scheme | Joint gain (pp) | Min–max same-named order pairs | Middle 80% linked-weight stress | New only | Returning P04 only | Returning P10 only |
|---|---:|---:|---:|---:|---:|---:|
| Within interface | +1.431 | +1.205 to +1.779 | +1.098 to +1.788 | +2.211 | +0.814 | +0.314 |
| Full session | +0.992 | +0.883 to +1.101 | +0.672 to +1.423 | +0.588 | +2.572 | +1.255 |
| Within run | +0.906 | +0.702 to +1.109 | +0.502 to +1.404 | +1.342 | +1.063 | +0.156 |

The scheme-equal middle 80% was **+0.884 to +1.420 pp**. Each of 20,000 draws (seed 450) used one exponential weight per person, shared across both roles and arms. Each draw also sampled the observed paired order names with replacement within each scheme. Weighting reflects the 28 official new cells (six each for P04/P10, two each for the other eight people), while held visits 2 and 3 contribute four returning cells per person. Each role retains its assigned total mass. Absolute official old/candidate scores were 74.724/76.104% new and 79.882/80.911% returning. With 20/80 returning weights they were 85.817/86.574%.

Removing one person from both roles left a minimum **+0.797 pp** across all ten deletions and three schemes (full session, P09 omitted). Removing P04 left scheme gains +1.538/+0.846/+1.207; removing P10 left +1.359/+1.391/+0.826 in table order. Among 44 two-person deletions, the minimum was **+0.590 pp** (full session, P04/P09 omitted). Removing both returning people leaves that role undefined and was excluded.

The four shared sessions have distinct window permutations in the two validation protocols even for matching order names. As an additional sensitivity check, pairing every observed new order with every returning order from its scheme gave +0.903 to +1.841 within interface, +0.751 to +1.233 full session and +0.651 to +1.160 within run. These cross-pairs reweight observed predictions. All three role corner estimates were positive within each scheme, although the returning P10-only effect approached zero under within-run delivery.

The two roles reuse participants, visits and development data previously used for model selection. The range describes specified shifts among these people and observed orders. Two returning participants cannot establish private-population uncertainty. This score covers the historical full-system emulation with corrected eye/rest seed counts. Exact archive parity, unrelated returners and sealed accuracy remain unmeasured.

Registration 13649a5. Run a fresh project replay into `runs/joint450/replay.txt`, then run [the scorer](../../studies/18nov/joint450/score.py) with the project Python. It authenticates the historical lineage pins and each §430/431 three-seed output, reconstructs both arms from saved scores, checks count-filter identity and historical zero controls, and reproduces the owner role endpoints and all eight official order deltas. [All order, role and deletion results](joint450_result.json).
