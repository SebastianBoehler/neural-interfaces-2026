# Lane D · Can cap250 teach itself within a session?

Registered 19 November 2026 before fitting or scoring. First decision uses the corrected Friday full0 bank
(Lane NI-V cohort A, shared fresh Sinc and FBLight seed 0). The corrected within0 bank is secondary.
This is the whole-session, label-free look-ahead probe in cross_opus §5. It is an optimistic data-access
test for this model and feature view, with no mathematical bound on other estimators.

- **Whole-session Gaussian.** Use log diagonal band power from the existing theta/alpha/beta1/beta2/broad
  covariance banks in official row order. Center within each session and fit a single tied-diagonal Gaussian
  with soft class responsibilities from cap250's terminal posterior; shrink each class mean toward the pooled
  session mean with 25 pseudo-rows, and mix the within-class variance equally with the mean within-person
  variance of the nine other people. Fit once on every row, including the future and the scored row.
  Apply a uniform class prior and add 0.25 times the Gaussian log likelihood ratio, divided by the effective
  feature count (five bands, so divide by 41 channels), clipped to ±2, to stock log probabilities.
  Open no gold labels until both sets of decisions are written. Prediction +0.15 pp (80% −0.45,+0.75)
  on full0 and +0.30 pp (−0.40,+1.00) on within0. Stop D-a/D-b if full0 <+0.50 pp, run both at full
  scale if ≥+1.50 pp, otherwise mark the ceiling undecided and time-box a pilot. The second teacher in
  cross_opus can be tested only when Lane C supplies an authenticated matching source-excluded bank.
- **Purity diagnostic.** Report terminal confidence (max), top-two margin, agreement with independently
  scored members and age (fifths within visit), per-cell true and predicted class counts, and coverage.
  Training selector, if needed: on other people only, among k=1…K and margins
  0.00,0.05,…,0.95, choose greatest coverage at ≥88% precision on ≥30 windows; break ties by higher k,
  then higher margin. Report independent held purity and class quotas; the largest pooled held subset
  passing 88% is a descriptive envelope only. Predict stock 74–80% pooled purity on full0, ≤55% coverage
  at ≥88% purity, and <100 selected past windows by age 190 in the median visit after equal quotas.
- **D-a, conditional.** Gate cap250's existing Adam refit inputs by the selected k and margin on
  the strictly earlier rows, then keep the highest-margin rows at the smallest available class quota.
  Skip the refit if any class is absent. Predict +0.20 pp (80% −0.45,+0.85). **D-b, conditional.**
  Use the same Adam refit on balanced pseudo-labels. Project stock strict-prefix responsibilities
  onto equal soft class mass with Sinkhorn and take a maximum-log-probability hard assignment with
  per-class counts differing by at most one. Predict +0.05 pp (−0.65,+0.75).
- **D-c.** Stream a copy of exact cap250. For each decision from the 20th row onward, project the raw
  terminal probabilities of the delivered prefix including the current row onto equal soft class mass.
  Emit only the current row's adjusted argmax and keep stock state and raw posterior history.
  Fixed unit temperature and tolerance 1e−8 on normalized class mass; no tuned parameters.
  Predict held-new +0.10 pp (80% −0.60,+0.80) and Scherer −0.10 pp (−0.70,+0.50).
  Class balance is a prior to test. The official metric weights class recalls equally and does not
  constrain predicted proportions. No method uses future rows in any causal arm.
- **Prior art.** Stock TSEM/SessionEM already operate on confident class templates in core PCs and
  ocular features. The offline test uses all soft weights, a separate raw band-power view, source scale
  shrinkage and a bounded correction on the complete cap250 terminal output. §252 and §253 trained a
  new classic personal classifier, including confidence and eye-agreement selection. D-a/D-b instead
  alter the existing Adam targets within the actual cap250 trajectory. §259 started from unsupervised
  balanced K-means in source LDA geometry. D-b/D-c constrain the existing class probabilities directly
  and need no matching from cluster IDs to classes.
- **Validation.** Authenticate corrected bank, labels, feature and order inputs. Require v3 replay OK,
  terminal argmax and zero-strength identity, pinned-order reproduction and people-nested paired 80%
  and 95% intervals. The offline banks have one order per scheme, so their intervals resample people
  only. Live Scherer uses eight fresh NI-V full-session orders, paired people/order resampling, a
  bitwise off-vs-original smoke and prefix/reset checks. Compare against corrected held-new banks
  where available. Report P04/P10 separately. A nonpositive upper paired bound refutes a causal
  gain on that cohort; a crossing interval stays undecided. No competition upload.

Input-format clarification before opening labels. `CausalPers.step()` returns probability rows before
fusion and log score rows after fusion. Normalize the former and softmax the latter. Require terminal
argmax identity after conversion. The initial run stopped at this format guard before writing decisions.

**First decision.** Corrected Friday FULL-SYSTEM DEV, P01–P10, 28 cells. Gaussian whole-session fit
changed full0 **+0.292 pp** (paired people 80% −0.060,+0.639; 34 fixes / 26 regressions) and within0
**+0.372 pp** (−0.061,+0.716; 36 / 26). Full0 P04 −0.382, P10 +1.060 pp. Stock reproduced at
73.739% and 78.324%. Reversed row order reproduced the Gaussian scores to 1e−12. `replay/check.py`
printed OK. Blind predictions SHA-256 `9e992ebaeda79e89d43fcf95aa668078dbc1aeafae6f971263e99cfb7d16fc72`.
The full0 point estimate trips the registered stop for D-a/D-b. More independent supervision remains
an open test. D-c is separate from the refit mechanism and proceeds.

**D-c execution lock.** For Scherer, replay the actual D-c copy on each NI-V full order (71000–71007).
Seal both its emitted predictions and the stock argmax of the exact terminal scores from those calls.
The candidate only intercepts emission downstream of the terminal stage. In the independent same-device
smoke, compare untouched stock, switches-off and D-c input bytes, compare D-c emissions with an independent
prefix replay, and compare reset/prefix and batch delivery. Keep work counters. An upstream divergence
invalidates the claimed same-input control. For PROTEUS, use corrected full0 and within0 terminal banks
with each other scheme and order explicitly scoped. No source threshold is used by D-c.

**Purity input lock.** Agreement counts four independently scored but correlated upstream families
from the corrected Friday inputs, frozen FBCNet-EA transformer and shallow-EA seed means, frozen
Sinc seed 0 and matched FBLight seed 0. They all use the same pinned delivery. Count matches to the
emitted cap250 pseudo-class; the resulting sample estimates terminal pseudo-label quality. The live
Adam target shadow can assign different classes at refit time. Select the source-only rule within
each leave-one-person-out fold and report its actual held coverage and purity. Diagnostic quotas
use equal predicted-class counts within each session and break ties in delivery order.

**Bank purity results.** Each row in [the cell CSV](laneD_purity_cells.csv) gives selected, correct,
and true and pseudo-class counts for one participant/session/interface cell. The descriptive
envelope uses held outcomes to pick a grid point and has no deployment meaning.

| Corrected order | All terminal rows | Descriptive ≥88% purity coverage | Source-only coverage / purity | With session quotas |
|---|---:|---:|---:|---:|
| full0 | 73.71% | 50.73% at 88.27% | 50.81% / 86.06% | 38.31% / 89.48% |
| within0 | 78.34% | 58.23% at 88.36% | 61.32% / 85.69% | 45.20% / 88.75% |

On full0, 4-of-4 agreement alone is 86.18% pure at 24.52% coverage. For posterior max ≥0.85,
purity is 94.10% at 19.77% coverage. Fifths in delivery order are 72.90 / 72.37 / 71.67 / 76.50 /
75.14% pure. Among sessions that reached 190 deliveries, the median equal-quota prefix held 58.5
windows on full0 and 79.5 on within0. The old 100-window result was for a classic classifier.

**D-c corrected Friday results.** Full0 74.256% vs 73.739%, **+0.517 pp** (80% +0.176,+0.807;
95% +0.031,+0.947; 30 fixes / 16 regressions). P04 +1.400 and P10 +0.327 pp. Within0 **−0.205 pp**
(80% −0.456,−0.004; 95% −0.599,+0.105; 20 / 25). P04 +0.375 and P10 −0.202 pp. All
19 pre-activation and replayed 35-row prefixes matched. Blind bank SHA-256
`1576e7f53a741e889f58b0df7c0f3be52545b3ec254a1951180c7686cbdefac6`. These are two corrected
full-system trajectories with one order per scheme, as described in [NI-V](NIV.md). Fresh
held-new orders still need corrected upstream producers.

Calibration scope. Each diagnostic threshold uses only other people’s labels. The saved
leave-one-person-out prediction banks for those source people may have trained on the diagnostic
fold's held person. This is a threshold-transfer diagnostic on existing banks. A future
adaptive refit needs a calibration bank fit without that held person throughout.

**External execution.** At 13:31 CET, Galvani accepted smoke 3136727, eight-order Scherer
array 3136728 and on-cluster collector 3136729. Smoke was pending on the per-user job limit
and the other jobs were dependency-pending. Release `ni26_lane_d_prefix_v2_20261119` writes
`result.json` after all banks validate. Source manifest SHA-256
`72ffcd487ba87ef8f7bdf73213905af97a3de3b9371b1ab6ff7af6269086d660`. The copied Python source
retains the original bytes as a prefix, and every package asset matches stock bitwise.
The Leonardo variant is `scripts/cloud/leo_lane_d.sh`.

**Smoke and resubmission.** The local real Scherer 44-window CPU smoke passed bitwise stock/off
terminal scores, labels, member votes and counters. D-c matched the upstream scores and counters,
changed decisions, and passed reset and 25-window batch-prefix parity. The initial Galvani smoke
3136727 could not dereference the stock ZIP symlink outside the container bind. Both dependent
jobs were cancelled by Slurm. The signed ZIP is now materialized inside the bind. Galvani
accepted replacement jobs **3136765** (RTX smoke), **3136766** (array), **3136767** (collector)
with the same source and ZIP hashes.

**RTX smoke correction, before external scoring.** Job 3136765 reached 120 windows per actor and
matched work counts, then stopped on independent terminal posterior bytes. This comparison is
sensitive to the stock-repeat numerical differences seen on RTX in earlier cap250 qualification.
Emit the signed stock archive byte-for-byte as `off.zip`. In the active arm smoke, bracket D-c by
independently loaded signed stock/off actors. Require identical stock/off and candidate upstream
argmax labels and work counts. Require maximum stock repeat terminal drift ≤5e−4 and candidate
upstream drift ≤max(1e−5, 2× measured stock repeat drift). Require exact same-trajectory D-c
decisions and reset-prefix labels. The main Scherer comparison uses one raw terminal trajectory
per exact package stream, so these independent reload differences do not enter the paired scores.

**Leonardo after the network switch.** The corrected local CPU smoke passed, with zero independent
stock repeat or candidate upstream drift. The disabled ZIP hashes to the signed cap250 archive.
The active archive is byte-identical to the v2 active archive. Galvani became unreachable while
the v3 launch was still waiting for the release-directory command, before any Slurm submit.
Repeat the corrected smoke, paired fresh Scherer orders and collector on Leonardo A100 with the
same signed stock, D-c package, raw input and NI-V order definitions. This is A100 evidence.

Leonardo accepted smoke **64847774**, eight-order array **64847775** and collector
**64847776**. Release `$WORK/ni26/runs/ni26_lane_d_prefix_20261119`, source manifest SHA-256
`e9f914abfbf3077cc91b27dd9a4ec3e720c1fa5728380b8003c4f9ad4e3b66e4`. The smoke was pending
priority at submission. Galvani's RTX run is recorded as a failed conformance attempt.
