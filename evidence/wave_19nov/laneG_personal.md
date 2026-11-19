# Lane G · Stronger personal model per public person

Registered Thursday 19 Nov 2026 (eduVPN off; Leonardo only) BEFORE any arm was scored. Strategy step change 1
(`evidence/review_19nov/strategy_opus.md`, S1) and Lane A's stated limit (`laneA_returning.md`: heads trained on one
visit understate production, so a higher dose may look flat only because the head is weak).

Hypothesis: for returning people the package's personal evidence is weak (a light personal head h on the 256-PC view plus
the own-space T/C/S members, all fitted without augmentation, trained on recorded-order streams but scored on causal
shuffled ones). A stronger per-person model, fitted on ALL of that person's public sessions with heavy augmentation and
with training streams matched to delivery (causal running statistics, shuffled order), raises returning-cell accuracy
enough that a higher dose (Lane A) pays more.

## What the package loads (read from submissions/lean_v66/submission.py, weights.npz of cap250)

Per public person j (row order `route_names` = P01..P10): `route_ps1_W (10,3,256)`, `route_ps1_c (10,3)` = personal head h
(`Route.step`: softmax of the causally standardised 256-PC view, enters the mix at g r a); `own_prior, own_csp, own_t_mu,
own_t_sd, own_t_W, own_t_c, own_c_W, own_c_c, own_s_W, own_s_c` = OwnStage T (tangent LR, 9 bands), C (FBCSP LDA), S (slow
LDA), applied as exp(beta r mean log members). The stock recipe (`replay/rlib3.ps1_heads`, `studies/12nov/ownspace.Personal`,
`niv/enrol.py`) uses the person's sessions with whole-session recentring/standardisation and no augmentation.

## Arms

* **S0 (stock recipe)**: re-implemented recipe, fitted on the training visits only. Check: on a single-visit public person
  (P01) it must reproduce the package's stored row to within fitting tolerance (parity receipt in the smoke).
* **G1 (drop-in, zero extra runtime)**: same keys and shapes. Training = every training visit played as R replicas in
  random full-session delivery order, causal running recentring (prior = mean of the visits' session means, kref 2) and causal
  standardisation, as the stage runs at test time; replicas 1..R-1 carry the package's own augmentation (`strong`: roll ±15
  samples, noise 0.2 sd, 10 % channel dropout). h on the PC256 views of those replicas (LR C 0.1 as stock); T LR C 0.01,
  C FBCSP + LDA 0.5, S LDA 0.7 as stock. No hyper-parameter is chosen on held data.
* **N (neural, extra runtime; phase 2)**: per-person fine-tune of the best frozen neural member on all of the person's
  sessions with strong augmentation, stored as a TorchScript file and run on routed windows only. The shipped nets already
  saw P04/P10 visits 2-3, so N cannot be scored honestly from shipped weights; N is only scored with a source-fold pretrain.
  N is registered here, not yet scored; its own result gets its own lines in this card.

## Instrument 1 · INDIVIDUAL-MODEL (this card's decision)

P04 and P10 (the only multi-visit public people). Splits: train {1} -> test 2, {1} -> test 3, {1,2} -> test 3 (the last
is the production-like regime). Test visit delivered in the 8 pinned orders (within0-3, full0-1, runs0-1, `seen_session`
permutations of Lane A), one window per call, state causal and reset per session. Per window each arm yields a personal
posterior: h alone, own-stage mean log members alone, and the product p ~ h * exp(own) as the single-number proxy of the
evidence the package injects. Score = balanced accuracy per participant/session/interface cell, then mean over cells
(equal cell weight; P04 and P10 separately; 20/80 P04/P10 reported next to it). Gain = G1 minus S0 on the same split and
order. Limit stated now: the 256-PC view basis (`tsem_V`, `route_V_hi`) and the TangentCore were fitted on all public data,
including the test visits; this leak is common to S0 and G1 (paired) but makes the h component optimistic for both.
The own-space members have no population fit (own_prior is rebuilt from the training visits only).

### Prediction (numeric, before scoring)

S0 product posterior BA (train {1}): P04 about 58 %, P10 about 72 %; train {1,2} -> 3: P04 60 %, P10 77 %.
G1 minus S0, product posterior, equal-cell mean over the three splits: **+2.0 pp** (80 % interval -1.0 .. +5.0); P04
+1.5, P10 +2.5. Components: own-stage T +2.5, C +1.5, S +1.0, h +1.0. Augmentation alone (without causal-order matching)
about two thirds of the gain.

### Decision rule (fixed now)

* ADVANCE to the full system (swap G1 arrays into the Lane A fold packages, score with the dose sweep) if the equal-cell
  product gain is >= +1.5 pp, positive on BOTH persons, and >= 6 of 8 orders non-negative on the 20/80 gain.
* REFUTED in this form if the equal-cell gain is < +0.5 pp or negative on both persons (then a neural member N or a
  different feature set is the only route; do not close the lane, close G1).
* Otherwise UNDECIDED (bars inside the +-0.5 pp noise of a two-person instrument do not close a direction).
Intervals: orders resampled within person x the three splits; two people only, so descriptive 80 %.

## Production (all public sessions)

G1 arrays for all ten public people (P04 and P10 on three visits, the others on one), written as `personal_G1.npz` in
route_names order, and spliced into a byte-for-byte copy of the sealed cap250 ZIP by `laneG/make_pkg.py` (only
`route_ps1_*` and `own_*` replaced; shapes, dtypes and every other array asserted equal; submission.py unchanged, so
no extra code path and no extra runtime: the timing check is identical code on identical shapes). Gate/gallery arrays
(`route_gal`, `bio_*`, `v6_tpri`) are NOT touched, so recognition is identical to stock by construction (asserted).

## Full-system consumption (pending)

Lane A's conditional returning instrument (Galvani, needs VPN) takes a package copy per fold; the G1 arrays for the fold
(visit 1 only) are exactly what Instrument 1 trains, so the full-system arm is "Lane A dose k with G1 fold arrays".
Prediction: equal-cell gain over stock at dose 1/16 +0.3 pp, at k = 4 +1.0 pp (the first-visit head is the weak form).
Registered here only as a pointer; it is scored on Galvani after the VPN is back, labelled FULL-SYSTEM DEV.

## Amendment 1 (19 Nov ~14:25 CET) · written AFTER seeing the local G1 run, BEFORE scoring any new arm

What the first scoring showed (local Mac replica of the Leonardo arrays, same code; Leonardo copy follows): stock product
posterior 77.16 % BA over the six splits; **g1 (registered form) -4.01 pp equal-person [80 % -4.36, -3.69], -2.54 pp at 20/80,
0/8 orders non-negative, negative on both persons**; g1a (augmentation with stock statistics) -4.37; g1c (causal-order matching,
no augmentation) -0.35 [-0.54, -0.16] product, own-stage mean +0.51 [+0.22, +0.80], T -0.38, C +0.34, S -0.47. By the rule fixed
above, **G1 is REFUTED in its registered form** (augmentation = the package's `strong`, incl. 10 % channel zeroing).
Cause (diagnosis, not yet a measured fix): `strong` is made for networks. A zeroed channel makes a covariance singular up to the
1e-3 trace ridge, and the tangent-space log of that matrix puts huge entries into every T feature; T falls -7.8 pp, h -3.8 pp.
The augmentation breaks the feature class, so G1's negative says "this augmentation", not "stronger personal models".
Going past the rule, plainly: AGENTS says a large-ceiling idea gets several forms. Three more forms, fixed now, same split,
orders, scorer and stock control; no choice among them on held data (all three are reported):

* **g2**: g1's causal shuffled-order replicas (R = 5, replica 0 clean) with a covariance-safe augmentation: roll +-15 samples,
  Gaussian noise 0.2 sd, per-channel gain lognormal sd 0.2 (electrode gain / impedance drift); no channel zeroing.
* **g4**: noise 0.2 sd and roll +-15 only (isolates the gain jitter).
* **g1c** stays in the table as the "no augmentation" control, stock as the paired control.

Prediction (numeric, before scoring): g2 product gain over stock +0.6 pp equal-person (80 % [-1.0, +2.0]); g4 +0.4 [-1.0, +1.8];
P(g2 >= +1.5 pp, the ADVANCE bar) about 15 %. Reasoning: the personal members read their own person's second and third visits;
the visits differ mainly in covariance scale and a few channels, which gain jitter imitates; but g1c (matching alone) is flat,
so I expect little. Same decision rule as above (ADVANCE >= +1.5 pp both persons and >= 6/8 orders; REFUTED < +0.5 or negative
on both; else UNDECIDED). The production arrays are NOT cut from a form that is not ADVANCE or UNDECIDED-positive.

## Amendment 2 (19 Nov ~14:40 CET) · after the g2/g4 scoring (local replica; Leonardo copy follows), before the hybrid is scored

g2 (gain jitter + noise + roll) product **-3.66 pp** equal-person [-3.96, -3.35], g4 (noise + roll) **-1.51** [-1.76, -1.26], both 0/8 orders
non-negative: **g2 and g4 REFUTED** by the fixed rule (negative on both persons); g1c still -0.35 product, own-stage +0.51 / orders 4/8.
Prediction miss: I expected g2 at +0.6 and g4 at +0.4. Every augmented replica pulls T, C and h away from the clean test windows
(T -4.5 / -1.2, C -3.1 / -1.0, h -4.0 / -2.4 for g2 / g4; the cost grows with the augmentation strength), so for covariance-class personal
members "augment heavily" is wrong at this data size (about 190 windows per visit). One component moved the other way: the slow-course
LDA S under g4: +0.76 pp [+0.57, +0.94], 8/8 orders, both persons positive (P04 +0.92, P10 +0.61); g2 S +0.00.
One exploratory arm, registered now, chosen BECAUSE of that component result (so selection-biased on the very 6 splits; it can only
be confirmed on other people, i.e. the NI-V Scherer returning enrolment): **gS** = stock arrays with only own_s_W / own_s_c taken
from g4. Prediction: product gain over stock +0.15 pp [-0.2, +0.6] (S is a third of the own-stage mean log and the stage enters at
beta 0.1 x ramp in the package, so any full-system effect is <= 0.05 pp). Not an ADVANCE candidate by itself; it is built as the
package copy only to show the format and to confirm on Scherer if that arm exists. Production arrays of the refuted G1 are kept
only as the format proof (laneG/make_pkg.py); they are not a candidate.

State of the lane after this: **the registered hypothesis (heavy augmentation makes the covariance/FBCSP personal model stronger) is refuted in
three forms and is not the route**; what is NOT tested is the other half of the brief, a fine-tuned neural personal member (arm N), which
needs a source-fold pretrain (the shipped nets saw P04/P10 visits 2-3), and a head with its own regularisation search. Keep both open.

## Results (INDIVIDUAL-MODEL, P04 + P10, 6 splits x 8 pinned orders; Leonardo arrays 64848092-97 and 64849648/719, local Mac replica identical to 2 decimals)

Gain over the stock recipe (same split and order), product posterior h x own-stage; equal-person mean [80 % interval over orders], 20/80, P04 / P10, orders non-negative.
Stock product BA 77.16 % (h 70.66, T 63.49, C 64.64, S 59.76, own mean-log 71.17).

| arm | verdict (rule fixed above) | product, equal-person | 20/80 | P04 / P10 | orders >= 0 | own-stage only |
|---|---|---|---|---|---|---|
| g1 (registered: strong aug, causal replicas) | REFUTED | -4.01 [-4.36, -3.69] | -2.54 | -6.46 / -1.57 | 0/8 | -3.42 |
| g1a (aug, stock statistics) | REFUTED | -4.37 [-4.62, -4.13] | -2.92 | -6.79 / -1.95 | 0/8 | -4.60 |
| g2 (gain jitter, amendment 1) | REFUTED | -3.66 [-3.96, -3.35] | -2.57 | -5.47 / -1.85 | 0/8 | -2.65 |
| g4 (noise + roll, amendment 1) | REFUTED | -1.51 [-1.76, -1.26] | -1.89 | -0.88 / -2.14 | 0/8 | +0.01 |
| g1c (causal replicas, no aug) | undecided / flat | -0.35 [-0.54, -0.16] | -0.49 | -0.10 / -0.59 | 0/8 | +0.51 [+0.22, +0.80] |
| gS (exploratory, amendment 2: stock + S member of g4) | selection-biased, see below | +0.66 [+0.43, +0.91] | +0.70 [+0.44, +0.96] | +0.60 / +0.73 | 7/8 | +1.25 [+1.06, +1.43] |

Reading. (1) The registered claim, "heavy augmentation raises the covariance/FBCSP personal model", is refuted in four forms: every augmented replica
costs T, C and the PC256 head (g2 T -4.5, C -3.1, h -4.0; g4 T -1.2, C -1.0, h -2.4), more the stronger the augmentation. Causal-order
matching alone does nothing (T -0.4, C +0.3, S -0.5, h -0.6). At about 190 windows per visit the clean window distribution of the next visit is
better matched by clean training windows than by perturbed ones. (2) Only the slow-course LDA S gained, under noise augmentation (+0.76 [+0.57, +0.94],
8/8 orders, both persons); with the stock T/C/h it lifts the own-stage mean log by +1.25 and the product by +0.66 (predicted +0.15: the prediction
missed; the larger own-stage gain suggests that S was over-confident and that noise acts as regularisation). gS was chosen because of that
component, on these same splits, so +0.66 is an upper-biased estimate and is NOT an ADVANCE claim. The ADVANCE bar (>= +1.5 pp product) is not
met by any arm. Instrument limits: two people; the 256-PC view basis was fitted on all public data (common to every arm, favours h); a
full-system effect needs Lane A's returning instrument (pending, Galvani) and is expected to be smaller than the individual-model one because
the own stage enters at beta 0.1 x ramp and h at 1/16.

## Production packages (all public sessions, format of the package's personal stage)

`laneG/make_pkg.py` splices per-person rows into a byte-for-byte copy of the sealed cap250 ZIP (SHA-256 9c4d5538...7d9117ed): only the listed
arrays change, every other array and archive member (submission.py included) is asserted equal, gate/gallery/dose keys asserted unchanged
(recognition identical to stock by construction), `check_package` passes. Packages in `runs/laneg/` (gitignored, receipts in `laneG/`):
`lane_g_gS.zip` sha256 87539dc0e35e0a526291533c4ea5ea5c887adc2f18c4dddb09f79e279f19ff88 (own_s_W/c from the g4 fit on all sessions of each of the ten
people: P04 and P10 three visits x 5 replicas, the other eight one visit); `lane_g_g1_refuted.zip` sha256 37dfc1dd...abae (format proof of the
refuted G1; do not upload). Runtime: the code path and tensor shapes are identical to stock (no submission.py change), personal-stage cost
7.14 -> 7.21 ms/window (gS) and 7.53 -> 7.45 ms/window (G1), i.e. zero within timer noise; no extra runtime per routed window.
Not built: arm N (neural personal member), see above.

## Next decisive steps

1. Confirm or drop gS on the NI-V Scherer returning enrolment (9 other people; the only data not used in this selection) when evidence/wave_19nov/NIV.md has it.
2. Full-system: Lane A fold packages with the gS (and stock) fold arrays, 8 orders, dose k in {1, 4} (Galvani, after the VPN).
3. If a personal-model route is pursued: arm N with a source-fold pretrain (needs Galvani's D-EArN banks), and per-member temperature / regularisation chosen inside the
   training visits only (an own-stage calibration is equivalent to a per-member beta, which Lane A's dose sweep partly covers).

## NI-V full-system score of gS vs cap250 (pre-registered 19 Nov, before any NI-V number exists)

Candidate `runs/laneg/lane_g_gS.zip` (sha256 87539dc0...19ff88) via `score_candidate.sh laneG_gS cap250` (B-new 36 tasks, enrolment, B-ret, zero-routing, collector;
full-session shuffle seeds 71000-71007 headline). Mechanism: S member regularised by noise augmentation; own stage enters at beta 0.1 x ramp, so it acts only for
returning people. Prediction: B-new delta ~0 (|d| < 0.2 pp; recognition identical by construction, no returning stage), B-ret delta +0.1 to +0.3 pp (the individual
+0.66 shrinks by the beta 0.1 weight and selection bias), sealed-weighted (0.57 new / 0.43 ret) about +0.1. Refutation rule: returning-arm delta 80 % lower
bound <= 0 AND sealed-weighted 80 % lower bound <= 0 -> do not upload gS. Veto: any new-arm delta 80 % upper bound < -0.3 or a zero-routing recognition change. Scherer instrument limits
(22/41 interpolated channels, stock BA ~47 %) apply: sign and veto evidence only. Dev P04/P10 full-system returning replay: no scorer exists (NIV.md: not recomputed); not run.

## Result: NI-V score of gS vs cap250 is null BY CONSTRUCTION (silent no-op, 19 Nov); gS is NOT tested there

`niv/enrol.py` (`person_rows` -> `own_arrays`) rebuilds `own_s_W/own_s_c` (all `own_*`) for every enrolled Scherer person and appends them
as new rows; gS only replaced rows of the ten PUBLIC people, which a Scherer person never routes to as the own stage (a returning Scherer
person is enrolled as row 11+). The NI-V delta gS minus cap250 is therefore exactly 0.00 (`ni26_niv_v1_20261119/out/report_laneG_gS_vs_cap250.json`,
jobs 3137801-3137805). That is no evidence for or against gS; the NI-V pre-registration above did not apply. gS acts only for the 10 public
people (P04, P10 are the only ones with held visits), so the next section scores it there.

## Full-system returning score of gS (pre-registered 19 Nov, before any number; Lane A instrument)

Instrument: Lane A's returning replay (`laneG/gs_worker.py` on `laneA/dose_worker.py` machinery, release
`/mnt/lustre/work/gehler/gfh527/ni26_laneG_gs_returning_20261119`): fold packages P04_s23/P10_s23 (visit 1 enrolled, visits 2, 3 held), causal EA
members, real refits, count filter, 8 orders (within0-3, full0-1, runs0-1; full0/full1 = full-shuffle primary), cells P04|2, P04|3, P10|2, P10|3.
gS arm = stock fold package with ONLY the held person's `own_s_W/c` row replaced by the g4 fit on VISIT 1 ONLY (the production gS used all
visits and would leak the held ones; it is not what is scored here). Arms: stock, ctl (stock row rewritten through the same code path),
gS. Gain = gS minus stock per (order, cell). Limits: two people, one training visit (production gS sees three for P04/P10), gS was selected on
the same individual-model dev (selection bias remains; the held visits and the g4 recipe are the same instrument family).
Sanity (fail loudly): zero arm == authenticated native zero; ctl == stock bitwise on every tape and exactly 0.0 BA; gates, mix and recognition tapes
identical; gS must change scores and at least some decisions on routed windows.
Prediction: weighted (0.2 P04 + 0.8 P10) gain +0.2 pp, equal-cell +0.2, P04 +0.3, P10 +0.1 (the own stage enters at beta 0.1 x ramp; the +0.66 individual
product gain shrinks, and P10 recognition is late, age 28); full-order primary within 0.1 of the all-order mean. 80 % interval about +-0.5.
Refutation rule: weighted gain with 80 % lower bound (orders resampled within person, 4000 draws) <= 0 -> do not upload gS (people+orders nested
interval of 2 people reported, descriptive). An interval straddling 0 with positive mean is "undecided", not closed (AGENTS).
