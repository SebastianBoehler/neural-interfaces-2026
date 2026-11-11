# Review, 5 Nov 2026 (night of the sealed opening): audit, research, gates, upload plan

Inputs: a 37-agent workflow (code audits with independent verification, five research blocks, three fresh-eyes
ideators, synthesis, two-lens adversarial challenge of each shortlisted idea), plus our own saved-probability gates.
Agent findings are marked "agent" where we did not re-run them ourselves. Scratch evidence:
`/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/95eb4b0e-c5db-4ae9-81b3-afeeeac7221a/scratchpad/`
(`own/` = ours, `p1/` = agents). Every control below reproduced cand_nxif exactly (s011 sel +2.91).

## 1. Live sealed-phase facts (checked 4–5 Nov, public API and Codabench source)

- Phase 30483 is live: 1 upload per UTC day, 5 in total, until 21 Nov 23:59 CET. **The daily quota resets at
  00:00 UTC (01:00 CET)**: verified in the Codabench source, not guessed. A failed run uses no quota.
- **New worker image `lionelyneuro/neural-compet:v4-sealed`**, unpublished (Docker Hub has only v2/v3). Every
  proof we have (real worker 1024097/1024447/1026185, Apptainer replica) is v3. The sealed task was rebound on
  4 Nov 21:31Z (task 36954): same ingestion program, new scoring program, about 25 KB of private input data (agent).
- **Fail-over fix:** if a TorchScript package fails, upload `sealed_candidate_ensemble_all_r19e_g.zip`
  (`125503e8…`, no `.pt`, real worker 1024097, labels identical to jit_g). Not `sealed_r19e_jit_g.zip`, which
  also carries a TorchScript member.
- **Cohort now private.** The organisers removed "20 participants × 6 sessions" and "ten new participants"
  (website commit 44ea882, FAQ 6f1a05f9). PROTEUS has 22 participants. The 57 % unseen share has no basis.
- **Organiser sealed baselines** (from scratch, public 6-person train split, no alignment): EEGNet 0.4927,
  SincShallowNet 0.5876. The same EEGNet scores 0.326 on the public split: the sealed set is much easier than
  P04/P09. A cand_nxif score below about 0.60 would point to a broken run (image, montage, units), not the model.
- Owner API (`/api/submissions/<id>/`) returns 10-decimal scores; the public board shows 2 decimals (agent, from
  the Codabench serializers). Paired differences between our own uploads are exact on the final objective.
- Board at 01:39 CET (2 decimals, 12 entries): JLShen 0.81, anarghya 0.78, lorenzoveronese 0.77, foreist 0.76,
  huytungst 0.76, teddythebuilder 0.75, aimarsilvan 0.72, adoroodchi 0.71, judeswallis 0.71, noiroao 0.68,
  syehsyoh 0.66, arnaultcaillet 0.48. Durations 32–1,414 s. **Our day-1 entry was not on the board at 01:39**
  (status unknown to this session: still queued, failed, or not uploaded).
- Rules fact (not a recommendation): quotas are per participant; a team keeps its best member score.
  Registration closes 5 Nov 23:59 CET. Earlier decision: no teammates.

## 2. Does the code do what we intend? (audits, each finding independently verified)

- **submission.py (frozen 25d51a6e): no correctness, causality or state-leak bug.** Staged cand_nxif parameters
  match the docs. Robustness probes on one session (amplitude ×10 / ×0.1, flat Cz, NaN window, permuted named
  montage, spikes) change 0–16 labels. One weakness: SessionEM never forgets (two P10 sessions in one stream:
  session 3 drops 0.995 → 0.930); a real sealed stream is one session, so this is a pessimistic proxy.
  CPU-only fallback would take about 29 min for 21k windows.
- **Dev harness: valid.** Metric = official cell mean; folds keep the held-out person out of everything that
  matters; stream order and the causal rebalancing/EM simulation equal submission.py. Two documented leaks
  (teacher EA prior G and EM event prior from all windows) are each worth < 0.1.
- **Teacher pipeline:** dev features equal the packaged teacher input. Three train/test mismatches remain:
  the corpora keep their native references (PROTEUS is M1), corpora with ≤ 4 s task intervals are windowed
  cue+0–4 s (PROTEUS cue+1–5 s), and pretraining EA is full-session (test is causal, k 5).

## 3. What the evidence really says

- **P10 drives 88 % of cand_nxif's +2.91 sel** (its weight in sel is 65 %; agent, re-derived from per-cell
  deltas). Per person over all 10 (all_loso, diagnostic): mean +1.12, se 1.25, 6 of 10 positive; without P10
  +0.32. P09 −5.9 and P06 −2.1 grow with the teacher weight.
- **Sealed forecast for the teacher (cand_nxif − R19e): +1.0 ± 1.2 points**, not +2.9.
- **The teacher also helps a second seen person (new tonight):** seen_session, own session 1 in training:
  P04 +1.14, P10 +2.31 at w 5 (P04 −0.72 at w 3).
- **Teacher weight, person-equal:** without P10 the curve is flat from w 2 to w 5 (s011: w 3 +0.82,
  w 5 +0.99; s611: +1.06 / +0.83) and falls steeply above w 5.5 (w 7 −0.14). **w 7 is dominated; w 3 is only
  a mild hedge** (w 3 − w 5 forecast −0.1 ± 0.5).

## 4. Gates run tonight (all on saved probabilities; all closed unless stated)

| Idea | Result (s011 sel Δ over R19e; cand_nxif +2.91) |
|---|---|
| Log-linear (PoE) pooling of the teacher | best +1.65 |
| Teacher pre-rebalanced before mixing | +2.80 |
| Map fitted with boosting weights (members' out-of-fold error) | +3.06, but dla 77.5 vs 78.0; independence 0.44 → 0.46 |
| Participant-balanced map | +2.37 |
| Pool-aware stacked map (agent) | −1.8 |
| Per-session label-free teacher gate | no statistic predicts harm (teacher skew r −0.40 with P05/P04 counterexamples) |
| Common-average-referenced input to the existing trunks | +2.48 (P09 still −4.9) |
| EM λ 0.4–0.7 with the teacher | best person-equal λ 0.7, but ds −1.6; not better than a115 |
| Agents: NAP/WCCN/CMN teacher normalisation, position-scheduled weight, seed-dispersion gate, m14r person-equal, forward-only teacher bundle (TTA, 4 Hz high-pass, feedback surgery), emx at w 3 | all ≤ base or refuted on prior evidence |

## 5. What survived

**a115: cand_nxif with `rb_alpha` 1.15 (data-only, frozen code).** α 1.15 comes from the round-3 explorer core
(chosen on dev people); P04 and P09 were never used to choose it.

| s011, Δ vs R19e | cand_nxif | **a115** | emx |
|---|---:|---:|---:|
| seen_session P04 / P10 | +1.14 / +2.31 | +1.43 / +2.03 | +0.07 / +2.05 |
| ds (P10 later sessions) | +2.63 | +2.35 | +1.77 |
| dla, person-equal (8) | +1.77 | +1.96 | +2.29 |
| all_loso without P10, person-equal (9) | +0.32 | +0.86 | +1.07 |
| P09 / P04 all_loso | −5.9 / +1.9 | −1.8 / +1.6 | +0.8 / −0.6 |

On the hidden pretraining seeds 6–11, P09 behaves the same (−5.4 → −1.3). Mechanism (round 2/3 profiling): the soft
teacher leaves a session class bias that α 1 under-corrects; α > 1 gains sit in the most skewed sessions.
a115 keeps the returning-person score and repairs most of the teacher's worst new-person loss.
**Honest size:** on the 8 dev people the α curve is flat from 1.0 to 1.25 (s011 dla7 0.84 / 0.99 / 0.68 at
α 1 / 1.15 / 1.25; s611 0.68 / 0.46 / 0.60), ds falls slowly (2.63 / 2.35 / 2.04), seen P04 rises with α. The case
rests on P09 and seen P04: a cheap, low-risk arm, not a breakthrough.
Package: `runs/sealed/cand_nxif_a115` (local, `scripts/set_params.py … rb_alpha=1.15`): harness 1.0 / 1.0,
3 of 2,655 public labels changed, `weights.npz` `7eda1d06…`, `submission.py` and `teacher.pt` unchanged.
**Staged 5 Nov 07:00:** official A100 replay 3055253 (102 s) and worker replica 3055254 (68 s), files unchanged,
identical scores (in-sample 0.96463, as cand_nxif). Upload ZIP `runs/upload/sealed_cand_nxif_a115.zip`
`841e7605…` (byte-identical to both replays' submission.zip).
emx (adds EM shrink 0.85, n0 40) gains more for new people but loses about 0.7 on returning people; its EM half
is negative on every readout except dla (agent challenge).

**Open, needs Galvani: reference-matched teacher (wave 11).** Three agents independently ranked it first among new
levers (p 0.07–0.18): after EA, PROTEUS's M1 reference is a session-dependent rotation the trunks never saw.
`scripts/export_moabb.py --ref M1` (new, unit-tested: a common native-reference signal cancels to 6e-8; default
path byte-identical) re-references every corpus to a spline-interpolated M1. `runs/diag/wave11.sh` (not in git)
exports, pretrains EEGNeX + IFNet seeds 0–5 (variants m7m1, and m7m1d with Dreyer2023 native) and screens against
the all-A100 base pair. The pre-registered gate is in the script header. The test-time CAR gate above lowers its
prior but does not test it.

**Board, 5 Nov ~21:00 CET (98 entries):** seohee 0.825; eight entries at 0.805–0.810; 0.783–0.797; us 0.779 at rank
17. Several entries may be one team (team members appear separately). Simple models on the sealed set: a plynk
tangent-space model 0.762, FBCNet 0.63–0.64, EEGNeX 0.556, CBraMod 0.513 (agent, from model names and the board).

## 5b. Approach tournament (5 Nov, person-equal scorer `scratchpad/own/pe_metric.py`)

Metric: J = 0.5 × person-equal all_loso (10 people) + 0.5 × person-equal seen_session (P04, P10), deltas over R19e.
cand_nxif J +1.38 (new +1.03, seen +1.73), a115 +1.65 (new +1.56, seen +1.73).

- **CAR for the deep members: closed.** Seeds 0–2 on Galvani, each CAR member swapped for its M1 twin: all deep
  CAR dJ −0.29, with CAR v2 −0.52; an M1 + CAR mixture +0.26 ± 0.37 (noise). CAR helps v2 alone (+1.85) but not the ensemble.
- **Cleanup ablations: the ensemble is already lean.** Removing any component costs J: EM −1.85, teacher −1.38,
  ShallowConvNets −1.37, FBCNets −1.22, rebalancing (α 0) −5.75. Only the trainable-FIR FBCNets are removable at no
  cost (+0.08 to +0.25, noise).
- **Existing alternative architectures** (EEGTCNet, EEGITNet, MSVTNet; 45–48 % alone) lower a115 when added.
- **Newly trained deep families** (ATCNet from the m7r trunk, EEGNeX, IFNet, Inception-ERP): none is a new class; a
  deep-only ensemble is far worse (J −4.5 to −8.5). EEGNeX in the trainable-FIR slot of a115: J +2.17 (≈70 configs scored).
- **Seen-person specialist: closed again.** Cross-session identification reaches at most 10/12 sessions after 60 windows
  and cannot reject new people (AUC 0.5–0.65).
- **Wave 11 (M1-re-referenced teacher corpora): closed.** Best s05 +2.47 vs +2.58 for the hardware-matched base.
- **Trained-in per-session latent normalisation (BEETL 2021 winner): the first large member-level jump.**
  Batch norm uses per-session statistics in training and causal running statistics of the session at test.
  FBCNet alone: 59.1 → 66.0 (all_loso, person-equal) and 58.1 → 64.1 (seen). ShallowConvNet: 58.0 → 62.4 / 58.3 → 65.4.
  EEGNeX: no gain. In the ensemble (seeds 0–2, plain twins = controls at a115 level):
  fbcnet_ea_rot05 replaced by the session-normalised FBCNet (k 2) J +2.14; the same nets at k 5 added
  ("arm D") **J +2.43** (new +2.07, without P10 +1.57, seen +2.80) vs a115 +1.65. Ensembles of session-normalised nets
  only are worse (J +0.75), so the heterogeneous members still matter. Needs a ~20-line submission.py change (a
  session-stateful TorchScript member). Build, a pre-registered fresh-seed confirmation (seeds 3–5) and a code review
  started 5 Nov ~20:15 CET.
  - **Confirmation (seeds 3–5): narrow FAIL of the pre-registered bar.** D − a115 = +0.30 J (bar +0.40); new9 +0.98
    (≥ +0.76 ✓), seen +2.39 (≥ +1.73 ✓), plain-twin control +0.02 (✓). The member-level gain replicates (FBCNet alone
    58.5 → 65.7); the ensemble keeps about 40 % of the dev estimate. Unbiased estimate: about +0.3 J over a115, mostly
    from returning people. P06 and P10 lose about 2 points each.
  - **Package `runs/sealed/cand_sbn`** (`submission.py` 6e526e3a: +29/−2 lines, class SJit; 12 SJit members = 6 seeds
    × k 2/5 trained on all 10 people; everything else = a115): harness 1.0 (2,655/2,655), 14 labels changed vs a115;
    old packages byte-identical with the new code; official A100 replay 141 s and worker replica 99 s, files unchanged,
    identical scores; CPU/CUDA agreement 1.0. ZIP `runs/upload/sealed_cand_sbn.zip` `cee75861…`. Estimated sealed
    runtime 20–23 min. Code review: safe; never run on the real v4-sealed worker. Audit recipe: docs/AUDIT_SBN.md.
    Uncommitted.
  - **Structure search (selection on member seeds 0–2, judged on 3–5):** REC = D + session-normalised FBLight (k 5),
    SincShallow (k 20) and raw FBCNet (k 20): held-out **J +2.27** (new +2.16, new9 +1.65, seen +2.37) vs D +1.95 and
    a115 +1.65. Paired by seed: REC − a115 +0.56 (SE 0.15, 6/6), REC − D +0.14 (SE 0.14). Two-fold estimate of the
    subset search about +2.14. All 8 slots session-normalised (sbn8): held-out +2.31 but new +1.20 (below a115), with
    the gain on the two seen people. 89 % of 56 subsets beat D held out. REC build started 6 Nov ~00:05 CET.

## 5c. Solution-space map and the strategic re-framing (6 Nov, 01:00–02:30 CET)

- **Board (102 entries, 6 Nov 00:10 CET):** 0.825; 0.810 ×2; 0.809; 0.808 ("LDA"); 0.806 ×3; 0.805; 0.797 ×2; … us
  0.779 (rank 18). All entries are still first uploads.
- **Constraints we imposed on ourselves, not the rules:** runtime (we used 835 of 3,600 s), package size (10 MB of
  15 GB), the code freeze, stateless members. The rules also allow unlabelled in-domain PROTEUS EEG (rest runs,
  inter-trial periods, trial tails), which we never used. pyriemann 0.12 is importable on the worker replica.
- **Coverage audit:** the Riemannian/tangent-space core (v2) was designed in one day on 18 Oct and never redesigned
  (3.6 % of saved experiments are linear). Never tried: many/narrow bands, sub-window or delay-embedded covariances,
  an AIRM reference, shrinkage estimators, MDM/FgMDM, FBCSP+LDA, TS ensembles, nonlinear heads, TSMNet SPD-BN.
- **Board reading:** tangent-space entries span 0.628–0.762 (v2-like recipes 0.628/0.673, plynk 0.762). Our stack is
  about v2 + 16 per cell on dev, so our v2 alone is probably about 0.62–0.66 sealed. Our stack works on the sealed
  set, but our core is about 10 points below the best TS recipe. Board score correlates with runtime (Spearman 0.66);
  all entries ≥ 0.80 run 430–3,410 s.
- **Regime check (P04/P10 later sessions):** our margin over v2 does not shrink on later sessions; no regime knob
  moves dev by more than about 1 point.
- **Upload design:** no current candidate reaches 0.81 under any plausible dev-to-sealed mapping (P ≈ 2 %). The most
  informative next upload is a calibration probe, **cal_v2** (v2 + rebalancing only; frozen code, `weights.npz`
  only): about 0.62–0.67 → stack fine, build a much stronger core; near 0.78 → the stack is broken on sealed.
  Package `runs/sealed/cal_v2`: harness 1.0, `weights.npz` `b33c16c6…`; official A100 replay 3060075 (68 s) and
  worker replica 3060076 (71 s), files unchanged, identical scores. ZIP `runs/upload/sealed_cal_v2.zip` `cdab9b17…`.
- **REC package staged (6 Nov ~03:00 CET):** `runs/sealed/cand_rec` = cand_sbn + session-normalised FBLight (k 5),
  SincShallow (k 20) and raw FBCNet (k 20), 6 seeds each, trained on all 10 people (export 3059663). Harness 1.0
  (0/2,655 differ), 22 labels changed vs a115, 10 vs cand_sbn; every SJit member traces cleanly (saved state zero,
  stream vs eager ≤ 5e-5). Official A100 replay 3059664 (171 s) and worker replica 3059665 (146 s): files unchanged,
  identical scores. ZIP `runs/upload/sealed_cand_rec.zip` `ab6adf65…` (11.3 MB). Expected sealed runtime about
  30 min (2.2× a115 in the replica). The independent review died on a usage limit; its SJit mechanism is shared
  with the reviewed cand_sbn.
- **Tangent-space core search (48 configs on Galvani CPU; harness reproduces v2 exactly): a much stronger causal core.**
  car_ov17_fs2_t0.1_le_k2 + LR C 10: common-average reference with fresh covariances, 17 overlapping bands 1–58 Hz,
  full-window plus two half-window covariances per band, fixed trace shrinkage 0.1, causal log-Euclidean recentering
  (k 2), slow block. Standalone person-equal all_loso **69.3 vs 58.8** for v2; seen 66.9 vs 57.7. Selection on one half
  of the people, judged on the other: +8.5 and +10.0. Drivers in order: shrinkage (+6), sub-window covariances (+3.3),
  CAR (+1.7 to +4.8), higher C / smaller k, overlapping bands (+0.6). Not helpful: OAS/LW, AIRM reference (equal, 9×
  slower), LDA/SVM heads, per-band ensembles, dropping the slow block (−3.8). In the a115 stack (core replacing v2 at
  weight 1–2): J 2.1–2.7 vs 1.65. Side findings: the 5 Nov "CAR +1.85" applied CAR only to the slow block (the feature
  cache ignores the input); models.TangentSpace with stretch=True is not causal at test time (unused by v2).
  A REC + new-core arm (pre-registered choice on people halves) and its build started 6 Nov ~02:10 CET.

## 5d. Ground-up redesign from the bottleneck (6 Nov, Galvani; scratch `scratchpad/gu/`)

Base: REC + core (J 76.09 absolute; new 76.82, seen 75.36). One calibration point: sealed ≈ J + 3.65 (cand_nxif).
- **Bottleneck = per-session class directions.** A head fitted on the session's own labels over the core's tangent
  features adds +6.0 J (new +7.6) and sits in the session's top 10–30 unsupervised components. Causally, even with true
  labels, only about half is reachable (+3.0; nothing before window ~96). Closed after cross-validation: label shift
  (−1.35; rebalancing already leaves 0.07 class-share error), per-session member weighting (+0.02), a global stacker
  (−4.8), the combiner re-tune (8,197 candidates, none ≥ +0.4 held-out both ways).
- **Design A, label-free session adaptation: TSEM survives.** Per-session Gaussian class templates in the training
  top-128 PCs of the core's causal tangent features, seeded by the 70 % most confident past outputs of the full
  pipeline with a per-class quota, multiplied in with exponent 0.15·t/(t+60). Nested LOPO over 264 configs +1.73 J
  (new +2.02, 8/10 ≥ 0; seen +1.44); halves +0.86 / +2.63; other PCA basis +1.57 to +2.02; 11/14 sessions positive.
  Seed quality drives it (weak base +0.3, strong +1.4, true labels +3.7), so base gains are amplified. Head
  self-training stays closed (windows are near-orthogonal in the 44k-dim space).
- **Design B, a learned session-conditioned network: closed.** Context branch −1.8 to −2.7 per seed vs SessionBN;
  meta-trained on 659 external sessions it only reaches parity; frozen −4.5. It relearns rebalancing, nothing more.
- **Core 2 (round 2):** core1 + a Hilbert (analytic-signal, complex) 7-band core, mean at slot weight 2: +1.57 J over
  REC (sd 0.26, 6/6 member seeds, both halves); with per-session z-scoring +1.65. Searched on all 10 people: shrink to
  about +0.7 ± 0.4 over REC + core. Round 2 final: both cores with per-session z-scoring (Z+HS, lr10, w1.5) is +1.74
  (6/6); not adopted, because z-scoring overlaps with TSEM (arm3 + TSEM 77.87 vs arm4 + TSEM 78.20 absolute, seeds 3–5).
  Closed in round 2: delay embedding, more/wider bands, 3 sub-windows, other shrinkage, MDM heads, CAR+M1 concat.
- **Running (6 Nov ~11:30 local):** pre-registered TSEM confirmation (exact SVD, fresh seeds, on REC + core2; bar
  J ≥ +1.0, new and seen ≥ +0.5, ≥ 7/10 people ≥ 0) plus three add-ons; the REC + core2 package; an SPD network
  with per-session SPD batch norm (TSMNet-style, gated vs the core standalone and ≥ +0.5 J in the stack).
- **TSEM pre-registered confirmation: PASS** (Galvani 3061045; exact-SVD bases; streaming class == batch, λ 0
  bit-identical). On REC + core2 arm4 with fresh member seeds 3–5: **+1.71 J** (new +2.38, seen +1.04, 8/10 people
  ≥ 0; halves +3.16 / +1.32); single seeds +1.49 / +1.25 / +1.30; pooled +1.46. On the per-session z-scored core
  (arm3) only +0.89 (the two overlap), so arm4 is the base. Hedge pc256 λ 0.2: +1.30 to +2.01, more people negative.
  Add-ons on top of TSEM (Galvani 3061049, grids fixed beforehand, nested LOPO + halves, bar +0.3): gated pseudo-label
  head −0.22 (TSEM already holds that signal), causal session-PC view +0.16 (halves −0.05 / +0.20), agreement-ranked
  seeds +0.20 (new −0.38): all rejected. TSEM ships as pre-registered (exact PC128, λ 0.15, ρ 0.7, m0 20, shrink 0.7).
- **SPD network with per-session SPD batch norm (TSMNet-style, own float64 PyTorch; prefix-causal, stream == batch):
  closed.** Standalone J 54 vs core 68.5 and SessionBN FBCNet 66.8; negative in the stack at every weight, seed and
  half (−0.1 to −3.3). Diagnostics: the core's strength is its ov17 sub-window input plus the slow block and a convex
  head on recentred tangent vectors; end-to-end SGD with chunk-statistics BN and a learned BiMap loses ~4.5 more.
- **Adversarial audits (6 Nov; `scratchpad/screens/audits.json`, judge `judge_verdict.md`): no leakage, causality or
  label bug in TSEM or core2.** Sizes shrink: core2 vs the packaged (pooled) REC is +1.30, not +1.57 (the paired
  number used weaker single-seed REC bases); after selection on the same people about +0.8 to +1.1. TSEM: the tail of
  the 128-PC basis is poorly determined (24 of 128 directions differ between randomized and exact SVD; J unchanged at
  +2.14 vs +2.18, people move by up to 1), and its family was chosen on all 10 people: forecast +1.2 to +1.4.
  Revised path ≈ 77.4–77.9 J (0.81–0.815 at full transfer); 0.83 now about 8–12 %. Three small pre-registered screens
  follow: a causal Gram-Wiener posterior filter before TSEM (+0.3 expected), a person-equal Sw target for TSEM's
  covariance, and person-equal sample weights for the two cores.
- **Screens (6 Nov afternoon, `scratchpad/screens/`): all three FAIL their pre-registered bars.** Gram-Wiener filter:
  +0.62 on the fresh base, but permuted-Gram nulls give +0.47 (generic shrinkage interacting with TSEM; TSEM's
  response to any input change is noisy at ±0.2–0.3 J). Sw target: +0.47 J but all from seen P04 s3; new −0.13, P09
  −2 on every base. Person-equal core weights: core1 −0.83 standalone (4/10). TSEM on four fresh bases: +1.17 / +0.94
  / +1.38 / +1.69 (mean +1.30), as forecast.
- **Ground-up round 2 (6 Nov, `scratchpad/gu2/`): nothing credible to add to cand_rec_ts2_tsem.** Interface-aware
  rebalancing/EM/TSEM: true-interface oracle ≤ +0.10 (pooling beats specificity at ~95 windows per interface;
  interface undetectable within a session, 48–50 %). Multi-view: no new view adds per-session class information
  beyond TSEM's PC128 (oracle ≤ +0.5); cross-view seeding / co-training / CCA ≤ +0.18; a second TSEM stage on PCs
  129–256 passed narrowly (+0.419 vs +0.40) but the verifier judged it not credible (fresh-base mean +0.20, λ-
  sensitive, a duplicate of the PC256 hedge): kept only as a possible late variant. Returning-person head: oracle
  +2.2–3.4 on seen (90 % P04), but open-set identification fails on PROTEUS (P05 resembles P04 s1 more than P04's
  later sessions) and false accepts cost −1 to −3.6. First principles (feedback-locked, cue-late, ERD/ERS envelope,
  EMG, ECG, IAF alignment, lane-geometry gaze): no view adds beyond PC128; label-free versions ≤ +0.14. Reading: the
  label-free per-session layer is saturated at 82 % seed precision (+0.1 J per seed-precision point); the next lever
  is cross-subject base accuracy, judged after TSEM.
- **Sealed check (6 Nov): cand_rec_ts2_tsem scored 0.8197 (cand_nxif 0.7795), so dev J gains transfer about 1:1.**
- **Round 3, base accuracy for new people (6–7 Nov, `scratchpad/gu3/`): nothing passes, nothing is variant-grade;
  the fresh bases remain unseen.** Screens on B2_pool / B2_s345 after TSEM:
  - CMMN spectral normalisation −0.85 / −0.99.
  - Domain-generalisation heads (mixed effects, group DRO, Fishr, person bagging, synthetic people): negative on both
    cores. On the Hilbert core only, the best is fishr1 at +0.36 / +0.22, with JG1 negative and P10-driven.
  - External within-subject contrasts (Shin2017 MI + arithmetic, latent / joint / anchor fits and a C-sweep): +0.4–0.5
    only on seen people (new −0.35, JG1 negative); every arm of the C-sweep is negative.

  Side finding: the shipped sklearn heads are effectively early-stopped, and fully converged heads are 0.1–0.6 J worse
  in the stack.
- **Post-TSEM re-tune of the chain's global settings (7 Nov, `scratchpad/retune/`, 243 configs): FAIL.**
  - In-sample best is α 1.3 + TSEM λ 0.2 at +0.39, but nested LOPO gives −0.21 and the half-splits −0.31 / −2.22.
  - Pre-registered confirmation on fresh bases: +0.52 / +0.17 / −0.14 / +0.08, mean +0.16 (bar +0.3).
  - The shipped EM λ, slot weight and teacher weight are already optimal; α 1.3 alone is +0.17 on average, positive on all four fresh bases.
  - Keep the shipped settings; α 1.3 + λ 0.2 is at most a near-equal upload variant.
- **Returning-person routing by class-direction fingerprints (7 Nov, `scratchpad/route/`).**
  - **Idea:** class directions are personal, so a session's pseudo-labelled class contrasts should match its own
    person's public sessions and nobody else's.
  - **PROTEUS, true labels:** clean separation at every window count; impostors ≤ 0.22, genuine 0.40–0.54 at the end.
  - **PROTEUS, pseudo-labels:** soft pseudo-labels bias the fingerprint toward the 3-session people (P05 → P04 0.32).
    Top-70 % seeds with debiasing fix it (margin +0.04), but this fix was found post hoc on the same 4 genuine probes.
  - **External check (Lee2019, Stieger2021):** at 1 % false accepts, only 10–42 % of genuine sessions are accepted.
  - **Exploratory routing (weight 0.25, from window 60):** dJ +0.98 / +0.99, new 0.00 with 0 impostor routings, seen
    +2.0 (P04 +3.2 / +4.0, P10 ~0).
  - **Sealed estimate:** +0.2–0.3 points realistic, +0.85 if every returning session is accepted; downside ≈ −0.01.
  - **Next:** under Force_Best only the upside counts, so it is being frozen, confirmed on fresh bases and prepared as
    an upload variant. The track allows within-session adaptation, so this must be declared in the method description.
  - **Pre-registered confirmation on fresh bases:**
    - **Config:** q70 debiased seeds, PC256, threshold 0.20, ramp 0.10, start window 60, weight 0.25.
    - **Results:** dJ +0.73 / +0.86 / +0.80 / +0.79. New is 0.00 on all four bases, with 0 impostor routings. Seen is
      +1.45 to +1.72 (P04 +2.6 to +3.5, P10 about 0). Real − null is +0.94 to +1.19, and the truncation test passes.
    - **Sealed estimate:** about +0.6 points, so about 0.826, if returning people route like P04/P10; +0.2–0.3 at
      external acceptance rates.
    - **Risk:** the closest new-person impostor (P06 → P04, max 0.175–0.193) sits only 0.007–0.025 below the
      threshold. The fresh bases vary member seeds only; the people are still the same 10.
    - **Package data:** `scratchpad/route/pkg/` (route_pkg.npz, route_stream.py; parity with the simulation exact).
  - **Independent review of sealed_cand_rec_ts2_tsem_route.zip (`34f60eca…`): no blocking defect.**
    - The ZIP matches the build bit for bit and is causal and reset-safe. With weight 0 it equals the baseline byte
      for byte. Replays are identical; runtime is about 42 of 60 min.
    - **Model risk: P04 attracts.** P04 is the weakest public person (gallery norm 3.0 vs 4.4–6.5 for the others), so
      a session with no class contrast scores 0.40 against P04 and gets full weight. A weak, BCI-inefficient new
      person would probably be routed to P04's head; the dev gate could not test this.
    - **Estimated harm:** −0.7 to −1.1 per affected session at weight 0.25, about −0.1 to −0.2 on the cell mean.
    - **Possible fix (not built):** a null gallery or minimum-contrast check before routing.
- **Routing v2 (7 Nov, `scratchpad/route2/`): the P04 attractor is fixed; the gain is P04-only.**
  - **Changes:** the shape score cos(P c, P G_p) projects out the population and debias directions; standardisation
    is robust (median/MAD, clip 4); τ 0.22.
  - **Stress tests:** constant, noise, random-label, weakened and artifact sessions route nowhere (v1 sent most of
    them to P04). Stieger FAR is 0.42 %.
  - **Pre-registered confirmation on fresh bases** (new 0.00, 0 impostor routings everywhere):
    - V-fix (w 0.25): seen +0.66 / +1.19 / +1.21 / +1.32; fails the seen bar on B2_s012.
    - V-gain (w 0.5): seen +1.85 / +1.72 / +2.14 / +2.10, passes every bar; P04 +4.0 to +5.1, P10 −1.1 to +0.2.
  - **Sealed estimate (V-gain):** +0.8 by the bar's arithmetic. It is −0.13 (−0.22 to +0.26) if the other returning
    people respond like P10, whose base is strong.
  - **Reading:** personal heads help where the population model is weak for that person (P04, 51 %) and are neutral or
    harmful where it is strong. A per-person trust weight from the public LOSO deficit is the natural next step.
- **Personal models with oracle identity (7 Nov, `scratchpad/personal/`): PS1 passes its pre-registered bar.**
  - **PS1:** the person's own LR head on the PC256 view (C 0.1), placed in the B2 slot instead of core1 at 3×.
  - **Seen gain on fresh bases from window 0:** +4.23 on B2_s012 (P04 +7.1, P10 +1.3), fresh mean +4.03.
  - **From window 60:** +2.46.
  - **Null (a wrong person's head):** −3.
  - **What mattered:** where the head plugs in. In the slot, it improves TSEM's seeds.
  - **Closed:** full 44k-dim MAP adaptation, Hilbert personal heads, a personal gaze-EM prior, time-varying weights.
  - **More sessions:** no evidence that 3 sessions beat 1.
  - **Sealed:** ≈ +1.06 × TAR − 0.86 × FAR, at most about +1 J. Routing v3 (v2 gate + PS1 in the slot + shadow
    unrouted tail for the fingerprint) is in progress.
- **Routing v3 (7 Nov, `scratchpad/route2/prereg3.txt`): PASS on all four fresh bases.**
  - **Design:**
    - **Gate:** the v2 gate reads an unrouted shadow tail.
    - **Ramp:** r = clip((s − 0.22)/0.05, 0, 1).
    - **Personal model:** PS1 is injected into the core1 slot at 3× of a separate rebalancing → SessionEM → TSEM tail.
  - **Results:**
    - dJ +1.12 / +1.06 / +1.04 / +0.99; seen +1.98 to +2.23 (P04 +3.4 to +4.2, P10 +0.0 to +0.7).
    - New is 0.00 with 0 impostor routings on every base.
    - Real − null +1.63 to +1.77.
  - **Stress tests and false accepts:**
    - All stress tests route 0 sessions, including with feedback through the full chain.
    - Stieger FAR 0.42 %, at a mean r of 0.085 for the false accepts.
    - A false accept costs about 0 at the typical r of 0.1, and −1.3 to −1.9 at r = 1.
  - **Sealed estimate:** central +0.33 if the 9 other returning people behave like P10; +0.9 if they route like the
    P04/P10 mean; +0.1 to +0.4 at external acceptance rates.
  - **Status:** the package build has started (cand_rec_ts2_tsem_route3).
- **Gated early boost "GEB" (7 Nov, `scratchpad/eb2/`): passes all pre-registered gates and replaces EB.**
  - **Mechanism:** EB's h(t) is multiplied by g = clip((0.35 − m̄)/0.15, 0, 1), where m̄ is the running mean margin of
    SessionEM's input over windows 0..t−1. Sessions that are already confident early get no boost.
  - **Cross-fit on fresh bases:** +0.54 (new +0.88, 10/10 ≥ 0; seen +0.20; P10 +0.01). Joint sign-flip null
    p = 0.001.
  - **Honest increment over EB:** +0.18 (paired p = 0.09), mostly seen; P10 is +0.5.
  - **Permuted-gate null:** real − null +0.24 (below the +0.3 bar). A constant gate gives +0.39.
  - **With routing v3:** +1.80 against the shipped chain, versus +1.53 for EB + routing and +1.05 for routing only.
  - **Cost:** 2 weight keys and ~20 code lines; no extra chain.
- **Lean / single-model study on the route3geb chain (7 Nov, `scratchpad/lean/`): no lean or single model matches.**
  - **Leave one family out:** every member family costs ≥ 0.3 J except the TF FBCNets (+0.10 / +0.17).
  - **Lean configurations:**
    - slot only −6.4;
    - slot + teacher −5.0;
    - slot + SessionBN FBCNet −4.2;
    - core1 only −9.3.
  - **One concat LR over both cores:** standalone 69.9 / 68.2, but chain-only −7.4 against the full system and −0.3
    in place of the slot.
  - **MLP heads:** all below the slot.
  - **Runtime:** the package computes 20 weight-0 nets that are never used (2.7 min wasted). Halving the SessionBN
    seeds (3 of 6 per group) changes dJ by −0.02 (range −0.17 to +0.13).
  - **Robust lean rebuild:** about 29 min instead of 41.9, as a weights-only change (not built or replayed).
  - **Reading:** our strength is the ensemble plus session adaptation; no single component is near the leader's
    single-model strength.
- **Overnight research programme (11 Nov, 29 pre-registered runs, `runs/archive_scratch_11nov/night/`): nothing passes.**
  - **Best fresh result:** +0.07, against a noise calibration of −0.22 to +0.10.
  - **Predictions were well calibrated:** 43 of 54 fell inside their 80 % ranges.
  - **Diagnostics:**
    - errors are spread over all pairs (MI↔WORD 37 %, SUB↔WORD 36 %, MI↔SUB 27 %);
    - WORD is the weak class (recall 73.7 %, seed precision 78 %);
    - true-label TSEM seeds are now worth only +0.93;
    - P04 accounts for 263 of 622 new-person errors;
    - on WORD errors the deep families are right about 41 % of the time and core1 18 %.
  - **Refuted:**
    - run-level without-replacement priors (−0.76);
    - SUB/WORD-specific TSEM covariance;
    - Dawid–Skene member reliability;
    - common-mode / mastoid EMG views (≤ +0.11 even with true labels);
    - mirrored-MI training (−0.25);
    - seed-rule, shrinkage, pair-exponent and family-column refinements;
    - longer rebalancing windows (−0.44; the 30-window scale is load-bearing).
  - **Reading:** the label-free session stage is at a local optimum. Further gains need new base information, i.e.
    better members for WORD and for P04-like people.
- **ST-EEGFormer member (11 Nov, Galvani 3085914, `runs/archive_scratch_11nov/steeg/`): FAIL.**
  - **Model:** ST-EEGFormer-small (25.3 M parameters, MIT weights), authors' fine-tuning recipe, 50 epochs, two seeds.
  - **Standalone:** new 53.85 and J 53.79, against a prediction of 60 (55–66). That is 15 points below core1, as with
    CBraMod, LaBraM and REVE.
  - **In the chain:** +0.16 nested (prediction +0.0, range −0.3 to +0.4). At w 0.5 it gives new +0.21 and seen −0.06;
    w 1 costs seen −0.54.
  - **Gate:** fails on both the standalone and the nested bar, so the fresh bases were left unscored.
  - **Reading:** this closes the foundation-model member line. About 2.4 k windows from 9 people cannot adapt a large
    pretrained encoder beyond our tangent and filter-bank cores.
- **Late second-opinion head after the full route3geb chain (11 Nov, `scratchpad/secop/`): FAIL.**
  - **Heads tested** (Codex's lead): channel power, Hann + lag energy, Fisher-z 8–13 Hz rank correlation, rank
    logistic on log covariance.
  - **Blend weights:** 5 / 10 / 15 % into the final posterior.
  - **Result:** every config is negative on exploration and on fresh bases. The pre-registered pick (channel power, 5 %)
    gives −0.21 fresh (null p 0.98).
  - **Shuffle control:** shuffled head posteriors lose as much as the real ones, so the heads add no window-specific
    information to our chain.
  - **Why:** they help only weaker bases (core1 or small ensembles: +0.15 to +0.26). Codex's +0.60 is a three-person
    slice effect on a weaker base.
  - **Scratch archive:** macOS /private/tmp cleanup has started deleting scratch files. The full scratchpad (3.3 GB) is
    archived in `runs/archive_scratch_11nov/` (gitignored).
- **Single deep model replacing the 30-net ensemble (7–8 Nov, `scratchpad/single_deep/`): FAIL.**
  - **Best case:** the wide FBCNet (18 bands, 64 filters; 67.6 new standalone) is −2.8 to −3.1 J through the chain,
    even at a post-hoc weighting.
  - **Other variants:** longer training with augmentation, fully fine-tuned EEGNeX/IFNet and an FBCNet with a tangent
    branch all lose 3–12 J.
  - **Why:** the ensemble adds about +6.5 J over cores only, and one model recovers at most about half. The gain is
    diversity across members, not one strong member.
  - **Runtime with one net instead of 65:** about 1.1–1.5 ks, near the leader's 1,022 s, but 3–6 J worse.
- **Core training crops and head iteration budget (7 Nov, `scratchpad/coreaug/`): both FAIL.**
  - **Task-period crops as extra training rows:** nested −0.47; cross-fit on fresh −0.46. Cue-locked sub-window and
    slow blocks make shifted crops a distribution shift; only P04 gains.
  - **Iteration budget:** the shipped tol stop (≈ 15 iterations) is already optimal. it5 −1.9, it8 −0.7, it30 −0.4,
    inner-LOPO early stopping −0.1 to −0.2.
- **Consolidation on routing v3 + GEB (7 Nov, `scratchpad/consol/`): FAIL.**
  - **Primary arm, picked by a frozen rule on exploration:** α 1.3, fresh dJ −0.23 (sign-flip p 0.86).
  - **Other arms:** TSEM λ 0.2 −0.07; both −0.37.
  - **Why:** GEB already absorbs the new-person gain of α 1.3, and routing replaces its seen gain. Keep the shipped
    values.
- **Personal deep models in the routed tail (7 Nov, `scratchpad/pdeep/`): no gain beyond PS1.**
  - Personal fine-tuned FBCNet/EEGNeX/IFNet are stronger standalone (72–75 vs PS1 71) and improve the routed mix by
    +1 to +2.3 BA.
  - TSEM absorbs it: seen +0.3 to +0.55 after TSEM, below the gate.
  - The only gate-passing arm, a personal teacher map, failed confirmation (P10 −0.5). Sealed value about 0.
- **Early routing on top of v3 (7 Nov, `scratchpad/route_early/`): FAIL, not shipped.**
  - **Mechanism:** a two-factor latched gate from window 10 (fingerprint nearest = v3 top person, with shape
    score ≥ 0.12). It routes all 4 seen sessions at t12–22.
  - **Result:** seen +0.40 to +0.92 over v3, but the pre-registered P10 bar fails on B2_s1 (−0.47).
  - **Stress:** not clean — random posteriors fire 2/140; a view artifact fires 1/14.
  - **Sealed:** about 0 to +0.1; small capped early weights do not help.
  - **Bug found:** the inherited band-correlation fingerprint slices assume 820 columns per band but the export has 861,
    so earlier "alpha+beta" numbers refer to shifted columns. The package would have shipped exactly the evaluated
    feature.
- **Early-window boost "EB" (7 Nov, `scratchpad/early/`): passes on fresh bases. Found in a post-hoc second round;
  adversarial verification running.**
  - **Diagnosis:** the stages add only +0.6 / +3.7 / +6.0 / +7.0 points in bins 0–9 / 10–19 / 20–39 / 40–59, against
    +10 later. SessionEM slightly hurts in 10–19. There is no early class bias.
  - **Mechanism:** with h(t) = clip((60 − t)/30, 0, 1), the SessionEM exponent ×(1 + 2h) with nmin 20, and the TSEM
    exponent ×(1 + 4h).
  - **Fresh-base results:** dJ +0.67 / +0.42 / +0.14 / +0.21 (mean +0.36). New +0.53 to +1.09; seen −0.50 to +0.41,
    with P10 −0.5 on every base.
  - **With routing v3:** +0.48 over v3 alone (exploratory). The two are complementary, since routing helps returning
    people and EB helps new people.
  - **Closed:** position-dependent member weights (−0.39) and an early rebalancing prior (−0.25).
  - **Adversarial verification (`scratchpad/verify_eb/`): not refuted, but the size is overstated.**
    - **Reproduced exactly;** causal.
    - **New-person gain is real** (joint person-sign-flip null p = 0.002): about +0.6 ± 0.2, coming from windows 20–39.
    - **At the J level** p = 0.07. Honest dev gain is about +0.2 to +0.25: cross-fit +0.27, early-only hybrid +0.25,
      nested on the fresh bases +0.17.
    - **Seen people** about −0.15 ± 0.4. EB hurts where the early base is already strong; P10's loss is the same 2–3
      windows on every base.
    - **Robustness:** a broad plateau across neighbouring configs; the transition point at 40–60 works, 80 collapses.
    - **Sealed estimate:** about +0.25 ± 0.25 points, i.e. 0.8197 → about 0.822.
- **SessionBN GPU round for new people (7 Nov, `scratchpad/deep2/`, 34 Galvani jobs, about 9 GPU-h): all FAIL.**
  - **SessionBN teacher:** replacing the teacher costs about −1.0; fusing it with the current one is neutral.
  - **SessionBN EEGNeX/IFNet fine-tuned on PROTEUS in the trainable-FIR slots:** nested +0.28 with both halves
    positive, but the plain twins already give most of that.
  - **SessionBN twins of the remaining plain-BN members:** −0.18.
  - **SessionBN FBCNets on CAR input:** +0.15.
  - Fresh bases untouched.
- **Third tangent core after TSEM (7 Nov, `scratchpad/core3/`): FAIL.**
  - 77 saved non-sz cores × {add, replace Hilbert} × weights {2, 2.5, 3}: nested LOPO −0.24, halves +0.22 / −0.79.
  - The best in-sample core (fb12 Hilbert, +0.29) lowers TSEM seed precision and is over the runtime budget.
  - Fresh bases untouched.
- **Expected path (before the audits):** REC + core2 ≈ 76.9 J (0.805 full transfer / 0.79 half), + TSEM ≈ 78.4 ± 1.0 (0.82 / 0.80).
  0.83 needs ≈ 79.35: full transfer, TSEM holding in the package, and about +0.8 J more. Architect's estimate 15–25 %.

## 6. Recommended upload plan (each upload still needs your authorisation)

**Superseded on 6 Nov by 5d** (4 uploads left): U2 = REC + core2 (also measures the dev-to-board slope), U3 = U2 +
TSEM (paired, so the 10-decimal difference measures TSEM on sealed), U4 = U3 + whatever passes its gate, U5 = final
best or hedge. If U2 − cand_nxif ≤ 0.008 the board does not reward dev gains: re-plan before U3. The list below is
the 5 Nov plan, kept for the record.

1. **Day 1: cand_nxif.** The 00:17 check-in never uploaded. Uploaded on 5 Nov 06:45 CET with the user's
   authorisation as Codabench **1034698**: **0.7794596196**, 834.7 s, rank 9 of 46 (top: seohee 0.83, then
   0.81 ×3, 0.79 ×3). The v4-sealed image runs the TorchScript teacher. A 3–5 point gap is beyond data-only
   knobs, so an approach tournament (six families, person-equal scorer `scratchpad/own/pe_metric.py`) started
   at 08:00. First member-level finding: CAR lifts v2 alone by +1.85 all_loso, but is neutral inside the
   ensemble (re-referenced views had been dropped on 30 Oct by argument, never tested).
2. **Day 2: a115**, once its official replay and worker replica pass. Fallback if they cannot run in time:
   cand_nxif_emx (staged, replica-checked). Forecast: S(a115) − S(cand_nxif) ≈ +0.2 ± 0.5.
3. **Day 3, branch on d = S(a115) − S(cand_nxif):**
   - d > 0: more rebalancing helps on the sealed cohort. Continue the line search with α 1.25 (a125, same build), or
     emx if new people look dominant.
   - d ≤ 0: the cohort rewards the plain teacher. Upload the wave-11 teacher if it passes its gate, else w 3.
   - Expect |d| ≈ 0.2–0.5. Data-only arms move the score by tenths, not points.
4. **Days 4–5:** the best measured direction, one step further, or the best combination. Keep one upload for a
   gated new-information arm if we are more than about 1.5 points below 3rd place.
- **Drop from the plan:** R19e as a scheduled arm (it wins only if the teacher is net harmful, about 15–20 %), w 7 and
  m7r4 (dominated). Keep r19e_g only as the failure fallback.

Every arm that might score best needs its audit recipe (a115 = the cand_nxif recipe plus one `set_params.py` call).
