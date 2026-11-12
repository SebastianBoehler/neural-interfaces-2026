# Ledger: what is closed, and where new runs are pre-registered

**How to use it.**
- Before repeating an idea, look it up in the index below. A "closed" verdict covers only the recipe that was tested;
  the cited section says which.
- **No gate or closure is final (user rule, 12 Nov; §52).** A gate is a threshold on the evidence of its day. Before
  a closure is used as a reason not to run something, check what actually ran: the code path, the model class, alone
  or in the chain, and on which version of the system. Pre-registration stays: state the rule first, then say plainly
  when and why you go past it.
- New runs are pre-registered at the bottom of this file **before** they are scored: mechanism, numeric prediction
  with an 80 % range, and the result that would refute it. Fill in the outcome and the verdict (hit, miss, refuted)
  afterwards, including for misses.
- Sections cited as "§n" are in [PREDICTIONS_11NOV.md](PREDICTIONS_11NOV.md) (the full record of 11 Nov). Earlier
  closures are in [REVIEW_5NOV.md](REVIEW_5NOV.md). Scripts cited there as `runs/archive_scratch_11nov/<path>` are
  tracked as `studies/11nov/<path>`; the chain replay is `replay/`.

Numbers are dev J (0.5 new + 0.5 seen) unless stated. Anything below +0.3 is inside the noise of the dev metric.

## Index of 11 Nov

### Adopted

| What | Result | Section |
|---|---|---|
| Fast solver (same outputs, half the runtime) | bitwise parity on 2,655 windows; 0.53× runtime | §1 |
| Label-free identity gate before the personal heads (routing v4.1) | dev +0.58; **sealed +0.267** (0.8162) | §11, §14 |

### Positive but not built

| What | Result | Section |
|---|---|---|
| TTT: nets fine-tuned in the session on the chain's labels | 4 families: new +0.6 to +0.7, seen −0.5; one net per family: new +0.4 to +0.65 | §24, §28, §31 |
| PFT: nets fine-tuned on a returning person's own public session | seen +1.4 to +2.0 (P04 only; P10 about 0) | §29, §31, §32 |
| Personal class templates as a TSEM prior | +0.19 (P04 only) | §19 |
| Rest periods as a fourth class in net training (REST4) | +0.1 to +0.3 | §26 |
| Slot weight 4.5 for the personal head | +0.12, mixed signs | §22 |

### Closed

| Direction | Result | Section |
|---|---|---|
| **Members and features** | | |
| ST-EEGFormer (fourth pretrained foundation model) | 53.9 % alone; +0.16 nested | §4 |
| Pair specialists for the WORD confusions (W12) | no chain gain | §6 |
| Scherer 2015 as member training data / in core1's head / as an eighth teacher corpus | members +1–3 alone, chain 0; core1 −0.9 to −1.1; teacher +3.3 alone, chain −1.0 | §7, §21, §18 |
| SessionBN versions of untried net families (SBNX) | no chain gain | §12 |
| Amplitude-envelope coupling core | 50.4 % alone; −0.6 to −2.5 | §25 |
| Cross-frequency coupling core | 58.4 % alone; −0.6 to −2.8 | §26 |
| Weight-space edits of core1 (ten hand edits, block gains, 60-step Weco search) | none passes | §13 |
| Temperature-calibrated extra members; log-linear pool with fitted weights | extras still negative; pool +1 at mix level, −1 to −2 in the chain | §31 |
| Re-weighted member families for seen sessions | −0.0 to −2.9 seen | §31 |
| **Session stage** | | |
| 29 pre-registered stage arms (count priors, pair views, Dawid–Skene, seeds, quotas, schedules) | 0 pass; nulls −0.22 to +0.18 | §5 |
| Recency-limited TSEM seeds; forgetting recentring reference | no gain | §23, §20 |
| Second Weco search over the whole session stage (92 evaluations) | best +0.6 on its half, negative on every hidden measure | §13, §31 |
| Unsupervised clustering of a session into classes | 47 % at best (chain 77 %) | §30 |
| **Labels and fine-tuning** | | |
| Label feedback rounds for TTT | label accuracy 77.3 % again; +0.07 | §31 |
| Fine-tuned nets as a final multiplicative stage with chain labels | −0.5 to −2.2 (true labels +1.6) | §28 |
| Better labels for past windows (two-directional smoothing, run balance) | 75.4–77.3 % against 76.6 % causal | §28 |
| **Sequence** | | |
| Graz cue-order prior | real (56 % by position) but ≤ +0.4 with oracle run structure | §9 |
| Reconstructing the Graz cue generator (247 million seeds × 12 variants) | no match | §27 |
| Order fingerprint of the sealed stream; order-robust re-tune | every reordering hypothesis rejected | §2, §3 |
| **Returning people** | | |
| Raw spectral biometric (correlation score) | cannot reject new people early; replaced by §11 | §10 |
| Similarity routing for new people | weighted heads lose to the pooled model | §16 |
| More regularised personal head; person-upweighted core; personal recentring prior | wins at core level, loses 0.30 in the chain; refuted | §17 |
| Personal head on ocular and slow-potential features | 72 % alone for P10; −1.5 to −7 seen as a final stage | §31 |
| **Bounds** | | |
| State across sessions, slow potentials, feedback in online runs, external corpora | nothing usable; see the section | §33 |
| GPU-batched eigendecomposition | speed probe only; runtime is not binding | §8 |

### Diagnostics worth knowing

| Finding | Section |
|---|---|
| Who gains from a personal head (ten-person proxy): gain follows the person's weakness; most of it is lost across sessions | §15 |
| Error anatomy: pair confusions; the true class is the runner-up in 73 %; WORD is the weak class; P04 holds 42 % of new-person errors | §5 |
| The chain matches a supervised within-session model; P04 is low-signal; members share errors | §30, §31 |
| Dev "seen" over-weights P04 relative to the sealed set | §31 |

## New pre-registrations

> **Clock correction (12 Nov, 01:40 UTC by `date -u`).** From the addenda of §47 on, the "written … UTC" stamps in
> §47–§57 were estimated and run ahead of the real clock by up to about 2 h 20 min (the last one says 03:55 UTC and was
> written at about 01:25 UTC). The order of pre-registration and result is unaffected: every one of those sections
> was committed and pushed before its result existed, and the git commit times are the authoritative record. Stamps
> written after this note come from `date -u`.

<!-- Template:

## N. NAME: one-line description (written DATE TIME UTC, before any result; code PATH)

**Why.** …
**Mechanism.** …

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| … | … | … |

| Item | Observed | Verdict |
|---|---|---|
-->

## 34. View of 12 Nov and plan (written 11 Nov 21:55 UTC, after `replay/check.py` printed OK; no new chain result yet)

**Independent reading of the record** (two own checks, not pre-registered, both exploratory: `studies/12nov/`):
- `slow_xsubj.py`: slow-potential / gaze features decode 57–59 % within a session and 47–49 % across people
  (P02 BrainHero 93 % within, 71 % across). The gaze-to-class mapping is mostly personal. No shared shortcut; agrees
  with §33.
- `anatomy.py` (B2_pool, v3 chain): new people by window position 72.6 % (0–30), 78.6 (30–60), 77.3 (60–100),
  77.9 (100–150), 75.7 (150+). The chain stops improving after window 30–60 although the unlabelled evidence triples.
  New people without P04: 83.1 %; P04 53.6 %. Seen: P10 91, P04 69.
- **Every group sits at or above its within-session supervised level in core1's features** (new 77 vs 76; P04 seen 69
  vs 55–59 with its own labels; P10 91). I find no label-free source the record missed. The +3 gap to the leader
  equals our true-label ceiling, so it needs information we do not have, not a better stage.
- **What the record does not price:** the returning-person branch. It is about 43 % of sealed sessions (46 of 106 if
  the cohort text holds), the only place where true labels of the same person exist, and every verdict on it rests on
  two people, one at ceiling (P10) and one low-signal (P04). The eight single-session people (40 sealed sessions)
  are unmeasured in the chain.

**Plan.**
1. LADDER (§35): does the label limit of in-session learning also hold for returning sessions, where the chain's labels
   are far more accurate? Cheap, local, decides whether in-session learning belongs behind the identity gate.
2. If the user agrees to build v5: build it as **two paired uploads**, PFT first (v4.1 + personal nets behind the
   gate: its sealed difference to v4.1 measures the eight single-session people directly, which dev cannot), TTT
   second. Each upload is then a candidate and a measurement.
3. Honest expectation: v5 as specified is +0.3 to +0.6 sealed (rank 6–8). I have no mechanism for +1.0 yet.

## 35. LADDER: in-session head on true vs chain labels, by new and returning sessions (written 11 Nov 21:55 UTC, before any result; code `studies/12nov/ladder.py`)

**Why.** "Fine-tuning is label-limited" (§28) was measured on new people (labels 77 % right) and with nets. In a
returning person's session the labels come from a chain that contains a model trained on that person's true labels
(P10: 91 % right), an independent source. If the limit is label accuracy, a self-labelled in-session head should pay
for P10 even though it does not for new people.

**Mechanism.** After the full v3 chain, a causal per-session logistic head (128 PCs of core1's features, z-scored on
the session's past windows, C 0.1, refit every 5 windows from window 20) is multiplied into the output:
out ∝ chain × head^(λ·t/(t+60)), λ ∈ {0.5, 1}. Labels for the head: true past labels (ceiling) or the chain's own past
outputs (hard labels, weighted by confidence). Bases B2_pool and B2_s345; the better λ is reported for both.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| True labels, new | +3.0 (+1.8 to +4.5) | < +1: the "+3 J ceiling" is not reproducible this way |
| True labels, seen (P04 / P10) | +2.0 (+0.5 to +3.5); P04 +3, P10 +1 | — |
| Chain labels, new | +0.1 (−0.4 to +0.5) | ≥ +0.8: the label limit is not what stops new people |
| Chain labels, seen | +0.3 (−0.5 to +1.0); P10 +0.5 (−0.3 to +1.2), P04 0.0 (−1 to +1) | P10 ≤ 0 on both bases: accurate labels do not unlock in-session learning either |

Reading rule: P10 ≥ +0.5 on both bases with chain labels → in-session learning is built behind the identity gate
(with the personal nets); otherwise the returning branch gets PFT only.

| Item | Observed | Verdict |
|---|---|---|
| True labels, new (λ 0.5; B2_pool / B2_s345) | +1.88 / +2.11 (λ 1: +0.77 / +1.00). P04 +6.7, P10 +4.8 to +5.0, P09 +3.3 to +4.3, P06 +2.6 to +3.1; P02 −1.6 to −2.2 | **Hit**, at the low edge (predicted +3.0, range +1.8 to +4.5) |
| True labels, seen (λ 0.5) | +0.52 / +0.52; P04 +1.3 / +1.0, P10 −0.3 / 0.0 | **Hit on the mean at the low edge; miss on both people** (predicted P04 +3, P10 +1) |
| Chain labels, new (λ 0.5) | −0.81 / −0.59 (λ 1: −1.8 / −1.7); 2 and 4 of 10 people ≥ 0 | **Miss, below the range** (predicted +0.1, −0.4 to +0.5) |
| Chain labels, seen (λ 0.5) | −1.69 / −1.30; P10 −0.3 / 0.0, P04 −3.1 / −2.6 | **Refuted** by the pre-set rule (P10 ≤ 0 on both bases) |
| Head alone, window accuracy (new / seen) | true labels 71.5 / 69.5; chain labels 66–67 / 67 (the chain: 77 / 80) | — |

**Reading (11 Nov 22:00 UTC; log `runs/ladder_12nov.log`).**
- **Returning sessions are saturated in core1's features.** Even true labels of the running session add only +0.5 to
  the seen score, and nothing for P10. The personal head already carries what an in-session supervised head would
  add. In-session learning is not built behind the identity gate; the returning branch keeps PFT only (the nets are a
  different feature space; P04 +2.4 to +4.5 there).
- **For new people true in-session labels are worth about +2 with this head, and only for weak sessions** (P04, P10,
  P09, P06). For strong sessions (P02, P01, P05) an in-session head trained on true labels lowers the chain: the
  population chain is better than 150 own labels. The "+3 J" in STATE §5 is closer to +2 by this route.
- A self-labelled head is negative everywhere. The label limit holds for returning sessions too. Closed.
- **What this does to the plan:** on dev neither branch has head-room that a session-stage or head design could
  reach. The remaining positive levers are the two that change the nets (TTT, PFT). Where the sealed errors sit
  (returning or new sessions) is not measurable on dev; a sealed measurement is the user's decision (§34, point 2).

## 36. REPCEIL: is the 76 % within-session level a property of the data or of core1's features? (written 11 Nov 21:59 UTC, before any result; code `studies/12nov/repceil.py`)

**Why.** Steer from the user (11 Nov evening): a team scores 0.846 under the same rules, so a +3 method exists; the
ceilings in STATE §5 were all measured inside our representation (core1's PCs, our nets) and are ceilings of that
representation, not of the data. The Discord export (`runs/archive_scratch_11nov/discord_11nov/`) names no top team's
method. So the first question is how much class information a session holds when it is decoded in representations
we never used as a within-session model.

**Test.** Per session (14), leave-one-run-out (6 runs, so every fold crosses a run and often the interface),
supervised, with that session's own labels only. Score: mean over sessions of the interface-cell balanced accuracy.
Representations, each with its best of three regularisation strengths (chosen on the mean, a mild optimism that is
acceptable for a ceiling):
- R0 reference: core1's 128 population PCs (the space of §30 and of TSEM), logistic regression.
- R1 session-own tangent space: 9 bands (1–4 … 40–58 Hz and 4–40 Hz), common-average reference, covariance of the
  full window, reference = the session's own log-Euclidean mean, all 861 tangent coordinates per band.
- R2 filter-bank CSP (6 filters per class pair and band, fitted in the fold) with LDA.
- R3 slow time courses: 0.1–4 Hz, 8 samples per second, all channels, shrinkage LDA.
- R4 late fusion of R1 + R3, and R5 of R0 + R1 + R3 (product of out-of-fold posteriors).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| R0 | 73 (70–76): lower than §30's 75.9 because folds are whole runs | — |
| Best single of R1–R3 | 75 (71–79) | — |
| Best fusion (R4 / R5) | 79 (75–83) | — |
| Best fusion minus R0 | +6 (+2 to +9) | ≤ +2: the 76 % level is a property of the data at this label count, and a better within-person representation is not where +3 is |
| Sessions where the best fusion beats R0 by ≥ 5 | 8 of 14 (5–11) | — |

Reading rule: best fusion ≥ R0 + 6 → core1's PC space leaves within-person information unused; the next step is to
put that representation behind the chain's labels (TSEM in the richer space) and into the population model. Between
+2 and +6 → worth one chain test. ≤ +2 → the representation is not the lever; the population model (invariance across
people) is.

| Item | Observed | Verdict |
|---|---|---|
| R0, core1's population PCs | 74.93 | **Hit** (predicted 73, range 70–76) |
| Best single of R1–R3 | FBCSP 76.01 (own tangent 74.53; slow 61.60) | **Hit** (predicted 75) |
| Pre-registered fusions R4 / R5 | 67.28 / 70.68 | **Miss, below the range** (predicted 79): the slow LDA's posteriors are too sharp for a product of two or three |
| Four-way fusion R0 × R1 × R2 × R3 (in the script before the run, not in the table above) | geometric 81.36, arithmetic 82.37; minus R0 **+6.44 / +7.44**; 8 of 14 sessions ≥ +5 | By the reading rule: ≥ R0 + 6 |
| Subsets (exploratory addendum) | R0 × R1 79.0–79.4; R0 × R2 × R3 80.7–81.5; R1 × R2 × R3 80.6–81.0 | Robust to the pooling rule and to leaving any one member out |
| Per session, R0 → four-way | P04 56.9 → 80.0, 60.4 → 72.0, 54.7 → 62.2; P08 79.4 → 89.3; P02 78.8 → 90.0; P01 75.1 → 84.7; P10s3 80.4 → 88.5; P05 73.1 → 69.5; P10s1 88.0 → 84.4 | — |

**Reading (11 Nov 22:05 UTC; log `runs/repceil_12nov.log`, posteriors `runs/repceil_12nov.npz`).**
- **The 76 % level was a property of core1's PC space, not of the data.** With the same 160 labels, session-own
  spatial filters (FBCSP), the session-own tangent space and slow time courses together reach 81–82 %, against 74.9 %
  in the population PCs and 76.6 % for the label-free chain on the same sessions (session mean).
- **P04 is not a low-signal person.** FBCSP alone decodes P04's first session at 74 % (population PCs: 57 %). P04's
  class information lies in directions the population basis does not keep. STATE §5 ("P04: 55–59 % with their own
  labels") holds only for core1's features.
- **Consequence for returning people:** the personal head is a logistic regression on the population PCs, the space
  that loses most for exactly the people who need it. A personal model in the person's own space is the first test
  (§37).
- **Consequence for new people:** the label-free target is no longer "match 76 %"; the session holds about 5 points
  more than the chain extracts. Whether any of it can be reached with the chain's labels is a separate test (§38,
  to be written after §37).
- Limits: three regularisation values per representation chosen on the mean; leave-one-run-out is not causal; 14
  sessions of 10 people.

## 37. XSESS: a returning person's model in their own space (written 11 Nov 22:06 UTC, before any result; code `studies/12nov/xsess.py`)

**Mechanism.** For P04 and P10 (dev seen: session 1 → sessions 2–3) three personal members are fitted on session 1
only and streamed causally over the later sessions: (T) tangent space of 9 bands after recentring to the session's
own reference (training: whole-session mean covariance; test: running mean with the public session's mean as 2
pseudo-windows), logistic regression; (C) FBCSP filters of the recentred public session, LDA; (S) slow time courses
with causal standardisation, shrinkage LDA. In the v4.1 chain replay the personal head's posterior for the true
person is replaced by a plain average: V1 = (PS1 + T + C) / 3, V2 = (PS1 + T + C + S) / 4, V3 = (T + C) / 2.
Exploration bases first; fresh bases only if the rule below is met.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Standalone, later sessions, mean of T and C averaged: P04 / P10 | 58 (52–64) / 74 (66–82) | — |
| Share of the within-session level kept across sessions (T + C: within 69 / 80 by §36) | 85 % (70–100) | < 60 %: own-space models are session-bound, like the PC head |
| Chain, V1, seen | +1.0 (−0.5 to +2.5); P04 +1.5 (−1 to +4), P10 +0.5 (−1 to +2) | seen ≤ 0 on both exploration bases |
| Chain, V2 minus V1 | +0.3 (−1 to +1.5): P10's slow view helps, P04's does not | — |
| New people | 0.00 (no new-person session is routed) | any change |

Rule: best variant ≥ +1.0 seen on both exploration bases with P10 ≥ 0 → confirm on the four fresh bases; then the same
members are fitted for all ten people for a package candidate.

| Item | Observed | Verdict |
|---|---|---|
| Standalone across sessions, T + C (session 1 → 2, 3): P04 / P10 | 61.0 / 66.9 (with S: 67–69 / 68–77). The PS1 head alone: 60.8 / 80.8 | **Hit / hit at the low edge** (predicted 58 / 74). For P10 the population-PC head is 14 points better than the own-space members |
| Share of the within-session level kept | P04 88 %, P10 84 % | **Hit** (predicted 85 %) |
| Chain, V1 / V2 / V3, seen (B2_pool; B2_s345) | −0.80 / −0.80 / −0.92; −0.68 / −1.07 / −1.32 (P04 −1.0 to −2.1, P10 0.0 to −0.8) | **Refuted** (seen ≤ 0 on both bases) |
| New people | 0.00 | Hit |
| Exploratory, after the result (`xsess_poe.py`, 20 arms): the members as a tempered product inside the head's slot, PS1 × (T·C·S)^(0.5/3) | seen +1.07 / +1.45: P04 +2.1 / +2.9, P10 0.0 / 0.0. As a final stage (β 0.25): P04 +4.5 to +4.8, P10 −2.4 to −2.9 | Not pre-registered; the best of 20 arms on two people |

**Reading (11 Nov 22:12 UTC; logs `runs/xsess_12nov.log`, `runs/xsess_poe_12nov.log`).**
- Own-space personal members do cross sessions (they keep 84–88 % of their within-session level), but averaging
  them into the personal head's slot dilutes the head and loses. As a tempered product they add +2 to +3 for P04 and
  nothing for P10: the same pattern as PFT (P04 gains, P10 does not).
- This fits §36: P04's class information lies outside the population PCs; P10's lies inside them (PS1 alone 81 %).
  By the §36 gap (four-way fusion minus R0) five of the eight single-session people look like P04, not like P10
  (P02 +11, P08 +10, P01 +10, P03 +9; P07, P09 +3; P06 0; P05 −4), so the lever may be larger on the sealed set than
  the dev mean says. Dev cannot settle it.
- Status: a returning-person lever of about the size of PFT, to be confirmed on fresh bases together with PFT if a
  returning-person package is built. Not the +3 mechanism.

## 38. OWNTTT: session-own filters trained inside the session, on true and on the chain's labels (written 11 Nov 22:14 UTC, before any result; code `studies/12nov/ownttt.py`)

**Why.** §36: with its own labels a session holds 81–82 % in session-own representations; the chain extracts 76.6 %.
TSEM, the stage that brought +3.6 sealed, is a generative model on confident seeds in the population PCs. The same
seeded scheme was never run in the session's own space (spatial filters and the full tangent space re-learnt per
session). The nets' TTT is the gradient analogue and gave new +0.6.

**Mechanism.** After the full v3 chain, every 5 windows from window 30, three members are refitted on the session's
past windows: FBCSP + LDA (C), session-own tangent space + logistic regression (T), slow time courses + LDA (S); all
referenced to the running mean covariance of the past windows. Labels: true past labels (ceiling) or the chain's
past outputs restricted to the 70 % most confident windows per predicted class (as TSEM's seeds). Fusion:
out ∝ chain × (geometric mean of the members)^(λ·t/(t+60)), λ ∈ {0.25, 0.5, 1}, member sets TC and TCS.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| True labels, new, best λ | +4.0 (+2 to +6) | < +2: the §36 surplus does not add to the chain |
| True labels, seen | +1.5 (0 to +3): P04 +4, P10 0 | — |
| Chain labels, new, best λ | +0.3 (−0.6 to +1.2) | ≤ +0.3 on both exploration bases: label-free learning in the session's own space is closed |
| Chain labels, seen | 0.0 (−1 to +1) | — |
| P(chain labels ≥ +0.8 new on both bases, ≥ 6 of 10 people ≥ 0) | 15 % | — |

Rule: chain labels ≥ +0.8 new on both exploration bases → confirm on the four fresh bases with λ and member set fixed.

| Item | Observed | Verdict |
|---|---|---|
| True labels, new, best λ (B2_pool / B2_s345) | TCS λ 0.5: +2.94 / +2.71; TC λ 0.25: +2.53 / +2.70 (8 of 10 people ≥ 0). P04 +6 to +11, P10 +5 to +7, P01 +3 to +5, P03 +2 to +7; P05 −3 to +1, P02 −0.6 to 0 | **Hit**, lower half (predicted +4.0, range +2 to +6) |
| True labels, seen | TCS λ 0.25–0.5: +1.82 to +2.50 (P04 +2.6 to +3.7, P10 +1.0 to +1.3) | **Hit** (predicted +1.5); P10 above its prediction (0) |
| Chain labels, new, best λ | TC λ 0.25: −0.14 / −0.47; TCS λ 0.25: −0.28 / −0.62; larger λ −1.7 to −6.0 | **Refuted** (≤ +0.3 on both bases) |
| Chain labels, seen | −0.92 to +0.50 at λ 0.25; −2.7 to −8.2 above | Hit (predicted 0.0) |
| Members alone, causal, new people (window accuracy) | true labels: T 69.8, C 72.0, S 57.5, together 76.7; chain labels: together 66–67; the chain: 77.3 | — |

**Reading (11 Nov 22:22 UTC; log `runs/ownttt_12nov.log`).**
- **Label-free learning in the session's own space is closed** in the seeded form that made TSEM work. Members
  trained on the chain's seeds reach 66–67 % alone, ten points below the same members on true labels and below the
  chain itself, so no fusion weight helps. With LADDER (§35) and TTT (§24, §31) that is three feature families in
  which the chain's labels buy nothing or +0.6.
- **The true-label ceiling is about +2.5 to +3 J in this space too** (and +2 for returning sessions, where the
  population-PC head gave +0.5): the §36 surplus is real and adds to the chain, but only with labels.
- **Where that leaves the +3 question.** In-session learning on our own labels is not the mechanism, in any of the
  three spaces. What remains open: (a) a population model that is right for more people before any session stage
  (P04 holds 74–80 % of class information in its own space and gets 54 % from the population model: the failure is
  the population's coverage of people, not P04's signal); (b) returning people in their own space (§37 follow-up and
  PFT: P04-like people gain +2 to +4, and by the §36 gap five of the eight single-session people look P04-like).

## 39. POPCURVE: new-person accuracy against the number of training people (written 11 Nov 22:30 UTC, before any result; code `studies/12nov/popcurve.py`)

**Why.** §36/§38: P04 holds 74–80 % of class information in its own space and gets 54 % from the population model.
The failure is the population's coverage of people. If coverage is the lever, new-person accuracy must still be
rising steeply at nine training people. This was never measured.

**Test.** core1 (the strongest member; `car_ov17_fs2_t0.1_le_k2`, logistic regression C 10) refitted for every
held-out person on k ∈ {2, 3, 5, 7} randomly drawn other people (4 draws each) and on all 9; causal test features;
score = person-equal cell-mean BA, alone and after the packaged rebalancing (α 1.15, W 30). No other member or stage.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| k = 9, core alone / after rebalancing | 67.5 (65–70) / 72 (69–75) | — |
| k = 2 / 3 / 5 / 7, core alone | 54 / 58 / 62.5 / 65.5 (each ±3) | — |
| Gain from 7 to 9 people, after rebalancing | +1.8 (+0.8 to +3.0) | ≤ +0.5: the curve is flat; more people of this kind would not help |
| Gain per doubling of people (log fit over k ≥ 3) | +5 (+3 to +7.5) | ≤ +2 |
| P04 as the held-out person, 3 → 9 people | +3 (0 to +7): P04 gains less than the others | — |

Reading: ≥ +4 per doubling means ten more people of the same kind would be worth about +4 on the base before any
session stage, more than every stage change since TSEM. Then the next bets are sources of people (PROTEUS-like
external recordings re-referenced into this montage; synthetic people by recombining sessions' spatial structure),
and the sealed package's tenth person is already worth part of a step. ≤ +2 per doubling closes "coverage".

| Item | Observed | Verdict |
|---|---|---|
| Run | First run (k = 2, 3, 5, 7 × 4 draws) stopped after 70 min without output; rerun with k = 3, 5, 7 × 2 draws and k = 9 (70 fits, 30 min; `runs/popcurve_12nov.log`) | Design reduced after the stop; no result had been seen |
| k = 9, core alone / after rebalancing | 69.29 / 69.19 | **Hit / miss, low** (predicted 67.5 / 72): rebalancing adds nothing to core1 alone |
| k = 3 / 5 / 7, core alone | 56.34 / 60.71 / 65.43 | **Hit** ×3 (predicted 58 / 62.5 / 65.5, each ±3) |
| Gain from 7 to 9 people | +3.86 core, +3.26 after rebalancing | **Miss, above the range** (predicted +1.8, +0.8 to +3.0) |
| Gain per doubling (log fit, k ≥ 3) | **+8.13 core, +7.48 after rebalancing** | **Miss, above the range** (predicted +5, +3 to +7.5) |
| P04 held out, 3 → 9 people | +10.4 (40.5 → 50.9) | **Miss, above** (predicted +3): P04 gains as much as anyone |
| Per person, 3 → 9 | P07 +28, P10 +20, P08 +16, P03 +11, P01 +12, P04 +10, P09 +10, P06 +9, P05 +7, P02 +6 | Everyone rises |

**Reading (12 Nov 00:25 UTC).**
- **New-person accuracy is limited by the number of training people, and the curve is not flattening**: about +2
  points per added person at nine people, +8 per doubling. This is the largest slope measured in this project, larger
  than any model or stage change since TSEM.
- It is a core-level number. How much survives the session stages is not measured (the stages add most where the
  base is weak); the chain-level slope needs a core-only chain at each k.
- The sealed package trains on ten people, one more than every dev fold, so part of one step is already in the
  sealed score.
- **It names the missing ingredient for new people: more people who did these tasks.** The one external corpus with
  all three tasks (Scherer 2015, nine patients, 22 of 41 channels interpolated, high-passed at 4 Hz) lowers core1 by
  3–4 points (§21) and helps the nets by 1–3 (§7): it does not behave like nine more PROTEUS people. Open: other
  public recordings of these tasks (search running), and whether variation between people can be imitated in
  training (§46).

## 40. ADD: do personal nets (PFT) and own-space personal members add up for a returning person? (written 11 Nov 22:40 UTC, before any result; code `studies/12nov/add.py`)

**Why.** Both levers help P04 and not P10 (§31, §37). If they carry the same information, a returning-person package
needs only one; if they add, P04-like people gain about twice as much.

**Test.** The v4.1 chain replay on the two exploration bases, all arms against the frozen-twin system of §31 (five
net families replaced by the three-seed frozen nets). PFT = the head-and-BN-affine variant (`pfh60`), all five
families; OWN = PS1 × (T·C·S)^(0.5/3) in the personal head's slot (the §37 follow-up arm, fixed here); BOTH.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| PFT, seen (P04, P10) | +1.4 (+2.4, +0.5), as §31 | — |
| OWN, seen (P04, P10) | +1.2 (+2.5, 0.0), as §37 | < +0.5 on both bases: the §37 arm was a pick of 20 |
| BOTH, seen | +2.0 (+1.0 to +3.0); P04 +4 (+2 to +6), P10 +0.3 (−0.7 to +1.3) | BOTH < max(PFT, OWN) + 0.3 on both bases: the levers are redundant |

| Item | Observed | Verdict |
|---|---|---|
| PFT, seen (P04, P10), B2_pool / B2_s345 | +0.94 (+1.84, +0.03) / +1.20 (+2.37, +0.03) | **Hit**, a little low (predicted +1.4) |
| OWN (in-slot, β 0.5), seen | −0.13 (+0.27, −0.54) / +0.01 (+0.29, −0.26) | **Refuted** (< +0.5 on both): on the frozen-twin base the in-slot arm of §37 gives nothing; it was a pick of 20 |
| BOTH, seen | +2.01 (+3.75, +0.26) / +1.88 (+3.49, +0.26) | **Hit** (predicted +2.0); not redundant (BOTH − PFT = +0.7 to +1.1) |

**Reading (11 Nov 23:43 UTC; log `runs/add_12nov.log`).** PFT replicates. The in-slot own-space arm is unstable
(+1.1 to +1.5 on the shipped nets, 0 on the frozen twins, +0.7 to +1.1 on top of PFT): not a form to build on. The
final-stage form is the stable one (§43).

## 41. PSEL: per-person choice of the personal model's space, tested on ten people (written 11 Nov 23:35 UTC, before any result; code `studies/12nov/psel.py`)

**Why.** The user (11 Nov, late): push on "the ceilings were ours, not the data's". The place where labels exist is
the returning branch, and it has only ever used the population PCs. §36/§37: P04's information is outside that
space (own-space members help P04 by +2 to +5 and hurt P10 by up to −3 as a final stage), P10's is inside it. If the
right space can be read off a person's public session, each returning person gets the model that suits them. Dev
"seen" has two people; this test uses ten.

**Test (a proxy with stated limits).** For each of the ten people, session 1 is split: runs 1–3 play the public
session (labels used), runs 4–6 the later session. The chain is the leave-one-person-out v3 chain's output on runs
4–6 (replay rows; the population has never seen the person, and there is no session change). Personal members are
fitted on runs 1–3 only and streamed causally over runs 4–6: P = logistic head on the population PCs (the PS1
recipe), and T, C, S of §37. Final stage: out ∝ chain × (geometric mean of the members)^β, β ∈ {0.25, 0.5}.
**Gate statistic, from runs 1–3 only:** leave-one-run-out accuracy of T·C·S minus that of P. Rule fixed now:
gap ≥ +5 → own-space stage (T·C·S); gap ≤ −5 → population-space stage (P); otherwise both (P·T·C·S).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| P stage for everyone, mean gain on runs 4–6 (better β) | +2.0 (0 to +4) | — |
| T·C·S stage for everyone | +3.0 (0 to +6) | — |
| P·T·C·S stage for everyone | +4.0 (+1 to +7) | — |
| Selected by the gate, minus the better of the three uniform stages | +0.5 (−1 to +2) | ≤ −1: choosing per person is worse than one recipe for all |
| Spearman, over ten people, between the gate gap and (gain of T·C·S − gain of P) | +0.5 (+0.1 to +0.8) | ≤ 0: the public session does not say which space suits the person |
| People for whom the best uniform stage lowers the chain by more than 1 | 2 (0–4) | — |

Reading: if own-space stages add ≥ +3 on the ten-person mean and no rule protects the people they hurt, the returning
package uses P·T·C·S with a small β; if the gate separates, it uses the gate. The cross-session size stays unknown
for eight people (P04 keeps 88 %, P10 84 % of the within-session level, §37).

| Item | Observed | Verdict |
|---|---|---|
| LOSO chain on runs 4–6 (B2_pool / B2_s345) | 82.50 / 82.33 | — |
| P stage for everyone | +2.00 / +1.92 (β 0.25 / 0.5: +2.00, +1.67; +1.83, +1.92) | **Hit** (predicted +2.0) |
| T·C·S stage for everyone | β 0.25: **+4.17 / +4.17**, 9 of 10 people ≥ 0 (P07 −1.7); β 0.5: +3.67 / +3.83 | **Hit** (predicted +3.0, range 0 to +6) |
| P·T·C·S stage for everyone | β 0.25: +4.17 / +4.08, nobody below −1; β 0.5: **+5.00 / +5.08** (P07 −1.7) | **Hit** (predicted +4.0, range +1 to +7) |
| Per person, P·T·C·S β 0.25 (B2_pool) | P01 +8.3, P02 +0.8, P03 +5.0, P04 +5.8, P05 +2.5, P06 +4.2, P07 +0.8, P08 +5.0, P09 +3.3, P10 +5.8 | Ten of ten ≥ 0 |
| Selected by the gate, minus the best uniform stage | −0.09 / −0.42 (β 0.25); −0.33 / −0.25 (β 0.5) | Inside the range (predicted +0.5, −1 to +2), but **no value**: the gate never beats one recipe for all |
| Spearman(gate gap, gain of T·C·S − gain of P) | +0.28 / +0.21 (β 0.25); +0.15 / +0.35 (β 0.5) | Hit at the low end; too weak to use |
| Members alone on runs 4–6 | T 56.7–88.3, C 63.3–91.7 (P08), S 43–89 (P02), P 59–87 | — |

**Reading (11 Nov 23:32 UTC; log `runs/psel_12nov.log`).**
- **First broad positive of this line: on ten people, own-space personal members add +4 to +5 to the chain, twice
  what a population-PC personal head adds (+2), and nobody is hurt at β 0.25.** It is not a P04 effect: P01, P03,
  P08, P10, P06 and P09 each gain +3 to +9.
- The per-person gate is dropped: three-fold accuracy on 100 windows is too noisy, and almost everyone gains anyway.
  The recipe is P·T·C·S for everyone.
- **Limits, all three matter.** (1) No session change: across sessions P04 kept the gain and P10 lost 2–3 as a final
  stage (§37). (2) The population had never seen the person; in the package it has, so part of this gain is already
  in the base. (3) Half a session of public data (100 windows; the package has 190 or 570).
- Next: the same stage across real session pairs of P04 and P10 on the LOSO chain (§42), which removes limit 1 for
  two people and leaves 2 and 3 as in this test, so the two are comparable.

## 42. XPAIR: the own-space stage across real session pairs (written 11 Nov 23:34 UTC, before any result; code `studies/12nov/xpair.py`)

**Test.** P04 and P10, three sessions each. Personal members (P, T, C, S as §41) fitted on one whole session A and
streamed over another session B (6 ordered pairs per person), and fitted on two sessions and streamed over the third
(3 per person). Chain = the LOSO v3 chain's output on B, as in §41. Stages P, T·C·S and P·T·C·S at β 0.25 and 0.5.
The matching within-session numbers are §41's for the same two people (P·T·C·S β 0.25: P04 +5.0 to +5.8, P10 +5.8
to +8.3).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| One public session, P·T·C·S β 0.25, mean of 12 pairs | +1.5 (−0.5 to +3.5); P04 +3.5, P10 −0.5 | ≤ 0: the stage does not survive a session change for either person |
| Share of the within-session gain kept | 25 % (0–60) | — |
| Two public sessions → the third, mean of 6 | +2.5 (0 to +5); better than one session by +1 (−0.5 to +2.5) | — |
| P stage (population-PC head), one session, mean of 12 | +2.0 (0 to +4): the head crosses sessions better than the own-space members for P10 | — |
| Pairs where P·T·C·S β 0.25 lowers the chain by more than 1 | 3 of 12 (1–6) | ≥ 7 |

Rule: mean of 12 ≥ +1.5 and two-session training ≥ +2 → a returning-person package candidate is specified
(own-space stage behind the identity gate, with PFT if §40 shows they add), with its dev confirmation on the real
seen protocol and the fresh bases.

| Item | Observed | Verdict |
|---|---|---|
| One public session, P·T·C·S β 0.25, mean of 12 pairs (B2_pool / B2_s345) | **+10.35 / +10.35**; P04 +13.8, P10 +6.9 | **Miss, far above the range** (predicted +1.5, −0.5 to +3.5) |
| Share of the within-session gain kept | more than all of it (within: P04 +5 to +6, P10 +6 to +8) | **Miss** (predicted 25 %) |
| Two public sessions → the third, mean of 6 | +11.98 / +11.90 (β 0.5: +15.4 / +15.1); +1.6 over one session | Above the range; the step from one to two sessions is a **hit** (predicted +1) |
| P stage, one session, mean of 12 | +5.94 / +5.67 (P04 +6.9, P10 +5.0) | Above the range (predicted +2.0) |
| T·C·S against P, per person | P04 +13.9 against +6.9; P10 +6.0 against +5.0; together +13.8 and +6.9 | — |
| Pairs lowered by more than 1 | 0 of 12 | Below the range (predicted 3) |

**Reading (11 Nov 23:40 UTC; log `runs/xpair_12nov.log`). Read the size with care.**
- **My prediction was wrong because I priced the stage against the shipped chain, and this test's base is the
  person-naive chain** (the population has never seen the person, no personal head). Against that base any personal
  information is worth a lot: the population-PC head alone gives +5.8. The numbers are gains over "nothing personal",
  not over v4.1.
- What does carry over: **across real sessions the own-space members hold about twice the personal information of the
  population-PC head** (+9.9 against +5.8; together +10.4), for both people, in every one of 12 session pairs, and a
  second public session adds +1.6 more. The session change does not remove it (§37's worry).
- In absolute terms for P04: person-naive chain plus the stage reaches 74.2 and 71.3 on sessions 2 and 3; the shipped
  v3 chain with the person in the population and the PS1 head reaches 72.5 and 65.0. For P10 it is 86.8 and 88.3
  against 91.0 and 90.9. So over the shipped chain the stage is worth about +4 for P04 and nothing for P10, which is
  what §37 measured directly (P04 +4.5 to +4.8, P10 −2.4 to −2.9 at β 0.25).
- **The increment that matters for a package is "own space on top of the population-PC head".** Ten-person proxy
  (§41, P·T·C·S minus P): +2.2 at β 0.25, +3.3 at β 0.5; eight of ten people ≥ +0.8; P10 is the one clear negative
  (−2.5). Across sessions (this test): P04 +6.9, P10 +1.9. Dev "seen" contains the best case and the worst case of
  the ten.
- **Estimate for the sealed set:** +2 to +3 per recognised returning session, times about 0.38 of sessions, gives
  **+0.8 to +1.1** if the eight single-session people behave like the ten-person proxy; +0.4 to +0.55 if they behave
  like the dev mean of P04 and P10. This is the first lever in the record with a credible +1.

## 43. DOSE and CONF: the own-space final stage on the real v4.1 chain (code `studies/12nov/dose.py`; logs `runs/dose_12nov.log`, `runs/conf_12nov.log`)

**Process note.** The dose-response is exploratory. The confirmation's predictions below were composed before the
fresh-base run, but the ledger write failed on a script assertion and the run started in the same command, so this
text reached the file only after the result (11 Nov 23:50 UTC). The predictions are unchanged from what was composed;
the only evidence for that is the session transcript. Treat the confirmation as pre-specified, not as pre-registered
on disk.

**Dose-response (exploration bases B2_pool / B2_s345; real seen protocol, session 1 → sessions 2–3).**
out ∝ chain × (T·C·S)^(β/3) on the sessions of recognised returning people.

| β | Shipped v4.1: seen (P04, P10) | With PFT nets, against the frozen twins |
|---|---|---|
| 0.05 | +1.35 (+3.2, −0.5) / +1.08 (+2.7, −0.5) | +1.49 / +1.88 |
| **0.10** | **+1.87 (+4.2, −0.5) / +1.72 (+3.7, −0.2)** | +1.61 (+4.3, −1.0) / +2.15 (+5.3, −1.0) |
| 0.15 | +1.47 (+4.5, −1.6) / +1.46 (+4.2, −1.3) | +2.13 (+5.3, −1.0) / +2.67 (+6.4, −1.0) |
| 0.25 | +0.96 (+4.8, −2.9) / +1.08 (+4.5, −2.4) | +1.75 / +1.76 |
| 0.40 | −0.20 (+5.1, −5.5) / +0.19 (+5.6, −5.2) | +0.69 / +0.95 |

P04's gain saturates near β 0.15; P10's loss grows with β and is at the noise level up to β 0.10. Single members:
C alone at β 0.05 gives +1.4 to +1.5; T alone +0.3; the three together give the smoothest curve.

**Confirmation (four fresh bases; the arm fixed after the dose-response: T·C·S, β 0.10, no other change).**

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Shipped v4.1 + stage, seen, mean of four fresh bases | +1.6 (+0.8 to +2.4); P04 +3.8 (+2.5 to +5), P10 −0.4 (−1.2 to +0.4) | mean < +1.0, or fewer than 3 of 4 bases ≥ +1.0, or P10 < −1.0 on any base |
| With PFT nets (against the frozen twins) | +1.9 (+1.0 to +2.8) | < the stage alone − 0.3 |
| New people | 0.00 | any change |

Limit: the fresh bases differ in net seeds only; the own-space members and the two people are the same. This
confirms robustness to the base, not transfer to other people.

| Item | Observed (B2_s012 / B2_s0 / B2_s1 / B2_s2) | Verdict |
|---|---|---|
| Shipped v4.1 + stage, seen | +1.74 / +2.39 / +2.53 / +1.07, **mean +1.93** (dJ +0.97); 4 of 4 ≥ +1.0 | **Hit; passes** |
| P04 / P10 | +3.73, +5.06, +4.82, +2.39 (mean +4.00) / −0.26, −0.28, +0.25, −0.26 (mean −0.14) | **Hit, hit** |
| With PFT nets, against the frozen twins | +2.67 / +2.01 / +2.27 / +2.41, mean +2.34 (dJ +1.17); PFT alone +1.30; P10 −1.0 on every base | **Hit**; above the stage alone |
| β 0.15 for comparison (not the fixed arm) | +2.00 / +1.72 / +2.94 / +1.60 | — |
| New people | 0.00 | Hit |

**Reading (11 Nov 23:50 UTC).**
- **On all six dev bases the own-space final stage adds +1.1 to +2.5 to returning sessions (mean +1.9, dJ about
  +0.95), and with the personal nets +2.3 (dJ +1.17).** It is the largest returning-person gain since routing itself
  and the first lever of this session with a dev size near +1 J.
- It is still P04's gain: P10 is at −0.1 (and −1.0 with PFT). What is new against PFT's "P04 only" is the ten-person
  evidence (§41, §42): own-space members add to eight of ten people within a session, and P10 is the one clear
  negative of the ten.
- **Sealed estimate: +0.4 to +0.9** (dJ +0.95 to +1.17 at the usual 0.46–0.7 transfer, or +2 to +3 per recognised
  returning session if the eight single-session people follow the ten-person proxy). With TTT for new people (about
  +0.2) a package could reach roughly 0.821 to 0.827. Not built.
- Not done: better own-space members (these are a first version: 9 bands, whole-window covariances, fixed
  regularisation), training on all public sessions of P04 and P10 (+1.6 in §42), the package form (per-person
  filters and heads, causal recentring, catch-up at recognition), runtime.

## 44. MEMSWEEP: better own-space members (written 11 Nov 23:55 UTC, before any result; code `studies/12nov/memsweep.py`)

**Why.** The members of §37–§43 are a first version (9 bands, whole-window covariances, fixed regularisation,
uncalibrated posteriors). Selection is on the ten-person proxy of §41 (T·C·S stage over the LOSO chain, both
exploration bases, best β of 0.1 / 0.25 / 0.5 / 1), then checked on the 12 real session pairs of §42. Only a variant
that is better on both goes to the real seen chain; the seen protocol (P04, P10) is not used for selection.

Variants: V0 current; V1 core1's 17 overlapping bands; V2 two half-window covariances added to T; V3 CSP with 4
filters per pair; V4 LDA shrinkage 0.3 for C; V5 T with C 0.003; V6 T with C 0.03; V7 temperature of each member
fitted on the public part by leave-one-run-out (then β acts on calibrated posteriors); V8 = V1 + V7.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| V0 on the ten-person proxy (best β) | +4.2 (the §41 value) | — |
| Best variant minus V0, ten-person proxy | +0.6 (0 to +1.5) | — |
| Best variant minus V0, 12 session pairs | +0.5 (−1 to +2) | — |
| V7 (calibration): people lowered by more than 1 at its best β | 0 (0–1), against 1 for V0 at β 0.5 | — |
| P(a variant beats V0 by ≥ +0.5 on both) | 35 % | — |

Rule: adopt the variant with the best ten-person mean only if it also beats V0 on the 12 pairs; otherwise V0 stays.

| Item | Observed | Verdict |
|---|---|---|
| V0 on the ten-person proxy, best β | +4.17 (β 0.25); 12 pairs +10.54 (β 0.5) | Hit |
| All variants, ten people / 12 pairs (best β) | V1 +4.50 / +11.08; V2 +4.12 / +10.64; V3 +3.83 / +10.62; V4 +4.21 / +10.73; V5 +4.08 / +10.67; V6 +4.12 / +10.44; V7 +4.38 / +10.99; **V8 +4.71 / +11.57** | Best minus V0: +0.54 and +1.03: **hit, hit** (predicted +0.6, +0.5) |
| Fitted temperatures (median T / C / S) | 0.43 / 4.7 / 8.0 (the grid's upper end): the tangent head is under-confident, both LDA members are far too sharp | — |
| V7, people lowered by more than 1 at its best β | 0 (V0: 1) | Hit |
| By the rule V8 is adopted for the chain check. Real seen chain, exploration bases, best β (`dose2.py`) | V8 +1.52 to +1.66 (β 0.25–0.5; P10 −1.3 to −3.6); V7 +1.61 to +1.75; V1 +2.03 to +2.04 (P10 −1.3 to −1.6); V0 at β 0.10: +1.72 to +1.87 (P10 −0.2 to −0.5) | **V8 is not better than V0 on the real chain** |

**Reading (12 Nov 00:00 UTC; logs `runs/memsweep_12nov.log`, `runs/dose2_v*.log`).** The members are at a plateau:
nine variants lie within half a point of each other on ten people, and on the real chain every one gives +1.5 to
+2.0 with P10 between −0.2 and −1.6. V0 at β 0.10 stays the reference because it costs P10 least. More member
engineering is not where the next gain is; the lever's size is set by how many returning people are P04-like.

## 45. V5R: the own-space stage as the package will run it (written 12 Nov 00:25 UTC, before any result; code `studies/12nov/v5_replay.py`)

**Why.** The user asked for the package candidate (12 Nov) and for exploration to continue. §43 applied the stage to
every window of a returning person's session with the true identity. The package only knows what the identity gate
says: nothing before window 10, a ramp r in [0, 1] afterwards, 87 % of returning sessions accepted, and about 2 % of
new-person sessions accepted by mistake.

**Mechanism (the package form).** For every window where the merged gate of v4.1 routes (r > 0) to gallery person
q: out ∝ routed tail × (T_q · C_q · S_q)^(β·r/3), β 0.10, members V0 of §37 fitted on q's public sessions of the fold
(seen fold: session 1; leave-one-person-out folds: all sessions of the other nine). Covariance sums run from window
0 for everyone; the person's reference prior enters only when the stage is applied. No other change to v4.1.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Seen, mean of six bases | +1.7 (+1.0 to +2.4): a little below §43's +1.9, the stage starts at recognition | < +1.0 |
| P04 / P10 | +3.5 (+2.2 to +4.8) / −0.2 (−1.0 to +0.5) | P10 < −1.0 on more than one base |
| New people, mean of six bases | 0.00 (−0.05 to +0.05); at most one of 14 new sessions touched | < −0.1 |
| dJ | +0.85 (+0.5 to +1.2) | — |

Pass → this replay is the reference the package must reproduce (stage posteriors and labels on the public windows).

| Item | Observed | Verdict |
|---|---|---|
| Seen, six bases | +2.00 / +1.98 / +1.59 / +1.98 / +1.87 / +1.31, **mean +1.79** | **Hit** (predicted +1.7) |
| P04 / P10, mean | +3.51 / +0.07 (P10 between −0.28 and +0.26) | **Hit / hit** |
| New people | 0.000 on every base; no new-person session is routed | **Hit** |
| dJ, mean | **+0.89** (+0.66 to +1.00) | **Hit** (predicted +0.85) |
| Gate on the four seen sessions (B2_pool) | P04 session 2 routed only from window 64 (share 0.60, mean ramp 0.86); P04 session 3, P10 sessions 2 and 3 from window 10 (shares 0.95, 0.69, 0.95) | — |

**Reading (12 Nov 00:28 UTC; log `runs/v5_replay_12nov.log`).** The package form keeps the gain: +1.79 on returning
sessions against +1.93 with the true identity from window 0. This replay is the reference for the package
(`runs/sealed/cand_v5_own`, built on Galvani 3090163; check 3090168, A100 replay 3090169, worker replay 3090170).
One of four seen sessions is recognised late and one only on 69 % of its windows: recognition is now a lever of its
own (§46).

**Addendum, 12 Nov 02:35 UTC: the package reproduces the study code.** `scripts/check_own_stage.py` on the Galvani
build (job 3090168): outputs equal to cand_v41_bio on every unstaged window; member log posteriors equal
`ownspace.Personal` to 1.6e-12; staged outputs equal base × exp(β r log members) exactly; all 14 public sessions are
staged from window 10 to their own person. A100 and worker-image replays identical (§ upload log). Open: paired
timing (3090327), independent review.

## 46. Toward 0.84: recognition head-room and virtual people (written 12 Nov 00:35 UTC, before any result)

**The user's target (12 Nov): a sealed score above 0.84**, +2.4 over 0.8162. In sealed points that needs, for
example, +3.5 on returning sessions (43 % of sessions → +1.5) and +1.6 on new-person sessions (57 % → +0.9). In hand:
the own-space stage, dev +1.8 on returning sessions (§45). Not in hand: anything for new people beyond TTT's +0.6 dev.

**46a. ORACLE: what perfect recognition of returning people would add (diagnostic; code `studies/12nov/oracle_gate.py`).**
v4.1 and v4.1 + stage with the true identity at ramp 1 from window 0 on the seen sessions, nothing routed on new
people, against the real gate.

| Check | Prediction (80 % range) |
|---|---|
| v4.1 with the oracle gate, seen, minus v4.1 | +0.5 (0 to +1.2) |
| v4.1 + stage with the oracle gate, minus the same with the real gate | +0.6 (0 to +1.5): the late session of P04 and the 69 % session of P10 recover |

**46b. ROTAUG: virtual people for core1 by random spatial rotations (code `studies/12nov/rotaug.py`).**
§39: each added real person is worth about +2 at core level. Between people, recentred covariances differ, among
other things, by how sources project to the sensors. For every training session two virtual copies are added whose
tangent blocks are rotated in sensor space, L → R L Rᵀ with R = exp(ε A), A a random skew matrix scaled so that each
channel mixes with the others at total amplitude ε; one R per session copy (a "virtual person"); the slow block is
left unchanged. The nets already train with rotation augmentation (`rot05`); core1 never did. core1 refitted per
held-out person on the nine others plus the copies; score as §39 (core alone, k = 9, reference 69.29).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Best ε of 0.1 / 0.2 / 0.4, core1 new-person BA minus 69.29 | +0.8 (−0.5 to +2.5) | ≤ +0.3 at every ε |
| People improved at the best ε | 6 of 10 (4–8) | — |
| P(≥ +1.5, the size of most of one real person) | 20 % | — |

Rule: ≥ +1.5 → the rotated core goes into the chain replay in core1's slot; between +0.3 and +1.5 → one more
variant (rotation of the slow block as well, four copies); ≤ +0.3 → closed.

| Item | Observed | Verdict |
|---|---|---|
| 46a. v4.1 with the oracle gate, seen, minus v4.1 (six bases) | **−0.24** (−0.53 to +0.69; P04 −0.66, P10 +0.19) | **Miss, below the range** (predicted +0.5, 0 to +1.2) |
| 46a. v4.1 + stage with the oracle gate, minus the same with the real gate | **−0.43** (+1.36 against +1.79; P04 +3.13 against +3.51, P10 −0.40 against +0.07) | **Miss, below the range** (predicted +0.6) |

**Reading of 46a (12 Nov 00:32 UTC; log `runs/oracle_gate_12nov.log`).** Knowing the person from the first window
is worse than the real gate on dev. The personal head and the stage are poor in a session's first windows (their
standardisation and reference have almost nothing to work with), and the real gate's delay and ramp protect against
that. Earlier or more certain recognition is not a lever on dev; the only recognition loss left is returning
sessions the gate never accepts (about 13 % by the external estimate), which dev cannot measure. Closed as a
direction; the package keeps the v4.1 gate unchanged.
| 46b. ROTAUG, ε 0.1 / 0.2 / 0.4 (two rotated copies per session; reference 69.29 reproduced) | 69.25 / 69.15 / 68.19: **−0.04 / −0.14 / −1.10**; people improved 5 / 3 / 2 of 10 | **Refuted** (≤ +0.3 at every ε; predicted +0.8, range −0.5 to +2.5: inside the range at the low end for the two weak strengths, below it for the strong one) |

**Reading of 46b (12 Nov 03:05 UTC; `runs/rotaug_12nov.log`).** Random rotations of the recentred tangent blocks do
not stand in for people: nothing at small strength, a loss at the strength the nets train with. Soft verdict (§52):
closed for unstructured rotations on core1 alone; the structured ones (cap shift, gain) and the nets (§51) are still
running.

## 47. SYNTH: virtual people for the population model (written 12 Nov 00:45 UTC, before any result; code `studies/12nov/virt.py`; Galvani CPU jobs)

**Why.** The user (12 Nov): more data, synthetic generation, keep exploring. §39: +8 points per doubling of training
people at core level. Data search (12 Nov, two passes): no public EEG set with our tasks and more than a few people
beyond what is already used; the Bordeaux and Graz mental-task studies are not public; the closest new sets are
ds007554 (30 people, arithmetic and arm imagery in 3-minute blocks, another cap) and an MEG set with our exact
tasks (17 people). So the test is whether between-person variation can be generated.

**Generators** (each adds two virtual copies of every training session to core1's training rows; score as §39/§46b,
core1 alone, nine training people, reference 69.29):
- **rot** (§46b, running locally): random rotation of the tangent blocks, ε 0.1 / 0.2 / 0.4.
- **cap**: the cap sits differently on a different head. Electrode positions (standard 10-05 on a sphere) are rotated
  by a random rotation vector with standard deviation σ degrees per axis; the signals at the shifted sites are
  obtained by spherical-spline interpolation (a smooth 41 × 41 mixing A); covariances become A C Aᵀ, the session
  reference and all tangent blocks are recomputed, the slow block is mixed by A. σ 3 / 6 / 10.
- **mix**: a virtual person whose class means lie between two real people: x′ = x + λ (μ_B,k − μ_A,k) for the windows
  of class k of person A, B another training person, λ uniform in [0, λmax]; λmax 0.5 / 1.0.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| cap, best σ, minus the reference | +1.0 (−0.5 to +3.0) | ≤ +0.3 at every σ |
| mix, best λmax | +0.3 (−1.0 to +1.5) | ≤ +0.3 at both |
| rot, best ε (§46b) | +0.8 (−0.5 to +2.5) | as §46b |
| P(any generator ≥ +1.5) | 25 % | — |

Rule: a generator at ≥ +1.5 is rerun with four copies and then carried into the chain replay in core1's slot;
+0.3 to +1.5 gets one refinement; below that it is closed. If all three close, generated variation of these kinds
does not stand in for people, and a learned generator would need a corpus of many people in this montage (the
87-person Dreyer 2023 set covers 27 of our 41 channels, motor imagery only).

**Addendum, 12 Nov 00:52 UTC, before any §47 result: a fourth generator, gmix.** "mix" only moves a person's class
means toward another real person's. gmix draws a new person from the between-person family: the three class means of
the copy are the grand class means plus Σ_j w_j (μ_j,k − grand_k) over the training people j, with the same random
weights w_j ~ N(0, γ²/(n−1)) for all three classes (γ 1 matches the observed spread between people); the session's
own windows keep their within-class scatter. γ 1.0 / 1.5. Prediction: best γ +0.5 (−1.0 to +2.0) over the reference;
refuted if ≤ +0.3 at both. It counts toward "any generator ≥ +1.5" (now 30 %).

**Corrections and inputs before any §47 result (12 Nov 01:25 UTC).**
- **cap generator, bug caught by its own sanity line.** The first three cap jobs (Galvani 3090209–3090211) printed
  "sigma 0: max abs feature difference 61": at zero shift the copies did not equal the originals. Cause: MNE's
  spherical-spline matrix smooths with its default regularisation (at identical positions max |A − I| = 0.64). With
  `alpha=1e-11` it is exact at zero shift and well conditioned at 3–10° (condition number 2–4; electrode spacing is
  about 20°). The three jobs were cancelled before they printed a result and resubmitted with the fix.
- **No more PROTEUS data exists.** Second search (primary sources: NEMAR, the dataset's repository, the competition
  repository and site history): nm000290 v1.0.0 (10 people, 14 sessions) is the only release; the "coming soon"
  wording was about that first release. Rule 04 allows public datasets whose terms permit competition use, declared
  in the write-up. ds007554 is CC0 (30 people, 4.5 GB; per session one 3-minute arithmetic block and one arm-imagery
  block with nominal event times); the MEG set with our tasks is CC0, 116 GB, no EEG. Neither is fetched.

**Literature pass and a fifth generator (12 Nov 01:40 UTC, before any §47 or §49 result).** A search of the
few-subject literature (primary sources read where marked in the session notes) found:
- With fewer than ten training domains, tuned pooled training is hard to beat: invariance penalties (IRM, V-REx,
  group DRO) are at or below it on DomainBed and on EEG (WOODS); hierarchical priors over subjects equal pooled ridge
  for an unseen subject (Jayaram 2016) or add +0.7 (Fazli 2011, 83 subjects); no source shows generated subjects
  beating a tuned linear baseline with fewer than ten subjects (a Riemannian VAE lost 3–4 points with an SVC). This
  lowers my prior for E3 and E6 of §49 to about 10 % each.
- Alignment is the one large documented effect (+4 to +5.6), and our recentring is its Riemannian form.
- The one augmentation with consistent cross-subject gains at 7–9 subjects is left-right channel reflection with a
  label swap (+1.4 to +3.8, EEGNet). It needs classes that mirror into each other; ours do not, and the mirrored
  core of 11 Nov (§5, arm A25) lost because it destroys the lateralisation of hand imagery. Closed, not rerun.
- Rotation and cap-shift copies survive recentring only as an orthogonal rotation of the recentred covariances
  (so "rot" and "cap" test the same family, cap the structured part of it). Reported gains for spatial rotations are
  small or absent at 19–32 electrodes.
- A whole-subject frequency shift is suggested but unverified across subjects. **fsc** is added: two copies of every
  training session with all 17 band edges scaled by 1/(1+p) and (1+p), a person whose rhythms run slower or faster;
  p 0.10 / 0.20. Prediction: best p +0.5 (−1.0 to +2.0); refuted if ≤ +0.3 at both.

**Two more generators at the user's suggestion (shifted or noisy variants of the data we have), 12 Nov 02:05 UTC,
before any §47 result.** **gain**: per-channel gain jitter, C → D C D with D = diag(exp(σ n)), σ 0.1 / 0.3 (contact
quality differs between people and sessions; after recentring this is again a rotation, structured by channel).
**wn**: added sensor noise, C → C + s · (mean channel variance) · I per band, s 0.5 / 2.0 (a person with a weaker
signal; not a rotation: it flattens the eigen-spectrum). Predictions: gain best σ +0.3 (−1.0 to +1.5); wn best s 0.0
(−1.5 to +1.0). Time-shifted windows are not rerun: as extra training rows they lost 0.47 nested on 7 Nov
(REVIEW_5NOV, core training crops), because the cue-locked sub-windows and slow features of a shifted window are a
different distribution.

| Item | Observed | Verdict |
|---|---|---|
| Reference on Galvani (no copies; 3090208) | 69.29, per person identical to the local run | Machines agree |
| mix, λmax 0.5 / 1.0 (3090254, 3090255) | 68.73 / 67.86: **−0.56 / −1.43** | **Refuted** (predicted +0.3; ≤ +0.3 at both) |
| rot, ε 0.1 / 0.2 (local, §46b) | 69.25 / 69.15: −0.04 / −0.14; ε 0.4 still running | No effect so far |
| gmix, γ 1.5 (3090257) | 67.48: **−1.81**; γ 1.0 still running | Negative |
| cap, σ 3° / 6° (3090251, 3090252; sanity at zero shift 1.5e-3) | 68.86 / 68.07: **−0.43 / −1.22**; σ 10° still running | Negative so far (predicted +1.0) |
| fsc, p 0.10 (3090259) | 68.69: **−0.60**; p 0.20 still running | Negative so far (predicted +0.5) |
| cap, σ 10° (3090253) | 67.77: **−1.52** | cap is negative at every σ (−0.43 / −1.22 / −1.52): **refuted** |
| gmix, γ 1.0 (3090256) | 68.34: **−0.95** | gmix is negative at both γ (−0.95 / −1.81): **refuted** |
| gain, σ 0.1 / 0.3 (3090328, 3090329) | 69.28 / 68.87: **−0.01 / −0.42** | **Refuted** (≤ +0.3 at both; predicted +0.3) |
| wn, s 0.5 / 2.0 (3090330, 3090331) | 66.20 / 66.04: **−3.09 / −3.25** | Below the range (predicted 0.0, −1.5 to +1.0): added sensor noise is clearly harmful |
| fsc, p 0.20 (3090373; the first job on a v100 node produced no output in an hour) | 67.61: **−1.68** | fsc is negative at both p (−0.60 / −1.68): **refuted** |
| **All seven generators** (rot, cap, gain, wn, fsc, mix, gmix), every strength | best is gain 0.1 at −0.01; every other setting is below the reference | **P(any ≥ +1.5) did not happen; no generator even reaches +0.3** |

**Reading of §47 (12 Nov 02:10 UTC by `date -u`).** Generated variation does not stand in for people on core1.
Spatial transforms (rotation, cap shift, gain) are neutral at small strength and harmful at larger ones; spectral
scaling, mixing between people and sensor noise are harmful throughout. Together with §49 (no people-efficient
estimator) and §53 (external same-task people lower accuracy even on real shared channels), the people curve of §39
stands as a statement about real people of this protocol, of whom there are ten. Soft verdict (§52): closed for
core1 alone; the nets are still running (§51, §56, §57a).

## 48. CHAINSLOPE: does the people curve survive the session stages? (written 12 Nov 00:58 UTC, before any result; code `studies/12nov/chainslope.py`)

**Why.** §39 is a core-level slope. Stronger members have repeatedly failed to move the chain (STATE §5), and the
stages add most where the base is weak, so the slope after the stages may be much smaller. That number prices every
"more people" lever (§47 and any new data).

**Test.** As §39 (core1 refitted on k = 3, 5, 7 other people, two draws each, and on all 9), then a core-only chain
on the held-out person's sessions: rebalancing (α 1.15, k 5, W 30) → SessionEM (λ 0.5, the packaged ocular and
slow-potential view) → TSEM (λ 0.15, the 128 PCs of the fold's own scaled training features). No other member, no
early boost, no routing.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Core-only chain at k = 9 | 76 (73–79); the full chain is 80.2 | — |
| Stages' gain over the core at k = 9 / at k = 3 | +7 / +9 (each ±3): larger where the base is weak | — |
| Slope per doubling after the stages | +5 (+3 to +7.5), about 0.65 of the core slope | < +2.5: the stages absorb most of what more people give |

| Item | Observed | Verdict |
|---|---|---|
| Core alone, k = 3 / 5 / 7 / 9 | 56.34 / 60.71 / 65.43 / 69.29 (identical to §39) | Reproduced |
| After rebalancing and SessionEM | 58.25 / 62.22 / 66.80 / 70.01 | — |
| Core-only chain (+ TSEM) | 58.65 / 62.50 / 67.24 / 70.10 | k = 9: **miss, far below** (predicted 76, range 73–79) |
| Stages' gain over the core at k = 9 / k = 3 | +0.8 / +2.3 | **Miss** (predicted +7 / +9): on core1 alone the stages add almost nothing |
| Slope per doubling after the stages | **+7.32** (core +8.13); 7 → 9: +2.85 | **Hit**, upper end (predicted +5, range +3 to +7.5); not refuted |

**Reading (12 Nov 03:28 UTC by `date -u`; Galvani 3090258).**
- **The people curve survives the stages in this test:** +7.3 per doubling after rebalancing, SessionEM and TSEM,
  against +8.1 before. More real people would raise the adapted score nearly as much as the raw one.
- The test is weaker than intended. A chain built on core1 alone gains only +0.8 from the stages (the full chain
  gains about +10 on the member mix): core1's posteriors are far over-confident (§31), and the stages are tuned for
  the softer mix. So the slope is measured where the stages do little; the full-chain slope is still not known.


## 49. PEFF: people-efficient estimators and noise for the population model (written 12 Nov 01:10 UTC, before any result; code `studies/12nov/peff.py`)

**Why.** The user (12 Nov): add noise, try sample-efficient methods. §39 says the estimator is starved of people: a
pooled logistic regression treats 2,400 windows as independent although they come from nine people, so it trusts
directions in which people differ. Estimators that count people, not windows, should need fewer of them.

**Setup.** Per held-out person: core1's features of the nine others, standardised, reduced to 512 PCs (so that every
estimator fits in seconds); the held-out person's causal features go through the same map. Reference inside this
harness: E1, logistic regression on the 512 PCs (best of C 0.01 / 0.1 / 1). Outer reference: the full 44k-feature
core1, 69.29 (§39).

| Estimator | What it changes |
|---|---|
| E2 person-balanced | every training person has the same total weight |
| E3 random-effects LDA | covariance = within-class + τ × covariance of the people's class-mean deviations; class means = person-equal means; τ 0 / 30 / 100 / 300 |
| E4 consistency selection (on the 44k features) | keeps the m features whose class contrasts agree most across people (largest across-people t), then the core's own head; m 2,000 / 8,000 |
| E5 people-bagging | 12 heads, each on 6 of the 9 people, averaged |
| E6 group DRO | 8 rounds; people re-weighted by exp(their loss) |
| E7 isotropic noise | two noisy copies, σ 0.5 / 1.0 of each PC's standard deviation |
| E8 person-jitter noise | two copies with noise whose per-PC scale is the across-people spread of that class's mean |

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| E1 (512 PCs) against the full core | −1.5 (−4 to +0.5): the PCs lose a little | — |
| Best of E2–E8 minus its reference (E1, or the full core for E4) | +1.0 (−0.5 to +2.5) | ≤ +0.3 for every estimator |
| E3, best τ > 0, minus τ = 0 | +1.0 (−0.5 to +3) | ≤ 0 |
| E7 isotropic noise minus E1 | 0.0 (−1 to +1): it is ridge by another name | — |
| P(any estimator ≥ +1.5 over its reference) | 30 % | — |

Rule: an estimator at ≥ +1.5 is rebuilt on the full features and carried into the chain replay in core1's slot.

| Item | Observed | Verdict |
|---|---|---|
| E1, 512 PCs, logistic regression (C 0.01 / 0.1 / 1) | 65.30 / 65.24 / 64.33; against the full core's 69.29: **−4.0** | At the lower edge (predicted −1.5, range −4 to +0.5): the 512 PCs lose four points |
| E2 person-balanced | 65.05 (−0.25 against E1) | — |
| E3 random-effects LDA, τ 0 / 30 / 100 / 300 | 66.38 / 65.28 / 65.24 / 64.96 | **Refuted** (τ > 0 is −1.1 below τ = 0). Plain LDA with person-equal means is +1.1 over the logistic head in this space |
| E4 consistency selection on the 44k features, m 2,000 / 8,000 | 60.14 / 66.62 against 69.29 | **Refuted** (−2.7 at best) |
| E5 people-bagging / E6 group DRO | 65.12 / 65.23 (−0.2 / −0.1) | No effect |
| E7 isotropic noise, σ 0.5 / 1.0 | 65.24 / 65.22 (−0.06 / −0.08) | **Hit** (predicted 0.0) |
| E8 person-jitter noise, 1× / 2× | 64.24 / 63.81 (−1.1 / −1.5) | Negative |
| Best people-efficient estimator minus its reference | none above +0.3 (E3's τ = 0 is the control, not the method) | **Refuted** by the pre-set rule |

**Reading (12 Nov 02:55 UTC; Galvani 3090241, `runs/diag/virt/peff_3090241.log`).**
- **No estimator that counts people instead of windows beats the pooled fit**, and noise does nothing. This matches
  the literature pass (§47): with fewer than ten training people a tuned pooled model is hard to beat. The people
  curve reflects information that only real people bring.
- Side finding worth keeping: **reducing core1's 44k features to 512 PCs costs four points on new people.** The
  class information is spread over the long tail of features. TSEM (128 PCs) and the personal head (256 PCs) work in
  a space that has lost part of it, the same message as §36 from another side.
- Soft verdict (§52): closed in this form, on this member, at core level. Not tested: the same estimators on the
  full feature set (only E4 ran there) and inside the chain.

## 50. UPW: a returning person's own rows up-weighted in the population core (written 12 Nov 01:55 UTC, before any result; code `studies/12nov/upw_core.py`, `upw_chain.py`)

**Why.** §39 read from the inside: the most valuable training person is the test person. Having session 1 in the
population lifts P10 from 82 to 91 and P04 from 54 to 69. At core level (§17) up-weighting the person's rows helped
P10 across sessions (+2.1 at weight 5, +4.4 at 20) where the personal head did not (−1.0), and was mixed for P04; it
was never scored in the chain. The package can hold one core head per public person (1 MB each) and swap it in at
recognition.

**Test.** Seen folds (all other people + session 1 of P04 / P10): core1's head refitted with the person's own rows at
weight w ∈ {3, 5, 10, 20}; w = 1 must reproduce the stored core1 posteriors. In the v4.1 chain replay core1's
posterior on the seen sessions is replaced by the up-weighted one (as if recognised from the first window; the TSEM
view and the personal head are unchanged). With and without the own-space stage (§45). Exploration bases first.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Harness: w = 1 against the stored core1 | equal to 1e-6 | not equal: no result is read |
| Core alone on the later sessions, best w, P04 / P10 | +0.5 / +3 (as §17) | — |
| v4.1 chain, best w, seen | +0.5 (−0.5 to +1.5): P10 +1.0 (−0.5 to +2.5), P04 0.0 (−1.5 to +1.5) | ≤ 0 on both exploration bases at every w |
| On top of the own-space stage | adds +0.3 (−0.7 to +1.3) to the stage's +1.8 | — |

Rule: ≥ +0.8 seen on both exploration bases with P04 ≥ −0.5 → fresh bases; then per-person cores join the v5
candidate.

| Item | Observed | Verdict |
|---|---|---|
| Harness, w = 1 against the stored core1 | max difference 3.0e-8 | Valid |
| Core alone on the later sessions (window accuracy; w = 1: P04 57.0, P10 76.4) | w 3: 57.8 / 77.7; w 5: 57.6 / 77.9; w 10: 56.0 / 77.4; w 20: 56.8 / 77.9 | P10 +1.0 to +1.5 at every weight, P04 +0.6 to +0.8 at 3–5 then below: smaller than §17's core-level proxy (predicted +0.5 / +3) |
| v4.1 chain, seen, w 3 / 5 / 10 / 20 (B2_pool; B2_s345) | −0.25, −0.13, +0.01, +0.01; −0.26, +0.01, 0.00, 0.00 | **Refuted** (≤ 0 on both bases at every weight, to within 0.01) |
| On top of the own-space stage (stage alone +2.00; +1.98) | +1.73, +1.99, +1.87, +1.86; +1.85, +1.86, +1.99, +1.85 | Adds nothing (predicted +0.3) |

**Reading (12 Nov 01:58 UTC; log `runs/upw_chain_12nov.log`; a first run of the scorer stopped on a name clash in
my script before any result).** Up-weighting the returning person's rows makes core1 slightly better alone and
changes nothing in the chain: the personal head and the stage already carry what the up-weighted core adds. Closed
in this form (posterior swapped in from the first window; the TSEM view unchanged).

## 51. NETAUG: virtual people for the session-normalised nets (written 12 Nov 02:25 UTC, before any result; code `studies/12nov/netaug.py`, scorer `netaug_score.py`)

**Why.** The user (12 Nov): the October closures on the nets should not hold us back; try the approaches and their
combinations again with what we now know. Reading the code with that in mind:
- The October tests (mixing, frequency scaling, weight averaging, MixStyle, adversarial, DRO) ran on the plain nets.
  The shipped members are SessionBN nets (5 Nov), and `SessionNet._train` applies only the per-window rotation:
  **`cmix` and `fsc` are ignored there**, so neither was ever tested on the current nets.
- The rotation the nets train with (`rot05`) draws a new random rotation for every window. A person is a
  transformation shared by all their windows; per-window rotation is noise around the same people. A SessionBN batch
  is made of chunks of 32 windows of one session, which is exactly the unit a virtual person needs.
- §39: people are the bottleneck; scoring is now in the chain.

**Arms** (four SessionBN families of the chain: FBCNet EA k2, FBLight, SincShallow, raw FBCNet; seeds 0–2; twins =
the frozen wave-2 nets of §31):
- **crot05 / crot10**: one random rotation per 32-window chunk (ε 0.05 / 0.10) instead of one per window.
- **cfsc15**: one frequency-scaling factor in [0.85, 1.15] per chunk (never applied to these nets before).
- **combo**: the shipped per-window rotation + chunk rotation 0.05 + chunk frequency scaling 0.15.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Standalone new-person BA, best arm minus twin, mean of families | +0.8 (−0.5 to +2.5) | ≤ 0 for every arm |
| crot05 minus twin (the clean "shared instead of per-window" comparison) | +0.5 (−0.7 to +1.8) | — |
| Chain, all four families swapped, best arm, dJ (exploration bases) | +0.2 (−0.3 to +0.7) | ≤ 0 on both bases |
| P(chain dJ ≥ +0.5 on both exploration bases) | 12 % | — |

Rule: chain ≥ +0.5 on both exploration bases → fresh bases, then the families are retrained for a package. My prior
is low because stronger nets have not moved the chain before (§7, §12); the arms are run because the mechanism was
never tested in this form and costs only queue time.

| Item | Observed | Verdict |
|---|---|---|
| Twins through the same code, new / seen (3 seeds) | FBCNet 66.04 / 64.12; raw FBCNet 63.42 / 59.34; SincShallow 61.75 / 62.11; FBLight 57.94 / 63.01 | Control |
| Standalone new, arm minus twin: crot05 / crot10 / cfsc15 / combo | FBCNet −0.30 / −1.45 / +0.60 / −0.31; raw FBCNet −3.14 / −9.20 / +0.01 / −2.57; SincShallow +0.28 / −0.94 / −0.38 / (combo did not finish); FBLight +0.59 / −1.26 / −0.29 / −0.94 | Best arm, mean of families: cfsc15 −0.02, crot05 −0.64: **refuted** (≤ 0 for every arm on the mean; predicted +0.8) |
| crot05 minus twin, the clean comparison | −0.64 on the mean of families (+0.59 to −3.14) | **Miss, below the range**: one rotation per chunk is not better than one per window, and it hurts the raw-input family |
| Chain, one family swapped (B2_pool, B2_s345) | all within −0.53 to +0.52; the largest, FBLight combo, is +0.52 / +0.14 | Inside the noise |
| Chain, all four families swapped, against all twins | crot05 −0.18 / −0.09; crot10 −0.81 / −0.85; cfsc15 −0.19 / −0.12 | **Refuted** (≤ 0 on both bases for every arm) |

**Reading (12 Nov 03:28 UTC; `runs/netaug_score_12nov.log`; Galvani packs 3090340, 3090364–3090368, twins 3090369).**
Virtual people do not help the SessionBN nets either, alone or in the chain. Chunk-wise rotation is no better than
the shipped per-window rotation; frequency scaling, never run on these nets before, is neutral alone and slightly
negative in the chain. With §47 this closes generated variation for both member types in the forms tested.


## 52. Standing rule: soft gates, and reviews that hunt silent no-ops (12 Nov 02:45 UTC)

**From the user, 12 Nov:** cleanups and reviews should find bugs like the one in §51; none of our gates or decisions
should be hard; the codebase exists only to optimise the final submission.

**What changed in the repo.**
- `experiments/session_bn.py`: `SessionNet._train` now asserts that `cmix` and `fsc` are not set instead of ignoring
  them (no shipped recipe sets them; pinned submission files unchanged).
- An audit for the same defect class is running over `experiments/`, `scripts/`, `replay/` and `studies/`: options
  parsed but not applied, overrides that drop a parent's behaviour, cache keys that omit a setting, scorers whose
  baseline differs from the stated one, leakage in folds. Findings and their effect on recorded verdicts go below.
- `AGENTS.md` carries both rules for every later session.

**Closures re-read under this rule** (status: what was actually tested, and what is now running or still open):

| Closure | What was actually tested | Now |
|---|---|---|
| Net training tricks (mixing, frequency scaling, weight averaging, MixStyle, adversarial, DRO), October | plain nets, one trick at a time, scored alone or in old mixes; mixing and frequency scaling never ran on SessionBN nets | chunk-wise rotation and frequency scaling running (§51); mixing, weight averaging and MixStyle on SessionBN nets still untested |
| Rotation augmentation "adopted" (`rot05`) | one random rotation per window | per-chunk rotation running (§51); rotated copies for core1 running (§46b) |
| Individual-frequency band shift | "closed, no run" (the peak estimate was judged unstable) | frequency-scaled virtual people running for core1 (§47 fsc) and the nets (§51) |
| Scherer 2015 in the cores (§21) | its 30 channels spline-interpolated to our 41 (22 interpolated), slow blocks filled with means | a core on the 19 channels both caps really share, trained on 18–19 people, was never tested: open |
| "P04 is a low-signal person" | within-session decoding in core1's features only | revised (§36): 74–80 % in its own space |
| Personal models for returning people | logistic head on the population PCs only | own-space stage built (§41–§45); up-weighted personal core running (§50) |
| Similarity routing for new people (§16) | spectral identity score, personal heads in the population PCs | own-space members of other people were not tried: open, low prior |
| Extra members lower the chain (§25, §26, §31) | members added at fixed stage settings | stage settings re-fitted with the member in place were tried only by two overfitting searches: open |
| Graz cue-order prior (§9) | multiplied into a stage output with one fitted temperature | a calibrated (non-parametric) fusion at the final stage was not tried: open, bounded near +0.4 |
| In-session learning on the chain's labels | nets (§24), population-PC head (§35), own-space members (§38), each properly run | stays closed; three independent families agree |
| Earlier or perfect recognition (§46a) | oracle identity from window 0, six bases | stays closed on dev |

| Audit finding | Affected verdicts | Action |
|---|---|---|
| **MixStyle never ran on ShallowConvNet** (confirmed by test: output identical with it on and off). Its hooks attach to the first two convolution modules; in ShallowConvNet those are wrapped in a combined layer that calls the functional form, so the hooks never fire. In FBCNet, FBLight and SincShallow only one convolution is hooked | R18 (October, `docs/archive/OPTIMIZATION_R10.md`): two of the six rows behind "MixStyle is neutral (±0.4)" are baseline reruns; the closure rests on four rows on plain nets | Recorded. MixStyle on the SessionBN nets remains untested (re-read table above) |
| **Segment recombination reached only the first corpora of the teacher's pretraining** (confirmed by test): it augments the first half of each batch, and the batch is built corpus by corpus, so of seven corpora the last three were never touched | Wave 8, 2 Nov ("segment recombination does not help the teacher", −0.48, closed): the corpora with arithmetic and word tasks never received it | Recorded; open, low priority (the teacher is pretrained once) |
| core1 variants entered the chain as posteriors only (`studies/11nov/rt4/score_core.py`): TSEM's view and the routing features stayed the original core1's | §20 FORGET (the changed reference would also change those views; the closure tested part of the mechanism), §13, §21 | Recorded; §20 is "closed for the posterior only" |
| Latent traps, no recorded run found on them: SessionNet also dropped the co-training loss and AdaBN; `ts_core2.parse` ignored unknown flags and never used its reference-mean token; `alib.mix` ignored a misspelled family in `wscale`; several module constants are bound as default arguments, so assigning them has no effect (`gm.LO`, `gm.HI`, `gm.BEM`, `gm.NMIN`, `gm.BTS`, `R3.TAU`, `R3.T0`); exists-skips whose file names omit settings (`ttt.py`, `sbnx.py`, `rest4.py`, `pft.py`); a fit cache whose key omits co-training | none found | The first three now assert (12 Nov). The default-argument constants and the cache key are listed here as traps, not changed |
| NETAUG (§51) had no control through its own code path, and the stored SincShallow twin has one seed and no seen-session rows locally | §51, before any result | In-harness twins added (tasks 100–111, Galvani 3090355); the scorer must use them |
| Scorers average whatever seeds exist (`dose.py`, `add.py`, `pft_score.py`): SincShallow has one seed in the PFT and frozen-twin files | §31, §40, §43: arm and twin share the same seed set, so the recorded differences stand | Recorded |

**Checked and clean** (audit): the PFT and TTT "head" scopes train the intended parameters; `run_cv.resolve` fails on
an unknown suffix; the runs 1–3 / 4–6 proxies split as described; `repceil.py` has no label leakage; the monkeypatches
of `v5_replay.py`, `oracle_gate.py` and `dose.py` take effect. **Not covered:** the export and build scripts, the
identity model's internals, the untracked scratch of the 29 stage arms and the two automated searches, and October
code deleted on 11 Nov.

## 53. SCH19: do external same-task people count as people when only real channels are used? (written 12 Nov 03:20 UTC, before any result; code `studies/12nov/sch19.py`)

**Why.** §39 says each added person is worth about +2; §21 closed Scherer 2015 in the cores, but with 22 of 41
channels interpolated (§52 re-read). The clean test uses only the 19 channels both caps have, bands from 4 Hz up
(Scherer is high-passed there) and no slow block, for both corpora alike.

**Test.** core1's recipe on 19 channels, 14 bands (4–58 Hz), three covariance windows; logistic head, C 10.
R19: PROTEUS only, leave one PROTEUS person out (nine training people). S19: the same plus Scherer's nine people at
row weight 1.0 and 0.3. Context: PROTEUS → Scherer (train on all ten, test on Scherer's people) and leave-one-person-
out inside Scherer.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| R19 (19 channels, no slow content) | 63 (58–67), against 69.29 with everything | — |
| S19 minus R19, best weight | +1.0 (−1.5 to +3.5) | ≤ 0 at both weights: Scherer's people do not act as people even on real channels |
| PROTEUS → Scherer, balanced accuracy | 45 (38–52) | — |
| Leave-one-person-out inside Scherer | 45 (38–52) | — |
| P(S19 − R19 ≥ +2) | 25 % | — |

Reading: ≥ +2 reopens external same-task data (a shared-channel member with 19 people; ds007554 on its own 19
shared channels becomes worth fetching). ≤ 0 means the corpus differs in more than its cap.

| Item | Observed | Verdict |
|---|---|---|
| R19 (19 real channels, 4–58 Hz, no slow block; nine PROTEUS people) | 58.38 | **Hit**, low edge (predicted 63, range 58–67): the reduced space costs 11 points against 69.29 |
| S19 minus R19, Scherer rows at weight 1.0 / 0.3 | **−3.18 / −3.65**; 2 of 10 people up at either weight | **Refuted** (≤ 0 at both; predicted +1.0) |
| PROTEUS → Scherer | 43.0 (sessions 27.5 to 57.1) | **Hit** (predicted 45) |
| Leave-one-person-out inside Scherer | 46.2 (35.8 to 58.3) | **Hit** (predicted 45) |

**Reading (12 Nov 03:30 UTC; log `runs/sch19_12nov.log`).** With no interpolated channel in play, Scherer's nine
people still lower new-person accuracy by more than three points. The corpus is not "nine more people of our kind":
a model trained on all ten PROTEUS people decodes Scherer's sessions at 43 %, and Scherer's own people generalise to
each other at 46 % in the same features. Patients, another protocol and another amplifier make it a different
population. §21's closure now holds for the clean version too; by the same reasoning ds007554 (another cap, block
design, nominal event times) is not worth fetching as training rows.

## 54. TSEMK: is the session stage limited by its 128 components? (written 12 Nov 03:20 UTC, before any result; code `studies/12nov/tsemk.py`)

**Why.** §49: 512 PCs of core1's features lose four points against the full features on new people; §36: the
population PCs lose a person's own directions. TSEM, the stage worth +3.6 sealed, sees 128 PCs. A 256-PC hedge gave
+0.30 on 6 Nov; larger spaces were not tried.

**Test.** core1's scaled features per fold, PCs recomputed (harness check: the chain with my first 128 PCs must
reproduce the reference J within 0.05). TSEM in k ∈ {64, 128, 256, 512, 1024} PCs with λ ∈ {0.15, 0.10, 0.07} in the
v3 chain, exploration bases; everything else unchanged (routing still uses its own 256-PC view).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Best (k > 128, λ) minus the shipped (128, 0.15), dJ | +0.3 (−0.4 to +1.0) | ≤ 0 on both bases for every k > 128 |
| k = 64 | −0.5 (−1.5 to 0) | — |
| P(≥ +0.5 on both exploration bases) | 15 % | — |

| Item | Observed (B2_pool / B2_s345) | Verdict |
|---|---|---|
| Harness | my PCs correlate 1.0000 with the stored ones in every one of the 128 columns; k = 128 reproduces the chain exactly (dJ 0.00) | Valid |
| k = 64, λ 0.15 / 0.10 / 0.07 | −1.30, −1.23, −1.29 / −1.25, −1.04, −1.32 | **Hit** (predicted −0.5, range −1.5 to 0) |
| k = 256 | −0.47, −0.14, −0.60 / −0.14, −0.17, −0.45 | Negative |
| k = 512 | −0.08, −0.15, −0.13 / −0.05, **+0.41**, −0.22 | Mixed |
| k = 1024 | +0.04, +0.03, −0.32 / +0.02, −0.18, −0.21 | About zero on dJ |
| k = 1024, λ 0.15, by protocol | **new +0.61 / +0.59** (8 and 7 of 10 people ≥ 0; both halves positive on B2_pool, G1 +0.85 and G2 −0.11 on B2_s345); **seen −0.54 / −0.54** | Not pre-registered as a split |
| Best (k > 128, λ) minus the shipped setting, dJ | no setting ≥ +0.5 on both bases; the best consistent one is +0.04 / +0.02 | Inside the range (predicted +0.3, −0.4 to +1.0); not refuted by the letter of the rule (not ≤ 0 on both), no pass |

**Reading (12 Nov 01:45 UTC; log `runs/tsemk_12nov.log`; PCs cached in `runs/tsemk_pcs.npz`).**
- 128 components are the optimum for the stage as one setting: fewer is clearly worse, more is neutral on dJ.
- **But the two halves of dJ move in opposite directions at k = 1024: new people +0.6 on both bases, returning
  sessions −0.5 on both.** The package runs recognised returning people through a second tail with its own TSEM, so
  the two can differ. This split was not pre-registered; it is confirmed or dropped by §54b.

**54b. Confirmation, written 12 Nov 01:46 UTC before any fresh-base result (`TSEMK_SPLIT=1 … tsemk.py`).** TSEM with
1,024 components and λ 0.15 for new-person sessions; the shipped 128 components for the seen protocol (unchanged, so
its change is exactly 0). Four fresh bases.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| New people, mean of four fresh bases | +0.45 (+0.1 to +0.8) | mean < +0.3, or fewer than 3 of 4 bases > 0 |
| People ≥ 0, mean | 7 of 10 (6–9) | < 6 |
| Both halves (G1, G2) ≥ 0 on the mean | yes | either half < −0.2 |

Pass → a v5 variant with the larger TSEM view for the unrouted tail is specified (new +0.45 is dJ +0.22, about +0.15
sealed, with a runtime cost to be measured).

| Item (54b) | Observed (B2_s012 / B2_s0 / B2_s1 / B2_s2) | Verdict |
|---|---|---|
| New people, 1,024 components for new-person sessions | +0.08 / +0.31 / −0.49 / +0.41, **mean +0.08**; 3 of 4 bases > 0 | **Refuted** (mean < +0.3; predicted +0.45, range +0.1 to +0.8: below it) |
| People ≥ 0 | 6 / 7 / 4 / 6 of 10 | Below the prediction (7) |
| Halves, mean | G1 +0.07, G2 +0.09 | — |
| Seen | +0.00 / +0.14 / +0.26 / +0.00 (the seen sessions' own view is unchanged; the small changes come through the gallery statistics taken from the new-person outputs) | — |

**Reading (12 Nov 02:05 UTC; log `runs/tsemk_split_12nov.log`).** The +0.6 for new people on the two exploration
bases does not hold on fresh ones (+0.08). It was a pick from fifteen settings split after the fact, and the fresh
bases say so. TSEM keeps its 128 components. Closed.

## 55. SEQCAL: the cue-order prior with a learned fusion (written 12 Nov 03:20 UTC, before any result; code `studies/12nov/seqcal.py`)

**Why.** §9 closed the Graz cue-order prior at ≤ +0.4 J with one temperature and one weight; §52 re-read: a fusion
that learns the calibration was never tried. This is a ceiling with the true interface and run position.

**Test.** For Graz windows: a multinomial logistic fusion of the chain's log posterior (3) and the pattern class of
the window's run position (one-hot, 3), fitted on the other people's Graz windows (leave one person out; seen
sessions use the fusion fitted on all new-person rows of other people). BrainHero windows unchanged. v3 chain,
exploration bases.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| dJ with the true run structure | +0.3 (0 to +0.8) | < +0.5: stays closed, now with a learned fusion |
| Graz window accuracy | +1.0 (0 to +2.5) | — |

| Item | Observed (B2_pool / B2_s345) | Verdict |
|---|---|---|
| dJ with the true run structure, learned fusion | **−0.67 / −0.74** (new +0.23 / −0.17; seen −1.56 / −1.30) | **Refuted** (< +0.5; below the predicted range 0 to +0.8) |
| Graz window accuracy, new people | 76.49 → 77.31 / 76.49 → 76.49 (the prior alone: 56.2) | Hit at the low end (predicted +1.0, range 0 to +2.5) |

**Reading (12 Nov 03:40 UTC; log `runs/seqcal_12nov.log`).** A fusion that learns its own calibration does no better
than §9's single temperature: at most +0.8 on Graz windows for new people, nothing in cell balanced accuracy, and a
loss on returning sessions (a fusion fitted on new-person outputs does not suit the sharper returning-person
outputs). The cue-order prior is real (56 % alone) and still worth nothing after the session stages. Closed for a
second, independent form of fusion.

## 56. NETPC and cmix on the shipped nets (written 12 Nov 03:20 UTC, before any result; code `studies/12nov/netaug.py`, tasks 120–131 and 200–211)

- **NETPC: the people curve for a net.** SessionBN FBCNet (the FB slot) and FBLight, seed 0, trained on k = 3, 5, 7
  random other people (two draws) per held-out person; k = 9 is the in-harness twin. Prediction: +5 per doubling
  (+2.5 to +8), flatter than core1's +8 because the nets are weaker and carry augmentation. If a net is much flatter
  than the core, the nets are the people-efficient members and deserve more weight when people are few; if equally
  steep, the bottleneck is common to both.
- **cmix4:** same-class mixing (α 0.4) on the four SessionBN families, never applied to them before (§51, §52).
  Prediction: standalone new +0.5 (−1 to +2) against the twin; chain as §51.

| Item | Observed | Verdict |
|---|---|---|

## 57. Follow-ups queued to use the free lanes (written 12 Nov 03:55 UTC, before any result)

**57a. swa50 on the shipped nets (Galvani, `netaug.py` tasks 140–151).** Weight averaging over the second half of
training (parameters and SessionBN running statistics alike) on the four SessionBN families; never run on them
(§52). Prediction: standalone new +0.3 (−0.7 to +1.3) against the in-harness twin; chain as §51.

**57b. BUNDLE: the small returning-person levers on top of the own-space stage (local, `studies/12nov/bundle.py`).**
Each was "positive, not adopted" alone and inside the noise (§19 personal class templates as a TSEM prior, +0.19;
§22 slot gain 4.5, +0.12). Under §52 a bundle of same-signed small effects is worth one test: template prior κ ∈
{0, 10, 20} × slot gain ∈ {3, 4.5}, all with the stage, v4.1 replay, exploration bases, then the best on the fresh
bases.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Best bundle minus the stage alone, seen (exploration) | +0.3 (−0.3 to +0.9) | ≤ +0.2 |
| P10 with the best bundle | ≥ −0.5 | < −1.0 |
| Fresh bases: bundle minus stage alone | +0.2 (−0.4 to +0.8) | ≤ 0 |

**57c. POPSTAGE: is the stage's value personal, or would any own-space members help at the end of the chain?
(local, `studies/12nov/popstage.py`).** The same three members (T, C, S) fitted on the nine other people's sessions
instead of the person's own, applied as the same final stage (β 0.05 / 0.10 / 0.25) to new-person sessions of the v3
chain. If this helps, the stage form is a way to add population members without disturbing the session stages (the
"extra members lower the chain" closure was for members inside the mix); if not, the stage's value is the personal
labels.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Population members alone on held-out people (T·C·S, window accuracy) | 55 (48–62) | — |
| New people, best β | 0.0 (−0.5 to +0.4) | ≥ +0.5 on both exploration bases would reopen population members as a final stage |

| Item | Observed | Verdict |
|---|---|---|
| 57b BUNDLE, every (κ, slot gain) with the stage, minus the stage alone (B2_pool; B2_s345) | κ 0 / gain 4.5: −0.66; −0.39. κ 10 / 3: −0.13; −0.26. κ 10 / 4.5: −0.67; −0.39. κ 20 / 3: −0.92; −1.05. κ 20 / 4.5: −0.66; −0.65 | **Refuted** (≤ +0.2; every bundle is below the stage alone). The stage and the template prior carry overlapping information; the stage alone stays |
| 57c POPSTAGE, population T·C·S alone on held-out people | 60.9 % window accuracy | Above the range (predicted 55, 48–62 at its edge) |
| 57c POPSTAGE as a final stage for new people, β 0.05 / 0.10 / 0.25 | −0.61, −1.50, −3.19 / −0.52, −1.20, −3.31; 2–3 of 10 people ≥ 0 | **Hit** on the prediction (0.0, −0.5 to +0.4, at its lower edge) and **the reopening is refuted**: population members hurt as a final stage too |

**Reading (12 Nov 01:55 UTC; logs `runs/bundle_12nov.log`, `runs/popstage_12nov.log`).** The own-space stage works
because of the person's own labels, not because of where it sits in the chain: the same three members fitted on
other people are a decent decoder alone (61 %) and still lower the chain at any weight. And the stage does not
stack with the two small returning-person levers of 11 Nov. v5 stays "v4.1 plus the stage, nothing else".

| Item (56, 57a) | Observed | Verdict |
|---|---|---|
| NETPC, SessionBN FBCNet, k = 3 / 5 / 7 / 9 (seed 0) | 52.36 / 60.21 / 63.88 / 65.28: +8.3 per doubling over the range, but 7 → 9 only **+1.40** | Above the predicted slope (+5) on the fit; **flattening at the top** |
| NETPC, FBLight | 50.13 / 56.18 / 57.70 / 58.50: +5.3 per doubling; 7 → 9 **+0.80** | **Hit** (+5); flattening |
| cmix4 on the four families, standalone new minus twin | FBCNet −1.44; raw FBCNet −0.38; SincShallow +0.17 (2 seeds); FBLight +0.06 | Prediction +0.5: **miss**, about −0.4 on the mean |
| cmix4 in the chain, all four swapped | −0.16 / −0.10 (new −0.45 / −0.33) | **Refuted** |
| swa50, standalone new minus twin | FBCNet −0.03; raw FBCNet −0.26; SincShallow −0.50 (1 seed); FBLight −0.29 | Prediction +0.3: **miss** |
| swa50 in the chain, all four swapped | −0.58 / −0.57 | **Refuted** |

**Reading (12 Nov 03:28 UTC; `runs/netpc_score_12nov.log`, `runs/netaug_score_12nov.log`).**
- **The nets flatten where core1 does not.** From seven to nine training people the two net families gain +1.4 and
  +0.8; core1 gains +3.9. Over the whole range their slopes are similar. So at ten people the linear member is the
  one still starved of people; the nets have used up most of what nine people offer them. One seed, two families.
- The three October tricks that had never run on the shipped nets (same-class mixing, frequency scaling, weight
  averaging) have now run on them, with controls through the same code: none helps alone or in the chain. Those
  closures now hold for the shipped nets too.


## 58. OCU: an explicit eye-movement branch (lead from the user, 12 Nov; written 12 Nov 01:57 UTC by `date -u`)

**The lead.** The user pointed at the rank-2 account's publications. A search (primary sources read: two ICTC 2025
papers, the lab's roster and publication list, the competition leaderboard) found: "Sungjun Wang" and "Seohee Kim" are
both listed as researchers of ETRI's EnSeption group, so ranks 1 (0.846) and 2 (0.830) plausibly come from one lab;
nothing public describes their entries. The lab's biosignal work is (a) reading detection from EOG glasses with
derived horizontal, vertical and magnitude channels, a ResNet-18 with channel attention and Mixup (Mixup +5.6 points
leave-one-subject-out; derived channels +4.6), and (b) two-branch fusion of EEG and ECG by attention (fusion +14
points over the best single branch, leave-one-subject-out). None of it uses test-time adaptation. The link between the
accounts and these people rests on name and lab membership only.

**Why it fits our data.** The display puts each task's cue at a fixed place, so gaze carries class information, and
its sign is shared by most people (REVIEW of 30 Oct: 8–9 of 10). We use it through a 48-dimensional hand-built view in
SessionEM and a 1 Hz waveform block in core1. No member learns from the eye-movement traces themselves.

**Step 1, probe (exploratory, run before this text; `studies/12nov/ocu_probe.py`, log `runs/ocu_probe_12nov.log`).**
Leave one person out, per-session centring, shrinkage LDA:

| View | Balanced accuracy across people |
|---|---|
| A: the shipped gaze view (horizontal and vertical frontal differences, 16 bins each) | 44.6 |
| B: A + magnitude | 44.2 |
| C: B + a second horizontal pair (FC5, C5 against FC6, C6) + eye-movement velocity statistics | **51.1** (P02 66, P03 58, P09 58; P04 36) |
| D: all 41 channels, 1 Hz waveform (core1's slow block) | 47.9 |

P02 and P03 share one clear pattern (horizontal: imagery +0.6, word −0.9; vertical: subtraction +0.55, imagery −0.5);
most others show the same signs weakly; P04 and P08 do not.

**Step 2, EMVIEW (pre-registered here, before any chain result; `studies/12nov/emview.py`).** SessionEM's view
extended from the shipped 48 features by the second horizontal pair (16 bins) and log velocity statistics (16 bins +
4), and the extension alone as a replacement of the gaze part. v3 chain, exploration bases.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Extended view, dJ | +0.2 (−0.3 to +0.8) | ≤ 0 on both bases |
| New people | +0.3 (−0.3 to +0.9) | — |
| P(≥ +0.5 dJ on both bases) | 15 % | — |

**Step 3, OCUNET (pre-registered here; `studies/12nov/ocunet.py`, Galvani).** A learned eye-movement branch: derived
channels (two horizontal pairs, two vertical differences, magnitude, their velocities; Savitzky–Golay smoothed,
session-centred causally), a 1-D ResNet with channel attention, Mixup; leave-one-person-out and the seen protocol,
three seeds. Scored alone, then in the chain three ways: as a mix member, as a final stage, and as SessionEM's view
(its log posteriors or embedding).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Alone, new people | 54 (47–61), at least +3 over view C's 51.1 | ≤ 51: the net learns nothing beyond the statistics |
| Mixup against no Mixup | +2 (−1 to +5) | — |
| Best chain integration, dJ | +0.3 (−0.3 to +1.0) | ≤ 0 on both exploration bases in all three forms |
| P(≥ +0.5 dJ on both bases) | 20 % | — |

| Step 2, EMVIEW: extended SessionEM views (B2_pool / B2_s345) | shipped + pair + velocity: −1.57 / −1.33; shipped + velocity: −1.35 / −0.98; shipped + pair: −1.14 / −0.83; pair + velocity + shipped event rates: −2.67 / −2.77 | **Refuted** (≤ 0 on both bases for every view; predicted +0.2). The template stage does not take a larger view at its shipped settings |

**Step 2b, COREFUSE (written 12 Nov 02:02 UTC, before any result; `studies/12nov/corefuse.py`).** Feature-level fusion,
the form the leaders' lab publishes: the 104 eye-movement features of view C join core1's 44k features as one block
(training sessions z-scored per session, test sessions causally), weighted g ∈ {3, 10, 30} against the scaled core
features. Leave one person out, core alone, reference 69.29.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Best g minus the reference | +0.5 (−0.3 to +1.5) | ≤ +0.3 at every g |
| P(≥ +1.5, then into the chain) | 15 % | — |

| Item | Observed | Verdict |
|---|---|---|
| Step 2b, COREFUSE: eye-movement block in core1, weight 3 / 10 / 30 (reference 69.29 reproduced) | 68.84 / 66.56 / 65.76: **−0.45 / −2.73 / −3.53** | **Refuted** (≤ +0.3 at every weight; predicted +0.5) |
| Same, per person at weight 3 | P02 +4.3, P09 +3.0, P01 +1.2, P06 +1.1, P08 +1.0; P04 −5.9, P03 −3.8, P10 −2.3, P05 −2.1, P07 −1.1 | The block helps the people who look and hurts the others by as much |

**Reading of steps 2 and 2b (12 Nov 02:10 UTC).** The eye-movement features are a real cross-person signal (51 %
alone) with a split population: for about half the people they add several points, for the other half they mislead
by as much, so a fixed fusion (in the template stage or in core1's features) nets out negative. What is not tested
yet is a fusion that can tell the two kinds of people apart inside a session, which is what an attention-based
two-branch model does and what the learned branch of step 3 is for.

| Item (step 3, OCUNET) | Observed | Verdict |
|---|---|---|
| Alone, new people: single seeds / mean of three seeds | Mixup 49.6, 50.0, 50.2 / 52.48; no Mixup 51.0, 50.6, 50.3 / **53.16** (P02 63, P08 63, P09 64, P03 54; P05 39) | Below the prediction (54; "+3 over view C's 51.1" not met: +2.1); not refuted (> 51) |
| Mixup against no Mixup | −0.7 | **Miss** (predicted +2): Mixup does not help here |
| Chain, as a mix member, weight 0.5 / 1 / 2 (B2_pool; B2_s345) | −0.01, −0.79, −1.62; −0.21, −0.61, −1.26 | Negative |
| Chain, as a final stage, β 0.05 / 0.1 / 0.25 | −0.78, −2.10, −6.89; −0.69, −2.30, −6.46 | Negative |
| Chain, SessionEM view = shipped 48 + the net's log posteriors (51) | **+0.21 / +0.33** (new +0.02 / +0.02; seen +0.41 / +0.65) | Positive on both bases, below +0.5; not refuted, no pass |
| Chain, SessionEM view = event rates + net log posteriors (19) | −0.62 / −0.58 | Negative |

**Reading of step 3 (12 Nov 03:28 UTC; Galvani pack 3090384; `runs/ocu_chain_12nov.log`).** A learned branch on the
eye-movement traces reaches 53 % across people, two points above the hand-built statistics, and Mixup does not help
it. In the chain it behaves like every other population member: negative in the mix and as a final stage. The one
form above zero is as three extra features of SessionEM's view (+0.2 / +0.3, all of it on returning sessions), which
is inside the noise; a fresh-base check is run before it is kept or dropped (§58b).

**58b. Fresh-base check of the one form above zero (written 12 Nov 03:30 UTC by `date -u`, before any fresh-base
result; `ocu_chain.py nomix fresh`).** SessionEM's view = the shipped 48 features + the eye-movement net's three log
posteriors; four fresh bases. Kept as a candidate only if the mean dJ is ≥ +0.3 and the mean seen change ≥ +0.4;
otherwise dropped. Prediction: dJ +0.15 (−0.2 to +0.5); P(kept) 25 %.

| Item | Observed | Verdict |
|---|---|---|
| Fresh bases (B2_s012 / B2_s0 / B2_s1 / B2_s2), dJ | +0.08 / +0.56 / +0.51 / +0.01, **mean +0.29** | Inside the range (predicted +0.15); **one hundredth below the +0.3 mark: dropped by the pre-set rule** |
| Seen / new, mean | +0.50 (P04 +1.06, P10 −0.07) / +0.08 | The seen mark (≥ +0.4) is met |
| All six bases | +0.21, +0.33, +0.08, +0.56, +0.51, +0.01: mean +0.28, none negative | — |

**Reading (12 Nov 03:33 UTC; log `runs/ocu_fresh_12nov.log`).** Three log posteriors of a learned eye-movement
branch, added to SessionEM's view, give a small gain that is non-negative on all six bases (+0.28 on the mean, P04's
again). By the rule it is dropped at +0.29 against +0.30; under §52 it is recorded as "positive, small, not built":
it would need a trained net inside the package for about +0.1 sealed. **The eye-movement lead does not explain the
leaders' margin on our system**: the signal is real (53 % alone across people), and every way of adding it at the
population level is worth between −7 and +0.3.

## 59. What the night of 11–12 Nov taught, and what follows (12 Nov 03:35 UTC by `date -u`)

**Scored tonight:** 26 pre-registered sections (§35–§58), about 60 arms. One lever passed: the own-space stage for
returning people (§41–§45; package cand_v51_own). Everything aimed at new people is at or below zero.

**Updated understanding.**
1. **New people are limited by how many real people we train on, and nothing substitutes for them.** +8 points per
   doubling at core level, +7.3 after the stages in a weak test (§39, §48). Seven kinds of generated variation, seven
   people-efficient estimators, noise, and the one external corpus with our tasks (even on real shared channels) all
   fail to act as people (§46b, §47, §49, §53). The nets have nearly used up what nine people give them (7 → 9:
   +0.8 to +1.4); the linear member has not (+3.9) (§56).
2. **The chain accepts only information it does not already have, and tonight the only such information was a
   person's own labels.** Population-level additions lose in every position: in the mix, as a final stage, as a view of
   the template stage (§57c, §58). Session-level learning on the chain's labels is closed in three feature families
   (§35, §38, TTT). The same three models are worth +1.8 when fitted on the person and −0.5 to −3.3 when fitted on
   others.
3. **Our dev metric resolves about ±0.3, and positives of that size do not survive.** +0.6 on two exploration bases
   became +0.08 on four fresh ones (§54b); small levers do not stack (§57b). Only effects near +1 on six bases, or a
   sealed measurement, should move a decision.
4. **Closures were worth re-reading, and the re-read is now done for the nets.** Three October tricks had never run
   on the shipped nets; they have now, with controls, and stay closed (§51, §56, §57a, §52).
5. **The leaders' margin is still unexplained.** Ranks 1 and 2 plausibly share one lab (name and roster only). Its
   published methods (eye-movement branches, two-branch fusion, Mixup) are worth −7 to +0.3 on our system (§58).
   Two accounts from one group would also have twice the sealed uploads to select from; whether that is within the
   rules is for the organisers, not for us to assume.
6. **What dev cannot tell us any more:** how the eight single-session returning people respond. The stage is +4 for
   P04 and 0 for P10; the ten-person proxy says most people are nearer P04. That gap is the whole range of the sealed
   estimate (+0.4 to +0.9), and only a sealed upload measures it.

**What follows.**
- **v5.1 is the next sealed measurement as well as the next candidate.** Its difference to v4.1 on the sealed set
  says whether the returning branch deserves the remaining effort (personal nets, more own-space) or not.
- **The one remaining new-person lever with a measured dev gain is in-session fine-tuning of the nets** (new +0.6).
  It needs a package form: the shipped nets are TorchScript traces whose normalisation is causal per window, while the
  dev recipe fine-tuned Python nets with batch statistics over the session so far. So the first step is a dev test
  of the package form (§60), not the build.
- **Lanes:** the montage-reordering check of v5.1 with the stage active (Galvani 3090406), then §60.

## 60. TTT3: does in-session fine-tuning survive the form a package can run? (written 12 Nov 03:40 UTC by `date -u`, before any result; code `studies/12nov/ttt3.py`, Galvani)

**Why.** §59: the one new-person lever with a measured dev gain (new +0.6 with four families, §31). The dev recipe
fine-tunes Python nets in training mode, with batch statistics over all windows so far and dropout on. The shipped
nets are TorchScript traces whose normalisation is the causal per-window formula and which expose only convolution
and output weights as parameters. Before any package work the lever has to hold in that form.

**Test.** The TTT2 run again on the same recipe, with four more variants on the very same trained nets (5 families ×
3 seeds, both protocols, chain labels unless stated): `pl20` the dev form (reference); `ple20` package form, all
parameters, lr 5e-4; `plelr20` the same at 1e-3; `plec20` package form, only the parameters a trace exposes, 1e-3;
`ore20` package form with true labels (ceiling).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| `pl20` here against §31 | reproduces: standalone new +4 to +6 per family | not reproduced: the harness differs |
| Package form, standalone new, mean of families (best of `ple20`, `plelr20`) | +4 (+2 to +6); within 1.5 of the dev form | < +2 |
| `plec20` against the best all-parameter variant | −0.3 (−1.5 to +0.5) | < −2: the affine parameters are needed, so traces must be re-exported |
| Chain, four families swapped, package form, new | +0.4 (0 to +0.8) | ≤ +0.2 on both exploration bases: not worth a build |
| P(chain new ≥ +0.4 on both bases) | 45 % | — |

Pass → the package work starts: trainable copies of one net per family, 20 steps every 20 windows, and a timing
replay on A100 (estimate from §31: about 28 extra minutes on the sealed set).

| Item | Observed | Verdict |
|---|---|---|
