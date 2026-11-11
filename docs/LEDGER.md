# Ledger: what is closed, and where new runs are pre-registered

**How to use it.**
- Before repeating an idea, look it up in the index below. A "closed" verdict covers only the recipe that was tested;
  the cited section says which.
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
