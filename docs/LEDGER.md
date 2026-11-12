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
