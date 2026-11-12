# Ledger: what is closed, and where new runs are pre-registered

**How to use it.**
- Before repeating an idea, look it up in the index below. A "closed" verdict covers only the recipe that was tested;
  the cited section says which.
- **No gate or closure is final (user rule, 12 Nov; §52).** A gate is a threshold on the evidence of its day. Before
  a closure is used as a reason not to run something, check what actually ran: the code path, the model class, alone
  or in the chain, and on which version of the system. Pre-registration stays: state the rule first, then say plainly
  when and why you go past it.
- New runs are pre-registered at the bottom of this file **before** they are scored (time stamps from `date -u`): mechanism, numeric prediction
  with an 80 % range, and the result that would refute it. Fill in the outcome and the verdict (hit, miss, refuted)
  afterwards, including for misses.
- Sections §1–§33 are in [PREDICTIONS_11NOV.md](PREDICTIONS_11NOV.md) (the record of 11 Nov); §34–§61 are in
  [PREDICTIONS_12NOV.md](PREDICTIONS_12NOV.md) (the night of 11–12 Nov). A section moves there once it is scored. Earlier
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

## Index of 12 Nov (sections in [PREDICTIONS_12NOV.md](PREDICTIONS_12NOV.md))

### Built and ready (not uploaded)

| What | Result | Section |
|---|---|---|
| **cand_v51_own**: v4.1 + own-space personal stage for recognised returning people (three models fitted on the person's public sessions in their own space, multiplied into the routed output at β 0.10) | dev **+0.89 J** on six bases (seen +1.79: P04 +3.5, P10 +0.1; new 0.00). Ten-person within-session proxy +4.2 to +5.1 over a person-naive chain. Package checked, replayed, reviewed, timed (1.08×). Sealed estimate 0.820–0.825 | §36, §37, §41–§45; `SEALED_UPLOADS.md` |

### Positive, small, not built

| What | Result | Section |
|---|---|---|
| Personal nets (PFT) on top of the stage | seen +2.3 with the stage against +1.9 (six bases); P10 −1.0 | §40, §43 |
| A learned eye-movement branch's log posteriors as three extra features of SessionEM's view | +0.28 on six bases, none negative; needs a net in the package | §58, §58b |

### Open (pre-registered, running or waiting)

| What | Section |
|---|---|
| In-session fine-tuning of the nets in the form a package can run | §60 (below) |

### Closed (in the form tested; §52 lists what each form was)

| Direction | Result | Section |
|---|---|---|
| **In-session learning on the chain's labels** | | |
| Self-labelled logistic head in the population PCs | new −0.6 to −0.8, seen −1.3 to −1.7 (true labels: +2 / +0.5) | §35 |
| Session-own members (FBCSP, own tangent space, slow) on the chain's seeds | new −0.1 to −0.6 (true labels: +2.5 to +2.9) | §38 |
| **People: generating or re-weighting them** | | |
| Virtual people for core1: rotation, cap shift, gain jitter, sensor noise, frequency scaling, person mixing, between-person combination | every strength ≤ 0 (best −0.01) | §46b, §47 |
| People-efficient estimators and noise for core1 (balanced, random-effects LDA, consistency selection, bagging, group DRO, isotropic and person-shaped noise) | none above its reference | §49 |
| Scherer 2015 as extra people on the 19 real shared channels | −3.2 to −3.7 | §53 |
| Virtual people for the SessionBN nets (chunk rotation, frequency scaling), same-class mixing, weight averaging | chain −0.1 to −0.9 | §51, §56, §57a |
| **Session stage** | | |
| TSEM in 64 / 256 / 512 / 1,024 components | 128 is the optimum; the 1,024-for-new-people split did not confirm (+0.08) | §54, §54b |
| Cue-order prior with a learned fusion (true run structure) | −0.7 | §55 |
| Bagging complete chains or stage settings | −0.5 to +0.1; the window-bag split did not confirm (+0.03) | §61, §61b |
| Extended eye-movement view for SessionEM | −0.8 to −2.8 | §58 |
| **Returning people** | | |
| Own-space members averaged into the personal head's slot; tempered in-slot product | −0.7 to −1.3; unstable (0 to +1.5) | §37, §40 |
| Per-person choice of the personal space by public-session CV | never beats one recipe for all | §41 |
| Nine variants of the own-space members | plateau, within 0.5 | §44 |
| Up-weighted personal core | chain 0.0 | §50 |
| Template prior and slot gain bundled with the stage | −0.1 to −1.1 against the stage alone | §57b |
| Perfect recognition from window 0 | −0.24 (worse than the real gate) | §46a |
| **Population members at other positions** | | |
| Own-space members fitted on other people, as a final stage | −0.5 to −3.3 | §57c |
| Eye-movement features in core1; learned eye-movement branch as member or final stage | −0.5 to −3.5; −0.0 to −6.9 | §58 |

### Diagnostics worth knowing

| Finding | Section |
|---|---|
| With its own labels a session holds 81–82 % in session-own representations, 74.9 % in the population PCs; P04 reaches 74–80 % (not a low-signal person) | §36 |
| New-person accuracy rises +8 points per doubling of training people (core1), +7.3 after the stages on core1 alone; the nets flatten from 7 to 9 people | §39, §48, §56 |
| Reducing core1's 44k features to 512 PCs costs four points | §49 |
| Own-space personal members hold about twice the personal information of the population-PC head across real sessions | §42 |
| The stage's value is the personal labels, not its position in the chain | §57c |
| Eye-movement features decode 51–53 % across people, with a split population (helps half, misleads half) | §58 |
| Exploration-base gains of +0.2 to +0.6 were zero on fresh bases three times | §54b, §58b, §61b, §59 |
| Ranks 1 and 2 plausibly share one lab (eye-movement and fusion work); their margin is unexplained | §58, §59 |
| Soft gates; closures re-read against what actually ran; audit of silent no-ops (two more affected closures) | §52 |
| What the night taught, in six points | §59 |

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

## 60. TTT3: does in-session fine-tuning survive the form a package can run? (written 12 Nov 03:34 UTC by `date -u`, before any result; code `studies/12nov/ttt3.py`, Galvani)

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

## 62. SEGFP: did the 10 Nov rescoring change how streams are segmented? (written 12 Nov 03:52 UTC by `date -u`, before any result; code `studies/12nov/segfp.py`)

**Why.** The independent review of 12 Nov (finding 7): the upload log reads the rescoring as a uniform drop, but our
three packages moved by −0.72 (no TSEM), −1.17 (with TSEM) and −1.05 (with routing), so the TSEM step lost 0.45. The
order hypotheses were tested and rejected on 11 Nov (§2); resets and segmentation never were. If the sealed streams
are now cut differently from what the contract says (per interface, per run), every session stage is tuned for the
wrong stream length and a re-tune could be worth more than anything else left. If instead sessions were merged
before the fix and are correct now, nothing is to be gained, but the drop is explained.

**Test (stage state only; the members' own causal state is left as it is, so finer cuts are lower bounds).** The
plain chain (mix → rebalancing → SessionEM → TSEM) on the two exploration bases with the stage streams cut (i) per
session (reference), (ii) per session × interface, (iii) per run, (iv) merged per person (P04 and P10 have several
sessions). Read at the SessionEM level (as cand_nxif) and at the TSEM level.

| Hypothesis | Predicted change at the SessionEM / TSEM level | Matches the sealed pattern (−0.72 / −1.17; TSEM step −0.45) if |
|---|---|---|
| Now per interface (ii − i) | −1.0 (−0.3 to −2.0) / −2.0 (−1.0 to −4.0) | both within 0.4 and the step within 0.3 |
| Now per run (iii − i) | −3 / −6 | same |
| Before merged per person, now per session (i − iv), on the people with several sessions | −0.4 / −0.7 (0 to −1.5) | same sign and ordering, size within a factor of two (only two people can show it) |

My prior: per interface matches 10 %; "merged before" is consistent 35 %; neither 55 %.

| Item | Observed | Verdict |
|---|---|---|
| Per session × interface (B2_pool / B2_s345): SessionEM level, TSEM level, TSEM step | −0.45 / −0.68; −1.13 / −1.41; −0.69 / −0.73. Sealed after the fix: −0.72; −1.17; −0.45 | **Matches by the pre-set rule on both bases** (each level within 0.4, the step within 0.3). Predicted −1.0 / −2.0: hit / hit |
| Per run | −1.77 / −1.92; −2.97 / −3.11 | Too large by a factor of 2.5: rejected |
| Merged per person | merging LOWERS the two people (−0.4 to −0.8 dJ), so "merged before, per session now" predicts a rise, not a drop | Rejected (wrong sign) |

**Reading (12 Nov 04:00 UTC; log `runs/segfp_12nov.log`).** A stream cut per session × interface reproduces the sealed
drop of both packages. Supporting facts: after the fix the organisers' FAQ no longer says windows are strictly
chronological within a session; the score's cells are session × interface; the public code's stream key is
`[subject, session]` with "one per recording unless the dataset groups them". Limits: only the stage state was cut
(the members' own causal state was not, so the true drop under this hypothesis would be larger); two non-independent
bases; three numbers. Not yet separated from a second explanation with the same flavour: one stream per session
whose runs come grouped by interface or by BIDS task instead of in acquisition order.

**62b (written 12 Nov 04:00 UTC, before any result; `studies/12nov/segfp2.py`).** Discriminating variants at the
stage level: (1) reset per session × interface; (2) reset per session × BIDS task (four streams); (3) one stream per
session, runs grouped Graz then BrainHero; (4) BrainHero then Graz; (5) grouped by BIDS task in alphabetical order.
Fingerprints: SessionEM level −0.72, TSEM level −1.17, step −0.45. Then, for the variants that match, the routed
packages: route3geb −1.05 after the fix and v4.1 − route3geb = +0.267 (both sealed), which need the gates re-run on
the cut or re-ordered streams. My prior: (1) 35 %, (2) 15 %, (3)–(5) together 30 %, none 20 %.

| Item | Observed | Verdict |
|---|---|---|
| (1) reset per session × interface (B2_pool; B2_s345): SessionEM, TSEM, step | −0.45, −1.13, −0.69; −0.68, −1.41, −0.73 | **Match** on both bases |
| (2) reset per session × task | −1.00, −2.16, −1.16; −1.34, −2.43, −1.08 | Too large: rejected |
| (3) one stream, Graz runs then BrainHero | −0.94, −1.48, −0.54; −1.00, −1.50, −0.51 | **Match** on both bases |
| (4) one stream, BrainHero then Graz | −0.61, −1.48, −0.87; −0.80, −1.39, −0.59 | Match on one base, step 0.12 outside on the other |
| (5) one stream, tasks in alphabetical order | −0.31, −0.09, +0.22; −0.31, −0.51, −0.20 | Too small: rejected |
| Sealed, after the fix | −0.72, −1.17, −0.45 | — |

**Reading (12 Nov 04:02 UTC; log `runs/segfp2_12nov.log`).** Every variant in which a session's windows arrive
grouped by interface reproduces the sealed drop, with or without a reset between the two groups; cutting or ordering
by task does not. The stage-level numbers cannot tell (1) from (3)/(4). With five variants and a ±0.4 tolerance a
chance match is possible, so the routed packages are the next fingerprints (62c).

**62c (written 12 Nov 04:02 UTC, before any result; `studies/12nov/segfp3.py`).** The full routed systems (v3 as
route3geb, v4.1) replayed with the gates run on the cut or re-ordered streams, for variants (1), (3), (4).
Fingerprints: route3geb after the fix −1.05 against its own pre-fix score; v4.1 − route3geb = +0.267 (sealed, both
after the fix; +0.58 on dev per session). A variant is supported if its v3 change is within 0.4 of −1.05 and its
v4.1 − v3 step within 0.25 of +0.27. Harness check: the identity variant must reproduce the stored chain exactly.

| Item | Observed | Verdict |
|---|---|---|
| Harness: identity variant | v3 80.0131 and 79.8474, v4.1 80.53 and 80.50: the stored values | Valid |
| (1) reset per session × interface: v3 change; v4.1 − v3 (B2_pool; B2_s345) | −2.93; +0.77 / −3.10; +0.79 | **Not supported** (sealed −1.05; +0.27) |
| (3) one stream, Graz then BrainHero | −3.50; +1.67 / −3.39; +1.39 | **Not supported** |
| (4) one stream, BrainHero then Graz | −2.87; +0.51 / −2.73; +0.65 | **Not supported** |

**Reading (12 Nov 04:05 UTC; log `runs/segfp3_12nov.log`).** The routed packages reject all three interface-grouped
variants: route3geb would have lost about three points, and it lost 1.05. The stage-level match of 62 / 62b was a
coincidence of two numbers, as cautioned. **Interface grouping is not what the fix changed.**

**New fact from the documents (primary sources, read 12 Nov; commit 8734b2f6 of 10 Nov 20:34 UTC).** The stream
definition is unchanged ("One stream is one participant × session"; code: `stream_by: [subject, session]`). What the
fix commit removed is every ordering guarantee: "There is no shuffling within a stream: runs are ordered by run
number and windows are strictly chronological within each run" became "Windows arrive one at a time (B = 1) within
each session stream, and the same evaluation order is used for all submissions"; "causal" now means "the current and
previously delivered windows"; "in recording order" and "chronologically" were deleted from three other pages.
So since 10 Nov the sealed windows of a session may arrive in a fixed order that is not chronological, and our
chain's use of order (run-level class balance in rebalancing, the Graz cue-order regularity, early-window logic)
rests on an assumption the organisers no longer make.

**62d (written 12 Nov 04:05 UTC, before any result; `SEGFP_ORDERS=1 … segfp3.py`).** Four within-session orders, with
stages and gates re-run on the re-ordered stream (members' own causal state unchanged): O1 a fixed random shuffle of
the whole session; O2 runs in order, windows shuffled inside each run; O3 runs in a random order, chronological
inside; O6 sorted by onset within the run across runs (first trial of every run, then the second, …). Fingerprints
(sealed, after the fix): SessionEM-level package −0.72; TSEM-level −1.17; route3geb −1.05; v4.1 − route3geb +0.27.
A variant is supported if all four are within 0.4 / 0.4 / 0.4 / 0.25. The 11 Nov test of O1–O3 (§2, a different
harness with the members recomputed) rejected all three; O6 is new. My prior that any of the four is supported: 15 %.

| Item | Observed | Verdict |
|---|---|---|
| O1 whole-session shuffle (B2_pool; B2_s345): plain SessionEM, plain TSEM, v3, v4.1 − v3 | −1.84, −2.61, −3.99, +1.94; −1.81, −2.66, −3.79, +1.93 | Not supported |
| O2 shuffle inside each run | +0.09, +0.02, −1.13, +0.80; +0.32, −0.27, −1.06, +0.40 | v3 fits (−1.05); the two plain levels do not: not supported |
| O3 runs in random order | −1.29, −2.57, −4.01, +1.30; −0.91, −2.38, −4.09, +1.82 | Not supported |
| O6 sorted by onset across runs | −2.08, −3.04, −4.76, +1.90; −2.14, −3.29, −4.82, +2.24 | Not supported |
| Sealed, after the fix | −0.72, −1.17, −1.05, +0.27 | — |

**Reading of §62 as a whole (12 Nov 04:08 UTC; logs `runs/segfp*_12nov.log`).**
- **No cut or order tested reproduces all four sealed numbers.** Ten variants across four families (per interface,
  per task, per run, merged per person; grouped by interface; shuffled in the session, in the run, runs permuted,
  sorted by onset). Any strong re-ordering would have cost our routed package three to five points; it lost 1.05.
  So the sealed windows are still delivered in, or close to, acquisition order per session, and the drop of 10 Nov
  has another cause (data or scoring), common to all teams.
- **The risk is real even though it has not materialised:** since 10 Nov the organisers guarantee no order, and our
  chain loses 3–5 points under a re-ordering. Nothing in the rules stops the final evaluation from using another
  order. This goes into STATE §2 as a hard bound.
- The stage-level "match" of §62 is a lesson in fingerprinting with two numbers: it met a pre-set tolerance on both
  bases and was wrong. The bases are not independent (§63), which made "both bases" worth less than it sounded.

## 63. Independent review of the approach: what is accepted and what changes (12 Nov 04:08 UTC)

A reviewer with the record and the code, and without my reasoning, ran five replay diagnostics of its own
(scratchpad `d1.py`–`d5.py`). Findings I checked or accept, and what they change:

1. **The dev "bases" are not independent.** B2_pool is the mean of B2_s012 and B2_s345, and B2_s012 the mean of the
   three single seeds; only the five SessionBN slots differ (5 of 15 of the mix weight). "Six of six bases" and "4 of 4
   fresh" mean subsets of the same nets on the same people. The ±0.3 yardstick measures seed noise; the uncertainty
   that matters is across people: per-person changes have a standard deviation of 1.5–3.5, so the standard error of a
   "new" change is 0.5–1.1, and "seen" is two people. **From now on results are reported as new (with a person
   bootstrap interval), P04 and P10 separately, not as "n of 6 bases".**
2. **My sealed estimate for cand_v51_own was too high.** P04 is 30 % of dev J and 3 of about 106 sealed sessions.
   Sealed gain ≈ 0.10 (P04's three sessions) + 0.33 × the gain of the eight single-session people. If they are like
   P10 the total is about +0.1; if like P04 about +0.7 to +1.2. v4.1's +0.27 points to the P10 side; the ten-person
   proxies point to the P04 side, but the proxy got both signs wrong where it can be checked against reality (the
   personal head: proxy P04 −1.7 and P10 +8.3, real +4.7 and 0.0). **Revised estimate: +0.3 (0.05 to 0.8), a sealed
   score of about 0.819 (0.817 to 0.824).** An upload is still worth it: the board keeps the best score, losses are
   bounded (the stage moves log-odds by at most about 0.7 nats), and it is the only measurement of that block.
   **Reading rule, fixed now:** v5.1 − v4.1 ≥ +0.4 → the remaining uploads go to the returning branch (personal nets,
   more own-space); ≤ +0.15 → the returning branch stops; in between → no further effort there, keep the package.
3. **"Only personal labels add" holds in a narrower form.** A synthetic calibrated member with *independent* errors
   passes through the chain at about 0.55 of its mix-level gain (70 % accuracy at weight 3: new +3.1). Real members
   trained on the same nine people score about a point worse than independent noise of equal accuracy: their errors
   are person-systematic and the stages amplify session-consistent error. So the rule is "stop adding members trained
   on PROTEUS's people; screen any candidate against the synthetic curve", and the open door is members from another
   population (the teacher is 50 % alone and carries weight 5).
4. **"New people are limited by the number of training people" is a core1-alone result.** core1 softened to the
   mix's confidence reaches 72.3 after the stages against the mix's 80.2; standalone core accuracy barely predicts
   chain value. The core-level gates of §46b, §47, §49, §53 and COREFUSE filtered on the wrong quantity, and §48
   measured the slope where the stages add 1–3 points, not 10. Status: unproven for the system, and not actionable
   (no more people exist). Nothing more is spent on it.
5. **The position effects are P04's.** Without P04, new-person accuracy by window is 72.7 (0–15), 80.6, 82.4, 82.3,
   86.0, 84.3 (150+): no plateau and no late drop. The first 15 windows, where no stage is active, cost about 0.9
   points; for new people only population information exists there.
6. **Contradictions in the record, corrected:** the "uniform drop" reading of the rescoring (our packages moved by
   −0.72, −1.17 and −1.05; tested in §62, unexplained); §47 said ds007554 was not fetched, but its EEG files were
   downloaded to Galvani on 2 Nov and tested as a teacher corpus (+1.67 alone, −0.98 in the chain; OVERNIGHT_2NOV);
   11 Nov §14's pre-set rule said the returning-person line stops at +0.267, and v5.1 was built anyway (under §52 that
   is allowed, but it was not said at the time); 11 Nov §32 calls the within-session proxy an upper bound, §42 found
   cross-session gains twice as large.

**The review's experiments, in its order:** (1) upload v5.1 as the measurement (the user's decision); (2) the
segmentation fingerprint (done, §62: nothing matches); (3) a ten-person proxy with the person's first runs inside the
population (not built: the nets need retraining per fold); (4) arithmetic-versus-rest and word-versus-rest heads on
external trunks as separate members instead of merged into the teacher (to check against the October archive
first); (5) warm-starting the personal head at recognition (≤ +0.2); (6) TTT only if the returning branch is dead.

## 64. WARM: warm-starting the personal head at recognition (written 12 Nov 04:24 UTC by `date -u`, before any result; code `studies/12nov/warm.py`)

**Why.** Review (§63, point 5) and §46a: with the true identity from window 0 the routed chain is *worse* than with
the real gate, because the personal head standardises its input with the session's own running statistics, which are
empty at the start. The public session offers a prior for those statistics.

**Test.** The head's input (256 PCs) is standardised with (k · public-session moments + running session moments) /
(k + n), k ∈ {5, 20}, instead of the cold running statistics. v4.1 chain with the real gate and with the oracle gate
from window 0, with and without the own-space stage; exploration bases. Reported per person.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Real gate, best k, seen | +0.1 (−0.3 to +0.4) | — |
| Oracle gate from window 0, best k, against the oracle gate cold (−0.24 against the real gate, §46a) | +0.4 (0 to +0.9): the cold-start defect closes | ≤ 0 |
| P(real gate ≥ +0.3 on both bases with P10 ≥ 0) | 15 % | — |

| Item | Observed | Verdict |
|---|---|---|
