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
- Sections §1–§33 are in [PREDICTIONS_11NOV.md](PREDICTIONS_11NOV.md) (the record of 11 Nov); §34–§65 are in
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
| In-session fine-tuning of the nets, package form (the form works; the trace-exposed parameters suffice) | chain new +0.2 with a person-bootstrap interval of −0.5 to +0.9; not built | §60 |

### Open (pre-registered, running or waiting)

None on 12 Nov 06:05 UTC.

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
| Warm-starting the personal head with public-session statistics | repairs the oracle's cold start, adds nothing over the real gate (−0.1 to −0.7) | §64 |
| **Population members at other positions** | | |
| Own-space members fitted on other people, as a final stage | −0.5 to −3.3 | §57c |
| Eye-movement features in core1; learned eye-movement branch as member or final stage | −0.5 to −3.5; −0.0 to −6.9 | §58 |

### Diagnostics worth knowing

| Finding | Section |
|---|---|
| **Independent review of the approach:** the six bases are not independent; report per person with a bootstrap interval; the v5.1 sealed estimate is +0.3 (0.05 to 0.8); upload reading rule | §63 |
| The 10 Nov rescoring: ten stream cuts and orders fingerprinted, none fits; the organisers removed the chronological-order guarantee | §62 |
| A false accept costs a new-person session 1.6 points with v4.1 and 2.2 with the stage; under 0.01 on the sealed score at the gate's rate | §65 |
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

## 66. WEIGHTS: is the chain tuned for the wrong population? (written 12 Nov 09:13 UTC by `date -u`, before any result; code `replay/grid.py`, `studies/12nov/weights.py`)

**Why.** The user (12 Nov, morning): we are now 10th; what is the limit we put on ourselves? One candidate from the
review (§63): dev J gives P04 about 30 % of the weight (half of "seen", a tenth of "new"), while P04 is 3 of about
106 sealed sessions. Every stage constant and member weight was chosen on J. If P04 pulled those choices, the chain
is tuned for a participant who barely counts.

**Test.** The retune grid of 6 Nov (rebalancing α × SessionEM λ × core-slot weight × teacher weight × TSEM λ, 243
settings; plain chain, B2_pool) scored per person, then read under two objectives: J (0.5 new + 0.5 mean of P04 and
P10) and a sealed-like S = 0.566 × new (ten people equal) + 0.434 × (0.87 × P10 + 0.13 × P04). Nested check: the
setting chosen under S on people half G1 (plus P10 as seen) is scored on half G2, and the reverse.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| The S-optimal setting differs from the shipped one | yes (60 %) | — |
| In-sample gain of the S-optimal setting over the shipped one, measured in S | +0.3 (+0.1 to +0.7) | — |
| Nested gain in S (mean of the two halves' new people, plus P10) | +0.1 (−0.2 to +0.4) | ≤ 0: the shipped constants are not P04's |
| P(nested ≥ +0.3) | 15 % | — |

| Item | Observed | Verdict |
|---|---|---|
| The S-optimal setting against the shipped one | the shipped setting is rank 6 of 243 under S (98th percentile; rank 23 under J) | It differs, by little |
| In-sample gain of the S-optimal setting, in S | +0.21 (83.07 against 82.86) | **Hit** (predicted +0.3) |
| Nested gain on the other half's new people | −0.88 and −2.00, **mean −1.44** | **Refuted** (≤ 0): what the grid offers is selection noise |
| Best setting for each new person alone (oracle) | +1.1 to +3.2 per person, mean +2.4 | Per-person settings would pay; nothing label-free chooses them |

**Reading (12 Nov 09:16 UTC; `runs/grid_12nov.npz`).** The chain is not tuned for P04 at the expense of everyone else:
under a sealed-like weighting the shipped constants are already at the 98th percentile of the grid, and choosing by
that weighting on one half of the people loses 1.4 on the other half. "We tuned for the wrong population" is not the
bottleneck. The same grid shows each person would gain 2–3 points from a setting of their own, which is the recurring
picture: the head-room is per person, and we have no label-free way to pick per person.

## 67. PTYPE: who transfers to whom? (written 12 Nov 09:17 UTC, before any result; code `studies/12nov/ptype.py`)

**Why.** The user's question, 12 Nov: what is the bottleneck? §63 says the members' errors are person-systematic; §36
says P04's information lies outside the population space. If people fall into a few types, a new person could be
served by the training people of their type; if every person is idiosyncratic, the population model is a compromise
nobody matches and only more people would help.

**Test.** Own-space members (T, C, S of §37) fitted on one person (all their sessions) and streamed causally over
every session of every other person: a 10 × 10 matrix of balanced accuracy (within-person entries only for P04 and
P10, across their sessions). Read: the best single source per target against the pooled nine-person members (60.9 %,
§57c); whether transfer is symmetric and clustered; whether a label-free statistic available inside the session
(agreement of the source's predictions with the chain's) identifies the best source.

| Check | Prediction (80 % range) |
|---|---|
| Mean single-source accuracy (off-diagonal) | 45 (41–50) |
| Best single source per target, mean over targets | 55 (50–60), below the pooled 60.9 for at least 7 of 10 targets |
| Spearman between a source's agreement with the chain in the session and its true accuracy there | +0.6 (+0.3 to +0.85) |
| Picking the source by agreement, against the pooled members | −3 (−8 to +2) |

No pass rule: this is a diagnostic. It turns into an experiment only if the best single source beats the pooled
members for most targets and agreement finds it.

| Item | Observed | Verdict |
|---|---|---|
| Mean single-source accuracy | 46.1 | **Hit** (45) |
| Best single source per target | 62 67 52 46 47 53 67 69 56 50, mean 56.8; below the pooled members' mean of 60.9 | **Hit** (55) |
| Spearman(agreement with the chain, true accuracy), mean over targets | +0.83 | Hit, at the top of the range |
| Source picked by agreement | mean 55.2, 5.7 below the pooled members | Hit (−3, −8 to +2): selection does not beat pooling (as 11 Nov §16) |
| Symmetry of transfer (ACC[i, j] against ACC[j, i]) | r = +0.80 | Not predicted |
| Within person, across sessions (diagonal) | P04 76, P10 81; nobody else's model gives them more than 46 and 50 | Not predicted |
| Chain accuracy as a new person (B2_pool) | 01: 79.0, 02: 90.5, 03: 75.0, **04: 53.6**, 05: 80.5, 06: 82.9, 07: 89.4, 08: 85.6, 09: 83.6, 10: 81.6 | Spearman with the best incoming transfer +0.78 |

**Reading (12 Nov 09:26 UTC; `runs/ptype_12nov.log`).** People are not interchangeable and the differences are
structured: transfer is symmetric, P01, P02, P07 and P08 serve each other at 56–69 %, and how well anyone transfers to
a person predicts how well the whole chain does on them as a new person (+0.78). P04 is the extreme: 76 % from their
own earlier sessions, at most 46 % from anybody else, 53.6 % in the chain against 83.1 % for the other nine. Selecting
similar people still loses to pooling everyone, so this is a map, not a method. What it says about the bottleneck is
in [REVIEW_12NOV.md](REVIEW_12NOV.md).

**Added 09:35 UTC (exploratory, B2_pool, LOSO).** No member reads P04: the eleven members score 37–51 % on P04
(core1 50.9, Hilbert core 51.0, the nets 37–48, the teacher 38.2) against 52–71 % on the other nine; the mix is 48.6
against 73.2, and the stages add 5 points for P04 and 10 for the others (chain 53.6 against 83.1). No re-weighting of
families can rescue a P04-like person; the information is absent from every model trained on other people.

## 68–71. System-level wave (written 12 Nov 09:35 UTC, before any result)

**Why.** The review of 12 Nov ([REVIEW_12NOV.md](REVIEW_12NOV.md)) names four self-made limits: one chain, a baseline
tuned on its test people, new members judged through the old stages at one fixed weight, and information only from
our ten people. The user's instruction, 12 Nov: address each with a build before any upload. Results are reported for
new people with a person-bootstrap interval, P04 and P10 apart.

### 68. TWOSYS: two complete chains on disjoint members, fused at the end (code `studies/12nov/twosys.py`)

Chain A = the two cores (+ teacher), chain B = the eight net slots (+ teacher), each with its own rebalancing,
SessionEM and TSEM; outputs multiplied (equal exponents; also 0.7/0.3 both ways). Mechanism: the stages amplify
session-consistent error inside one chain; two chains with different members may err on different sessions.

| Check | Prediction (80 % range) |
|---|---|
| Chain A alone / chain B alone, new | 77.5 (75–79) / 77.5 (75–79.5), against 80.2 |
| Fused, against the single chain, new | −0.3 (−1.2 to +0.6) |
| Lever | ≥ +0.5 on new with the bootstrap interval above 0; otherwise the single chain stays |

### 69. FAIR: stored candidates re-screened with the chain co-tuned (code `replay/fair.py`)

Every stored member that was closed at one weight against shipped constants (SessionBN versions of eight net
families, the augmentation and rest-class nets of 11–12 Nov, the two coupling cores, the eye-movement nets) is added
at weight 0.5, 1 or 2 while rebalancing strength, both stage strengths and the slot and teacher weights are re-chosen
(81 × 3 settings). Cross-fit: settings chosen on one half of the people, scored on the other; the baseline gets the
same procedure without the candidate. Null: a duplicate of an existing family as the candidate.

| Check | Prediction (80 % range) |
|---|---|
| Mean nested gain over the candidates, new | −0.1 (−0.5 to +0.3) |
| Candidates with a nested gain ≥ +0.8 and above the null | 1 (0–3) of about 30 |
| Lever | such a candidate that repeats on the other seed sets goes into a rebuilt mix |

### 70. SEQ: a learned sequence stage as a second design (code `studies/12nov/seqstage.py`)

A small causal recurrent model reads, window by window, the mix's log posterior, the 128 components and the 48
gaze and event features of a session and outputs the class; trained on the other nine people with random
orthogonal transforms of the components per pseudo-session, random window subsets and class-share shifts, so that it
must learn the adaptation rule inside the session and not fixed directions. It replaces rebalancing, SessionEM and
TSEM. Caveat: the training people's mix posteriors come from folds whose models saw the held-out person.

| Check | Prediction (80 % range) |
|---|---|
| Learned stage alone, new | 77 (72–81), against 80.2 for the hand-built stages |
| Product with the chain, against the chain, new | −0.3 (−2 to +0.7); P04 0 (−4 to +6) |
| Lever | ≥ +0.5 on new with the interval above 0, or ≥ +5 on P04 with nobody else below −1 |

### 71. FMVIEW: frozen foundation-model embeddings as a view of the session stage (code `studies/12nov/fmview.py`; extraction `studies/12nov/fmemb.py` on Galvani)

Embeddings from pretrained encoders without any fine-tuning on our people, reduced to components fitted on the
other nine people, used as a second TSEM view after the first and as extra columns of the first.

| Check | Prediction (80 % range) |
|---|---|
| Linear probe on the embeddings, leave one person out | 45 (38–55) |
| Second TSEM through the embedding view, new | 0.0 (−0.8 to +0.8) |
| Lever | ≥ +0.5 on new with the interval above 0 |

| Item | Observed | Verdict |
|---|---|---|
| §68 chain A (cores) / chain B (nets) alone, new | 74.1 / 77.6 (teacher in both); 74.1 / 77.0; 72.5 / 77.6 | A below its range, B a hit |
| §68 fused, against the single chain (B2_pool / B2_s345) | best arm (average, teacher with the cores) −0.69 [−0.97, −0.38] / −0.71; products −0.9 to −6.3; 0 to 3 of 10 people ≥ 0 | **Refuted as a lever** (predicted −0.3): pooling all members before the stages beats two chains fused after them |
| §69 FAIR, all 64 stored candidates (19 SessionBN-family nets; the run was stopped before two simpleconv settings and the tidnet family; 27 augmentation nets, 12 rest-class nets, 4 coupling-core heads, 2 eye-movement nets; B2_pool; `runs/fair_*12nov*.log`) | old screen: mean -0.15; **co-tuned search leaves the candidate out in 51 of 64 cases**; co-tuned cross-fit gain mean -0.04, range -0.73 to +0.15; the four duplicates of shipped members (null) 0.00 to −0.20 | **Hit** (predicted mean −0.1); candidates at ≥ +0.8 above the null: **0** (predicted 1, 0–3). The fixed-weight screen did not close any stored member wrongly |
| §70 SEQ, learned stage alone (on the mix), new | 64.6 / 65.6 against 80.2 / 80.1 for the hand-built stages; 0 of 10 people ≥ 0 | **Miss, far below the range** (predicted 77, 72–81) |
| §70 SEQ on the chain's output; product with the chain | −11.0 / −9.5; product −6.9 to −3.8; P04 −3 to −10 | **Refuted** in this form (400 steps, 24 dimensions). Two sanity variants running (untrained; 1,500 steps in 8 dimensions) |
| §70 SEQ sanity variants | untrained (0 steps) on the chain's output: +0.08 [−0.40, +0.53], i.e. neutral; 1,500 steps in 8 dimensions: −11.2 on the chain's output, −16.3 on the mix | The damage comes from training: a similarity learned on nine people does not carry to a tenth. Closed as built |

### 72. NETVIEW: the session stage through the nets' own features (written 12 Nov 09:40 UTC, before any result; code `studies/12nov/netemb.py` on Galvani, scored with `studies/12nov/fmview.py`)

The session stage has only ever looked through core1's 128 components and the gaze view; the nets enter as posteriors.
The classifier inputs of the four shipped SessionBN recipes are captured per leave-one-person-out fold (the net never
saw the held-out person), reduced to 64 components on training sessions, and used as a second TSEM view.

| Check | Prediction (80 % range) |
|---|---|
| Second TSEM through a net's feature view, best of the four recipes, new | +0.2 (−0.5 to +1.0) |
| Lever | ≥ +0.5 on new with the interval above 0 on both seeds |

**§71 FMVIEW scored (10:08 UTC; logs `runs/fmview_*.log`; embeddings `studies/12nov/fmemb.py`, Galvani jobs 3091224–3091227, rows checked against the replay).**

| Item | Observed | Verdict |
|---|---|---|
| Linear probe on frozen embeddings, leave one person out (middle layer; causal EA / per-window input) | CBraMod 42.0 / 42.9, LaBraM 35.9 / 35.4, REVE 44.2 / 39.9, ST-EEGFormer 38.8 / 37.2; **REVE at random initialisation 42.8** | Hit (45, 38–55), but pretraining adds nothing a random network of the same shape does not give |
| Second TSEM through an embedding view, new (8 views × 2 bases × 2 strengths) | −0.50 to +0.39; best ST-EEGFormer per-window +0.39 [+0.19, +0.59] on one base and +0.31 on the other; the random-init control −0.07 to −0.20 | **Hit on the prediction (0.0), refuted as a lever**: the best of 32 arms sits at selection-noise level |
| Embedding components appended to the first TSEM | −0.56 to +0.22 | Refuted |
| Probe as a final member | −1.4 to −3.9 | Refuted |

Reading: frozen EEG foundation models, as published, carry no more class information on this task than a random
network of their shape. The information "from thousands of other people" is not in their features in a form that
separates these three mental tasks.

**§72 NETVIEW scored (logs `runs/netview_*.log`; Galvani pack 3091237; grid chain without routing, new people; second TSEM through the net's classifier input, 64 components, λ 0.05 / 0.10 / 0.15; each cell B2_pool / B2_s345).**

| Recipe, seed | λ 0.10 | λ 0.15 | P04 at λ 0.15 |
|---|---|---|---|
| FBCNet SessionBN (fb), s0 | +0.56 [+0.21, +0.90] / +0.35 | **+0.85 [+0.50, +1.21] / +0.52 [+0.11, +0.90]** | +1.9 / +2.1 |
| fb, s1 | +0.59 [+0.17, +1.01] / +0.15 | **+0.74 [+0.25, +1.26] / +0.44 [−0.14, +1.02]** | +3.1 / +3.1 |
| FBCNet raw SessionBN (fbraw), s0 | +0.28 / +0.12 | +0.48 [+0.19, +0.78] / +0.32 | +1.9 / +1.7 |
| fbraw, s1 | +0.59 / +0.32 | +0.58 [+0.16, +0.99] / +0.39 | +2.4 / +2.3 |
| FBLight (fbl), s0 / s1 | +0.35 / +0.02; +0.45 / +0.10 | +0.36 / +0.14; +0.30 / +0.07 | −0.9; 0.0 |
| SincShallow (sinc), s0 / s1 | +0.04 / −0.20; +0.06 / −0.10 | +0.08 / −0.06; +0.30 / −0.08 | +1.2 to +1.8 |
| The net view alone in place of the 128 components | −0.6 to −1.9 | | |

Verdict: **lever met for the fb recipe** (≥ +0.5 with the interval above 0 on both seeds, on B2_pool; on B2_s345 +0.52 and
+0.44). Prediction (+0.2, −0.5 to +1.0) hit at its top. Unlike the foundation and random views (§71: −0.5 to +0.4, no
pattern), every FBCNet arm is positive at λ 0.15, the gain grows with λ, and P04 gains 2–3 points. λ 0.15 is the edge of
the tried range and was picked after seeing it: the confirmation below fixes it in advance.

### 73. NETVIEW2: confirmation on fresh nets, after the full routed chain (written 12 Nov 10:10 UTC, before any result; code `studies/12nov/netview2.py`; Galvani packs 3091329, 3091330)

Fixed arm: the fb recipe's classifier input exactly as stored by `netemb.py` (64 components fitted inside the fold on
four training sessions, no further rotation), second TSEM at λ 0.15 applied **after the full v4.1 replay chain with
routing** (`A.system`). Fresh nets: fb seeds 2–5 (independent trainings). Secondary: fbraw seeds 2–5; the seen
protocol (P04, P10) on fb seeds 0–1.

| Check | Prediction (80 % range) |
|---|---|
| New people, mean over fb seeds 2–5, B2_pool | +0.55 (+0.2 to +0.9) |
| Same, B2_s345 | +0.40 (0.0 to +0.8) |
| People ≥ 0 at the seed mean | 7 of 10 (6–9) |
| Seen (P04, P10) | +1.0 (−0.5 to +2.5) |
| **Refutation** | mean over fresh seeds < +0.25 on B2_pool, or its person-bootstrap 10th percentile ≤ 0 |
| Exploratory (on seeds 0–1, labelled as such): λ 0.2 / 0.3; two nets' views in succession; stored components against the fmview rotation | — |

| §73 exploratory (fb seeds 0–1, the ones that chose the arm) | Observed |
|---|---|
| After the full routed chain, stored components, mean of the two seeds, B2_pool / B2_s345 | λ 0.10: +0.54 [+0.27, +0.81] / +0.48; **λ 0.15: +0.55 [+0.19, +0.95] / +0.49 [+0.09, +0.90]**; λ 0.20: +0.47 / +0.43; λ 0.30: +0.33 / +0.04. The optimum is 0.10–0.15; the routed chain keeps the gain |
| Two views in succession (fb s0 then s1; fb then fbraw), or concatenated | +0.16 to +0.44: a second view adds nothing |

### 74. DVIEW: is the net view's gain a property of discriminative directions? (written 12 Nov 10:19 UTC, before any result; code `studies/12nov/dview.py`)

**Why.** §72–§73: the session stage gains about +0.5 for new people when it looks through a net's features, and
nothing through foundation or random features. The nets' features are trained to separate the classes across
people; core1's 128 components are unsupervised. If the mechanism is "a view whose directions separate classes for
somebody", it can be built from core1's own 256 components with the training people's labels, without new nets. The
union of per-person directions is the part aimed at P04-like people: a person nobody resembles on average may still
separate along one other person's directions.

Views, per leave-one-person-out fold from the cached 256 components (training rows centred per session), each as a
second TSEM after the full routed chain at λ 0.10 / 0.15: (a) pooled shrinkage LDA, 2 directions; (b) PLS-DA, 16
components; (c) per-person LDA directions of the nine training people, 18, orthonormalised; (d) null: 18 random
orthonormal directions of the same 256 components.

| Check | Prediction (80 % range) |
|---|---|
| (a) pooled LDA, new | 0.0 (−0.4 to +0.4) |
| (b) PLS-DA 16 | +0.2 (−0.3 to +0.7) |
| (c) per-person directions, new | +0.3 (−0.2 to +0.8); P04 +1 (−1 to +3) |
| (d) random null | 0.0 (−0.3 to +0.3) |
| Lever | a view ≥ +0.4 on new above the null on both bases, interval above 0 → combine with or replace the net view |

**§74 DVIEW scored (10:20 UTC; `runs/dview_12nov.log`; after the routed chain, B2_pool / B2_s345).**

| View | λ 0.10 | λ 0.15 | Verdict |
|---|---|---|---|
| (a) pooled LDA, 2 | −0.02 / −0.14 | −0.21 / −0.19 | Hit (0.0) |
| (b) PLS-DA, 16 | +0.45 [+0.07, +0.80] / +0.54 [+0.20, +0.90] | +0.60 [+0.22, +0.99] / +0.47 [+0.17, +0.74] | Above the prediction (+0.2) |
| (c) per-person LDA directions, 18 | +0.33 / +0.12 | +0.28 / +0.22; P04 +0.7 / 0.0 | Hit (+0.3); no P04 effect |
| **(d) random 18 directions (null)** | **+0.28 [+0.02, +0.52] / +0.25 [+0.09, +0.42]** | +0.22 / +0.11 | **Miss: the null is not zero** (predicted 0.0) |

**This changes how §72–§73 must be read.** A second TSEM pass through *random* directions of core1's own components
gains +0.25 for new people: part of any second-view gain is generic (a second pass strengthens the session stage, as the
5 Nov "permuted-Gram null" of +0.47 and the grid's preference for TSEM λ 0.2 already showed). Against this null the
net view (+0.49 to +0.55 after the routed chain) and PLS-DA (+0.45 to +0.60) are +0.2 to +0.3 above it, which is
at the edge of resolution.

**Addendum to §73, written 10:20 UTC, before any fresh-seed result exists** (the files are still training): the
§73 rule stays as written. Added comparisons, run on the same bases: (i) random 64-direction projections of core1's
256 components, five draws, as a second TSEM at λ 0.15; (ii) the first TSEM alone at λ 0.20 and 0.25, no second pass.
**The net view counts as a lever of its own only if its fresh-seed mean exceeds the better of (i) and (ii) by ≥ +0.25
with a person-bootstrap 10th percentile above 0.** Otherwise the finding is "the session stage is slightly too weak",
worth about +0.25, and the cheapest form (no net features) is preferred.

**§73 nulls (`runs/viewnull_12nov.log`; after the routed chain, new people, B2_pool / B2_s345).**

| Null | Observed |
|---|---|
| (i) random 64 directions of core1's 256 components, second TSEM λ 0.15, five draws | single draws −0.27 to +0.58 / −0.14 to +0.55; **mean +0.14 / +0.20** |
| (ii) first TSEM at λ 0.20, no second pass | **+0.64 [+0.25, +1.04] / +0.30 [+0.09, +0.51]** |
| (ii) first TSEM at λ 0.25 | +0.37 / +0.16 |

Two lessons. One random view moves new people by −0.3 to +0.6: a single view arm at +0.5 is inside the spread of
random views, and §72's "every FBCNet arm positive" is the evidence that matters, not any single arm. And the
plain stage strength (λ 0.15 → 0.20) is worth as much as any view so far on B2_pool. By the addendum, the net view
must reach the better null + 0.25, i.e. about +0.89 on B2_pool and +0.55 on B2_s345.

### 75. LAMS: is the session stage too weak? (written 12 Nov 10:21 UTC, before any result; code `studies/12nov/lams.py`)

§73's null (ii) found TSEM λ 0.20 worth +0.64 / +0.30 for new people on two bases. λ 0.15 was set on 6 Nov, before
routing, the identity gate and SessionBN reweighting. Test: the full routed v4.1 replay at λ 0.12, 0.15, 0.20, 0.25,
0.30 on all six bases, new and seen; cross-fit λ over the two people halves.

| Check | Prediction (80 % range) |
|---|---|
| λ 0.20 against 0.15, new, mean of six bases | +0.40 (0.0 to +0.8) |
| λ 0.20 against 0.15, seen (P04, P10) | −0.3 (−1.5 to +1.0) |
| λ chosen on one half of the new people, scored on the other, mean of the two directions and six bases | +0.2 (−0.3 to +0.6) |
| Lever | cross-fit ≥ +0.25 with seen not below −0.5 → λ goes into the package with whatever else is built |

| §75 item | Observed | Verdict |
|---|---|---|
| λ 0.20 against 0.15, new, per base | +0.64 (pool), +0.30 (s345), +0.03, −0.04, −0.05, +0.03 | the two exploration bases were the lucky ones |
| Mean of six bases, new | λ 0.12 −0.45; **λ 0.20 +0.15 [−0.12, +0.41]**; λ 0.25 +0.02; λ 0.30 −0.26 | **Miss** (predicted +0.40) |
| Seen at λ 0.20 / 0.25 / 0.30 | −0.12 / −0.91 / −1.48 | Hit |
| Cross-fit over people halves | −0.07 (−0.79 to +0.52) | **Refuted** (< +0.25): λ 0.15 stays |

Lesson recorded for every view result above: B2_pool and B2_s345 sit high for stage changes; the other four bases
(only the SessionBN slots differ) halve or erase the effect. **NETVIEW2 (§73) is therefore reported on all six bases
(changed in the script before any fresh result was scored); B2_pool stays the pre-registered primary.**

**§73 on six bases, before the fresh seeds (`runs/netview2_fb01_six.log`, `runs/viewnull_six.log`; new people, λ 0.15).**

| Base (routed chain) | net view, choosing seeds 0–1 | random 64 directions, mean of 5 | TSEM λ 0.20 |
|---|---|---|---|
| B2_pool (80.18) | +0.55 | +0.14 | +0.64 |
| B2_s345 (80.10) | +0.49 | +0.20 | +0.30 |
| B2_s012 (80.31) | +0.15 | −0.11 | +0.03 |
| B2_s0 (79.57) | +0.40 | +0.07 | −0.04 |
| B2_s1 (80.52) | −0.17 | −0.40 | −0.05 |
| B2_s2 (80.55) | −0.11 | −0.33 | +0.03 |
| **Mean** | **+0.22** | **−0.07** | +0.15 |

Over six bases the net view is +0.29 above the random null, but its absolute gain is +0.22, and it is ≤ 0 on the two
strongest bases. The bases that gain are the weaker ones: a second view partly repairs what a stronger set of
SessionBN seeds already gets right.

**§73 NETVIEW2 scored (10:29 UTC; `runs/netview2_fb_fresh.log`, `runs/netview2_seen.log`; fresh fb nets seeds 2–5, Galvani pack 3091329).**

| Check | Observed | Verdict |
|---|---|---|
| New, mean over fresh seeds, B2_pool (primary) | **+0.20 [−0.25, +0.68]**, 5 of 10 people ≥ 0 | **Refuted** (rule: < +0.25 or 10th percentile ≤ 0) |
| Same on the other five bases | +0.32, −0.06, +0.21, −0.05, −0.18; six-base mean **+0.07** (random null −0.07) | Miss (predicted +0.40 on B2_s345 and above) |
| Per person, consistent over bases | P05 +0.7 to +3.8, P10 +1.0 to +2.3; P01 −1.0 to −1.8 | Person-specific, cancels in the mean |
| Seen (P04, P10), choosing seeds 0–1 | P04 +0.3 to +2.4, P10 −0.8 to 0.0 | Inside the range; two people |
| fbraw fresh seeds (secondary) | not run: the pack was pre-empted after the fb tasks | — |

Verdict: the nets' own features as a second session-stage view are **not a lever**. The choosing seeds' +0.5 was
selection on two favourable bases; fresh nets give +0.2 on the primary base and +0.07 over six. Closed in this form.

### 76. SELECT: can a session choose its own chain variant without labels? (written 12 Nov 10:30 UTC, before any result; code `studies/12nov/select.py`)

**Why.** Every stage change of 12 Nov moved single people by ±1–4 points and the mean by ~0 (§66 per-person optimum
+2.4; §73 P05 +3, P01 −1.8). If a label-free statistic of a whole session's output can tell which variant fits that
session, the per-person head-room becomes reachable. The criterion must be computable causally at the end of the
session or progressively; here first as a diagnostic at session end (upper bound for a causal version).

Variants: the routed chain at TSEM λ 0.12 / 0.15 / 0.20 / 0.25 / 0.30, plus two second passes through random
directions (§73 nulls), B2_pool and B2_s345, new people. Criteria per session and variant: (a) mean top posterior;
(b) balance of predicted classes (entropy; true classes are balanced by design); (c) agreement with the other variants'
majority; (d) separation of the predicted classes in the 128 components (Fisher ratio).

| Check | Prediction (80 % range) |
|---|---|
| Oracle: best variant per session, new | +2.5 (+1.5 to +4) |
| Best criterion, picking per session, against λ 0.15 | +0.2 (−0.4 to +0.7) |
| Lever | a criterion ≥ +0.5 on both bases → a causal version is built and pre-registered |

| §76 item (10:31 UTC; `runs/select_12nov.log`; B2_pool / B2_s345) | Observed | Verdict |
|---|---|---|
| Oracle: best of seven variants per session, new | +1.44 [+1.03, +1.84] / +1.00 [+0.70, +1.29] | **Miss, low** (predicted +2.5): the variants are too alike for per-session choice to matter |
| Pick by agreement with the variants' majority | +0.63 [+0.21, +1.06] / +0.32 [+0.13, +0.52] | below the lever on B2_s345; it is a vote among variants |
| Pick by confidence / class balance / Fisher separation | +0.07 / +0.27 / +0.10; −0.14 / +0.14 / +0.21 | Hit (+0.2): no criterion |

Reading: per-session choice among session-stage variants has at most 1–1.4 points of head-room, and label-free
criteria reach a fraction of it. The per-person spread seen all day is noise of the stage, not a missed setting.
The large losses (P04: every member 37–51 %) sit before the stages, in the members.

