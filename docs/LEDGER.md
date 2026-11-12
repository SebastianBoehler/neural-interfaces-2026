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

### 77. BANDMAP: where does P04's signal live, and where does transfer to P04 fail? (written 12 Nov 10:34 UTC, before any result; code `studies/12nov/bandmap.py`)

**Why.** User, 12 Nov: next wave on both lines; this is the first step of the new-member line. P04 is read at 76 % by
their own earlier sessions and at 37–51 % by every member trained on others (§67, §77 addendum of §67). A member aimed
at P04-like people needs a feature family where P04's classes separate **and** where the separating directions are
shared with somebody. Per band (nine bands) and family (tangent space T, CSP C, plus slow S): accuracy from the
person's own other sessions (P04, P10) and from the pooled other nine people (all ten targets), causal, as §37.

| Check | Prediction (80 % range) |
|---|---|
| P04 own-session, best single band | 65 (55–72), in 8–30 Hz |
| P04 from others, best single band | 45 (40–52) |
| Bands where P04 from others ≥ 50 % | 0 (0–2) |
| Others' mean from others, best band | 55 (50–60) |
No pass rule: a diagnostic. It turns into a member only if a band/family reads P04 from others ≥ 55 %.

| §77 item (10:37 UTC; `runs/bandmap_12nov.log`, `runs/bandconf_12nov.log`) | Observed | Verdict |
|---|---|---|
| P04 from own other sessions, best single band/family | CSP 4–8 Hz 60.6; slow S 59.3; CSP 4–40 57.8; best tangent band 4–40 Hz 54.9 | **Miss** (predicted 65 in 8–30 Hz): P04's own signal is in theta and slow potentials, not in mu/beta |
| P04 from the other nine, best single band/family | CSP 4–40 49, CSP 13–20 47 | Hit (45) |
| Bands where P04 from others ≥ 50 % | 0 of 19 | Hit |
| Others' mean from others, best | CSP 8–13 51.2, S 50.6 | Hit at the low edge (55) |
| Below-chance transfers | P04 slow S **30** (others' mean 50.6); P05 CSP 8–13 **24**, CSP 4–40 26 | not predicted |
| Are they class permutations? | P05 alpha CSP: SUB and WORD swapped (WORD → SUB 73 %), best permutation 50; P04 slow: no consistent permutation across sessions (best 40–43) | P05's is a swap; P04's is not |

**Reading.** No band or feature family carries P04's classes in a direction shared with any of the other nine
people (best 49 %, best under any class permutation 53 %). P04's own separable signal is theta CSP and slow
potentials, and in exactly those families P04 runs against the population (slow 30 %). The new-member line for
P04-like people, from our nine training people, is closed in feature space as well as in model space (§67).
The P05 swap is a second kind of person difference (a class relabelling in alpha); the chain already reads P05 at 80.5.

### 78. PNET: personal nets on top of the v5.1 stage, six bases (written 12 Nov 10:38 UTC, before any result; code `studies/12nov/pnet.py`)

**Why.** User, 12 Nov: build the personal-net component of the returning branch now (no package). Dev evidence so far
is on two bases and partly in older forms (§40 in-slot, §43 dose). Test in the exact v5.1 form: the five net families
on seen sessions replaced by their fine-tuned versions (fitted on the person's session 1; stored outputs of 11 Nov
§29, three seeds), then the routed chain and the own-space final stage at β 0.10, on all six bases; reference = the
same chain with the frozen twins and the stage. Variants: head and BN affine only (`pfh60`), all parameters 60 steps
(`pf60`).

| Check | Prediction (80 % range) |
|---|---|
| pfh60 + stage minus twins + stage, seen, six-base mean | +0.3 (−0.3 to +0.9); P04 +0.8, P10 −0.3 |
| pf60 + stage, same | +0.2 (−0.6 to +1.0); P10 −0.8 |
| **Lever** | ≥ +0.3 seen with P10 ≥ −0.5, on at least four of six bases → personal nets trained for all ten people on all public sessions (Galvani) and kept as a ready component |

| §78 item (10:41 UTC; `runs/pnet_12nov.log`) | Observed | Verdict |
|---|---|---|
| Stage on the frozen twins, seen, per base | +1.71 to +2.25 | the v5.1 stage replicates on all six bases |
| pfh60 + stage minus twins + stage, seen, six-base mean | +0.17 (P04 +0.53, P10 −0.18); per base −0.11 to +0.41 | Hit (+0.3, −0.3 to +0.9) |
| pf60 + stage, same | +0.25 (P04 +1.13, P10 −0.63) | Hit (+0.2); P10 below −0.5 |
| Bases meeting the lever | pfh60 1 of 6, pf60 0 of 6 | **Refuted**: the per-person nets are not trained |

Reading: with the own-space stage in place, personal nets add +0.2 for returning people, all of it P04's, and cost
P10 up to 1 point. The stage already holds what the nets would add. The returning branch on dev ends at v5.1.

### 79. TRUST: can a returning session tell how much to trust its personal stage? (written 12 Nov 10:42 UTC, before any result; code `studies/12nov/trust.py`)

**Why.** User, 12 Nov: push on the personal stage. Its dev gain (+1.7 to +2.3 seen on six bases) is P04's: P04
gains up to β 0.25 (+4.8), P10 loses as β grows (−0.5 at 0.10, −2.9 at 0.25), §43. One β for all wastes P04's
head-room and costs P10. If a causal, label-free signal inside the session says whether the personal model still
fits (its three members agree with each other; it agrees with the chain; it is confident), β can follow it. This is
the per-person trust weight proposed on 7 Nov and never built (review §4).

**Diagnostic first.** Per returning session: the best β of 0 / 0.05 / 0.10 / 0.15 / 0.25 / 0.40 (true labels) and
three end-of-session signals: (a) agreement of T, C and S with each other, (b) agreement of their mean with the routed
chain, (c) their mean top posterior. Data: the four real seen sessions (P04 s2, s3; P10 s2, s3) on six bases, and the
12 cross-session pairs of §42 (one public session → another, LOSO chain) for more points.

| Check | Prediction (80 % range) |
|---|---|
| Best β, real seen sessions | P04 0.25–0.40; P10 0–0.05 |
| Signal separating the two people on the real sessions | (a) yes, (b) no, (c) no |
| Spearman(signal (a), best β) over the 12 pairs | +0.4 (−0.1 to +0.8) |
| **Lever** | if a signal separates the people and ranks the pairs ≥ +0.5, a trust-scaled β (0 to 0.25, map fixed from the 12 pairs) is pre-registered and scored on the six bases |

| §79 diagnostic (10:44 UTC; `runs/trust_12nov.log`) | Observed | Verdict |
|---|---|---|
| Best β, real seen sessions (three bases) | P04 s2 0.40, s3 0.15–0.25 (+7 to +8); P10 0–0.05 (β 0.15 costs up to −2.6) | Hit |
| Signals on the real sessions, P04 / P10 | T·C·S agreement 0.50–0.52 / 0.58–0.61; **agreement with the chain 0.57–0.59 / 0.74**; confidence 0.75–0.77 / 0.81–0.82 | All three separate the people; (b) most clearly. Miss on (b) and (c) |
| Spearman with the best β over the 12 pairs | T·C·S agreement −0.70; **agreement with the chain −0.65 (with the gain at 0.15: −0.94)**; confidence −0.70 | **Opposite sign to the prediction (+0.4)** |

The sign is the reverse of my mechanism: the stage pays most where the personal model *disagrees* with the chain,
because that is where the chain is weak (P04) and the person's own model knows better. The pre-set lever asked for a
rank correlation ≥ +0.5; I go past its letter on purpose, because a strong negative rank is as usable as a positive
one. The map is fixed now from the 12 pairs, before the real-chain run: β = 0.40 at a running chain agreement ≤ 0.50,
0.10 at ≥ 0.70, linear in between; the running agreement starts from 10 pseudo-windows at 0.65.

### 80. TRUST2: personal stage with a trust-scaled β (written 12 Nov 10:44 UTC, before any result; code `studies/12nov/trust2.py`)

The v5.1 replay with β_t from the map above (causal running agreement of the personal members' mean with the routed
output before the stage), against fixed β 0.10, six bases. Then, if it passes, the false-accept cost (§65 protocol).

| Check | Prediction (80 % range) |
|---|---|
| Seen, six-base mean, against v5.1 | +0.4 (−0.2 to +1.0); P04 +0.9, P10 0.0 |
| New (false accepts only) | 0.00 (−0.05 to +0.02) |
| **Lever** | seen ≥ +0.3 on ≥ 4 of 6 bases with P10 ≥ −0.3 and new ≥ −0.05 |

| §80 item (10:46 UTC; `runs/trust2_12nov.log`) | Observed | Verdict |
|---|---|---|
| Mean β the map gives (B2_pool) | P04 s2 0.19, s3 0.31; P10 0.12 / 0.10 | the signal separates the people as intended |
| Seen against fixed β 0.10, six bases | +0.01, −0.26, −0.12, +0.14, +0.40, +0.55; **mean +0.12** (P04 +0.01, P10 +0.23) | **Refuted** (lever 2 of 6 bases; predicted +0.4) |
| New | 0.000 on every base | Hit |

Reading: in the routed chain P04's gain is already saturated at β 0.10 (the gate's ramp and the personal head carry
the rest); a larger β for P04 adds nothing on average, and the small P10 recovery is the only gain. The personal
stage on dev is at its plateau: nine member variants (§44), per-person space (§41), personal nets (§78) and a trust
weight (§80) all land within ±0.3 of v5.1.

### 81. TTT3B: in-session fine-tuning for new people, three more seeds (written 12 Nov 10:47 UTC, before any job ran; Galvani, `runs/diag/ttt/galv/ttt3.py` = `studies/12nov/ttt3.py`, tasks 15–29)

**Why.** User, 12 Nov: keep the GPUs busy. The only member-level change that has ever been positive for new people in
the chain is package-form in-session fine-tuning on the chain's labels (§60: +0.27 / +0.17 on two bases, three seeds,
person interval −0.5 to +0.9; true labels +1.8). Seeds 3–5 halve the seed part of its noise and allow six-base
scoring with six seeds. The person part of the interval stays.

| Check | Prediction (80 % range) |
|---|---|
| Chain, five families swapped, package form `ple20`, six seeds, new, six-base mean | +0.15 (−0.15 to +0.45) |
| **Lever** | ≥ +0.3 on the six-base mean with ≥ 4 of 6 bases positive → a v6 with in-session fine-tuning is specified (runtime check first) |

### 82. INPOP: the personal stage for ten people when the person is already in the population (written 12 Nov 10:48 UTC, before any result; code `studies/12nov/inpop.py`)

**Why.** The open question of the returning branch: what the stage gives a returning person whose one public session
is in our training set (the sealed situation for eight people). Proxy for all ten: session 1, runs 1–3 are public
(in the nets' training, stored outputs of §32) and fit the own-space members; runs 4–6 are the "later" stream (90
windows each). Mini-chain on runs 4–6 as their own stream: the five net families from the in-population nets (three
seeds), core1, Hilbert core and teacher from the LOSO outputs, rebalancing, SessionEM and TSEM; then the stage at
β 0.10 and 0.25, no gate (recognition assumed). Reference arm: the same chain with LOSO nets (person-naive).

| Check | Prediction (80 % range) |
|---|---|
| Stage gain on the in-population chain, β 0.10, mean of ten | +1.5 (0 to +3) |
| People with a gain ≥ +1 | 5 of 10 (3–8) |
| In-population nets against LOSO nets, mini-chain, mean | +2 (0 to +5) |
| **Validity** | the proxy is informative only if P04's stage gain exceeds P10's, as on the real cross-session sessions (+3.5 / +0.1); otherwise it is recorded as uninformative |

| §82 item (10:50 UTC; `runs/inpop_12nov.log`; B2_pool / B2_s345 / B2_s012) | Observed | Verdict |
|---|---|---|
| In-population nets against LOSO nets, mini-chain | +5.1 / +4.5 / +4.8 | Above the prediction (+2) |
| Stage gain, in-population chain, β 0.10 | **+3.2 [+2.0, +4.4] / +3.8 [+2.4, +5.3] / +3.3 [+2.0, +4.5]** | **Miss, high** (predicted +1.5, 0 to +3) |
| People ≥ +1 | 6 / 7 / 7 of 10; per person (B2_pool) P01 +8.3, P02 +0.8, P03 +4.2, P04 +6.7, P05 +4.2, P06 +2.5, P07 −0.8, P08 −0.8, P09 +0.8, P10 +5.8 | Hit (5) |
| Stage gain, β 0.25 | +5.2 / +5.6 / +5.1; P07 −3.3 | — |
| Validity: P04 above P10 | +6.7 against +5.8 (real cross-session: +3.5 against +0.1) | Passes by the letter, barely |

Reading. Within a session, with the person's first runs already in the population, the personal stage adds +3 to
+4 points for seven of ten people. Across sessions on the real data it kept about half of that for P04 and nothing
for P10 (whose chain is at 91–92 % once their first session is in training). So the sealed value of v5.1 for the
eight single-session people lies between "P10-like" (≈ 0) and "half the within-session gain" (≈ +1.5 to +2 for
their sessions, ≈ +0.6 to +0.8 sealed points if they are 43 % of the board). The proxy cannot narrow this further;
only the sealed set can.

### 83. NOISE: why are a person's own labels worth +3 and the chain's labels nothing? (written 12 Nov 10:58 UTC, before any result; code `studies/12nov/ownnoise.py`)

**Why.** User, 12 Nov: understand why own data is worth so much and use it. The facts: ~90 labelled windows of a
person are worth +5 in the nets and +3 to +4 more in the own-space stage (§82); the same in-session learner on true
labels gives new people +2.5 to +2.9 (§38), on the chain's labels −0.1 to −0.6, though those labels are 77–85 %
right. Two explanations predict different cures. (1) **Error count**: 15–20 % wrong labels are too many for small
in-session models; cure: fewer, cleaner labels. (2) **Error placement**: the chain is wrong on the windows that look
like the other class, so the learner copies the chain's decision boundary and adds nothing; cure: labels whose
errors are independent of the features (another view, another label source).

**Test.** §38's learner (T, C, S refitted every 5 windows on the chain's most confident windows per class), new
people, two bases, three label sources on exactly the same selected windows: (a) the chain's labels; (b) true labels
with the chain's errors moved to random windows of the same true class (same count and confusion at every refit);
(c) true labels. Fusion as §38 (T·C·S, λ 0.25 and 0.5).

| Check | Prediction (80 % range) |
|---|---|
| (c) true, new | +2.8 (+2 to +3.5), as §38 |
| (a) chain, new | −0.4 (−0.8 to 0), as §38 |
| **(b) chain-count errors at random places, new** | **+2.0 (+0.5 to +3)**: placement, not count, is the problem |
| Reading | (b) ≥ +1.5 → explanation 2; the next build is a label source with feature-independent errors. (b) ≤ +0.5 → explanation 1; the next build is cleaner, fewer labels |

| §83 item (11:03 UTC; `runs/ownnoise_12nov.log`; new people, λ 0.25, B2_pool / B2_s345) | Observed | Verdict |
|---|---|---|
| (a) chain labels (16 % wrong on the selected windows) | −0.28 / −0.62 | Hit (−0.4) |
| (b) the same errors moved to random windows | +0.39 / 0.00 | **Miss** (predicted +2.0) |
| (c) true labels **on the same selected windows** | **+0.55 / +0.73** (λ 0.5: +0.62 / +0.80) | **Miss, far below** the +2.8 of §38, which used true labels on **all** past windows |
| P05 | −3 to −10 with every label source, true labels included | a within-session learner fails for P05 |

**Reading: neither explanation; the windows are the problem.** Decomposition on B2_pool: chain labels −0.28 →
errors moved to random places +0.39 (placement: +0.7) → no errors at all on the same windows +0.55 (count: +0.2) →
true labels on all windows +2.9 (§38; **which windows: +2.4**). The chain's confident windows are the ones it already
gets right; labelling them, even perfectly, teaches the learner nothing the chain does not know. The value of a
person's own data lies in labels for the windows the population model finds hard. A returning person's public
sessions supply exactly those (every window labelled), which is why the personal stage works (§82: +3 to +4 within
session). For a new person no label-free source can supply them: where the chain is uncertain, its labels are near
chance by construction. Self-training inside the session is closed **in principle**, not only in the forms tried.
This also says where a bigger step for new people would have to come from: information about the hard windows that is
independent of the chain (another sensor view, the task structure), not more adaptation on the chain's own beliefs.

### 84. RESID: a personal stage that learns where the population chain is wrong (written 12 Nov 11:03 UTC, before any result; code `studies/12nov/inpop2.py`)

**Why.** §83: a person's labels are valuable on the windows the population model gets wrong. The v5.1 stage ignores
that: it multiplies a fixed share of the personal members into every window. A stacked personal model fitted on the
public windows, with the population chain's own (out-of-sample) posterior as input next to the personal members,
can learn this person's correction: when to overrule the chain and toward which class. Proxy of §82 (ten people;
runs 1–3 public, runs 4–6 later). Public-side inputs: the person-naive chain on runs 1–3 and the personal members
cross-fitted run by run; later side: the in-population chain and the members fitted on runs 1–3. Logistic regression
on the six log posteriors (+ products of chain and member logs), C 0.1, output mixed with the chain at weight w ∈
{0.5, 1}.

| Check | Prediction (80 % range) |
|---|---|
| RESID against the v5.1 stage (β 0.10) on the in-population chain, mean of ten | +0.3 (−0.8 to +1.2) |
| People where RESID beats the stage | 5 of 10 (3–7) |
| **Lever** | ≥ +0.8 on three bases with ≥ 6 of 10 people ahead → test on the real seen sessions (P04, P10) in the v5.1 replay |

| §84 item (11:05 UTC; `runs/inpop2_12nov.log`; B2_pool / B2_s345 / B2_s012) | Observed | Verdict |
|---|---|---|
| v5.1 stage on the in-population chain | +3.17 / +3.83 / +3.25 | as §82 |
| RESID w 0.5 | +4.92 / +5.58 / +4.92 | — |
| **RESID against the stage** | **+1.75 [+1.00, +2.50] / +1.75 [+0.83, +2.58] / +1.67 [+0.83, +2.50]**; ahead 8 / 7 / 8 of 10 | **Lever met** (predicted +0.3, −0.8 to +1.2: miss, high) |
| w 1.0 | +1.50 / +1.42 / +1.50 against the stage | — |

On the within-session proxy the stacked correction beats the v5.1 stage by +1.7 for eight of ten people. By the rule
it goes to the real cross-session test.

### 85. RESID2: the stacked correction on the real later sessions of P04 and P10 (written 12 Nov 11:05 UTC, before any result; code `studies/12nov/resid_seen.py`)

v5.1 replay, six bases, seen protocol. Public side: session 1 of the person, with the person-naive (LOSO) chain's
posterior and the personal members cross-fitted run by run (six folds). Later side: the routed chain before the
stage and the members fitted on all of session 1. The correction replaces the v5.1 stage where the gate routes, at
the gate's ramp: out ∝ chain × R^(w·r), w 0.5 (fixed by §84; 1.0 reported).

| Check | Prediction (80 % range) |
|---|---|
| Seen against v5.1, six-base mean | +0.5 (−1.0 to +2.0); P04 +1.0, P10 0.0 |
| **Lever** | ≥ +0.5 on ≥ 4 of 6 bases with P10 ≥ −0.5 → RESID becomes the returning-person stage candidate (v5.2) |

| §85 item (11:08 UTC; `runs/resid_seen_12nov.log`, `runs/resid_seen_w1_12nov.log`) | Observed | Verdict |
|---|---|---|
| v5.1 stage on the real later sessions, six bases | +1.31 to +2.00 seen (P04 +2.4 to +4.0, P10 −0.3 to +0.3) | replicates |
| RESID w 0.5 against v5.1 | −0.42, −0.41, −0.40, −0.40, +0.13, +0.39; **mean −0.18** (P04 −0.16, P10 −0.20) | **Refuted** (0 of 6 bases) |
| RESID w 1.0 against v5.1 | mean −0.85 (P04 −1.25, P10 −0.45) | Refuted |

Reading: the stacked correction wins +1.7 within a session and loses across sessions. What it learns is the
person-naive chain's errors on session 1; on a later session the chain is person-aware (the person's session 1 is in
training and the personal head routes), and its errors are different ones. The simple product stage does not depend
on the chain's error pattern and survives. This is the third time a within-session proxy has over-promised for the
returning branch (§32, §41, §84); the cross-session behaviour of a personal model is only measured on P04, P10 and
the sealed set.

### 86. HARDVIEW: another sensor view, consulted on the hard windows (written 12 Nov 11:20 UTC, before any result; code `studies/12nov/hardview.py`)

**Why.** User, 12 Nov, after §83: try another sensor view for the hard windows. §83 says a gain for new people needs
information about the windows the chain finds hard that does not come from the chain. The 6 Nov multi-view work asked
whether views add to TSEM over a whole session (≤ +0.5 oracle); it never asked about the hard windows alone.

**Views** (no EOG channels exist; the frontal and temporal electrodes carry eye and muscle activity), each a
population classifier (leave one person out; features z-scored causally within the session; logistic regression):
V1 gaze/event features (48, as SessionEM); V2 muscle: log power 30–40 and 40–58 Hz per channel (82); V3 log band
power per channel, nine bands (369); V4 temporal profile: log broadband power in eight half-second segments per
channel (328); V5 slow time courses (as the S member). **Hard windows:** the third of each session's windows with the
lowest top posterior of the routed chain (a causal per-window quantity).

**Fusion:** out ∝ chain × P_V^(w · g), g = 1 when the chain's top posterior is below c; w ∈ {0.25, 0.5, 1},
c ∈ {0.5, 0.6, 1.01} (1.01 = every window). The arm (view, w, c) is chosen on the two exploration bases and scored
on the four others.

| Check | Prediction (80 % range) |
|---|---|
| Chain accuracy on its hard windows, new people | 55 (48–62) |
| Best view on the hard windows | 42 (37–48) |
| Best view on the hard windows where the chain is wrong (chance 50 % among the two other classes) | 28 (20–35) for the true class |
| Chosen arm on the four fresh bases, new | +0.1 (−0.3 to +0.5) |
| **Lever** | ≥ +0.3 on the fresh four with the person-bootstrap 10th percentile above 0 |

### 87. FOCAL: population nets trained to emphasise their hard windows (written 12 Nov 11:21 UTC, before any job ran; `studies/12nov/netaug.py` tasks 160–183, Galvani)

**Why.** §83: the value lies in the hard windows. Every member is trained with plain cross-entropy, which spends most
of its gradient on windows the nets already get right. Focal loss (weight (1 − p_true)^γ, γ 1 and 2) shifts training
toward the hard windows of the training people. Never tried in this project. Four SessionBN recipes × two γ × three
seeds, both protocols; scored with the fair co-tuned screen (§69) and as a replacement of the shipped family.

| Check | Prediction (80 % range) |
|---|---|
| Nets alone, new, against the twin recipe | −0.5 (−2 to +1) |
| Replacing the shipped family in the chain, new, six-base mean | 0.0 (−0.4 to +0.4) |
| **Lever** | ≥ +0.3 on six bases with the interval above 0 |

| §86 item (11:21 UTC; `runs/hardview_12nov.log`) | Observed | Verdict |
|---|---|---|
| Views alone, new people | gaze 52.4, band power 52.4, temporal profile 49.4, slow 47.8, muscle 36.9 | — |
| Chain on its hard third / elsewhere | 60.8 % / 84.5 % | Hit (55, 48–62) |
| Best view on the hard windows | band power 40.1 %, slow 38.9, gaze 36.8, temporal profile 34.5, muscle 33.2 | Hit (42) |
| On hard windows where the chain is wrong: view picks the true class / the chain's wrong class | 27–34 % / **38–46 %** | Miss: every view leans to the chain's wrong class |
| Chosen arm (muscle, w 0.25, c 0.5) on the four fresh bases | −0.71 [−1.02, −0.40]; every view's best arm −0.7 to −2.5 | **Refuted** |

Reading: the chain's hard windows are hard in every sensor view we have. Where the chain is wrong, gaze, muscle,
band power, temporal profile and slow potentials more often pick the chain's wrong class than the true one, so the
errors are not the chain's own: those windows look like another class in every signal (or the person was not doing
the cued task). No other view carries independent information about them. Together with §83 this bounds what any
label-free method can do for new people on these windows.

### 88. AMBIG: are the chain's hard windows hard for the person's own model too? (written 12 Nov 11:21 UTC, before any result; code `studies/12nov/ambig.py`)

§86: the hard windows look like another class in every population view. Two readings: (1) the windows are ambiguous
in the data (the person was not doing the cued task, or did it the way they do another task): then nothing helps;
(2) the information is there in the person's own space and only the mapping is missing: then the head-room is real
and only labelled own data unlocks it. Test: own-space members (T·C·S) trained on the other runs of the same session
with true labels, run by run, on all 33 new-person sessions; accuracy on the chain's hard third against the rest.

| Check | Prediction (80 % range) |
|---|---|
| Own model on the chain's hard windows | 52 (42–62) |
| Own model on the other windows | 76 (70–82) |
| Own model right where the chain is wrong (hard windows) | 40 (30–50) |

| §88 item (11:24 UTC; `runs/ambig_12nov.log`; new people, B2_pool) | Observed | Verdict |
|---|---|---|
| Own model on the chain's hard windows / the rest | **78.3 % / 85.3 %** (chain 60.8 / 84.5) | **Miss, far above** (predicted 52 on hard) |
| Hard windows where the chain is wrong (348): own model picks the true class / the chain's class | **72.4 % / 17.8 %** | Miss, far above (predicted 40) |
| Per person, own model on hard | 68 (P04) to 86 (P08); right where the chain is wrong 47 (P07) to 90 (P01) | every person |

**Reading: the hard windows are not ambiguous.** In the person's own space, with labels from the same session's other
runs, 72 % of the chain's errors are decoded correctly. The information is in the data; what fails is the mapping
from the population to this person. (Caveat: the own model uses later runs as well, so this is not causal; it is an
upper bound on what the session holds, not on what a causal method gets.) §83 showed that self-training on the
chain's confident windows with a linear learner cannot reach it. Untested: a learner that uses the session's own
cluster structure, not a decision boundary fitted to the easy windows.

### 89. LPROP: does the session's own geometry carry the labels from the easy windows to the hard ones? (written 12 Nov 11:24 UTC, before any result; code `studies/12nov/lprop.py`)

Within each new-person session: own-space features (9-band tangent space after recentring on the session mean; 30
principal components per session), a k-nearest-neighbour graph, label spreading. Seeds: the chain's easy two thirds,
labelled (i) with the true labels (upper bound for the seeding), (ii) with the chain's labels. Accuracy on the hard
third, non-causal first (a ceiling for the method); a causal version follows only if (ii) clears the chain.

| Check | Prediction (80 % range) |
|---|---|
| (i) true seeds, accuracy on the hard windows | 62 (52–72) |
| (ii) chain seeds, accuracy on the hard windows | 57 (48–65), chain 60.8 |
| **Go/no-go** | (ii) ≥ 64 % on the hard windows → causal version and chain fusion, pre-registered |

| §89 item (11:25 UTC; `runs/lprop_12nov.log`) | Observed | Verdict |
|---|---|---|
| (i) true seeds on the easy two thirds → hard windows | 50.9 % (k 7), 49.5 % (k 15) | **Miss** (predicted 62) |
| (ii) chain seeds → hard windows | 47.0 %, 44.9 % (chain 60.8) | **No-go** (< 64) |

Where the hard windows sit (B2_pool, new people): 48 % of the first fifth of each session (26 % of the last); equal
across interfaces; SUB and WORD more than MI; the confusion on them is spread over both other classes.

Reading: even perfectly labelled easy windows do not carry their labels to the hard windows through the session's own
neighbourhood structure. The hard windows are a different region of the person's space: decodable with labels from
hard-type windows (§88: 78 %), not reachable from the easy ones. For a new person the bottleneck is now precise: the
labels of their hard-type windows, which no label-free source in this data supplies.

### 90. WARMUP: how much of the loss is the session stages' warm-up? (written 12 Nov 11:26 UTC, before any result; code `studies/12nov/warmup.py`)

§89: 48 % of the first fifth of each session is hard, 26 % of the last fifth. The session stages (rebalancing,
SessionEM, TSEM) start empty and learn the session as it goes. Ceiling for a faster warm-up: run the stages over the
session twice in a row and score each window from the second pass, when the stages already hold the whole session
(hindsight; not allowed in a package, a bound only). Grid chain without routing, new people, six bases.

| Check | Prediction (80 % range) |
|---|---|
| Chain accuracy, first fifth / last fifth of the sessions | 72 / 84 |
| Hindsight stages, new, gain | +1.5 (+0.5 to +3) |
| Share of the gain in the first fifth | 60 % (40–80) |
| Reading | ≥ +1.5 → a faster causal warm-up is worth a study; ≤ +0.5 → warm-up is not the loss |

| §90 item (11:27 UTC; `runs/warmup_12nov.log`; grid chain, new people) | Observed | Verdict |
|---|---|---|
| Window accuracy, first fifth / last fifth (causal) | 69–71 / 75–76 | Miss on the last fifth (predicted 84: window accuracy, not balanced) |
| Hindsight stages, gain, six bases | +0.35 to +1.78; **mean +0.92 [+0.20, +1.60]** | Inside the range (+1.5, +0.5 to +3), lower half |
| Where | first fifth +3 to +5 points of window accuracy; last fifth ±0.5 | Hit: all of it is the warm-up |

Reading: a perfectly warmed-up session stage would be worth about +0.9 for new people (core1's own recentring warm-up
is not included; its features are cached). Between the two reading marks: a causal warm-up can recover only part of a
+0.9 bound, so it is a small lever (≈ +0.3), not a breakthrough. P05 loses with hindsight (−2.6 to −4.7): their
session changes over time, and late information misleads early windows.

**§81 addendum (12 Nov, before any seed-3–5 result was scored): seeds 6–8 (tasks 30–44) queued on Galvani as pack
`ttt3d` so the GPUs stay busy; they are scored together with seeds 3–5 under the same rule if they finish first.**

### 91. TTT4: in-session fine-tuning with terms that reach the hard windows (written 12 Nov 11:31 UTC, before any job ran; code `studies/12nov/ttt4.py`, Galvani)

**Why.** §83 and §89: pseudo-labels on the chain's easy windows do not teach the hard ones. TTT's loss only touches
windows through their (mostly easy, mostly right) pseudo-labels. Two unsupervised terms act on every window, hard ones
included: consistency between a window's prediction and its augmented version (time shift, noise, channel dropout;
pushes the decision boundary away from dense regions) and entropy minimisation. Variants, package form, four SessionBN
recipes, three seeds: ple20 (reference, §60), fix20 (pseudo-labels on augmented inputs + consistency), ent20
(pseudo-labels + entropy), cons20 (consistency only).

| Check | Prediction (80 % range) |
|---|---|
| Nets alone, new, fix20 against ple20 | +0.5 (−1 to +2) |
| Chain, four families swapped, new, best new variant against ple20, two bases | +0.1 (−0.3 to +0.5) |
| **Lever** | a variant ≥ +0.4 over the frozen nets in the chain on six bases with the interval above 0 |

### 92. POPPRIOR: a population prior for the session stage's class templates (written 12 Nov 11:36 UTC, before any result; code `studies/12nov/popprior.py`)

**Why.** §90: the session stage's warm-up costs up to +0.9 for new people, all in the first fifth of the session, and
TSEM starts from nothing at window 15. The personal template prior was tried for returning people (§19, P04 only);
a population prior for new people was not. Per fold, the training people's class means in the session-standardised
128 components (in the fold's own basis, from the cached training projections) are averaged and pull TSEM's class
means with m0p pseudo-windows (10, 30); TSEM may start at window 5 instead of 15 when the prior is present. Grid
chain without routing, new people, six bases.

| Check | Prediction (80 % range) |
|---|---|
| Best (m0p, start) against the shipped TSEM, new, six-base mean | +0.2 (−0.2 to +0.6) |
| Share of the gain in the first fifth | 70 % |
| **Lever** | ≥ +0.3 with the interval above 0, the arm chosen on the two exploration bases and confirmed on the four others |

| §92 item (11:37 UTC; `runs/popprior_12nov.log`) | Observed | Verdict |
|---|---|---|
| Every (m0p, start) on every base, new | −0.32 to −1.20; first-fifth window accuracy −0.6 to +0.6 | **Refuted** (predicted +0.2) |
| Chosen arm (10, 5) on the four fresh bases | −0.50 [−0.85, −0.17], 3 of 10 people ≥ 0 | Refuted |

Reading: the training people's class templates point the wrong way for a new person (P01 −2 to −4, P05 −1 to −4);
the same lesson as §67 (people do not transfer) at the level of the session stage. The warm-up cannot be shortened
with population knowledge.

**Lane fill (12 Nov, before any of these results; user: use up to 36 GPU lanes).** Queued: TTT4 seeds 3–5 (tasks
12–23, scored with seeds 0–2 under §91's rule), and the §73 secondary (fbraw fresh seeds 2–5, netemb tasks 9, 13, 17,
21), which the pre-emption had dropped. Nothing else is queued without a result that calls for it.

| §87 item (12:12 UTC; `runs/fair_focal_12nov.log`, `runs/focal_swap_12nov.log`, `runs/focal_v51_12nov.log`) | Observed | Verdict |
|---|---|---|
| Nets alone, new (focal1 / focal2 against the shipped recipe) | fb 67.3 / 67.1, fbl 58.9 / 59.4, fbraw 63.4 / 63.0, sinc 61.9 / 61.4; twins 66.0, 57.9, 63.4, 61.7 | Hit (about +0.5) |
| Added as members, co-tuned cross-fit | −0.16 to +0.23 | none passes |
| Replacing the four families, against the twins, new, six bases | focal1 **+0.07 [−0.19, +0.32]**, focal2 −0.27 | **Refuted** (lever +0.3) |
| Seen, without / with the v5.1 stage | focal1 +0.66 to −0.00 / mean +0.26 (P04 +0.48, P10 +0.04) | P04 again; the stage absorbs most of it |

