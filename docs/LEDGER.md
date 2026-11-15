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

**Read the "closed" tables with the map** ([SOLUTION_SPACE_MAP.md](SOLUTION_SPACE_MAP.md), 13 Nov). An audit of all
119 sections found that many closures rest on weak tests: new members scored as one more member at a fixed weight,
new cores swapped in as posteriors only, whole systems in one configuration, directions dropped within 0.01 to 0.1 of
a bar inside the noise, and returning-person levers dropped on P10 alone. Those are backlog, not verdicts:
`evidence/map_13nov/map_experiments_11_13nov.md` part 2 lists them, part 4 the positives never combined, part 6 the
ideas never run; `evidence/map_13nov/map_history_to_10nov.md` covers the closures before 11 Nov. A verdict of
"refuted" below a bar it missed by less than the noise should be read as "undecided".

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

### 70. SEQ: a learned sequence stage as a second design (code `studies/12nov/seqstage.py`) [review 13 Nov: confounded: fold-specific PCA bases pooled; withdrawn [§140]]

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

### 73. NETVIEW2: confirmation on fresh nets, after the full routed chain (written 12 Nov 10:10 UTC, before any result; code `studies/12nov/netview2.py`; Galvani packs 3091329, 3091330) [review 13 Nov: seen/J are routing v3 (A.system), not v4.1 [§140]]

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

### 75. LAMS: is the session stage too weak? (written 12 Nov 10:21 UTC, before any result; code `studies/12nov/lams.py`) [review 13 Nov: seen/J are routing v3 (A.system), not v4.1 [§140]]

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

### 76. SELECT: can a session choose its own chain variant without labels? (written 12 Nov 10:30 UTC, before any result; code `studies/12nov/selectvar.py`)

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

### 78. PNET: personal nets on top of the v5.1 stage, six bases (written 12 Nov 10:38 UTC, before any result; code `studies/12nov/pnet.py`) [review 13 Nov: personal nets from window 0 in both tails, not the package form [§140]]

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

**Review note (13 Nov 00:42 UTC, independent reviewer, confirmed; written by the v52-build session):** the §78 form is
not the package form. Dev swaps the personal nets from window 0 and also in the gate's shadow path, adds a SessionBN
shallow member the package does not have, and leaves slot 9 unswapped. Re-scored from the stored outputs in the
package form without the shallow member: +0.39 seen (P04 +0.33, P10 +0.45; per base +0.28 to +0.55); with it: +0.02.
Each person moves in steps of about 0.27 (one or two windows). §78 therefore neither supports nor contradicts §105.

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

| §73 secondary (`runs/netview2_fbraw_fresh.log`; fbraw fresh seeds 2–5) | +0.13, +0.14, −0.30, −0.06, −0.34, −0.55; six-base mean −0.16 | confirms the refutation |

### 93. DRO: person-robust objectives on the SessionBN nets (written 12 Nov 12:16 UTC, before any job ran; `studies/12nov/netaug.py` tasks 300–335, Galvani)

**Why.** User, 12 Nov: fill the free GPU lanes. The located loss is P04-like people (§67, §77): the nets fit the
average training person. GroupDRO over participants (η 0.1, 0.01) and a participant-balanced loss upweight the
worst-fitted training people. October tried them on plain nets only (R8: neutral; R18: adversarial hurts); never on
the SessionBN nets the chain relies on. Four recipes × three objectives × three seeds, both protocols; scored as
replacements against the twins (as §87).

| Check | Prediction (80 % range) |
|---|---|
| Replacing the four families, new, six bases, best objective | 0.0 (−0.4 to +0.4) |
| P04 as a new person | +1 (−1 to +3) |
| **Lever** | new ≥ +0.3 with the interval above 0, or P04 ≥ +2 with nobody below −1 |

### 94. EPOCHS: are the SessionBN nets trained too short or too long? (written 12 Nov 12:18 UTC, before any job ran; `studies/12nov/netaug.py` tasks 400–423, Galvani)

**Why.** User, 12 Nov: fill the remaining lanes. The leader's jump came with four times the runtime (review §4); our
nets' training length was set in October on a weaker chain and never revisited for the SessionBN recipes. Arms:
training length × 2 and × 0.5 (one-cycle schedule stretched), four recipes, three seeds, both protocols; scored as
replacements against the twins (as §87).

| Check | Prediction (80 % range) |
|---|---|
| × 2, new, six bases | +0.1 (−0.3 to +0.5) |
| × 0.5, new, six bases | −0.3 (−0.8 to +0.2) |
| **Lever** | ≥ +0.3 with the interval above 0 |

| §81 item (12:25 UTC; `runs/ttt3b_score_12nov.log`; seeds per family 6–8, ttt3d still running) | Observed | Verdict |
|---|---|---|
| Nets alone, new, package form `ple20` / `plec20` | +4.0 to +5.7 / +3.4 to +7.1 | replicates §60 |
| Chain, all families swapped, new, six bases, `ple20` | +0.20, −0.05, +0.06, −0.01, −0.02, −0.04: **mean +0.02** | **Refuted** (lever +0.3) |
| Same, `plec20` / `plelr20` | mean +0.16 / +0.14; 5 of 10 people ≥ 0 | below the lever |
| True labels (`ore20`), new | +1.77 to +2.09, 7–8 of 10 people ≥ 0 | the ceiling stands |
| Seen, dev form `pl20` | +0.53 to +0.79 (P04 +1.8 to +2.5) | P04 again |

Reading: with six to eight seeds the three-seed +0.27 of §60 shrinks to about 0.0–0.2. In-session fine-tuning on the
chain's labels is closed for new people; the +2 ceiling with true labels is §83's lesson again (labels on the hard
windows). The script's secondary "one fine-tuned seed" section failed on the uneven seed counts and is not reported.

| §93 item (13:02 UTC; `runs/dro_swap_12nov.log`; four families replaced, against the twins, six bases) | Observed | Verdict |
|---|---|---|
| GroupDRO η 0.1 / 0.01, new | −0.34 [−0.67, −0.01] / −0.22; P04 +0.5 / +0.7; P05–P07 −1.1 to −1.7 | **Refuted** |
| Participant-balanced loss, new | +0.04 [−0.27, +0.33], 7 of 10 ≥ 0; P04 +0.1 | Refuted (lever +0.3; P04 lever +2) |
| Seen | −0.05 to −0.67 | — |

Reading: up-weighting the worst-fitted training people does not reach a new P04-like person (+0.5 at most) and costs
the others. Robust objectives cannot supply what no training person carries (§67, §77). Process note: the first
scoring attempt failed on an unconverted name in the scorer (fixed in `focal_swap.py`) and on a brace pattern in a
remote copy; both now fail loudly, and the result above is from the corrected run.

### 95. DOSE6: the v5.1 stage strength on all six bases (written 12 Nov 13:16 UTC, before any result; `BETA=… studies/12nov/v5_replay.py`)

β 0.10 was chosen on the two exploration bases (§43), which overstate stage effects (§75). Before the first upload,
the dose is re-read on all six bases: β 0.05, 0.075, 0.10, 0.15, 0.20.

| Check | Prediction (80 % range) |
|---|---|
| Best β by six-base seen mean | 0.10 (0.075–0.15) |
| Best minus 0.10, seen | +0.1 (0 to +0.4) |
| **Rule** | the package changes only if another β beats 0.10 by ≥ +0.3 seen with P10 not worse by more than 0.3 |

| §95 item (13:21 UTC; `runs/dose6_*.log`; six-base means) | β 0.05 | 0.075 | **0.10** | 0.15 | 0.20 |
|---|---|---|---|---|---|
| Seen (P04, P10) | +1.18 (+2.31, +0.05) | +1.46 (+2.75, +0.17) | **+1.79 (+3.51, +0.07)** | +1.94 (+3.91, −0.04) | +1.18 (+2.81, −0.46) |

Verdict: hit. β 0.15 is +0.15 above 0.10 (rule: ≥ +0.3), and its extra is P04's. **The package keeps β 0.10.**

### 96. INPOP2: personal nets and the stage together, for ten people (written 12 Nov 13:23 UTC, before any result; code `studies/12nov/inpop3.py`)

**Why.** User, 12 Nov: how can the leaders score so high? The one information source we found that reaches the hard
windows is a person's own labelled data (§88: 78 % against 61 %). Returning people have it. If the sealed set holds
many returning sessions (the withdrawn cohort text said 43 %), a strong personal system is the most plausible place
for a +3 board margin. Dev measures it on two people only; the §82 proxy covers ten (within a session). Here: the
in-population chain with the nets replaced by their personally fine-tuned versions (stored §32 outputs, head-only
`pfh60` and full `pf60`, fitted on runs 1–3), with and without the v5.1 stage.

| Check | Prediction (80 % range) |
|---|---|
| PFT nets + stage against the stage alone, ten people | +1.0 (0 to +2.5) |
| People ahead | 6 of 10 |
| **Reading** | ≥ +1.5 with ≥ 7 of 10 ahead → the personal-net component is built for all ten people now, ready for upload 2 if upload 1 shows the returning branch pays |

| §96 item (13:24 UTC; `runs/inpop3_12nov.log`; B2_pool / B2_s345 / B2_s012) | Observed | Verdict |
|---|---|---|
| Personal nets alone on the in-population chain | pfh60 +1.8 / +2.2 / +1.8; pf60 +2.6 / +3.3 / +2.6 | — |
| Personal nets + stage against the stage alone | **pfh60 +2.6 / +2.4 / +2.4; pf60 +2.7 / +2.5 / +2.5**, 8 of 10 ahead, intervals +1.2 to +3.9 | **Above the range** (predicted +1.0); the reading's mark is met |
| Personal nets + stage against the in-population chain | +5.7 to +6.3 | — |

Reading, with the caveat this proxy has earned: within a session, with the person's first runs in the population,
the stage and personal nets add up (+6 for eight of ten people). On the real later sessions of P04 and P10 the same
pair added +0.2 over the stage (§78). Which of the two the eight single-session sealed people resemble is the one
question only the sealed set answers; upload 1 (v5.1) measures the stage's half of it. By the pre-set reading, the
personal-net component is now built for all ten people so that it is ready for upload 2.

**§96 build (12 Nov):** `scripts/build_pnets.py` fine-tunes each of the 24 shipped SessionBN nets (runs/sbn, trained
on all public windows) for each of the ten people on all their public windows (pf60 and pfh60), Galvani packs
`pnets_a`, `pnets_b`. Fails loudly if the loss does not fall or no parameter changes. Export to TorchScript, the
package code that swaps a recognised person's nets in, and the replay checks follow only if upload 1 shows the
returning branch pays (§63 reading rule).

## 97–98. Second and third systems from first principles (written 12 Nov 13:34 UTC, before any result)

**Why.** User, 12 Nov: stop extending the one chain; design new systems from first principles and score them whole.
What a design has to respect, from the measurements of 11–12 Nov: (1) session adaptation is the largest single
lever (+7 recentring, +3.8 TSEM); (2) population models alone reach 60–69 % for new people; (3) people differ in
structured ways (§67, §77); (4) the hard third of each session is decodable with the person's labels (§88) but not
from the easy windows (§89); (5) every model trained on the same people shares errors (§63). Each system is scored
alone (new people, LOSO; seen where it applies), then as a product with the chain; dev resolution is ±0.5–1.

### 97. S2: a nonlinear multi-view model with its own session stage (code `studies/12nov/sys2.py`)

Inputs per window, all z-scored causally within the session (no state across sessions): 9-band tangent space after
causal recentring (population prior from the training people; 256 principal components fitted on training rows),
per-channel log band power (369), slow time courses (64 components), gaze and event features (48). Model: a
two-layer network (512, 128; dropout 0.5; label smoothing; mixup across training people; five seeds), trained
leave one person out. Session stage of its own: count rebalancing, then TSEM in the network's 128-dimensional
embedding (λ 0.15). No member of the chain is used.

| Check | Prediction (80 % range) |
|---|---|
| S2 alone, no session stage, new | 68 (63–72), core1 69.3 |
| S2 with its own session stage, new | 75 (70–79), chain 80.2 |
| Product with the chain (exponent 0.5), new | +0.3 (−0.5 to +1.2) |
| **Reading** | S2 with stage ≥ 78 → a full second system worth developing; product ≥ +0.8 on six bases → candidate |

### 98. S3: a session-set model that learns the adaptation (Galvani; code `studies/12nov/sys3.py`)

First principles: the chain's stages are hand-made rules for "adapt to this session". S3 learns that rule end to end:
a causal transformer over a session's window embeddings (S2's inputs, projected) predicting each window's class from
the window and all earlier ones. Against §70's failure (a learned stage overfits nine people) it trains on synthetic
sessions: each training episode draws a real session and applies a random spatial rotation of the inputs (virtual
person), a random class-share shift and a random window subset; 20,000 episodes per fold. Pre-registered when the
code is written, before its first job.

**§98 predictions (written 13:35 UTC, before S3's first job; code fixed in `studies/12nov/sys3.py`, inputs `sys3_inputs.py`).** Ten folds × three seeds, 20,000 synthetic-session episodes each.

| Check | Prediction (80 % range) |
|---|---|
| S3 alone, new (balanced accuracy, mean of three seeds) | 72 (62–78); S2 with its hand stage as the comparison |
| Product with the chain (exponent 0.5), new, six bases | +0.2 (−0.6 to +1.0) |
| **Reading** | S3 ≥ S2-with-stage + 2 → learning the adaptation beats hand rules; S3 ≥ 78 → a system worth developing |

| §97 item (13:39 UTC; `runs/sys2_12nov.log`) | Observed | Verdict |
|---|---|---|
| S2 alone, new | **60.5** (P04 40.7, P05 46.0, P10 50.3; best P02 74.2) | **Miss, below the range** (predicted 68, 63–72; core1 69.3) |
| + rebalancing; + TSEM in its own embedding | 61.6; 62.2 (λ 0.15) | Miss (predicted 75) |
| Product with the chain, six bases | −2.5 to −6.0, 0–2 of 10 people ≥ 0 | Refuted |

Reading: a from-scratch multi-view network on 2,400 training windows is 9 points below core1 (a linear model on
44,239 tangent features of 17 overlapping bands, full and half windows) and its own session stage adds only +1.7
(the chain's stages add +10 to core1-level members). Two things the chain has that S2 lacks: the feature richness of
core1 (compressing to 256 components loses most of it, as §54 found at 512) and members whose errors the stages can
average. A smaller network does not substitute for either. S3 runs on S2's inputs, so it measures one thing only:
whether a learned adaptation beats S2's hand stage (62.2) on the same inputs.

**12 Nov afternoon: Galvani access lost (2FA rejected; the MLcloud access may have expired).** Results still on
Galvani and not fetched: §94 training length (finished), §91 TTT4 (finished or nearly), §96 personal nets (built),
§98 S3 (queued). **Deviation from §98, stated before any S3 result:** S3 now runs locally on the Mac CPU through
`scripts/run_capped.sh`, one fold at a time, seed 0 only (about 6 minutes per fold), instead of three seeds on
Galvani; the reading rules stay, with one seed's noise noted.

**12 Nov evening: compute moves to Leonardo (CINECA, EuroHPC), with the user's authorisation to use their two
allocations (AIFAC_P02_1636, AIFAC_P02_1675), balanced between them.** Work folder `$WORK/ni26`; packs via
`scripts/cloud/leo_pack.sh` (one A100 per pack). Re-runs, in order of decision value: §94 training length, §98 S3
seeds 1–2, §91 TTT4, then the §96 personal nets (the 24 shipped SessionBN nets must be retrained first; only 6
checkpoints exist locally). **Change against the Galvani runs, stated before any Leonardo result:** TTT's pseudo-labels
come from the B2_pool v3 replay chain (`studies/12nov/chain_final.npz`, rebuilt locally), not from the Galvani file;
every TTT comparison is against the frozen and `ple20` nets of the same run, so the rules are unchanged.

**§98 bug, found before any S3 result was scored (12 Nov):** the "small rotation" of the synthetic sessions used noise
of 0.15 per entry on a 737 × 737 matrix, which makes Q a random rotation (mean |diagonal| 0.16): every episode
scrambled the feature identities, the model could not learn, and the five local folds that ran sat at chance (25–47 %
window accuracy, loss ≈ ln 3). Fixed to 0.15/√d (mean |diagonal| 0.99) with an assertion that fails loudly if the
rotation is not small. The broken outputs were deleted; S3 restarts from scratch (Leonardo, seeds 0–2).

**12 Nov 14:55 UTC, Leonardo jobs queued (environment checked: torch 2.14.1+cu126, braindecode 1.8.1):** §94
training length `ep_a`, `ep_b` (1675); §91 TTT4 `ttt4_a`, `ttt4_b` (1636); §98 S3 with the rotation fix, seeds 0–2
`sys3` (1636); §96 personal nets `pnets_a`, `pnets_b` (1675; retrains the 18 shipped SessionBN nets whose checkpoints
are not local). Each line is copied home and scored by `scripts/cloud/leo_collect.sh` under its pre-set rule.

## 99. Review of 12 Nov evening and LAPREF (written 12 Nov 15:23 UTC, before any result)

Two independent reviews (literature and internal audit; summary in `docs/REVIEW_12NOV.md` §7) agree: no published
label-free method for new people is missing from the chain; the large unmeasured block is returning people; the only
untested axis with plausible ≥ +0.5 for new people is re-referencing for core1, closed on 30 Oct by an argument
("EA of a re-referenced signal is a rotation") that fails for tangent features of a population classifier — the same
kind of argument that wrongly closed CAR, which later shipped.

**LAPREF.** core1 rebuilt with a 4-nearest-neighbour surface Laplacian (`lap_ov17_fs2_t0.1_le_k2`) and with CAR and
Laplacian blocks concatenated (`cl_…`), heads lr10 as shipped; leave one person out and seen. Swapped into core1's
slot (posterior only; TSEM's view and the routing features keep the shipped core1 — a known limit, §52), routed chain,
six bases, person bootstrap.

| Check | Prediction (80 % range) |
|---|---|
| core1 alone, new: lap / cl (shipped car 69.3) | 67 (63–71) / 70 (67–73) |
| Chain with the swapped slot, new, six-base mean | lap −0.3 (−1.2 to +0.6); cl +0.2 (−0.5 to +0.9) |
| **Lever** | ≥ +0.5 new with the interval above 0 → rebuild the TSEM view and the routing features with the new core and confirm |

| §99 item (16:12 UTC; `runs/lapref_score.log`) | Observed | Verdict |
|---|---|---|
| Harness: shipped core1 rebuilt against the stored one | max difference 7e-9 (LOSO), 0 (seen) | the build path is the shipped one |
| core1 alone, new: lap / cl / shipped car | **58.9 / 66.2 / 69.3** | Miss, below (predicted 67 / 70) |
| Chain with the swapped slot, new, six-base mean | lap **−1.07 [−1.49, −0.68]**, 2 of 10 ≥ 0; cl **−0.48 [−0.77, −0.20]**, 4 of 10 ≥ 0 | **Refuted** |
| Seen | lap −0.5 to −2.1; cl −0.4 to −2.1 (P04 down to −4.2) | — |

Reading: the 30 Oct closure by argument was wrong in kind but right in outcome. A surface Laplacian removes the
broad spatial patterns core1 classifies on (69.3 → 58.9), and adding it next to CAR doubles the dimension without
new information (66.2). Re-referencing is now closed on evidence.

**Leonardo, 16:10 UTC:** the first resubmission ran; the SessionBN SincShallow tasks of §94, §96 and §91 failed
because `runs/pretrain` had not reached Leonardo (the copy that carried it timed out). Copied and resubmitted:
`ep_sinc`, `pnets_sinc`, `ttt4_sinc`. The FBCNet, raw FBCNet and FBLight outputs of §94 and the 360 personal-net
files are in.

### 100. PCROP: more training rows for the personal models from shifted windows (written 12 Nov 16:39 UTC, before any result; code `studies/12nov/pcrop.py`)

Both reviews (REVIEW_12NOV §7): the personal models learn from about 190 windows; shifted crops (`start0.5`, 0.5 s
earlier, never used anywhere; `start2`, 1 s later) were only tried for the population core. Here the own-space T
and C members are fitted on the public windows plus their shifted copies (each copy recentred as its own session);
S (slow, cue-locked) stays on the original windows. v5.1 replay, seen protocol, six bases, against v5.1.

| Check | Prediction (80 % range) |
|---|---|
| + start0.5 crops, seen, six-base mean | +0.2 (−0.3 to +0.7); P04 +0.4, P10 0.0 |
| + start0.5 and start2 crops | +0.1 (−0.5 to +0.7) |
| **Lever** | ≥ +0.3 seen with P10 ≥ −0.3 on ≥ 4 of 6 bases → part of the next candidate |

| §100 item (16:45 UTC; `runs/pcrop_05_12nov.log`, `runs/pcrop_052_12nov.log`) | Observed | Verdict |
|---|---|---|
| + start0.5 crops, seen, six bases | −0.53 to +0.41; **mean −0.04** (P04 −0.08, P10 −0.01) | **Refuted** (1 of 6 bases) |
| + start0.5 and start2 | −0.26 to +0.54; **mean 0.00** | Refuted |

Reading: more rows from shifted copies of the same trials do not help the personal models; the copies carry the same
trials' information, and the 190-window limit is a limit of trials, not of rows.

| §98 item (16:45 UTC; Leonardo pack 64284047, seed 0, after the rotation fix) | Observed | Verdict |
|---|---|---|
| Training loss at 16,000 episodes | 0.29 (from 1.10) | the model fits the synthetic sessions |
| Window accuracy on the held-out person, ten folds | **5–28 %** (P02 5 %, P08 5 %; best P05 28 %) | **Refuted**, and below chance (33 %) |

Reading: far below chance is not noise: on a new person the model systematically picks a wrong class. It learned
something that holds inside the training people's sessions and inverts on a new one (most likely the sequence
structure of the runs: a causal model over a session can exploit the cue order and class-count patterns of the
training sessions). Not debugged further: even a corrected version would have to beat the hand stages' +10 on a
member mix, and S2 showed the inputs are 9 points weaker than core1. Seeds 1–2 and the rest of the pack were
cancelled to free the lanes.

**16:48 UTC: framing check and the upload candidate.**
- Dev split by session index (review §7, B2_pool / B2_s345): as new people, P04's sessions score 56, 55, 50 and P10's
  77, 86, 82. Later sessions are easier for P10 and harder for P04: no general "later sessions are easier" effect
  on the two people dev can show.
- v5.1 rebuilt locally from the stored v4.1 package (`scripts/add_own_stage.py runs/sealed/cand_v41_bio
  runs/sealed/cand_v51_own beta=0.10`): `submission.py` 34eba3b0… as built on Galvani; `weights.npz` 6a1f35c1…
  instead of Galvani's e88bc3b3… (sklearn fits on macOS against Linux; the Galvani copy is out of reach). The local
  build is a new package and gets its own checks.
- **v5.1b** (`submissions/ensemble_v5b/submission.py`, SHA-256 0259d1ea…): v5.1's code plus one stderr line per
  session from the routed tail (windows, routed, degenerate, staged, errors, first error), so a silent no-op of the
  stage on the sealed run shows in the logs. Outputs must equal v5.1's; checked by `scripts/check_own_stage.py`
  against v4.1 on all public windows.

| §94 item (17:27 UTC; `runs/ep_swap_12nov.log`; Leonardo, four families replaced, against the twins) | Observed | Verdict |
|---|---|---|
| Training × 2, new, six bases | **+0.00 [−0.17, +0.19]**, 3 of 10 ≥ 0; seen −0.32 | **Refuted** (predicted +0.1; lever +0.3) |
| Training × 0.5, new | −0.60 [−0.94, −0.27]; seen −0.81 | Hit (predicted −0.3) |

The October training length is right; more training does not help the SessionBN nets.

**v5.1b integration check (17:27 UTC; `runs/sealed/cand_v51b_check.json`, local Mac):** equal to v4.1 on every
unstaged window, members equal to the study class (max difference 6e-14), output difference to the formula 0.0,
27 labels changed in sample (the same count as the Galvani build of v5.1), staged share 0.95 per routed session; the
new stderr line prints once per session (14 sessions). **Passed.** §96 personal nets: all 480 files built.

**v5.1b official replay on the Mac (18:44 UTC; `runs/sealed/cand_v51b_own_replay_mac2/replay.json`):** the warm-up phase's
ingestion and scoring programs on the package ZIP (sha256 4a0bc427…, 62.3 MB): files unchanged, 91.9 s on CPU,
**score 0.9932767697 — identical to the Galvani A100 and worker-image replays of the Galvani-built v5.1** (12 Nov
morning). The four warm-up sessions each print one `[own-stage]` line (windows 188–189, routed and staged 178–179,
degenerate 0, errors 0). Leonardo became unreachable at about 18:40 UTC (every login node times out, also for the
user); the A100 replay and the fetch of §91 wait for it. All polling loops were stopped.

| §91 item (18:50 UTC; `runs/ttt4_score_12nov.log`, second run; Leonardo; seeds fb/fbl/fbraw 6, sinc 5) | Observed | Verdict |
|---|---|---|
| Nets alone, new: ple20 / fix20 / ent20 / cons20 | +4.0 to +5.6 / −0.4 to +4.5 / +3.8 to +4.8 / −1 to **−25** (consistency alone collapses fbraw and fb) | — |
| Chain, all families swapped, new, six bases | ple20 −0.01 to −0.23 (mean −0.07); **fix20 +0.33, +0.31, +0.40, +0.01, +0.42, +0.31 (mean +0.30), every interval crossing 0**; ent20 −0.29; cons20 −0.23 | **Refuted** by the rule (≥ +0.4 with the interval above 0); fix20 is the best new-person arm of 12 Nov but inside the noise |
| Seen, fix20 | −0.5 to −1.3 | it would cost returning people |

Process note: the first scoring of §91 (18:58, from the collector after a failed copy) found no files and printed
+0.00 for every arm instead of failing. `ttt3_score.py` now asserts that it found something to score.

### 101. FIX2: refinements of the augmented-input fine-tuning (written 12 Nov 19:09 UTC, before any job ran; code `studies/12nov/ttt5.py`, Leonardo) [review 13 Nov: base counts here carry almost no information, read the person uncertainty in §139]

User, 12 Nov: follow up the best new-person arm of the day (§91 fix20: +0.30 on six bases, every interval crossing 0;
seen −0.5 to −1.3). Package form throughout. In a package it would run only on sessions the identity gate does not
route (new people), so it is scored for new people with seen left at the frozen nets. Variants: fixA (fix20 again,
the replicate), fixW / fixS (augmentation shift 5 / 30 samples, noise 0.1 / 0.3, channel dropout 0.05 / 0.2), fixPL
(pseudo-labels on augmented inputs, no consistency), fixT (only windows with chain confidence ≥ 0.6), fix40 (40
steps), fixLR (lr 2.5e-4). Four SessionBN recipes, seeds 0–2.

| Check | Prediction (80 % range) |
|---|---|
| fixA against frozen, new, six bases | +0.25 (−0.1 to +0.6): the replicate |
| Best other variant, new | +0.35 (0 to +0.7) |
| **Lever** | a variant ≥ +0.4 on six bases with the person-bootstrap 10th percentile above 0, *and* its seeds 0–2 agreeing with fix20's seeds 0–5 within 0.2 → a gated v5.2 component (fine-tuning only on unrouted sessions; runtime checked first) |

### 102. v5.2 preparation: personal nets as package members (written 12 Nov 19:28 UTC)

Not an experiment and not a candidate yet: by §63 a v5.2 is built only if the v5.1b upload shows the returning block
pays (v5.1b − v4.1 ≥ +0.4). To be ready within a day if it does: `scripts/export_pnets.py` traces the 24 × 10
personal SessionBN nets (pfh60, §96) at the package's k values (300 traces, about 80 MB) on Leonardo, each checked
against the eager personal net on the person's own sessions (< 1e-4). Package code (swap the recognised person's
traces in after the gate routes, warming their session statistics on the buffered windows) is written only after the
measurement. Dev value so far: +0.17 seen over the stage (§78), within-session proxy +2.4 (§96).

**§101 addendum (12 Nov evening, before any FIX2 result): seeds 3–5 (tasks 12–23) queued in parallel (`ttt5_c`,
`ttt5_d`), scored together with seeds 0–2 under the same rule, so a passing variant comes with six seeds.**

### 103. Under-explored axes from the reviews (written 12 Nov 19:38 UTC, before any result)

User, 12 Nov: widen the exploration inside the solution space. Three axes the reviews named that were never run on the
current chain, each local and cheap:

**103a. THIRDS: core1 with the window split in three** (`car_ov17_fs3_t0.1_le_k2`; shipped: full + two halves).
Swapped into core1's slot (posterior only), routed chain, six bases. Prediction: core1 alone 69.5 (68–71); chain new
+0.1 (−0.3 to +0.5). Lever ≥ +0.3 with the interval above 0.

**103b. IFACE: the interface as a session-stage variable** (oracle: the true interface label, which a package would
have to detect from the run structure). Arms: rebalancing kept separately per interface inside a session; teacher
weight per interface (3 / 5 / 7 each, chosen on one people half, scored on the other). Grid chain without routing,
new people, six bases. Prediction: per-interface rebalancing −0.1 (−0.5 to +0.3); per-interface teacher weight,
cross-fit, +0.1 (−0.3 to +0.4). Reading: an oracle below +0.5 closes the interface axis for good.

**103c. PROC: a labelled Procrustes oracle** (closes the Riemannian alignment family on evidence). In core1's 256
components (session-standardised), a shrinkage LDA trained on the other people; for each test session the class means
computed **with true labels** are rotated onto the training class means (orthogonal Procrustes) before classifying.
Prediction: LDA alone 66 (62–70); with the oracle rotation 74 (68–80). Reading: if the oracle gains < +5 over LDA
alone, a label-free rotation (which can only approximate it) is closed; if ≥ +8, a label-free version on TSEM's
templates is pre-registered.

**Throughput (12 Nov, 21:10 CET):** FIX2 (§101) re-launched as Slurm job arrays, one task per A100 (jobs 64304502 64304503),
instead of six tasks sharing one GPU slot's 8 cores (5–6 h per task); the 50 minutes already run were discarded, no
result had been written. Scoring moves to Leonardo too (`scripts/cloud/leo_cpu.sh`, a booster slot per scoring run),
once `replay/check.py` prints OK there.

| §103 item (20:11 UTC) | Observed | Verdict |
|---|---|---|
| a. THIRDS: core1 alone, new (shipped 69.3); chain, six bases | 68.6; **−0.62 [−0.89, −0.36]**, 2 of 10 ≥ 0 | **Refuted** |
| b. IFACE: rebalancing per interface; teacher weight per interface (cross-fit) | **−0.41 [−0.70, −0.12]**; **−0.01 [−0.20, +0.18]** (best in-sample +0.00 to +0.91) | **Refuted**; the interface axis is closed with an oracle below +0.5 |
| c. PROC: LDA in core1's 256 components; with the labelled Procrustes rotation | **64.6 → 89.1 (+24.4)**, every person +16 to +40 | Above the range (predicted 74); the reading's mark (≥ +8) is met |

Reading of c, with its caveat: the oracle uses the session's true class means computed over the whole session,
including the windows it then classifies, so most of +24 is "knowing the class means of this session". The question
it opens is narrower: does rotating the session into the population frame with *estimated* class means beat what
TSEM already does with the same estimates?

### 104. PROC2: Procrustes alignment with causal and label-free class means (written 12 Nov 20:11 UTC, before any result; code `studies/12nov/proc2.py`)

Per new-person session, from window 15 on: class means of the earlier windows only (causal), from (i) their true labels
(causal oracle), (ii) the chain's labels on its most confident windows (TSEM's selection rule); orthogonal Procrustes
rotation of the session's standardised 256 components onto the training class means; population LDA; the window's
posterior. (ii) is fused into the routed chain as a final member (exponent 0.25 / 0.5), six bases.

| Check | Prediction (80 % range) |
|---|---|
| (i) causal true-label means, alone, new | 78 (70–85) |
| (ii) label-free means, alone, new | 66 (62–70), LDA alone 64.6 |
| (ii) fused with the chain, new, six bases | 0.0 (−0.5 to +0.4) |
| **Lever** | ≥ +0.4 with the interval above 0 on six bases |

**Process note (§101, 21:20 CET):** the collector scored FIX2 while 20 of 24 array tasks were still running (only
FBLight's three seeds had landed): `squeue -j` on an array's master id stops matching once its tasks start, and a
failed squeue also read as "finished". That partial score (one family) is not a result and is not used. The collector
now waits by job name and treats a failed squeue as "still running".

| §101 item (21:14 UTC; `runs/ttt5_score_12nov.log`, full run; six seeds per family, one task per A100) | Observed | Verdict |
|---|---|---|
| fixA (fix20 replicated), new, six bases | **+0.36** (+0.15 to +0.49 per base); fix20 earlier +0.30 | replicates (within 0.2) |
| fixS / fixLR | **+0.37** (+0.24 to +0.56) / **+0.36** (+0.04 to +0.55; per-base interval above 0 on 3 of 6) | best arms |
| fixW / fix40 / fixPL / fixT | +0.27 / +0.19 / +0.01 / −0.18 | the consistency term matters (fixPL ≈ 0); confident-only labels hurt |
| Seen (would be gated off in a package) | −0.7 to −1.8 | — |
| **Lever** (≥ +0.4 with the interval above 0) | best mean +0.37 | **Not met, narrowly** |

Reading: augmented-input fine-tuning with the consistency term is a small, *replicated* gain for new people of about
+0.35 (two runs, three variants, six seeds, positive on every base), about +0.2 on the board if new people are 57 %
of cells. It misses the pre-set bar by 0.03–0.04 and costs 20–40 minutes of runtime per sealed run; under §52 it is
recorded as "positive, small, replicated, not built". A package would run it only on sessions the gate does not route.

### 105. Sealed expectation for v5.2 over v5.1b (written 12 Nov 21:43 UTC, before any upload of v5.2 is discussed) [review 13 Nov: its dev basis (§78/§119) is not the package form [§140]]

Inputs: upload 5 measured v5.1b − v4.1 = +0.361 (0.8197668759); with the assumed cohort (106 sessions, 46 returning,
3 of them P04's) the non-P04 returning sessions gained about +0.6 each, while the within-session proxy had promised
+3.2 for the stage (§82): a survival ratio of about 0.19 from proxy to sealed. v5.2 adds personal nets on routed
windows: +2.4 within session on the proxy (§96), +0.17 on the real later sessions of P04 and P10 (§78).

| Quantity | Prediction (80 % range) |
|---|---|
| v5.2 − v5.1b, sealed | **+0.20 (−0.10 to +0.50)**, i.e. a sealed score of about 0.822 (0.819 to 0.825) |
| Sealed runtime | about 1.4× v4.1's 18.8 min, 27 min (worst case 32 min) against the 60-minute limit |
| **Reading rule** | ≥ +0.15 → v5.2 is the base of the final package; ≤ 0.00 → the final package builds on v5.1b; in between → v5.2 stays the base only if nothing else is ready |

**Profile of the v5.1 replay (B2_pool; 21:45 UTC; `runs/profile_12nov.log`), for the research window.** New people
(LOSO): 76.6 % of windows right; interfaces equal (BrainHero 76.7, Graz 76.5); run types 75.4–77.3; by fifth of the
session 73.1, 80.5, 76.5, 77.6, 75.1 (the first fifth is the warm-up loss of §90; the last fifth dips); classes MI
81.0, SUB 75.2, WORD 73.5; P04 54 against 75–90 for the others. Returning (seen): 82.9 %; Acquisition-Graz 87.5 against
Online-Graz 79.2 (feedback runs are harder for returning people); WORD the weakest class in both.

**Runtime as a limit (user's question, 12 Nov):** the 3,600 s evaluation limit is official (competition YAML;
OFFICIAL_CONTRACT). The package is `submission.py` plus data files, inference-only; the worker image has no C/C++
compiler, nvcc or triton launcher, so own compiled code cannot be built there and a prebuilt binary cannot be
verified on the hidden worker. Within those rules the cost of in-session fine-tuning is an engineering question:
the v5.3 research (v52-build session) first measures torch-native batching of the 30 nets' fine-tuning (torch.func
vmap per family, CUDA graphs) on an A100 before any recipe is cut.


**Review note on §105 (13 Nov 00:42 UTC):** its dev input "+0.17 on the real later sessions (§78)" was not measured in
the package form (see the note under §78; package form +0.39, or +0.02 with the shallow member). The prediction stands
as written, with weaker support than stated. **Second review (evaluation path, 13 Nov ~01:00 UTC, confirmed): with
the personal nets only from the first routed window (window 10; 60–63 for P04 session 2), in the routed tail, and the
gate on the population shadow, which is what the package does, §119's +0.55 becomes −0.05 (P04 −0.10, P10 0.00). The
package-form dev support for v5.2 is about zero; §105's +0.20 needs restating on that basis (main session).**

### 106. V53: in-session fine-tuning for unrouted sessions, runtime-feasible forms (written 12 Nov by 21:49 UTC [stamp corrected from the commit time; it read 22:20 UTC], before any result; code `studies/13nov/ttt_reduce_score.py`, v52-build session) [review 13 Nov: base counts here carry almost no information, read the person uncertainty in §139]

User, 12 Nov: build v5.3 = v5.2 + the §101 fine-tuning on sessions the identity gate does not route, within ≤ 45 min
sealed (worst case). The full §101 form fine-tunes all 30 SessionBN members (20 Adam steps every 10 windows from window
30). Its cost on an A100 is being measured (`studies/13nov/ft_bench.py`, directly on the package's TorchScript traces).
Before any cut is built, each cut is scored in the chain (new people, six bases). Part A needs no new training, because
the stored per-seed outputs of §101 (`runs/ttt/probs5`, six seeds per family) give these forms per family:
`delta1` = frozen six-seed mean + the change of ONE fine-tuned seed (one fine-tuning per family, its change added to all
members of the family); `delta2` = the same with two seeds; `one6` = one fine-tuned seed next to five frozen ones; `all6` =
§101's arm (the reference, +0.36 / +0.37 / +0.36). Variants fixA, fixS, fixLR. Each seed choice is scored, and the
mean over the choices is reported.

| Check (new people, six-base mean, mean over the seed choices) | Prediction (80 % range) |
|---|---|
| fixA all6 (replicates §101 exactly; a code check) | +0.36 |
| fixA delta1 | +0.22 (0.0 to +0.40) |
| fixA delta2 | +0.28 (+0.05 to +0.42) |
| fixA one6 | +0.06 (−0.05 to +0.15) |
| **Rule** | the package uses the cheapest form whose six-base mean is ≥ +0.25 with ≥ 5 of 6 bases positive and whose measured A100 cost keeps the sealed worst case ≤ 45 min; if no reduced form passes, all6 if it fits the budget, otherwise the best form that fits, stated as below the rule |

Part B (pre-registered separately, before it runs, if needed): fewer refits or steps cannot be scored from stored
outputs and need new Leonardo runs of `ttt5.py` variants. Part B runs only if the A100 cost of the passing Part A form
does not fit the budget.

| §106 item (22:16 UTC; `runs/v53_reduce_score.log`; mean over seed choices, six bases) | Observed | Verdict |
|---|---|---|
| fixA / fixS / fixLR all6 | +0.36 / +0.37 / +0.36 | equals §101 (code check passed) |
| fixA delta1 | **+0.26** (bases +0.12 to +0.38, 6 of 6 positive; single seed choices −0.09 to +0.63) | Hit (+0.22, 0 to +0.40); **passes the rule** |
| fixA delta2 | +0.34 (+0.23 to +0.53, 6 of 6) | Hit (+0.28) |
| fixA one6 | +0.15 (−0.05 to +0.31, 5 of 6) | Miss, high (+0.06, up to +0.15) |
| fixS delta1 / delta2 / one6 | +0.26 (6 of 6) / +0.20 / +0.12 | — |
| fixLR delta1 / delta2 / one6 | +0.25 (6 of 6) / +0.28 / +0.10 | — |

Reading: one fine-tuning per family, with its change added to all six seeds of that family, keeps about 70 % of the gain
(+0.26 of +0.36). Two per family keep about 95 % with fixA, though not with fixS. The cheapest form passing the rule is
delta1: four fine-tunings per refit instead of 30. Which seed is chosen matters (single choices range from −0.1 to
+0.6 per base), so a package would average over that choice, which delta2 does. Seen people lose 0.4 to 1.8 in every
form, so the gate stays.

### 107. FTCOST: can the full §101 recipe be made cheap? (written 12 Nov by 21:54 UTC [corrected from the commit time; it read 22:55 UTC], before the run; code `studies/13nov/ft_bench2.py`, Leonardo A100)

User rule (relayed 12 Nov): research only until about 14 Nov 02:00 UTC, no package. Upload 5 used 21.5 of 60 min, so
about 38 min are free. First bench (`ft_bench.py`, job 64311968): fine-tuning through the TorchScript traces costs
20/58/99 ms per step for FBCNet at 30/110/190 windows, 8/10/13 ms for FBLight and 9/27/46 ms for SincShallow; CUDA-graph
capture fails on FBCNet. That bench trained the frozen FIR banks too (requires_grad forced on) and ran the clean forward
with grad, so it overstates the cost. Projected naively, the full recipe (30 nets, 16 refits per 190-window session)
costs about 6.5 min per unrouted session. Second bench: dev's trained parameters only, the clean forward without
grad, then (a) one trace, (b) one eager dev net carrying the trace's weights (equal to the trace: max |Δp| 0.0 on 40
real windows, all four families, Mac), (c) a family's six seeds in one step with torch.func (stack_module_state +
functional_call + vmap, one augmentation per seed).

| Check (A100, per Adam step) | Prediction (80 % range) |
|---|---|
| trace1 against the first bench, FBCNet at t = 110 | 0.4× (0.2–0.7×) |
| vmap6 against six × eager1, FBCNet at t = 110 | 0.35× (0.15–0.8×) |
| Projected full recipe with vmap per family (12 fb, 6 fbraw, 6 fbl, 6 sinc), per 190-window session | 60 s (25–150 s) |
| **Reading** | ≤ 25 s per unrouted session (60 sessions within about 25 min) → the full recipe is runtime-feasible and §106's cuts are not needed; otherwise the cheapest form passing §106's rule is costed with the same bench |

| §107 item (22:29 UTC; Leonardo A100 jobs 64312661 bench2, 64314298 bench3; seconds per Adam step, t = 30 / 110 / 190) | Observed | Verdict |
|---|---|---|
| trace1 (dev's trained parameters, clean forward without grad), FBCNet | 0.046 / 0.056 / 0.069: 0.95× the first bench at t = 110 | Miss (predicted 0.4×): the traces carry a fixed ~37 ms per step (TorchScript executor) |
| eager1, CPU augmentation as in dev / GPU augmentation | FBCNet 0.016 / 0.013 at t = 110; FBLight 0.016 / 0.006; SincShallow 0.032 / 0.031; the CPU augmentation alone 3–15 ms | eager is 3–4× cheaper than the trace |
| vmap over six seeds against six × eager1 (GPU augmentation), t = 110 | FBCNet 0.075 against 0.076; FBLight 0.027 against 0.038; SincShallow 0.211 against 0.185 | Miss (predicted 0.35×): no gain; the A100 is compute-bound at these sizes, not idle |
| Projected per 190-window session (16 refits, eager, GPU augmentation) | full recipe, 30 nets: **about 142 s**; delta1 (4 nets): **about 20 s**; delta2: about 39 s | Reading: the full recipe is **not** runtime-feasible (> 25 s); delta1 is |

Reading: batching does not make the full recipe cheap, because each FBCNet or SincShallow step already fills the GPU.
At 60 unrouted sessions the full recipe would cost about 140 min. delta1 (§106: +0.26) costs about 20 min, or about 35
min if all 106 sessions went unrouted. That is too much next to v5.2's own GPU cost (§112), so a package needs a cheaper
schedule as well: §108's cum10 (half the steps) or a refit every 20 windows. A package would also need eager networks
(braindecode is in the official worker image's requirements) and on-device augmentation (a different random stream from
dev, to be scored as its own arm).

### 109. RUNTYPE (explore track; first numbered 106, which V53 already used): calibration and feedback runs as separate session domains (written 12 Nov 21:57 UTC, before any result; code `studies/12nov/runtype.py`)

Reasoning from the profile: returning people are right on 87.5 % of Graz calibration windows but 79.2 % of Graz
feedback windows; in feedback runs the participant watches a moving display driven by an online decoder, which adds
visual and error-related activity to every window. The session stages pool both run types into one set of statistics.
Oracle (true run type; in a package the switch would have to be detected, e.g. as a change point in the window
statistics): rebalancing, and separately SessionEM + TSEM, kept per run type inside each session. Grid chain without
routing, six bases; new people.

| Check | Prediction (80 % range) |
|---|---|
| Rebalancing per run type, new | −0.2 (−0.6 to +0.2) |
| SessionEM + TSEM per run type, new | −0.3 (−0.9 to +0.3): fewer windows per statistic |
| **Reading** | an oracle below +0.5 closes the run-type axis; ≥ +0.5 → a change-point detector is pre-registered |


### 108. TTT6 (exploit track): arms that may raise the gain of the §101 fine-tuning (written 12 Nov by 21:59 UTC [corrected from the commit time; it read 23:20 UTC], before any job ran; code `studies/13nov/ttt6.py`, Leonardo, 24 tasks) [review 13 Nov: base counts here carry almost no information, read the person uncertainty in §139]

User, 12 Nov: spend the
free sealed runtime (about 38 of 60 min) on gain, not only on cost. Same nets, folds, augmentation and chain labels as
ttt5, four SessionBN families, seeds 0–5. Each arm is scored in the chain (new people, six bases) against the same
run's frozen nets; `fixA` is the in-run reference.

| Arm | What changes | Prediction against frozen, new, six-base mean (80 % range) |
|---|---|---|
| fixA | §101 again | +0.36 (+0.25 to +0.45) |
| cum10 | each refit continues from the previous one, 10 steps (half the cost) | +0.38 (+0.15 to +0.55) |
| cons2 | consistency weight 2 | +0.38 (+0.15 to +0.55) |
| soft | CE against the chain's probabilities (not its argmax × confidence) | +0.30 (+0.05 to +0.50) |
| step5 | a refit every 5 windows | +0.40 (+0.20 to +0.55) |
| head | only SessionBN affine + output layer trained | +0.25 (0 to +0.45) |
| view2 | two augmented views per step | +0.40 (+0.20 to +0.55) |
| **Rule** | an arm counts as a gain over fixA if it is ≥ +0.08 above fixA in the six-base mean and ahead on ≥ 5 of 6 bases; arms that cost less (cum10, head) count if they are within 0.05 of fixA; the best counting arm becomes the recipe for the costing in §107 |

### 110. CLUSTX (explore track): experts for person types, bounded by an oracle first (written 12 Nov 22:11 UTC, before any result; code `studies/12nov/clustx.py`)

Reasoning: people differ in structured ways (§67: transfer is symmetric; P01, P02, P07, P08 serve each other at
56–69 %; P04, P05, P10 are isolated). A single model pooled over everyone is a compromise; one expert per person type
could fit its members better, if a new session's type can be recognised. Experts = own-space T·C·S members (as §37)
trained on the leave-one-out cluster members: type A {01, 02, 07, 08}, type B {03, 06, 09}, and the pooled nine.
Per new session: (i) oracle = the expert with the best true accuracy; (ii) label-free = the expert agreeing most with
the chain; each fused into the routed chain as a final stage (β 0.10), six bases.

| Check | Prediction (80 % range) |
|---|---|
| Oracle best expert alone minus pooled alone, window accuracy | +3 (0 to +6) |
| Label-free pick fused, new, six bases | −0.2 (−0.8 to +0.3) |
| **Reading** | oracle < +2 → person-type experts closed; label-free ≥ +0.4 with the interval above 0 → a whole-system version is pre-registered |

### 111. SEEN-DOSE under the sealed evidence (explore track; written 12 Nov 22:12 UTC, before any result; code `studies/12nov/seendose.py`)

Reasoning: upload 5 showed the own-space stage pays about +0.6 per non-P04 returning session on the sealed set, so the
returning block is the one measured lever. Which form of the personal stage is strongest is still chosen on dev's
two people. Here the ten-person proxy with the person in the population (§82, runs 1–3 public, runs 4–6 later) is
used as the ruler for forms the proxy can rank: stage strength β 0.05–0.40, KREF 1/2/4 (prior weight of the public
mean in the recentring), the members used (T, C, S alone and together), and the personal nets on top (§96). The proxy
over-promises in size (§82 against upload 5: 0.19×), so only the ranking is read, and only forms that are also
≥ 0 on the real later sessions of P04 and P10 (v5.1 replay, six bases) are kept.

| Check | Prediction |
|---|---|
| Best proxy form against v5.1's (β 0.10, T·C·S, KREF 2) | +0.5 within session (0 to +1.2) |
| Same form on P04 / P10 real later sessions | P04 +0.3, P10 −0.2 |
| **Reading** | a form ahead on the proxy by ≥ +0.5 and ≥ 0 on both real people goes into the next candidate's design |

**§105 correction (22:21 UTC): the runtime estimate for v5.2 does not hold.** Leonardo A100, same node, official replay of
the public warm-up sessions (95 % of windows routed): v5.2 383.1 s against v5.1b 97.7 s, **3.9×** (the Mac CPU pair
said 1.36×; on CPU the tangent path dominates and hides the extra net calls, on GPU it does not). v5.1b took 1,287 s on
the real sealed worker; if the extra cost scales with routed windows, v5.2 would take about 48 min with 43 % of
sessions routed and about 84 min if all were: **over the 60-minute limit in the worst case.** v5.2 is not uploadable as
built. Profiling is running (`scripts/profile_v52.py`, job 64314420); the likely cost is the personal path running
30 traces window by window on the critical path, after the base nets, without the base path's overlap.


### 112. V52FAST: profile the v5.2 GPU runtime and bring it down (written 12 Nov 22:24 UTC, before the profile; code `studies/13nov/profile_v52.py`, v52-build session)

User, 12 Nov (via the top session): profile the v5.2 runtime and bring it down. Leonardo job 64311532, one A100 node:
the official GPU replay took 383.9 s for `cand_v52_pnets` against 97.8 s for `cand_v51b_own` (3.9×; on the Mac CPU 1.36×).
In the same job `check_v52.py --device cuda` streamed 2,655 windows in 383.1 s against 229.5 s (1.67×). The check
failed only on its 1e-4 tolerance for the personal posteriors (main 2.8e-4, switch 2.6e-4), with a reference that runs
on the CPU; every other condition passed. The profile (official `predict_streams` + loader on CUDA, CUDA-synchronised
component timers, cProfile) comes first. The fix is chosen from it.

| Target | Value |
|---|---|
| Paired official GPU replay, new build / `cand_v51b_own`, one node | **≤ 1.3×** |
| Outputs against v5.2 on the same device | identical labels on all public windows; personal member posteriors within 1e-5 (same arithmetic, other scheduling) or, if the arithmetic is batched differently, within the GPU-vs-CPU spread of the shipped base traces, stated with that number |
| Sealed worst case (v5.1b's 1,287 s on the real worker × the replay ratio) | stated for 43 % and 100 % routed sessions |
| Prediction | the 30 personal traces at batch size 1 cost launch overhead, not compute: personal-trace time ≈ 60–80 % of the extra; overlapping or batching them reaches 1.2–1.5× |

| §112 item (23:18 UTC; A100 jobs 64315681 profile, 64316388 ABBA, 64316760 check, 64316761 replay series) | Observed | Verdict |
|---|---|---|
| The 3.9× | a cold start: the first replay in a fresh job pays about 230 s of neuralbench/benchopt setup (`/tmp/neuralbench`); v5.1b alone, run first, took 325.1 s (64302270), and v5.2 ran first in 64311532 | the 3.9× was a measurement artefact |
| Paired ABBA after a throwaway warm-up replay | v5.1b 96.4 / 99.0 s, v5.2 137.9 / 136.6 s: **1.40×** (+53 ms per routed window) | — |
| Profile, official `predict_streams` on CUDA (v5.2) | predict 98.9 ms per window against 82.2; PersonalNets.proba 15.6 ms per routed call, 43 ms for the first (warm-up) call; Route3.step 34.8 against 18.9 ms; model deepcopy about 0.1 s per session | personal traces about half the extra (prediction 60–80 %: about right) |
| Fix: `submissions/ensemble_v52b` | the main thread runs the last routed person's traces for the current window while the CPU tangent thread is still working (`PersonalNets.prefetch`, only when those traces are exactly one window behind, so the arithmetic is v5.2's) | — |
| v5.2b against v5.2, official streaming, all 2,655 public windows | CPU: identical labels, max posterior difference **0.0**, 711 of 715 test-split calls prefetched; GPU: identical labels, **0.0**, 2,501 of 2,515 | equal (tolerance 0 met) |
| Paired GPU replay series (warm-up, v5.1b, v5.2b, v5.2, v5.2b, v5.1b) | 97.9; 98.2; 126.7; 146.8; 132.8; 105.7 s: **v5.2b 1.27× v5.1b** (v5.2 1.44× in the same series) | **Target met** (≤ 1.3×), narrowly; node noise about ±5 % |
| Sealed estimate (1,287 s × ratio; the replay sessions are 95 % routed windows) | v5.2b about **27 min** if every session is routed, about **24 min** at 43 %; v5.2 about 31 / 25 min | — |
| Why `check_v52.py` failed on GPU | the personal posteriors against a reference run as one CPU batch per session: 2.8e-4 (main), 2.6e-4 (switch), equal to the digit with deterministic algorithms (64314422). Shipped base traces, GPU against CPU on real streams: 5e-5; personal traces 4.7e-5 as a batch, 1.9e-4 window by window (64315682). On the Mac, a reference with the package's own batch schedule gives **0.0** and one batch 3.5e-5 | float32 numerics of batch shape and device, not logic. GPU `check_v52.py --device cuda --ref-schedule` on v5.2b (64317874): **V52 CHECK OK**, personal posteriors 5.2e-6 (main) and 2.5e-5 (switch), mixes ≤ 8e-7, all other conditions as before |

Not done: worker-image replay (needs Galvani; the image cannot be built on Leonardo).

**Explore results (22:42 UTC; Leonardo `runs/leo_cpu/`).**

| Item | Observed | Verdict |
|---|---|---|
| §104 PROC2: causal true-label means / label-free means, alone, new | 68.7 / 63.8 (LDA alone 64.6) | see below |
| §104 label-free Procrustes fused with the chain, six bases | −2.6 to −3.3 at exponent 0.25, −5.0 to −5.8 at 0.5; 0–3 of 10 people ≥ 0 | **Refuted**: rotating with estimated class means misleads; the §103c +24 was the true labels |
| §109 RUNTYPE: rebalancing per run type; SessionEM + TSEM per run type | −0.10 [−0.50, +0.31]; **−2.98 [−4.09, −1.89]** | **Refuted**; the run-type axis is closed (halving the statistics costs 3 points) |
| §110 CLUSTX, §111 SEEN-DOSE | crashed at import: `studies/12nov/select.py` shadowed Python's `select` module on Leonardo's path (stdlib `subprocess` imports `select`) | not results; the file is renamed `selectvar.py`, both resubmitted |

| Explore item (22:53 UTC; `runs/leo_cpu/clustx-64316017.out`, `seendose-64316018.out`) | Observed | Verdict |
|---|---|---|
| §110 CLUSTX alone, window accuracy: pooled / type A / type B / oracle pick / label-free pick | 56.5 / 55.8 / 50.3 / **58.6** / 57.5 | **Refuted**: the oracle best expert is +2.1 over pooled (bar +2: at the mark, not above) |
| §110 fused into the chain (β 0.10 on the summed T·C·S logs), six bases | pooled −3.2 to −4.1; label-free pick −3.0 to −4.2; oracle pick −2.7 to −3.9 | Refuted; person-type experts are closed |
| §111 SEEN-DOSE, proxy: v5.1's form (β 0.10, KREF 2, T·C·S) | +3.42 | as §82 |
| Best proxy forms | β 0.25 T·C·S (any KREF): **+5.2 to +5.3**, +1.7 to +1.9 over v5.1's form, 8 of 10 ahead (P07 −3.3); β 0.15: +4.4 to +4.6 (+1.0 to +1.2) | the proxy prefers a stronger stage |
| The same β on the real later sessions (§95, six bases, against β 0.10) | β 0.15: P04 +0.40, P10 −0.11; β 0.25: P04 −0.7, P10 −0.5 | **Rule not met** (≥ 0 on both real people): β 0.15 misses by 0.11 on P10 |

Reading of §111: on ten people the proxy ranks a stronger personal stage (β 0.15–0.25) above the shipped β 0.10; on the
two people with real later sessions the stronger stage helps P04 and costs P10 a little. Applying the sealed survival
ratio of the stage (0.19 from proxy to sealed, §105) to the proxy's +1.0 to +1.9 gives about +0.2 to +0.35 per
returning session, about +0.1 to +0.15 on the board. Recorded as the one form question a later sealed measurement
should settle (β 0.15 is the cautious choice); not adopted by the pre-set rule.


### 113. TTT7 (exploit track): runtime-feasible schedules with on-device augmentation (written 12 Nov 22:55 UTC, before any job ran; code `studies/13nov/ttt7.py`, Leonardo, 24 tasks)

§107: the full recipe costs about 142 s per unrouted session and delta1 about 20 s, which is still too much next to v5.2's
own cost. A package also has to draw its augmentation on the GPU, because CPU-drawn augmentation costs 3–15 ms per
step. Arms (the same nets, folds and labels as ttt5/ttt6, seeds 0–5, scored all6 and delta1/delta2 with
`ttt_reduce_score.py`):

| Arm | Change | Cost against fixA | Prediction, new, six-base mean, all6 (80 % range) |
|---|---|---|---|
| fixA | §101 (CPU-drawn augmentation), in-run reference | 1 | +0.36 (+0.25 to +0.45) |
| gA | fixA with GPU-drawn augmentation (another random stream, same distribution) | 1 | within ±0.08 of fixA |
| gE20 | a refit every 20 windows | 0.5 | +0.30 (+0.15 to +0.45) |
| gC10E20 | warm start, 10 steps, every 20 windows | 0.25 | +0.28 (+0.10 to +0.45) |
| gC5 | warm start, 5 steps, every 10 windows | 0.25 | +0.30 (+0.10 to +0.45) |
| gC10T20 | warm start, 10 steps, every 10 windows, from window 20 | 0.55 | +0.36 (+0.20 to +0.50) |
| **Rule** | gA within ±0.08 of fixA → on-device augmentation is accepted for a package. The package recipe is the cheapest arm (with delta1 or delta2) whose six-base mean is ≥ +0.25 with ≥ 5 of 6 bases positive and whose projected cost keeps the sealed worst case (v5.2's measured cost plus fine-tuning on every unrouted session) ≤ 45 min |

### 114–115. Explore wave 2 (first numbered 112–113, which the build session had used): whole systems built for the measured losses (written 12 Nov 22:57 UTC, before any full result; smoke runs on two people only checked that the code runs)

**114. PSYS: a personal system for returning people** (`studies/12nov/psys.py`). Reasoning: the sealed set showed that
returning people respond (+0.6 per non-P04 returning session for the stage); the v5.1 stage multiplies fixed personal
members into the chain. A whole personal system instead: the person's own tangent space (64 components from the public
runs), an LDA whose class means adapt within the session by causal soft EM, starting from the public class means.
Proxy of §82; scored alone, then fused into the in-population chain alone and next to the v5.1 stage.

| Check | Prediction (80 % range) |
|---|---|
| Personal system alone, ten people, window accuracy | 78 (72–84), in-population chain 82 |
| Stage + system (β 0.25) against the stage alone, proxy, three bases | +1.0 (0 to +2.5) |
| **Reading** | ≥ +1.0 on the proxy with ≥ 7 of 10 ahead → the real later sessions of P04 and P10 decide (≥ 0 on both) |

**115. EARLY: a window-local system for the start of a session** (`studies/12nov/early.py`). Reasoning: the first
fifth of a new session loses 3–7 points because the session statistics are not yet learned; a system that needs none
(relative band power per channel; tangent space at the training people's grand mean, no recentring) is available
from window 0. Scored on windows 0–15 and 0–30 against the chain, and fused with a weight fading out by window 30.

| Check | Prediction (80 % range) |
|---|---|
| Early system on windows 0–15 / chain | 55 (45–62) / 72 |
| Fused, new, six bases | −0.1 (−0.5 to +0.2) |
| **Reading** | fused ≥ +0.3 with the interval above 0 → kept for the next design |

| §114 follow-up (22:59 UTC; `studies/12nov/psys_seen.py`, run locally; meant as a one-base smoke but it ran all six bases because the replay head overwrites `sys.argv`, now fixed) | Observed | Verdict |
|---|---|---|
| Personal system (β 0.25, at the gate's ramp) on top of v5.1, real later sessions, six bases | seen −0.36 to −1.03, **mean −0.81 (P04 −0.06, P10 −1.57)** | **Fails the §114 rule** (≥ 0 on both real people) whatever the proxy shows |

Reading: an adaptive personal LDA adds nothing over the v5.1 stage on real later sessions and costs P10 1.6 points; its
in-session EM follows the person's drift, which is what hurts P10. The proxy result (Leonardo) is still recorded when it lands.

### 116–117. Explore wave 3 (written 12 Nov 23:04 UTC [date corrected 13 Nov: it read 13 Nov], before any full result; two-person smokes only checked that the code runs)

**116. BANDSEL** (`studies/12nov/bandsel.py`): people carry their classes in different bands (§77). Per-band population
models (tangent and CSP, nine bands) combined per new session with weights from their causal agreement with the chain,
against equal weights and an oracle (true accuracy so far); fused into the routed chain at β 0.10, six bases.
Prediction: band ensemble alone 58 (equal) / 59 (agree) / 61 (oracle); fused −0.1 (−0.5 to +0.3) for the label-free
version. Lever: ≥ +0.3 with the interval above 0.

**117. FAMSEL** (`studies/12nov/famsel.py`): per-session weights of the member families. 17 weightings (shipped, each
family dropped, each doubled) through the grid chain; per session the weighting is picked by true accuracy (oracle) or
by agreement with the majority of the 17 outputs. Prediction: oracle +1.5 (+0.8 to +2.5); label-free +0.1 (−0.2 to
+0.4). Reading: label-free ≥ +0.4 → a causal version is pre-registered; oracle < +1 → per-session member weighting is closed.

Numbering: this session uses §114–§119 for the explore track; the build session continues at §120.

| Explore wave 2 (23:08 UTC; Leonardo) | Observed | Verdict |
|---|---|---|
| §114 PSYS alone, proxy, window accuracy | P01 82, P02 72, P03 50, P04 49, P05 69, P06 79, P07 84, P08 85, P09 67, P10 72 (in-population chain about 82) | Miss, low (predicted 78) |
| §114 stage + personal system (β 0.25) against the stage alone, proxy, three bases | +1.00 / +0.92 / +0.83 | below the reading's +1.0; and the real later sessions gave −0.81 (P10 −1.6): **refuted** |
| §115 EARLY alone, windows 0–15 / 0–30 / after 30 | 44.8 / 40.5 / 43.7 % against the chain's 68.6 / 72.6 / 77.3 | Miss (predicted 55): without per-session normalisation a new person is barely decodable |
| §115 fused (weight fading by window 30), six bases | −1.15 to −1.51 (weight 0.5), −2.2 to −2.5 (weight 1.0) | **Refuted**: the session-start loss cannot be bought back without session statistics |

**§105 correction of the correction (23:09 UTC): the 3.9× was a cold-start artefact.** The first official replay in each
Leonardo job pays about 230 s of neuralbench/benchopt setup in a fresh /tmp (v5.1b alone, run first, took 325.1 s in job
64302270; in job 64311532 v5.2 ran first). Paired ABBA on one A100 after a throwaway warm-up replay (build session, job
64316388): v5.1b 96.4 s, v5.2 137.9 s, v5.2 136.6 s, v5.1b 99.0 s: **1.40×**, +53 ms per routed window, consistent
with the +70 ms of the 14-session check. Sealed estimate (1,287 s × the ratio): about 25 min at 43 % routed, about 30
min if every session were routed — inside the 60-minute limit. My "48 / 84 min, not uploadable" of 22:2x UTC is
withdrawn; the original §105 runtime estimate (27 min, worst case 32) stands. Lesson: never time a single cold replay.

| Explore wave 3 (23:14 UTC; Leonardo `runs/leo_cpu/bandsel-64317556.out`, `famsel-64317557.out`) | Observed | Verdict |
|---|---|---|
| §116 BANDSEL alone (equal / agree / oracle), ten people | 51.2 / 51.2 / 51.6 % | Miss, low (predicted 58–61; the two-person smoke was the easy pair) |
| §116 fused, six bases, label-free | −0.57 to −0.94 | **Refuted**; per-band weighting closed |
| §117 FAMSEL single re-weightings (six bases) | dropping core1 −2.0 to −2.9; dropping SessionBN FBCNet −0.5 to −2.1; dropping the plain TF FBCNets −0.30 to +0.36 (mean +0.09); doubling the SincShallow slot −0.7 to +0.4 | the shipped weights are near the best fixed choice |
| §117 per-session pick, oracle / label-free (agreement with the majority) | **+1.2 to +2.1** / −0.16 to +0.04 | Oracle above 1 (not closed), but no label-free criterion finds it (as §76) |

Reading: per-session choice among member weightings has 1–2 points of head-room, like per-session choice among stage
variants (§76), and agreement-based criteria cannot see it. The open question for this family is a criterion, not a
weighting; none is known.

### 118. SELMETA (explore): a learned selector for the 1–2 points of per-session head-room (written 12 Nov 23:15 UTC [date corrected 13 Nov: it read 13 Nov], before any full result; code `studies/12nov/selmeta.py`)

§76 and §117 both found 1–2 points available from choosing per session (stage variants, member weightings) and no
hand-made label-free criterion that finds them. Instead of a hand criterion, learn one: for each of the 17 weightings,
five label-free features of its output on the session's first 60 windows (relative to the shipped output); a ridge
model trained on the other people's sessions predicts each weighting's gain on the rest of the session; the session
switches at window 60 to the predicted best (or stays if no gain is predicted). Six bases, new people.

| Check | Prediction (80 % range) |
|---|---|
| Oracle from window 60 | +1.2 (+0.7 to +1.8) |
| Learned selector from window 60 | +0.05 (−0.3 to +0.3) |
| **Reading** | ≥ +0.3 with the interval above 0 → a package-form selector (all 17 weightings run in parallel; runtime) is pre-registered |

| §118 item (23:26 UTC; `runs/leo_cpu/selmeta-64318417.out`) | Observed | Verdict |
|---|---|---|
| Oracle from window 60, six bases | +0.96 to +1.36 | Hit (+1.2) |
| Learned selector from window 60 | **+0.00 on every base**: the ridge model never predicted a gain above 0, so no session switched | **Refuted**; the selector is a no-op, not a small effect (recorded as such) |

Reading: the 1–2 points of per-session head-room (§76, §117, §118) are invisible to label-free statistics of the first
60 windows, hand-made or learned. On average every alternative weighting loses, and nothing in the early outputs
separates the sessions where one wins.

### 119. RETSTACK (explore → design input): the returning-people stack at stage strength 0.15 (written 12 Nov 23:26 UTC [date corrected 13 Nov: it read 13 Nov], before any result) [review 13 Nov: personal nets from window 0 in both tails; package form ≈ −0.05 [§140]]

The returning block is the one lever measured on the sealed set. Two of its parts were tested apart: personal nets
(v5.2) and a stronger stage (§111: β 0.15 ahead on the proxy, P10 −0.11 on real later sessions). Here together, so the
next candidate's design (after the research window) rests on a joint measurement: (a) the ten-person proxy with the
in-population chain (`studies/12nov/inpop3.py`, BETA 0.15): personal nets + stage against the stage alone; (b) the real
later sessions of P04 and P10 in the v5.1 replay (`studies/12nov/pnet.py`, BETA 0.15): personal nets pfh60 + stage
0.15 against twins + stage 0.15; the β 0.10 numbers of §78 and §96 are the comparison.

| Check | Prediction (80 % range) |
|---|---|
| (a) proxy, nets + stage 0.15 over the stage 0.15 alone | +2.0 (+1.0 to +3.0) |
| (b) real, pfh60 + stage 0.15 against twins + stage 0.15, seen | +0.2 (−0.3 to +0.6); P10 −0.2 |
| **Reading** | if (b) is ≥ the β 0.10 combination (+0.17, §78) with P10 ≥ −0.3, the next candidate's stage strength is 0.15 |

| §119 item (23:37 UTC; `runs/leo_cpu/retstack_a-64319397.out`, `retstack_b-64319399.out`) | Observed | Verdict |
|---|---|---|
| (a) proxy, personal nets (pfh60) + stage 0.15 over the stage 0.15 alone | **+1.67 / +1.67 / +1.75** [+1.0, +2.4], 9 of 10 ahead (pf60 +1.6, 10 of 10) | Hit (+2.0, +1.0 to +3.0) |
| (b) real later sessions, pfh60 + stage 0.15 against twins + stage 0.15, six bases | **+0.39 to +0.79, mean +0.55 (P04 +0.81, P10 +0.28)**; the lever of §78 met on 6 of 6 bases (at β 0.10 it was +0.17 and 1 of 6) | **Hit, above the range** (predicted +0.2; P10 −0.2) |
| (b) pf60 + stage 0.15 | +0.36 (P04 +1.26, P10 −0.55), 0 of 6 | the head-only variant is the right one |
| Stage 0.15 alone on the twins against stage 0.10 on the twins | seen 82.10–82.36 against 82.35 | about −0.1: the stronger stage alone does not pay |

**Reading (design input for the next candidate, after the research window):** with the personal nets in place, the
stronger stage pays and P10 no longer loses: personal nets (pfh60) + own-space stage at β 0.15 is about +0.4 seen over
v5.2's β 0.10 (82.75 against 82.52 absolute on B2_pool), positive for both real people, and +1.7 over the stage alone
on the ten-person proxy. By §119's rule the next candidate's stage strength is 0.15, with the head-only personal nets.


| §108 item (23:40 UTC; `runs/ttt6_score.log`; new people, six bases; all6 / delta1 / delta2) | Observed | Verdict |
|---|---|---|
| fixA (in-run reference) | +0.37 / +0.27 / +0.35 | replicates §101/§106 |
| cum10 (warm start, 10 steps, half the cost) | **+0.46** (bases +0.16 to +0.72) / +0.26 / +0.34 | all6: +0.09 over fixA, **counts as a gain by the rule, narrowly: ahead of fixA on 5 of 6 bases** (cum10 − fixA: pool +0.13, s345 +0.09, s012 +0.30, s0 +0.01, s1 −0.04, s2 +0.04; check 13 Nov 00:03 UTC from `runs/ttt6_score.log`); s0 and s2 sit at +0.01 and +0.04, so most of the margin comes from one base (s012); delta forms equal fixA at half the cost |
| cons2 / soft / step5 | +0.32 / +0.27 / +0.21 (all6) | no gain |
| head | −0.16 | refuted: the full parameter set is needed |
| view2 | +0.25 / +0.11 / +0.15 | no gain |

### 130. STACK (explore track, gap 1 of `docs/SOLUTION_SPACE_MAP.md`): the replicated small positives for new people as one system (written 13 Nov 00:05 UTC, before any result; code `studies/13nov/stack.py`, Mac, stored outputs only) [review 13 Nov: base counts here carry almost no information, read the person uncertainty in §139]

Numbering: the explore track's §114–§119 are used, and the build session continues from §120, so the explore track
uses §130–§139 from here on. "Small levers do not stack" rests on one test of two levers. Here five levers for new
people are scored alone, all together, and with each one left out, in the routed chain (`A.system`) on six bases, with
a person bootstrap. Returning people already have their joint test (§119). The levers:

- **FT**: cum10 in-session fine-tuning (§108, the counting arm). The four SessionBN families are swapped for
  `runs/ttt/probs6` ttt6_*_cum10, against ttt6_*_base (the same nets frozen), new people only. In the package it would
  run on unrouted sessions; seen sessions keep the frozen nets here.
- **REST**: the rest-as-fourth-class ShallowConvNet (rest4_sh_r1, three seeds) as an extra member at weight 1. §26
  measured it against its own twin (rest4_sh_c3), but the twin's chain sits 0.7 J below the shipped chain. Here it is
  measured against the chain without it, which is what a package would ship.
- **OCU**: the eye-net log posteriors (oc_nomix, three seeds) appended to SessionEM's GE view (§58b form).
- **MIX**: the member mix raised to the power 0.9 and renormalised before the stages (11 Nov: +0.2 J).
- **PLS**: a second TSEM pass after the routed chain through 16 PLS-DA directions of core1's components, λ 0.15, new
  people only (§74; the random-direction null was +0.25 of its +0.5).

The reference for every arm is the stack base: the shipped chain with the four SessionBN families replaced by ttt6
base (frozen). The shipped chain itself is also reported, so that no lever is credited for repairing a weaker base.

| Arm (new people, six-base mean, against the stack base) | Prediction (80 % range) |
|---|---|
| FT alone | +0.45 (+0.30 to +0.60) |
| REST alone | +0.10 (−0.40 to +0.50) |
| OCU alone | +0.20 (−0.10 to +0.40) |
| MIX alone | +0.15 (−0.20 to +0.40) |
| PLS alone | +0.35 (+0.10 to +0.60) |
| All five | +0.70 (+0.20 to +1.20): the stage levers (OCU, MIX, PLS) all strengthen the session stages and overlap |
| Stack base against the shipped chain | 0.0 (−0.3 to +0.3) |
| **Rule** | A lever stays if leaving it out of the full stack costs ≥ +0.10 new (six-base mean), with ≥ 4 of 6 bases agreeing, and seen moves by no more than −0.15. The stack is a design input for the build after the window if its kept levers add ≥ +0.6 new on ≥ 5 of 6 bases with seen ≥ −0.15. If all five together land below the best single lever, small levers do not stack, and that is recorded as a second test of the claim |

| §130 item (13 Nov 00:11 UTC; `runs/stack_13nov.log`; new people, six-base mean against the stack base; [10, 90] person bootstrap) | Observed | Verdict |
|---|---|---|
| FT alone | +0.46 [−0.23, +1.16], 6/6 bases; seen +0.10 | Hit (+0.45) |
| REST alone | +0.34 [−0.01, +0.69], 6/6; **seen −0.51** | Hit (+0.10, range to +0.50); costs seen |
| OCU alone | +0.14, 5/6; seen +0.47 | Hit (+0.20) |
| MIX alone | +0.18, 5/6; seen +0.24 | Hit (+0.15) |
| PLS alone | +0.41 [+0.08, +0.75], 6/6; seen 0.00 | Hit (+0.35) |
| **All five** | **+0.86 [+0.25, +1.49], 6/6 (+0.75 to +0.98)**; seen −0.21; against the shipped chain +0.78, 6/6, seen +0.05 | Hit (+0.70). **Small levers do stack**: all five land above the best single lever (+0.46) |
| Stack base against the shipped chain | +0.08 (−0.46 to +0.35), seen −0.26 | Hit (0.0) |
| Leave-one-out: keep FT / REST / OCU / MIX / PLS | +0.40 (6/6, seen −0.22) / +0.33 (6/6, seen −0.11) / **+0.68 [+0.46, +0.91] (6/6, 10 of 10 people, seen +0.28)** / −0.10 (1/6) / −0.27 (0/6) | By the rule: OCU and REST stay; MIX and PLS go; **FT misses on seen** (−0.22 against the −0.15 bound) |
| ALL without PLS (FT, REST, OCU, MIX) | **+1.13 [+0.65, +1.61], 6/6 (+0.94 to +1.24)**, 8 of 10 people; seen −0.21 | Best arm of the run, chosen after the fact: a winner's-curse risk |

**Reading.** (1) The lever that looked weakest alone (OCU, +0.14) carries the stack: without it the other four drop
to +0.18. The eye-net view in SessionEM seems to make the stage use the fine-tuned and rest-trained members. (2) The
two stage levers that only strengthen the second pass (PLS, MIX) overlap with OCU and hurt in the stack. (3) FT's seen
cost is a replay artefact in mechanism, not a measured package effect. FT changes only the new-person outputs, and
`A.system` refits its routing gate on those outputs (`route(..., oof=sh["all_loso"])`). A package keeps its gate fixed.
By the rule as written, FT still fails the seen bound, and I say so plainly. Whether it goes in the design is settled
by the follow-up below, which measures the seen cost with FT on and off in the same subsets.

**§130b follow-up (written 13 Nov 00:12 UTC, before it runs; same script, `SUBSETS=`).** The subsets the rule and
the reading point to are scored on six bases: REST+OCU (the rule's kept levers), FT+REST+OCU, FT+OCU, and
FT+REST+OCU+MIX (the best arm again, as a control for the rerun). Predictions against the stack base, new, six-base
mean: REST+OCU +0.55 (+0.20 to +0.90); FT+REST+OCU +1.0 (+0.6 to +1.3); FT+OCU +0.75 (+0.4 to +1.1); FT+REST+OCU+MIX
+1.13 (identical code, so the same number). **Design input** (for the build after the window, not a candidate now): the
smallest subset within 0.10 of the best, with ≥ 5 of 6 bases and seen ≥ −0.15. If every FT subset misses the seen
bound, the design keeps FT only for unrouted sessions, and its seen cost is checked in the v5 replay with a fixed
gate before any build.

### 131. TTT8 (exploit track): follow-ups of cum10 (written 13 Nov 00:13 UTC, before any job ran; code `studies/13nov/ttt8.py` on `ttt6.py`, Leonardo, 24 tasks)

cum10 (warm start) counted narrowly in §108: +0.46, ahead of fixA on 5 of 6 bases. Warm start lets the steps add up
over a session, so the step count, refit spacing, learning rate and consistency weight trade differently than for
fixA. Same nets, folds, augmentation and chain labels as ttt6; new people, six bases, `ttt_reduce_score.py`
(all6 / delta1 / delta2) against the same run's frozen nets.

| Arm | Change (all warm start) | Cost against fixA | Prediction, all6, six-base mean (80 % range) |
|---|---|---|---|
| cum10 | §108 again, in-run reference | 0.5 | +0.46 (+0.30 to +0.60) |
| cum20 | 20 steps every 10 windows | 1 | +0.45 (+0.25 to +0.60) |
| cum10e5 | 10 steps every 5 windows | 1 | +0.50 (+0.30 to +0.65) |
| cum5e5 | 5 steps every 5 windows | 0.5 | +0.45 (+0.30 to +0.60) |
| cum10lr | learning rate 1e-3 | 0.5 | +0.40 (+0.15 to +0.60) |
| cum10c2 | consistency weight 2 | 0.5 | +0.45 (+0.25 to +0.60) |
| **Rule** | An arm replaces cum10 as the recipe if it is ≥ +0.08 above the in-run cum10 (all6, six-base mean) and ahead on ≥ 5 of 6 bases; an arm at cost 1 must also beat §108's fixA by ≥ +0.08. delta1/delta2 are read the same way for the cheap package form. Otherwise cum10 stays |

| §130b item (13 Nov 00:15 UTC; `runs/stack_13nov_sub.log`; new, six-base mean against the stack base / against the shipped chain) | Observed | Verdict |
|---|---|---|
| FT+REST+OCU+MIX (control: the same code as §130) | +1.13, 6/6; seen −0.21 / shipped +1.05, seen +0.05 | Identical to §130: the rerun is deterministic |
| FT+REST+OCU | +1.08 [+0.64, +1.54], 6/6 (+0.83 to +1.31); **seen −0.49** / shipped +1.00, seen −0.23 | Hit (+1.0); misses the seen bound |
| FT+OCU | **+0.86 [+0.30, +1.41], 6/6 (+0.71 to +1.01); seen +0.18** / shipped +0.78, seen +0.45 | Hit (+0.75); the only subset within the seen bound |
| REST+OCU | +0.61, 6/6; seen −0.34 / shipped +0.53, seen −0.08 | Hit (+0.55) |

**Verdict by the rule: no subset passes as written.** The two subsets within 0.10 of the best (+1.13) both miss the
seen bound (−0.21, −0.49), and FT+OCU, which keeps seen people safe, is 0.27 below the best. The seen cost comes with
REST: the rest-trained member is applied to seen sessions too, and it costs them about 0.3–0.5. **Design input for
the build after the window: FT+OCU (+0.86 new, 6 of 6 bases, seen +0.18; +0.78 over the shipped chain).** REST
(with MIX) is worth +0.27 more for new people only if it can stay off returning sessions. The v5 replay, where routed
sessions take the personal tail, decides that, and it decides before any build. Packaging costs: OCU needs the eye
net in the package, and FT needs run time (§107, §113).

### 132. TEACH2 (explore track, gap 2 of the map): the teacher line redesigned and re-scored in the current design (written 13 Nov 00:15 UTC, before any result; code `studies/13nov/teach2.py`, Mac, stored head outputs)

The teacher is the only member with partly independent errors, and its design dates from 1–2 Nov. The stronger
Scherer-trained teacher (m8s: +3.3 alone) lost −0.98 J in the chain at weight 5 on 11 Nov (§18), and no weight rescued
it. Two untested reasons: (i) its outputs carry a per-session offset that the stages amplify, and per-session
normalisation of the head outputs would remove it; (ii) the cognitive heads (rest, subtraction, word) drown in the
38-column map and would work better as a member of their own. Teachers are rebuilt from `runs/tsch/heads.npz` (seed
means of both trunks, balanced logistic map C 0.01 per left-out person), new people only, in two contexts: the
shipped chain and the STACK design input (FT+OCU, §130b). The reference is the rebuilt m7c teacher at weight 5 (the
harness check: alone 50.46 ± 1.5, as on 11 Nov).

| Arm (new, six-base mean against the rebuilt m7c teacher; shipped / FT+OCU context) | Prediction (80 % range) |
|---|---|
| Shipped teacher instead of the rebuilt one | 0.0 (−0.3 to +0.3) |
| m8s at weight 5 / 2 | −1.0 (−1.8 to −0.3) / −0.3 (−0.9 to +0.3) |
| m7c session-normalised (n7c), weight 5 | +0.2 (−0.5 to +0.8) |
| m8s session-normalised (n8s), weight 5 / 2 | +0.3 (−0.8 to +1.2) / +0.2 (−0.5 to +0.8) |
| Cognitive-heads member next to the teacher (cog7 / cog8 at weight 1, 2) | +0.1 (−0.3 to +0.4) |
| **Rule** | A variant counts if it is ≥ +0.30 new over the reference in **both** contexts, on ≥ 5 of 6 bases, with seen ≥ −0.15. If n8s counts and t8s does not, reason (i) holds: the Scherer teacher failed on session offsets, not on content. A counting variant becomes a design input (it needs only a different map in the package: the trunks stay) |

### 133. WEAK (gap 3 of the map, alongside the fine-tuning arms): fine-tune only sessions that look weak without labels (written 13 Nov 00:17 UTC, before any result; code `studies/13nov/weak.py`, Mac, stored outputs)

Fine-tuning gains concentrated on P04-like sessions on 11 Nov, and strong sessions lost. A package can only gate on
what it sees: at window 30, where fine-tuning starts, the session's mean top probability of the shipped member mix
over its first 30 windows. cum10 (FT) is applied below a fixed quantile of the dev new-people sessions (q33, q50,
q67), and the inverse gate (inv50, strong half only) is the control. Contexts: FT alone against frozen, and FT+OCU
against OCU (§130b's design input). Six bases, new people.

| Arm (new, six-base mean against the same context without FT) | Prediction (80 % range) |
|---|---|
| ungated (§130 FT; FT+OCU − OCU) | +0.46 / +0.72 (both measured) |
| q50: weak half only | +0.45 (+0.20 to +0.70) |
| q33 / q67 | +0.35 (+0.10 to +0.60) / +0.48 (+0.25 to +0.70) |
| inv50: strong half only (control) | +0.05 (−0.20 to +0.25) |
| **Rule** | A gate is worth adopting if it lands within 0.05 of the ungated arm (or above) in both contexts at ≤ 2/3 of the sessions, with ≥ 5 of 6 bases ahead of frozen. It saves run time, which pays for the rest of the design. If inv50 ≥ +0.25, the gains are not concentrated in weak sessions and gating is closed. If a gate is ≥ +0.10 above the ungated arm on ≥ 5 of 6 bases, strong sessions lose from FT and the gate is a gain, not only a saving |

### 134. OCU9 (explore → consolidation): is OCU's role in the stack seed luck? (written 13 Nov 00:18 UTC, before any job ran; `studies/12nov/ocunet.py` tasks 7–17 odd = nomix seeds 3–8, Leonardo)

In §130, OCU was +0.14 alone but +0.68 when left out of the stack (10 of 10 people), and that rests on three eye-net
seeds. Six more nomix seeds are trained, and §130b's FT+OCU is re-scored with all nine (`stack.py`, after the run).

| Check (new, six-base mean) | Prediction (80 % range) |
|---|---|
| OCU alone, nine seeds | +0.15 (−0.05 to +0.35) |
| FT+OCU against FT alone (OCU's share of the design input), nine seeds | +0.40 (+0.20 to +0.60); three seeds gave +0.40 (0.86 − 0.46) |
| **Rule** | If OCU's share in FT+OCU falls below +0.20 with nine seeds, the three-seed result was seed luck and OCU leaves the design input; otherwise it stays, scored with nine seeds |

## 135–137. Night plan, 13 Nov (written 13 Nov 00:23 UTC, before any of these jobs was submitted)

The user wants the GPUs busy overnight, with no session awake between waves. Everything below goes in at once: arrays
released now, scorings held by Slurm dependencies (`afterany`). Every table is written to `runs/night13/` on Leonardo,
and each scorer asserts its inputs, so missing files fail loudly instead of printing zeros. Arrays skip outputs that
already exist, so a resubmission is cheap. Track ratio: exploit §131, §135, §136; explore §132 (Mac), §133 (Mac),
§134, §137. Every scorer here uses `A.system`, the routing-v3 dev chain, not the shipped v4.1/v5 chain, and every
script prints that; the package-faithful replay is §138, built on the Mac tonight.

**Ceilings of the explore studies** (user, 13 Nov: size explore work by its ceiling, rank 1 is no upper bound):
- §137 TTT9, fine-tuning with better labels. Ceiling: fine-tuning on true labels gave +1.6 to +2.9 on 11 Nov, and
  session-own models reach 81–82 % with labels against 74.9 %. This study measures that ceiling for the current recipe
  (oracle arm) and takes a second label-free step towards it (labels from the stack chain instead of the v3 chain).
- §132 TEACH2 (Mac). Ceiling: an independent member at 70 % was estimated at +3.1 for new people.
- §134 OCU9: consolidation of the lever that carries the stack, not a ceiling study.
- Not tonight, with reasons. Teacher retraining: the pretraining corpora are not on Leonardo, and Galvani needs the
  user's 2FA. Gap 7 retests (MixStyle, EEGNeX in the filter slot): low ceilings (≤ +1 on the record), so they rank
  below. Gap 6, self-supervised training on the rest runs: high ceiling, but it cannot be specified and smoke-tested
  well within the hour. It is the first design for 13 Nov day.

### 135. TTT8F (exploit): cum10 and its neighbours on fresh nets (ttt8.py tasks 24–71 = seeds 6–17, account 1675)

The fine-tuned arms so far used seeds 0–5, the nets they were chosen on. Seeds 6–11 and 12–17 are two fresh
replications, scored separately (`ttt_reduce_score.py SEEDS=6-11` and `SEEDS=12-17`).

| Check (all6, new, six-base mean against the same run's frozen nets) | Prediction (80 % range) |
|---|---|
| cum10, seeds 6–11 / 12–17 | +0.40 (+0.20 to +0.60) each |
| **Bar** | cum10 replicates if ≥ +0.25 on both fresh sets with ≥ 5 of 6 bases; a neighbour replaces it only if it is ≥ +0.08 above cum10 on all three seed sets (0–5, 6–11, 12–17) |

### 136. STACKF (exploit): the stack on seeds that did not choose it (Leonardo CPU job after §135 and §134)

§130b chose FT+OCU and "without PLS" after seeing the results. Here they are scored once more, with FT from fresh nets
(ttt8 seeds 6–11) and the eye-net from fresh seeds 3–8 (`stack.py`, `FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8
FT_SEEDS=6-11 OCU_SEEDS=3,…,8`).

| Arm (new, six-base mean against the stack base) | Prediction (80 % range) |
|---|---|
| FT+OCU | +0.70 (+0.40 to +1.00); §130b +0.86 |
| FT+REST+OCU+MIX | +0.95 (+0.50 to +1.30); §130b +1.13 |
| **Bar** | the design input (FT+OCU) is confirmed if it is ≥ +0.50 on ≥ 5 of 6 bases with seen ≥ −0.15; below +0.30, §130b was mostly selection |

### 137. TTT9 (explore, high ceiling): fine-tuning with better labels (ttt9.py, LAB=true and LAB=stack, seeds 0–5, after a debug smoke)

Pseudo-labels for all fine-tuning so far came from the v3 chain on B2_pool (`chain_final.npz`: new 80.18). LAB=stack
uses the FT+OCU stack's outputs on B2_pool (`chain_stack.npz`: new 80.93, seen 79.60). LAB=true uses one-hot true
labels: the oracle ceiling, never shippable. Arms cum10 and fixA, against the same run's frozen nets; the v3-label
reference is §108/§131 on the same seeds 0–5.

| Check (all6, new, six-base mean) | Prediction (80 % range) |
|---|---|
| true labels, cum10 / fixA | +2.0 (+1.0 to +3.0) / +2.2 (+1.2 to +3.2) |
| stack labels, cum10 | +0.55 (+0.30 to +0.80); v3 labels +0.46 |
| **Bar** | true-label cum10 ≥ +1.5: label quality is the lever, and it gets several more forms (iterated rounds, per-session label filtering). Stack labels ≥ +0.10 over v3 labels on ≥ 5 of 6 bases: a round-3 label refresh is queued on 13 Nov. True-label gain < +1.0: the recipe, not the labels, limits fine-tuning |

**§137 addendum (13 Nov 00:31 UTC, before any TTT9 result):** wave 2 (`scripts/cloud/night13b.sh`, arrays 64324823
and 64324824, held until wave 1's TTT9 arrays end) repeats TTT9 on fresh seeds 6–11 (`runs/night13/ttt9_*_s6-11.txt`).
The bar of §137 is read on both seed sets; a label effect counts only if it holds on both.

### 138. PKG: the design inputs in the package-faithful replay (written 13 Nov 00:31 UTC, after a one-base smoke on B2_pool and before the six-base run; code `studies/13nov/pkg.py`)

The §130 scores came from `A.system` (routing v3, gate refit on the scored outputs). This replay is the uploaded
chain: routing v4.1 (route4 "max", t10), then the own-space stage on routed windows. The gate's calibration is fixed to
the shipped members' outputs. Fine-tuning (cum10) runs only on sessions the gate never routes. Returning people get
the personal nets (pfh60) and the stage at 0.15. OCU is applied everywhere. On B2_pool the gate routes all 762 seen
rows and none of the 2,655 new rows, so FT reaches only new people, as a package would apply it.

**Harness limit, stated before the run:** the personal nets and the FT twins are study nets (pft, ttt2, ttt6), not the
shipped members. Arms are compared only against arms with the same nets: P2, P3 and P4 against P1t. P1 against P0
mixes a net swap with the design and is not read as a result. The smoke (B2_pool) gave, against P1t: P2 (FT) new +0.71,
seen 0.00; P4 (OCU) new +0.32, seen +0.78; P3 (FT+OCU) new +0.92, seen +0.78 (P04 +1.32, P10 +0.25), J +0.85.

| Check (six-base mean against P1t) | Prediction (80 % range) |
|---|---|
| P3, FT+OCU: new / seen / J | +0.80 (+0.45 to +1.15) / +0.40 (0.00 to +0.80) / +0.60 (+0.30 to +0.90) |
| P2, FT alone: new | +0.60 (+0.30 to +0.90); seen 0.00 (FT does not reach routed sessions) |
| P4, OCU alone: seen | +0.40 (0.00 to +0.80) |
| **Bar** | FT+OCU is confirmed in the package chain if P3's J is ≥ +0.40 on ≥ 5 of 6 bases with seen ≥ −0.15. That makes it the design for the build after the window, with the eye net and the cheap FT form as package work |

**§138 addendum (13 Nov 00:32 UTC, before any six-base result):** the same replay on fresh seeds (FT from ttt8 seeds
6–11, eye-net seeds 3–8) is queued on Leonardo (job 64325439, after §135 and §134; `runs/night13/pkg_fresh.txt`). The
§138 bar is read on both seed sets.

**§137 round 3 (13 Nov 00:34 UTC, before any TTT9 result; `scripts/cloud/night13c.sh`, jobs 64325581 → 64325584 →
64325585):** after round 2 (stack labels) ends, a CPU job builds new labels from round 2's fine-tuned outputs plus the
nine-seed eye-net (`chain_stack2.npz`), then cum10 and fixA fine-tune on them (seeds 0–5, `runs/night13/ttt9_stack2.txt`).
Prediction: round 3 over round 2, cum10 all6, +0.05 (−0.10 to +0.20). **Bar:** ≥ +0.10 on ≥ 5 of 6 bases means iterated
self-training climbs, and more rounds are worth building into the package's run time; ≤ 0 means one refresh is the
whole label gain.

| §138 item (13 Nov 00:36 UTC; `runs/pkg_13nov.log`; six bases, against P1t: the same nets without FT and OCU) | Observed | Verdict |
|---|---|---|
| Routing | on every base the gate routes all 762 seen rows and none of the 2,655 new rows | FT reaches only new people, as in a package. Also: for new people, routing v4.1 leaves the grid chain unchanged, so the unrouted grid-chain studies of 12 Nov stand for new people (their seen numbers do not transfer) |
| P2, FT alone | new +0.50 [−0.13, +1.12], 6/6; seen 0.00 | Hit (+0.60) |
| P4, OCU alone | new +0.16, 5/6; **seen +0.43 (P04 +0.62, P10 +0.25), 5/6** | Hit (+0.40) |
| **P3, FT+OCU** | **new +0.72 [+0.27, +1.19], 6/6; seen +0.43, 5/6; J +0.58**. Per base J: pool +0.85, s345 +0.39, s012 +0.68, s0 +0.49, s1 +0.77, s2 +0.29 | Hit on the mean (+0.60). **The bar is missed narrowly**: J ≥ +0.40 on 4 of 6 bases (s345 is +0.39), though all six are positive. By the header rule this reads **undecided, leaning positive**. The fresh-seed run (job 64325439, `runs/night13/pkg_fresh.txt`) decides |
| P1/P1t against P0 | new −0.46 / −0.30, seen +0.38 | Not a result (net swap; see the harness limit above) |

**Reading.** In the package's own chain the design input holds up: FT carries the new-people gain (+0.50), the
eye-net view carries returning people (+0.43 seen, both P04 and P10 positive), and together they reach J +0.58 on the
same nets, with all six bases positive. For new people it is a little below §130's `A.system`
number (+0.72 against +0.86). On returning people the eye-net view adds to the
personal-nets stack instead of competing with it.

### 139. Review of 13 Nov on the fine-tuning paths: corrections, and TTT10 (written 13 Nov 00:39 UTC; TTT10 before any job ran)

**Correction 1: six bases are almost one system wherever slots 5–8 are swapped** (independent reviewer, confirmed
here). The TTT scorers (§101, §106, §108, §131, §135, §137) replace the four SessionBN slots with the run's own nets on
every base, so only slot 9 (weight 1/15) and the non-SessionBN members differ between bases. Per-person deltas
correlate 0.83–0.99 across bases. "Ahead on 5 of 6 bases" and base ranges therefore carry almost no information, and
every rule above that counts bases is restated in person terms:

| Result | New people, six-base mean | 80 % / 95 % person bootstrap | People ≥ 0 | Without the best person |
|---|---|---|---|---|
| fixA (§101/§108) | +0.37 | −0.20 to +0.92 / — | — | +0.17 |
| cum10 (§108) | +0.46 | −0.23 to +1.16 / −0.56 to +1.52 | 6 of 10 | +0.12 (P10 as a new person: +3.5) |
| §130 FT alone | +0.46 | −0.22 to +1.14 / −0.56 to +1.50 | 6 of 10 | +0.12 (P10); bases correlate 0.92–0.99 |
| §130 all five | +0.86 | +0.24 to +1.49 / −0.07 to +1.81 | 7 of 10 | +0.55 (P10 +3.7); 0.84–0.98 |
| §130 without PLS | +1.13 | +0.67 to +1.60 / **+0.45 to +1.85** | 8 of 10 | +0.86 (P10 +3.6); 0.77–0.97 |
| §130 OCU / MIX / REST alone | +0.14 / +0.18 / +0.34 | 95 % lower ends −0.16 / −0.13 / −0.21 | 7 / 6 / 8 of 10 | +0.06 / +0.11 / +0.17; bases correlate less (0.07–0.90) |

**Reading.** FT's gain for new people is mostly one person: P10 scored as a new person gains +3.5, and the other nine
average +0.12. The "counts narrowly, 5 of 6 bases" of §108 is restated as **undecided: +0.09 over fixA, inside the
person noise**. §130's stack holds up better on this measure: without PLS, its 95 % person interval stays above zero
and the nine people other than P10 gain +0.86. A real replication needs fresh fine-tuned seeds (§135, tonight) and
ideally other people. The §130b and §138 bars that counted bases are read in these terms when their fresh-seed runs
land.

**Correction 2: §108's "head refuted" closes only that form.** Its selector (`p.dim() > 1 and p.shape[0] >= 3`) took
the final weight matrix without the output bias, and it ran at 5e-4. The personal-head recipe that worked trains the
last two parameters at 2e-3. Head-only fine-tuning is the cheap form, so it is rerun here as TTT10.

**Fixes in code (13 Nov ~00:37 UTC):**
- Every fine-tuning output now stores its label file's name and hash (`labels`, from `ttt5.LABEL_TAG`).
  `ttt_reduce_score.py` prints the label sources, fails on mixed sources, and fails on oracle labels under a
  packageable prefix. No submit script of tonight exports `CHAIN_NPZ`; `ttt9.py` sets it inside the process from
  `LAB`, and `ttt10.py` refuses to start if it is set.
- `ttt6.run` now asserts the base-vs-harness difference (< 1e-6) instead of only printing it.
- Package items 4–8 of the review (the §78 form, the personal traces fine-tuned from retrained twins, the
  false-accept risk, the v5.3 choices, runtime against window index) went to the build session.

**TTT10** (`studies/13nov/ttt10.py`, tasks 0–23, after a debug smoke). Scope head2 = SessionBN affine + the last two
parameters. Arms: head2 (20 steps every 10 windows, each refit from the frozen net, 2e-3), head2c (warm, 10 steps,
2e-3), head2x (warm, 20 steps, 5e-3); in-run reference cum10.

| Check (all6, new, against frozen) | Prediction (80 % range) |
|---|---|
| head2 / head2c / head2x | +0.25 (−0.05 to +0.50) / +0.30 (0.00 to +0.55) / +0.30 (−0.10 to +0.60) |
| **Bar** | A head arm is the cheap form if it is within 0.10 of the in-run cum10 **and** its gain without the best person is ≥ +0.10. Base counts are not used |

| §134 item (13 Nov 00:39 UTC; Leonardo `runs/night13/stack_ocu9.txt`; eye-net seeds 0–8, FT seeds 0–5 as in §130) | Observed | Verdict |
|---|---|---|
| OCU alone, nine seeds | +0.15 [−0.14, +0.44], 8 of 10 people; seen +0.31 | Hit (+0.15) |
| FT+OCU, nine seeds | +0.86 [+0.28, +1.43]; seen +0.27; over shipped +0.77, seen +0.54 | identical to three seeds |
| OCU's share in FT+OCU (FT+OCU − FT) | **+0.40** | Hit (+0.40). **OCU stays in the design input; the three-seed result was not seed luck.** The FT side still has the §139 caveat (P10) |
| FT+REST+OCU+MIX, nine seeds | +1.08, seen −0.44 | as §130b: REST costs seen |

### 120. V53 package choices, fixed before any v5.3 code (written 13 Nov 00:42 UTC; review of 13 Nov, item 7; v52-build session)

No package before about 14 Nov 02:00 UTC. A fine-tuning package (v5.3 = v5.2b + in-session fine-tuning on unrouted
sessions) makes the following choices. Each differs from or is missing in the dev runs, so each is fixed here before
it is built:

1. **SessionBN state after a refit.** Dev recomputes the state by re-running the whole prefix (net2(X[:hi])) after each
   refit. The package does the same once per refit (one no-grad pass over the buffered windows), then advances window by
   window. Equal by the causal formula. Checked against a dev-form reference on the same device and batch shapes (as
   `check_v52.py --ref-schedule`), tolerance 1e-5.
2. **Slots 6 and 9.** Dev fine-tunes slot 6 (fb at k 2) and not slot 9 (the same nets at k 5). The package fine-tunes the
   k-2 traces and applies their change to the k-5 traces of the same seeds: the dev form scored with slot 9 also swapped
   is scored before the build. If that is below the slot-6-only form, the k-5 traces stay frozen.
3. **No silent no-op.** The package's forward paths run under no_grad, and a v5.2-style `except Exception` fallback would
   turn a broken fine-tune into the frozen output. The package therefore counts refits, steps and the parameter change
   per session in its stderr line. The replay check asserts that, on every unrouted public session (forced unrouted),
   refits happened, parameters changed (max |Δθ| > 0) and the fine-tuned posteriors differ from the frozen ones.
4. **Late recognition.** A session first routed after fine-tuning has started (P04 session 2: first routed at window 63)
   has no dev arm. Rule: fine-tuning stops at the first routed window, and v5.2b's path takes over from there. The cost of
   the fine-tuned windows before it is measured in the package-faithful replay (§138 form) before the build.
5. **On-device augmentation** (§113 gA against fixA) and the schedule (§113, §131) are taken from their scored arms
   only. The person-level reading of §139 (people ≥ 0, the gain without the best person) applies, not base counts.

### 140. Evaluation-path review of 13 Nov: three verdicts reverse; PKG2 in package form (written 13 Nov 00:49 UTC; PKG2 after a one-base smoke on B2_pool, before the six-base run)

Second independent reviewer (`evidence/map_13nov/review_evaluation_path_13nov.md`). The core replay is sound for new
people. The defects are in the studies around it. Below, what changes, and what I verified here.

**1. Seen and J from `A.system` do not hold on v4.1** (`replay/alib.py:88-96`: routing v3 with the gate refitted on
the arm's own outputs). New-people numbers are unaffected. On v4.1 with a fixed gate (reviewer's re-score): §58b's
eye-net view seen +0.51 → −0.33, dJ +0.28 → −0.14. In the stack, FT alone seen 0.00, OCU −0.03, FT+OCU −0.03,
FT+REST+OCU +0.31 (recorded −0.49). **§130b's seen verdicts reverse.** "REST costs seen" and "FT+OCU is the only
subset within the bound" are artefacts of the scoring path. The stack reads: **new +0.86 (all five) / +1.13 (without
PLS), with the person intervals of §139; seen unresolved on that path.** §73 and §75 say "full routed v4.1 replay"
but call `A.system`; their seen and J columns are v3. Also affected on seen or J: §54, §61, §87, §99, §132, §133 and the
seen columns of the TTT scorers. Per-base seen deltas of one arm range over about 1 point and one window is 0.07 J,
so a ±0.15 seen bound cannot be resolved on any single path.

**2. Personal nets were given to whole returning sessions from window 0, in both tails** (pnet.py, the first pkg.py,
the PFT arms of add.py and dose.py). A package swaps them in only after the gate routes (window 10 for three sessions,
60–63 for P04 session 2). Reviewer: §119 as coded +0.55 (P04 +0.81, P10 +0.28); in package form −0.05 (P04 −0.10,
P10 0.00). P10's +0.28 was one window inside windows 0–9. **§78, §119 and §138 P1–P4 do not measure the package. The
package-form dev support for v5.2 is about zero, so §105's expectation is restated: no dev evidence for a sealed gain
from the personal nets.** v5.1b's sealed gain came from the stage, not from personal nets. Passed to the build session.

**3. The six bases are about two independent ones** (B2_pool is the mean of s012 and s345; slots 0–4 and 10, two
thirds of the weight, are identical in all six; with the SessionBN slots swapped they agree on 99.0–99.8 % of labels).
Sixteen random-view null arms give a six-base-mean SD of 0.16 for new, and the best of 16 nulls reached +0.61 on the
exploration bases. "6/6 bases" in §106, §108, §130, §130b and §138 is not replication; fresh seeds are (§135, §136).

**4. §70's learned sequence stage is confounded** (`seqstage.py` pools fold-specific PCA coordinates, and 52–54 of
128 components flip sign between folds). The same fault is in `studies/11nov/clus/ceiling.py`, §30's "lda2" row.
The −11 to −16 result, and §98's lesson "a similarity learned on nine people does not carry to a tenth", are
withdrawn. **A learned session stage has never had a valid test.** It goes to the top of the explore backlog, in a
fold-consistent basis.

**5.** MIX's seen column in `stacklib.py` substitutes the personal head into the already-powered mix (seen changes
sign on one base when done before the power). LIKELY: `route4.py:54`'s KAPPA template prior adds κ·PM to the
numerator but not κ to the denominator (`replay/pl.py:109` does), so §19 (+0.19) and §57b used an over-scaled
template; all runs tonight use KAPPA 0. The grid chain has no gated early boost and sits 0.6–1.0 below the shipped
chain on new, so §69, §71, §90, §92, §103b, §109, §117 and §118 were measured against a weaker baseline.

**6.** Small: `segfp3.py` passes the gate's out-of-fold array in permuted order (≤ 0.07 J); `netaug_score.py`
compares arms with fewer seeds against a three-seed twin; `replay/score.py` reports JG1 and JG2 on different scales;
`weak.py` used an in-sample quantile and the B2_pool mix for every base (fixed before its run, §133 below).

**PKG2** (`studies/13nov/pkg.py`, rewritten). The gate is calibrated from the shipped members' population shadow, and
routing comes from that shadow. Personal nets enter the routed tail only, from the first routed window. FT runs on
sessions the gate never routes. There is no shallow personal member. Arms: T0 (FT twins in both tails), T1b (the
personal nets' frozen twins in the routed tail), T1 (personal nets in the routed tail, v5.2's form), T1w (the dev
form, from window 0 in both tails), F (T1 + FT), O (T1 + OCU), FO, R (T1 + REST), FOR. Smoke on B2_pool, against T1:
F new +0.57 / seen 0.00; O +0.19 / +0.12; FO +0.83 / +0.12 (J +0.48); R +0.40 / 0.00; FOR +0.99 / +1.03 (J +1.01);
T1 against T1b, seen −0.25 (P04 −0.52).

| Check (six-base mean; read new in person terms, seen as P04 and P10 separately) | Prediction (80 % range) |
|---|---|
| FO against T1: new / seen | +0.75 (+0.40 to +1.10) / 0.0 (−0.5 to +0.5) |
| FOR against T1: new / seen | +0.95 (+0.55 to +1.35) / +0.3 (−0.4 to +1.0) |
| T1 against T1b (personal nets, package form): seen | −0.1 (−0.5 to +0.3) |
| **Reading rule** | New counts if its 80 % person interval excludes 0 and the gain without the best person is ≥ +0.3. Seen is read as unresolved unless both P04 and P10 move the same way by ≥ 0.3. The fresh-seed run (job 64325439, now on this code) decides |

| §113 item (13 Nov 00:54 UTC; `runs/ttt7_score.log`, re-scored in person terms; new, all6, against the same run's frozen nets) | Observed | Verdict |
|---|---|---|
| fixA (in-run reference) | +0.34, 80 % [−0.22, +0.92], 6 of 10 people; without P10 (+2.2) +0.14 | replicates §101 |
| gA: fixA with GPU-drawn augmentation | +0.20, [−0.33, +0.72]; without P10 −0.02 | −0.14 against fixA: **outside the ±0.08 rule**, so on-device augmentation is not accepted by the rule. But both arms use the same distribution and differ only in their random draws, so −0.14 measures how much the fine-tuning gain moves with the draws. Every earlier arm reused the same seeded CPU draws, so this noise was never sampled; the ±0.08 rule was below it |
| gE20 (refit every 20) | −0.22 | refuted |
| gC10E20 (warm, 10 steps every 20) | +0.28, [−0.19, +0.77]; without P10 +0.04 | no gain |
| **gC5** (warm, 5 steps every 10, GPU-drawn, cost 0.25) | **+0.59, 80 % [+0.04, +1.14]**, 95 % [−0.25, +1.42], 5 of 10 people; **without P10 (+2.9) +0.34** | Above the range (+0.30). The cheapest arm, and the first fine-tuning form whose 80 % person interval excludes 0 and that holds ≥ +0.3 without P10. By §113's rule it is the package recipe, pending a fresh-seed replication (below) |
| gC10T20 | +0.33 | no gain |

**§113 fresh (written 13 Nov 00:54 UTC, before the job runs; `ttt7.py` tasks 24–71 = seeds 6–17, Leonardo):** the
whole TTT7 table again on fresh nets and fresh draws, scored as SEEDS=6-11 and 12-17
(`runs/night13/ttt7_s6-11.txt`, `ttt7_s12-17.txt`). Prediction for gC5 all6: +0.40 (+0.10 to +0.70) on each fresh set
(winner's-curse shrinkage from +0.59). gA − fixA: 0.00 (−0.20 to +0.20). **Bar:** gC5 is the fine-tuning recipe if,
on both fresh sets, its 80 % person interval excludes 0 or its gain without the best person is ≥ +0.25.

### 141. SEQ2 (explore, top of the backlog after §140): the learned sequence stage in a fold-consistent basis (written 13 Nov 00:58 UTC, after a one-fold smoke and before the full run; code `studies/13nov/seqstage2.py`, Leonardo CPU job)

**Ceiling, and why it could be large:** the hand-built session stages (rebalancing, SessionEM, TSEM) are worth about
+9.4 points over the mix for new people (70.8 → 80.2). Session-own representations with labels reach 81–82 %, and
the whole-session class-geometry oracle reaches 89. A stage that learns how to adapt to a session, instead of
following three fixed rules, aims at that gap. It has never had a valid test: §70 pooled fold-specific coordinates
(§140). This is the second form of the idea, with §70's model unchanged so that only the basis fix is tested. A third
form (a recurrent reader with random rotations per pseudo-session, as §70's text described) follows if this one
moves.

Smoke (fold P01 only, 20 steps): on the chain's output, P01 +1.6, the other people untouched.

| Check (B2_pool / B2_s345; new; 400 steps, d 24, seeds 0–2) | Prediction (80 % range) |
|---|---|
| Learned stage alone on the mix | 75 (70–79), against 80.2 for the hand-built stages; §70 gave 64.6 |
| Learned stage on the chain's output, against the chain | −0.5 (−3 to +0.8); §70 gave −11 |
| Product with the chain | 0.0 (−1.5 to +0.8) |
| **Bar** | A third form is queued if the stage on the chain's output, or its product, is ≥ +0.3 new with the 80 % person interval above 0, or if the stage alone on the mix comes within 2 points of the hand-built stages. Otherwise the basis fix alone does not rescue this model, and the record says that, not that learned stages fail |

| §140 PKG2 item (13 Nov 00:58 UTC; `runs/pkg2_13nov.log`; six bases; seen read as P04 and P10) | Observed | Verdict |
|---|---|---|
| Routing | all 762 seen rows and none of the 2,655 new rows routed, on every base | as before |
| FO (FT + OCU) against T1 | **new +0.86, 80 % [+0.31, +1.42], 7 of 10, without P10 +0.55**; seen +0.01 (P04 +0.19, P10 −0.18; bases −0.68 to +0.37); J +0.43 | Hit (+0.75): **new counts by the rule**; seen unresolved (prediction 0.0) |
| **FOR (FT + OCU + REST)** against T1 | **new +1.08, 80 % [+0.62, +1.51], 8 of 10, without P10 +0.83**; seen +0.61 (P04 +0.16, P10 +1.07; bases +0.11 to +1.03); **J +0.85** | Hit (+0.95 / +0.3): **new counts**; seen positive on all six bases but P04 only +0.16, so unresolved by the rule. **The v4.1 path reverses §130b's verdict: REST does not cost returning people here** |
| F, O, R alone against T1 | new +0.46 (without P10 +0.12) / +0.14 / +0.34; seen 0.00 / +0.01 / +0.05 | singles are small; the stack is where the gain is |
| **Personal nets, package form** (T1 against T1b, routed tail from the first routed window) | **seen −0.36 (P04 −0.78, P10 +0.06; bases −0.77 to −0.12)** | Prediction −0.1 (−0.5 to +0.3): hit. Unresolved by the rule (P04 and P10 disagree), **but negative on all six bases and −0.78 for P04**. The dev form (T1w against T1b) is −0.42 here as well: with this harness's twins neither form gains |

**Reading.** In the package's own chain, the new-people stack (FT + eye-net view + rest member) gives +1.08 for new
people, +0.83 without P10, with an 80 % person interval of +0.62 to +1.51. Returning people do not lose from it (+0.61,
positive on every base, but unresolved). This is now the strongest measured design input. The personal nets of v5.2 do not
gain in package form, and they cost P04 about 0.8 here. The fresh-seed run of this replay (job 64325439) and the
§113/§135 replications of the fine-tuning decide the next candidate after the window.

**Night incident (13 Nov 01:01 UTC).** The fresh-seed SincShallow tasks of §135 and §137 wave 2 (seeds ≥ 6) died at
start: the pretrained SincShallow trunks exist for seeds 0–5 only. `leo_array.sh` ended each task with `echo EXIT
$?`, so Slurm showed them as COMPLETED. The scorers refused to run on the missing files, as designed, so no zeros were
printed. The smoke tests used seed 0 and could not catch it. Fixes:
- `ttt5.make` uses the trunk of seed mod 6 for seeds ≥ 6 and prints it. Fresh SincShallow nets share their
  pretrained trunks with seeds 0–5 but have their own training seed; the other three families are fully fresh.
- `leo_array.sh`, `leo_cpu.sh` and `leo_dbg.sh` now exit with the task's own exit code.
- Only the sinc tasks were rerun (`scripts/cloud/night13d.sh`: arrays 64328025–64328027), and the six failed
  scorings were resubmitted behind them (64328028–64328033).

From now on a smoke covers the highest seed an array will use.

| §131 item (13 Nov 01:26 UTC; `runs/ttt8_score.log`; seeds 0–5, new, all6 against frozen; person terms) | Observed | Verdict |
|---|---|---|
| cum10 (in-run reference) | +0.46, 80 % [−0.24, +1.14]; without P10 (+3.5) +0.12 | identical to §108 (same nets and same draws, deterministic) |
| cum20 / cum5e5 / cum10c2 | +0.38 / +0.19 / +0.34; without P10 −0.08 / −0.19 / −0.02 | no gain |
| cum10e5 (10 steps every 5 windows, cost 1) | +0.61, [−0.24, +1.49]; without P10 (+4.9) +0.14 | +0.15 over cum10, but all of it is P10; the rule waits for the fresh sets (§135) |
| cum10lr (lr 1e-3) | +0.49; without P10 +0.12 | no gain |

**Reading.** On seeds 0–5 every warm-start variant is P10 plus about ±0.15 for the other nine. Variants of steps,
spacing, learning rate and consistency weight do not move the nine. The exception so far is gC5 of §113 (+0.34 without
P10), which differs in its augmentation draws. That points again at draw noise in the "without P10" figure. The
fresh-seed sets (§113 fresh, §135) decide whether any form gains for people other than P10.

| Night results (13 Nov 01:27 UTC; Leonardo `runs/night13/`; new, all6 against the same run's frozen nets; person terms) | Observed | Verdict |
|---|---|---|
| **§137 true labels (oracle), cum10**, seeds 0–5 / fresh 6–11 | **+1.89, 80 % [+1.26, +2.55], 9 of 10, without P10 +1.49 / +1.78 [+1.06, +2.52], without P10 +1.39** | **Hit (+2.0). The bar (≥ +1.5) is met: label quality is the lever, with a ceiling of about +1.5 for the nine people other than P10, four times what any label-free form reaches** |
| §137 true labels, fixA | +1.14, without P10 +0.81 | the warm start uses labels better than the refit-from-frozen form |
| §137 stack-chain labels, cum10, seeds 0–5 / fresh 6–11 | +0.20, without P10 −0.20 / **+0.00**, without P10 −0.40 | **Miss** (+0.55 predicted): labels from the FT+OCU stack chain are *worse* for fine-tuning than the v3 chain's (+0.46), although that chain is 0.75 more accurate. A plausible mechanism, not tested: the stack's labels already contain the fine-tuned nets' own errors (FT is in the stack), so the nets learn to agree with themselves. The bar for a round-3 refresh is not met |
| §137 round 3 (labels from round 2 + nine-seed eye-net), cum10 | +0.21, without P10 −0.21 | no climb: iterated self-training does not help in this form |
| §139 TTT10 head2 (bias included, lr 2e-3) | **−0.02**, without P10 −0.21; cum10 in-run +0.46 | **Refuted** in its reviewed form too: head-only fine-tuning does not carry the gain |
| §135 fresh cum10, seeds 6–11 / 12–17 | +0.32, without P10 −0.02 / +0.40, without P10 +0.12 | Below the bar (+0.25 on both sets with the rule's base count; in person terms: the nine people other than P10 gain about 0). **Fine-tuning on the v3 chain's labels is a P10 effect** |
| §113 fresh fixA, seeds 6–11 / 12–17 | +0.17 / +0.39; without the best −0.01 / +0.17 | the same; gC5 rows still scoring |

**Reading.** Fine-tuning works when its labels are right (+1.4 to +1.5 for the nine people other than P10). With the
chain's own labels it works only for P10. The label-free routes tried tonight, better chain labels and iterated
rounds, made it worse. This is the high-ceiling gap to attack next. The labels must be right on the windows the nets
get wrong, and errors independent of the nets are what the labels need. Next forms: labels from members that do not
include the nets being tuned (core1, Hilbert and teacher only); the true-label ceiling by window confidence, to find
which windows' labels matter; and fine-tuning only on windows where independent members agree.

### 142. TTT12 (explore, high ceiling): labels whose errors are independent of the nets being tuned (written 13 Nov 01:28 UTC, before any job ran; `ttt9.py` LAB=indep / agree / truelow, labels from `studies/13nov/make_labels2.py`, seeds 0–11, Leonardo)

**Ceiling, and why it could be large:** true labels give +1.4 to +1.5 for the nine people other than P10 (§137); the
v3 chain's labels give about 0 for them; the stack chain's labels, which contain the tuned nets' own outputs, give
less. Hypothesis: what limits fine-tuning is not label accuracy but that the labels share the tuned nets' errors.
Label files (B2_pool dev chain):
- indep: a chain without the three SessionBN families (core1, Hilbert, transformer, shallow nets, teacher). New
  77.4 against v3's 80.2: less accurate, but its errors are not the tuned nets'.
- agree: v3's labels, kept only where the independent chain agrees (91 % of new windows; label accuracy 78.7 % there,
  54.3 % on the other 9 %).
- truelow (ORACLE, never shippable): true labels on the low-confidence half (where v3 is right 62 %), v3 elsewhere.
  This decomposes the §137 ceiling.

| Check (cum10 all6, new, against frozen; seeds 0–5 and fresh 6–11; read without P10) | Prediction (80 % range) |
|---|---|
| indep | +0.35 (−0.10 to +0.80); without P10 +0.20 |
| agree | +0.50 (+0.10 to +0.90); without P10 +0.25 |
| truelow (oracle) | +1.4 (+0.8 to +2.0); without P10 +1.0 |
| **Bar** | A label-free source counts if, without P10, it is ≥ +0.30 on both seed sets: then it replaces the v3 labels in the fine-tuning recipe. truelow ≥ +1.0 without P10 means the ceiling sits in the hard windows, and the next forms aim there (labels on low-confidence windows from the independent members only) |

| §141 item (13 Nov 02:23 UTC; Leonardo `runs/night13/seq2.txt`; B2_pool / B2_s345, new) | Observed | Verdict |
|---|---|---|
| Learned stage alone on the mix | 67.6 / 68.4, against 80.2 / 80.1 hand-built (the mix is 70.8 / 70.5) | **Miss** (75 predicted): still below the mix itself |
| On the chain's output, against the chain | −8.0 / −8.8, 0 of 10 people | **Miss** (−0.5 predicted) |
| Product with the chain | −3.7 / −4.0 | Miss |

**Reading.** The basis fix did not rescue this model: in a fold-consistent basis it still loses 8–12 points, almost
as in §70. What fails is therefore this architecture or its training (attention over the session's windows,
trained with true labels on nine people, 400 steps), not only the basis. Per §141's rule the record says this form
fails; it does not say learned stages fail. A third form would need a different inductive bias (e.g. learning only the
parameters of the hand-built stages, per session type) and is not queued tonight.

### 143. Night wave of 13 Nov 02:25 UTC: fresh seeds for the stack's non-fine-tuning members (written before any job ran)

Since fine-tuning's share of the stack is a P10 effect (§135), the stack's gain for the nine other people rests on the
eye-net view (OCU) and the rest-class member (REST). Both have three seeds so far (OCU nine since §134).
- **OCU seeds 9–17** (`ocunet.py` nomix, tasks 19–35 odd; already smoke-tested on Leonardo).
- **REST seeds 3–8**: the rest windows (`data/proteus_windows_start-4.5.npz`) exist only on Galvani, so a serial job
  rebuilds them with the official pipeline (`scripts/export_windows.py -4.5`). A debug smoke of `rest4.py` follows, then
  rest4 shallow r1 and c3 at seeds 3–8 (tasks 21, 22, 27, 28, …, 51, 52).
- **PKG3, fresh everything**: the package-form replay (§140 arms) with FT seeds 6–11, OCU seeds 9–17 and REST seeds 3–8,
  none used in any choice (`runs/night13/pkg_fresh2.txt`). REST seeds are now named explicitly in `stacklib.py` and
  `pkg.py`, as OCU's are.

| Check (PKG3, against T1, six-base mean; read without P10) | Prediction (80 % range) |
|---|---|
| FO new / FOR new | +0.70 (+0.25 to +1.15) / +0.90 (+0.45 to +1.35); without P10 +0.45 / +0.65 |
| FOR seen | +0.3 (−0.4 to +1.0) |
| **Bar** | The new-people stack (FT+OCU+REST) is the design input for the build after the window if FOR's new gain without P10 is ≥ +0.40 with its 80 % person interval above 0 on these fresh seeds |

Not tonight, with reasons:
- The teacher line: the corpora are not on Leonardo and Galvani needs the user's 2FA.
- MixStyle and EEGNeX retests: low ceilings on the record and new code; they are next after the high-ceiling work.
- Self-supervised training on the rest runs: not specified yet.
- The fine-tuning line: its next form waits for TTT12's labels, which are scoring now.

| Night results 2 (13 Nov 02:25 UTC; Leonardo `runs/night13/`; new in person terms) | Observed | Verdict |
|---|---|---|
| **§138/§140 package replay on fresh seeds** (FT seeds 6–11, eye-net 3–8; REST seeds 0–2 not fresh), FOR against T1 | **new +1.06, 80 % [+0.54, +1.57], 8 of 10, without P10 +0.84; seen +0.84 (P04 +0.53, P10 +1.14; every base +0.49 to +1.43); J +0.95** | **Replicates PKG2** (+1.08 / +0.83 / J +0.85). Now P04 and P10 both move ≥ +0.3, so **seen counts by §140's rule**. **FT+OCU+REST is the design input for the build after the window** (§143's bar is read again on PKG3's fully fresh seeds) |
| same, R (REST alone) / FO / F | new +0.50 (without the best +0.26), seen +0.45 (P04 +0.42, P10 +0.47) / new +0.65 (without P10 +0.32), seen +0.06 / new +0.32 (without P10 −0.02) | The rest-class member carries the returning-people gain; FT adds only through P10 |
| same, personal nets in package form (T1 against T1b) | seen −0.40 (P04 −0.87, P10 +0.06) | replicates §140: **v5.2's personal nets cost P04** |
| §136 STACKF (fresh FT and eye-net seeds, `A.system`) | FT+OCU +0.65 [+0.04, +1.25]; FT+REST+OCU+MIX **+1.08** [+0.58, +1.57] | Hit for both (+0.70 / +0.95). The design input is confirmed on new (bar ≥ +0.50); its seen column is v3 and not read |
| **§142 truelow (oracle: true labels on the low-confidence half only)**, cum10, seeds 0–5 / 6–11 | **+2.03 [+1.41, +2.69], without P10 +1.62 / +1.84, without P10 +1.41** | **Hit, above** (+1.4): true labels on the hard half alone reach the full true-label ceiling of §137. **The whole ceiling of fine-tuning sits in the labels of the low-confidence windows** |
| §142 indep / agree labels, cum10, both seed sets | +0.24 / +0.20 and +0.29 / +0.17; without P10 −0.11 / −0.12 and −0.06 / −0.18 | **Refuted:** labels from members independent of the tuned nets do not help the nine other people either. On the hard windows every label-free source available is about as wrong as the tuned nets |
| §113 fresh gC5, seeds 6–11 / 12–17 | +0.38 [−0.15, +0.93], without P10 +0.16 / +0.59 [+0.09, +1.11], without P01 +0.38 | **Bar missed on 6–11** (the 80 % interval includes 0, and +0.16 is below +0.25), met on 12–17. Over three seed sets gC5 averages +0.52 and is the best fine-tuning form, but its gain for the nine is not reliable |
| §135 fresh cum10e5 / cum20 (12–17) | +0.39 (−0.02 without P10) / +0.35 on 6–11; +0.65 (+0.28) / +0.59 (+0.24) on 12–17 | no variant replicates on both fresh sets |

**Reading.** The new-people stack replicates in the package's own chain on fresh seeds: J +0.95, with returning people
gaining as well (the rest member). Fine-tuning's own share is P10. Its true ceiling (+1.4 to +1.6 for the nine)
needs correct labels on the hard half of the windows, and no label-free source tried tonight provides them. The
high-ceiling question for fine-tuning is now narrow: where can correct labels for the hard windows come from? Not
from the chain, not from independent members, not from iterated rounds.

### 144. REST2 (exploit): more of the rest-class lever in the package chain (written 13 Nov 02:26 UTC, before any result; `pkg.py` ARMS=FOR,FOR2,FORw2, Mac, stored seeds 0–2; fresh FBCNet rest seeds 3–8 queued on Leonardo as array 64335969)

The rest-class member carries the returning-people gain of the stack, and part of the new one (§140 fresh: R alone new
+0.50, seen +0.45, both P04 and P10 up). Two forms that may add: FOR2 adds the FBCNet rest-class net (rest4_fb_r1) as a
second member; FORw2 doubles the shallow rest member's weight.

| Check (against FOR, six-base mean, person terms) | Prediction (80 % range) |
|---|---|
| FOR2: new / seen | +0.15 (−0.20 to +0.50) / +0.10 (−0.40 to +0.60) |
| FORw2: new / seen | +0.05 (−0.30 to +0.40) / +0.10 (−0.40 to +0.60) |
| **Bar** | An arm replaces FOR in the design input if new without the best person is ≥ +0.20 over FOR and seen does not fall on both P04 and P10 |

| §144 item (13 Nov 02:31 UTC; `runs/pkg_rest2_13nov.log`; against FOR, six bases) | Observed | Verdict |
|---|---|---|
| FOR2 (+ the FBCNet rest-class net) | new −0.27 (without P10 −0.22); seen −0.08 (P04 −0.22, P10 +0.05) | **Refuted**: a second rest member costs new people |
| FORw2 (shallow rest member at weight 2) | new −0.14 (without P10 −0.19); seen −0.64 (**P04 −1.43**, P10 +0.13) | **Refuted**: a heavier rest member costs P04 |

**Reading.** The rest lever is at its best as one shallow member at weight 1. More of it hurts. FOR stays the
design input. The fresh FBCNet rest seeds (array 64335969) are no longer needed for this question; they are left to
finish as data.

**§120 addendum (13 Nov 01:23 UTC; review items 5, 6 and 8, v52-build session; details in `docs/V52_BUILD.md`):**
- Item 5: the 12 twin-based personal bases (raw FBCNet, SincShallow) were rebuilt from the shipped traces themselves:
  120 traces, traced vs eager 0.0, graft check 0.0. Corrected set of 300 in `runs/pnets_jit_shipbase`. `trained()` now
  fails unless asked to train, and `build_v52.py` refuses twin bases (tested: it lists the 12 old ones).
- Item 6: forced impostor personal nets on the 14 public sessions cost −0.06 points (worst −1.04). This is in-sample at
  a ceiling of about 99 %, so it is weakly informative.
- Item 8: per-window cost grows by 0.014 ms (v5.1b) and 0.021 ms (v5.2b) per window index (78 → 88 and 80 → 93 ms over
  577 windows, A100). A one-off ~11 s of TorchScript compilation when a person's traces are first used in a process,
  not per session.

### 105b. Sealed expectation for v5.2 over v5.1b, restated (13 Nov 02:41 UTC; replaces §105's +0.20)

§105's +0.20 (−0.10 to +0.50) rested on §78 and §119. Both gave the personal nets to whole returning sessions from
window 0, in both tails (§140). Package-form evidence:

| Source | Personal nets against their twins, seen (P04, P10) |
|---|---|
| Reviewer, from §119's stored outputs, package form | −0.05 (−0.10, 0.00) |
| PKG2 (§140), six bases | −0.36 (−0.78, +0.06), negative on all six bases |
| PKG fresh seeds (Night results 2) | −0.40 (−0.87, +0.06) |
| Impostor study (build session, in-sample impostor at about 99 %) | −0.06 points, worst −1.04 |

Limits: PKG uses study nets (pft), not the 120 rebuilt traces of `runs/pnets_jit_shipbase`. On the seen side one window
is about 0.07 J. **Restated expectation for v5.2 over v5.1b: −0.05 sealed (−0.30 to +0.10).** The dev seen change is
−0.05 to −0.40, J −0.03 to −0.20, scaled by §119's proxy-to-sealed ratio of about 0.2 for the returning-session
share. No dev number supports a gain, and every package-form number has P04 below zero. **v5.2 / v5.2b should not be
uploaded as built.** The returning-people gain of the current design comes from the rest-class member (§140 fresh:
R alone, seen +0.45, P04 +0.42, P10 +0.47), not from personal nets. The build session's runtime and guard work (§112,
§120, `docs/V52_BUILD.md`) carries over to the next package.

| §132 TEACH2 item (13 Nov 02:46 UTC; `runs/teach2_13nov.log`; new, against the rebuilt m7c teacher at weight 5; shipped / FT+OCU context) | Observed | Verdict |
|---|---|---|
| Rebuilt teacher against the shipped one | +0.07 / −0.07 | harness hit (0.0) |
| m8s (Scherer) at weight 2 / 5 | −1.56 / −1.40; −0.71 / −0.51 | replicates §18 in both contexts |
| Session-normalised m7c (n7c) / m8s (n8s, w2 / w5) | −0.33 / −0.14; −1.37 / −1.72 and −1.00 / −0.61 | **Refuted.** Per-session normalisation does not rescue the stronger teacher, so reason (i) of §132 fails: m8s does not fail on session offsets |
| Cognitive-heads member next to the teacher (cog7 / cog8, w1, w2) | −0.33 to +0.08; −0.22 to −0.16 | **Refuted** (≥ +0.30 needed in both contexts); seen columns are the v3 path |

**Reading.** Rebuilt from stored heads, none of the teacher redesigns gains in either context. The stronger teacher
(+3.3 alone) keeps losing about −0.5 to −1.7 in the chain, and the gap 2 forms possible without retraining are closed.
The ceiling argument for independent members (+3.1 for a 70 % independent member) is unchanged. It needs a teacher
trained differently (e.g. trained against the chain's errors), which needs the corpora on Galvani.

| §133 WEAK item (13 Nov 02:46 UTC; `runs/weak_13nov.log`; per-base, out-of-person thresholds; new) | Observed | Verdict |
|---|---|---|
| FT context: ungated / q50 / q33 / q67 / inv50 | +0.46 / **+0.38** (8 of 10 people ≥ 0) / +0.09 / +0.31 / +0.08 | q50 is 0.08 below ungated, and the rule needs 0.05: **not adopted by the rule** |
| FT+OCU context: ungated / q50 / q33 / q67 / inv50 | +0.72 / +0.54 (80 % [+0.17, +0.93], 8 of 10) / +0.17 / +0.44 / +0.17 | 0.18 below ungated: not adopted |
| Without P10 (from the per-person rows) | FT: ungated +0.11, q50 +0.22. FT+OCU: ungated +0.37, q50 +0.32 | gating the weak half helps the nine in the FT context and is neutral with OCU |

**Reading.** Gains concentrate partly in weak sessions: inv50 (strong half only) is +0.08 / +0.17, below the
closing mark of +0.25. Gating to the weak half keeps most of the gain at half the run time, with fewer people losing
(8 of 10 ≥ 0 instead of 6). It misses the rule's 0.05 margin, so it is recorded as a run-time option for the package,
not as a gain. Seen columns are the v3 path and are not read.

### 145. Deep night queue of 13 Nov 03:04 UTC (written before any of these jobs was submitted)

The parent session asked for a queue that carries itself for hours. Waves that need no pending result:
- **A. Replication sets B and C for the stack** (the design input read on three independent seed sets): ttt8 seeds
  18–29 (FT; tasks 72–119); eye-net nomix seeds 18–35 (tasks 37–71 odd); shallow rest-class seeds 9–20 (tasks
  6·seed+4). Then PKG on set B (FT 18–23, OCU 18–26, REST 9–14) and set C (FT 24–29, OCU 27–35, REST 15–20):
  `runs/night13/pkg_setB.txt`, `pkg_setC.txt`. Prediction for FOR against T1, each set: new +0.90 (+0.40 to +1.40),
  without P10 +0.65; seen +0.5 (−0.3 to +1.2). **Bar:** the design input stands if, on both sets, its new gain without P10
  is ≥ +0.40 and its 80 % person interval excludes 0.
- **B. Eye-net from two families**: the mix arm of `ocunet.py` (Mixup), seeds 3–11 (tasks 6–22 even), so the view can
  be built from both arms (scored on the Mac when they land).
- **C. §145 CONFHALF (fine-tuning aimed at the hard windows, label-free)**: v3 labels with the low-confidence half
  switched off (`chain_confhalf.npz`); ttt9 LAB=confhalf, seeds 0–11. Since §142 put the whole ceiling in the hard
  half's labels, the wrong labels there may also carry the damage. Prediction, cum10 all6, without P10: +0.20 (−0.15 to
  +0.55), against the v3 labels' +0.12 / −0.02 / +0.12. **Bar:** ≥ +0.30 without P10 on both seed sets → these labels
  replace v3's in the fine-tuning recipe.
- **D. The teacher line with real retraining** (gap 2): MOABB is not installed on Leonardo and the corpora are not
  there, so a serial job installs it and serial jobs download and window the corpora (`scripts/export_moabb.py`).
  Pretraining (12 seeds per trunk: the m7c control, +Scherer, and a cognitive-only teacher) is queued once the
  corpora exist. This is the one line here that runs for hours, and the first step may fail (download sizes, licence
  prompts); it is marked as an attempt.

Not queued, with reasons: the chain-level people curve needs every member retrained per subset (core1, Hilbert, about
65 nets, teacher map), which is a pipeline that does not exist yet; self-supervised trunks need a specified objective
(next design task).

**§145 submitted (13 Nov 03:06 UTC).**
- A–C (`scripts/cloud/night13g.sh`):
  - ttt8 sets B and C: 64338742, 48 tasks
  - eye-net nomix seeds 18–35: 64338744
  - eye-net mix seeds 3–11: 64338745
  - rest-class seeds 9–20: 64338746
  - CONFHALF: 64338747, 48 tasks
  - scorings: pkg_setB and pkg_setC (64338748, 64338749), confhalf_s0-5 / s6-11
  - GPU size: about 28 GPU-hours (fine-tuning tasks about 20 min, eye-net about 5 min, rest net about 3 min). With
    Leonardo's free GPUs they drain in about 1.5 h; the CPU scorings follow within about 1 h.
- D (`scripts/cloud/leo_corpora.sh`): MOABB install 64338786, then 9 of 14 corpus chunks (PhysionetMI, Cho2017,
  Lee2019_MI, Dreyer2023 1–44). The serial QOS refuses more than about ten jobs per user. Still to submit when slots
  free: Dreyer2023 45–87, Schirrmeister2017 1–7 and 8–14, Shin2017A, Shin2017B.
- Missing, to rebuild in the day once the corpora exist: the pretraining submit and the head-export scripts of the
  11 Nov teachers lived in an untracked scratch folder on Galvani.
- The honest size of tonight's GPU work: the queue does not carry many hours on its own. The one multi-hour GPU line
  (teacher pretraining, 12 seeds × 2 trunks × 3 teachers) waits on the corpus downloads.

### 146. PCURVE: the chain-level people curve, reduced form (written 13 Nov 03:45 UTC, after a one-draw smoke and before the five-draw run; code `studies/13nov/peoplecurve.py`, Mac)

**Why it matters (the ceiling of the whole data side):** core1 rises about +8 points per doubling of training people
(§39, §48). Whether the session stages keep that slope or flatten it has never been measured, and every data-side
idea (more people, synthetic people, unlabelled recordings, other corpora) is sized by it. **Cheapest honest form:**
retraining every member per subset would need a pipeline that does not exist, so a core1 surrogate (balanced LR on
core1's 256 exact-SVD components; it agrees with core1 on about 82 % of windows) is trained on 3, 5, 7 or 9 of the nine
training people (five draws below 9). It is scored alone and through the shipped stages as the only member. The
missing members make the absolute chain about 11 points lower than the shipped chain; the slopes are what is read.
Limit: each fold's PC basis was computed on all nine training people.

Smoke (one draw): core 55.7 / 60.6 / 63.1 / 64.4 and chain 59.9 / 66.6 / 68.3 / 69.3 at k = 3 / 5 / 7 / 9.

| Check (five draws) | Prediction (80 % range) |
|---|---|
| core slope per doubling (k 3–9) | +5.5 (+4 to +7) |
| chain slope per doubling (k 3–9) | +5 (+3 to +7) |
| **Reading rule** | Chain slope ≥ 0.7 × core slope: the stages do not flatten the people gain, so a data-side idea worth +x at core level should be worth about +x in the chain, and the data side stays the biggest ceiling (a tenth person ≈ +1.4 in the chain at +6 per doubling). Chain slope < 0.4 × core slope: the stages already substitute for people, and data-side ideas are worth much less than their core numbers |

| §146 item (13 Nov 03:46 UTC; `runs/pcurve_13nov.log`; new people; five draws below k 9) | Observed | Verdict |
|---|---|---|
| core surrogate, k 3 / 5 / 7 / 9 | 56.7 / 59.8 / 62.7 / 64.4 | **slope +4.9 per doubling** (k 5–9: +5.4); prediction +5.5, hit |
| chain (surrogate through the stages), k 3 / 5 / 7 / 9 | 61.9 / 64.5 / 67.4 / 69.3; stage gain +5.2 / +4.6 / +4.7 / +4.9 | **slope +4.7 per doubling** (k 5–9: +5.7); prediction +5, hit |

**Reading (the rule's first branch):** the chain slope is 0.95 of the core slope. **The session stages do not flatten
the people gain; they add a constant +5 at every number of people.** For the data side this means:
- a data-side idea worth +x at core level is worth about +x in the chain;
- one more real person is worth about +0.7 in the chain (log2(10/9) × 4.7);
- a doubling of effective people is worth about +4.7.

The data side is the largest ceiling on record. The earlier data-side substitutes (generated variation,
people-efficient estimators, Scherer rows) failed at core level, not in the chain, and that conclusion stands.
Unlabelled PROTEUS recordings (rest runs, other crops) and more corpora in the teacher are the routes left. Limit:
one member, a surrogate core, and a PC basis shared across subsets. The slope of the full multi-member chain may
differ.

### 147. SSL1 (explore, gap 6): self-supervised trunks from the unlabelled PROTEUS recordings (written 13 Nov 03:48 UTC, before any result; code `studies/13nov/sslnet.py`, scorer `ssl_score.py`, Leonardo, seeds 0–11 after a debug smoke)

**Ceiling, and why it could be large:** §146 shows the chain keeps core1's +4.7 per doubling of people; the stages do not
substitute for people. The unlabelled recordings are allowed for training and were never used. The Graz rest periods
alone add about half as many windows again as the labelled task windows. A trunk that learns the recordings' own
structure (montage, artefacts, session variation) is the cheapest stand-in for more people.
**Design:**
- Pretraining is per left-out person, on the nine training people's recordings only: no unlabelled data of the scored
  person is used.
- Data: task windows plus the 4 s before each Graz cue, aligned with the official windows' statistics.
- Objective: SimCLR between two augmented views (shift, noise, channel drop, amplitude) on the net's output layer
  widened to 64; 1500 steps of 256 windows.
- The trunk then initialises the shipped FBLight SessionBN recipe, trained and scored as usual against the same recipe
  from a fresh trunk (twin, same seed), with slot 8 swapped in the chain, new people.

| Check (12 seeds, new, six-base mean, against the twin) | Prediction (80 % range) |
|---|---|
| ssl alone against twin alone | +0.8 (−1.0 to +2.5) |
| in the chain (slot 8 swapped) | +0.05 (−0.20 to +0.30): one family is one fifteenth of the mix |
| **Bar** | the chain gain without the best person ≥ +0.10 with the person interval above 0 → SSL trunks for all four SessionBN families (a wave for each), then the stack; alone ≥ +1.5 with the chain flat → the trunk learns but one family cannot carry it, so the four-family wave goes anyway |

| §145 C item (13 Nov 03:48 UTC; Leonardo `runs/night13/confhalf_s*.txt`; cum10 all6, new, against frozen) | Observed | Verdict |
|---|---|---|
| CONFHALF (v3 labels, low-confidence half switched off), seeds 0–5 / 6–11 | **−0.32** (without P10 −0.59) / **−0.14** (without P10 −0.39); fixA −0.07 / −0.23 | **Refuted, below the range** (+0.20 predicted). Fine-tuning on the confident half only *hurts*; the hard half's labels carry fine-tuning's gain even though they are only 62 % right. Together with §142 (true labels on the hard half alone reach the full ceiling): fine-tuning learns from the hard windows. Removing their labels removes the signal, and correcting them gives +1.6 |

### 148. Teacher retraining queued (gap 2; written 13 Nov 03:50 UTC, before any job ran)

- Corpus downloads (serial QOS: four at a time, the rest wait):
  - PhysionetMI and Cho2017 are windowed.
  - Lee2019_MI was resubmitted at 16 GB (it ran out of memory at 8 GB).
  - Dreyer2023, Schirrmeister2017, Shin2017A, Shin2017B and Shin2018WG (word generation) are queued.
- Rebuilt from tracked code: `scripts/export_heads.py` (per-corpus head log-probabilities on the 2,655 PROTEUS
  windows, the format `teach2.py` reads; smoke on an 11 Nov FBCNet checkpoint: 19 outputs, as the m7r corpora give)
  and `scripts/cloud/leo_teach.sh` (EEGNeX and IFNet, seeds 0–11, the m7r recipe, then the head export).
- Queued behind the corpus jobs:
  - **m7c2**, the seven m7 corpora again on Leonardo (the control, and a check of the rebuild against
    `runs/tsch/heads.npz`): 64341544 / 64341545 → heads 64341546.
  - **cog2**, a cognitive-task teacher on Shin2017B (subtraction vs rest) and Shin2018WG (word generation vs rest),
    meant as a member of its own: 64341548 / 64341549 → heads 64341551.
- Scoring (`teach2.py` reading the new heads files, in the shipped and FT+OCU+REST contexts with the weight re-chosen)
  follows when the heads exist. Prediction: cog2 as an extra member, new, +0.2 (−0.3 to +0.7). The rebuild check:
  m7c2 within ±0.3 of the stored m7c teacher in the chain.

| §145 A item (13 Nov 05:16 UTC; Leonardo `runs/night13/pkg_set{B,C}.txt`; package-form replay; FOR against T1; three disjoint seed sets for all three members) | New [80 % person] / without P10 | Seen (P04, P10) | J |
|---|---|---|---|
| Set A (fresh FT and eye-net, REST seeds 0–2; "Night results 2") | +1.06 [+0.54, +1.57] / +0.84 | +0.84 (+0.53, +1.14) | +0.95 |
| **Set B** (FT 18–23, OCU 18–26, REST 9–14) | +0.55 [−0.04, +1.09] / **+0.30** | +0.94 (+1.54, +0.33) | +0.74 |
| **Set C** (FT 24–29, OCU 27–35, REST 15–20) | +0.80 [+0.26, +1.26] / +0.60 | +0.22 (+0.24, +0.19) | +0.51 |
| FO (no REST), sets B / C | +0.50 / +0.68; without P10 +0.20 / +0.38 | +0.45 / +0.27 | +0.48 / +0.48 |
| R (REST alone), sets B / C | +0.20 / +0.15 | −0.01 / +0.08 | +0.09 / +0.12 |

**Verdict by §145's bar (new without P10 ≥ +0.40 with the 80 % interval above 0, on both sets): met on C, missed on B**
(+0.30, interval reaching −0.04). Over the three independent seed sets the design input is positive every time:
- new +0.55 to +1.06, mean +0.80; without P10 +0.30 to +0.84, mean +0.58;
- seen +0.22 to +0.94, every set positive for both P04 and P10;
- J +0.51 to +0.95, mean +0.73.

The first measurement was the highest. Its strength is about two thirds of it, which is what the winner's curse leads one
to expect. REST alone replicates neither its new nor its seen gain (sets B and C about 0); its effect shows only
together with FT and OCU. **Reading: the design input stands as a design (positive on all three sets, both groups),
with an expected dev J of about +0.7, not +0.95.** Translated to sealed with the stage's survival ratio it is small.
With a new-people survival nearer the stage's own it would be about +0.2 to +0.4 sealed; neither ratio has been measured
for new-people levers.

| §147 SSL1 item (13 Nov 05:16 UTC; `runs/night13/ssl1.txt`; 12 seeds; slot 8 swapped; new) | Observed | Verdict |
|---|---|---|
| ssl alone against twin alone | 58.36 against 58.13 (+0.23) | Miss, low (+0.8 predicted) |
| in the chain | **+0.08, 80 % [+0.02, +0.14], 8 of 10 people ≥ 0; without the best +0.05** | Hit on the chain (+0.05 predicted), but below the bar (+0.10 without the best). The four-family wave is not triggered; small, consistent and positive, it stays recorded as an option to stack |

**Teacher retraining (§148), first release failed and resubmitted.**
- All 48 pretraining tasks and both head exports failed within three minutes at 04:31 UTC (`AssertionError: no
  windows for schirrmeister2017`): the pretraining was held behind the download jobs with `afterany`, and both
  Schirrmeister chunks had run out of memory at 8 GB. The failure was loud, as designed.
- Shin2018WG cannot be fetched by MOABB: it expects the raw zip files under `data/shin2018/raw`, which are only on
  Galvani, so **cog2 as specified cannot run**.
- Fixes and resubmission (05:15 UTC):
  - Schirrmeister in five chunks of 2–3 subjects at 16 GB (64347045–64347049).
  - `leo_teach.sh` now runs a pre-flight serial job that checks every listed corpus has windows; the arrays start
    only if it succeeds.
  - m7c2 with the same seven corpora as the shipped teacher (pre-flight 64347050 → 64347051 / 64347052 → heads
    64347053).
  - A changed arm under its own name, **cog1**: a cognitive teacher on Shin2017B alone, subtraction vs rest
    (64347054 → 64347055 / 64347056 → heads 64347058). The word-generation part is missing until the Shin2018 zips are
    copied from Galvani.
- Why no session woke on the failure: these chains have Slurm-held scoring but no local collector. A failure inside
  the chain therefore shows only in sacct or in the next check. The morning note now says this for every chain.

**§148 scoring, TEACH3 (written 13 Nov 06:18 UTC, after the chain completed and before any score; `studies/13nov/teach3.py`).**
The pre-flight passed, all 48 pretraining tasks and both head exports completed, and the heads have the stored layout
(m7c2: the same 19 corpus heads in the same order). Arms against the stored m7c teacher at weight 5, new people, in the
shipped and the FT+OCU+REST contexts: m7c2 replacing the teacher (rebuild check), and cog1 as an extra member at weight
0.5, 1 or 2.

| Check (six-base mean, new) | Prediction (80 % range) |
|---|---|
| m7c2 against stored m7c (rebuild check) | 0.0 (−0.3 to +0.3), both contexts |
| cog1 at its best weight | +0.15 (−0.3 to +0.6) |
| **Bar** | the rebuild is valid if m7c2 lies within ±0.3 in both contexts. cog1 counts if it is ≥ +0.30 in both contexts with the gain without the best person ≥ +0.15. A failed rebuild check voids the cog1 reading too |

| §148 TEACH3 item (13 Nov 06:23 UTC; `runs/teach3_13nov.log`; new, against the stored m7c teacher at weight 5; shipped / FT+OCU+REST) | Observed | Verdict |
|---|---|---|
| m7c2 alone / stored m7c alone | 50.69 / 50.46 | the retrained teacher reproduces the stored one alone |
| m7c2 replacing the teacher in the chain | −0.17 [−0.48, +0.12] / +0.19 [−0.09, +0.46] | **Rebuild check passed** (within ±0.3 in both contexts; prediction 0.0) |
| cog1 alone | 35.5 (a two-class subtraction-vs-rest teacher mapped to three classes) | — |
| cog1 as an extra member, weight 0.5 / 1 / 2, shipped | +0.02 / +0.01 / −0.02 | nothing in the shipped chain |
| cog1, FT+OCU+REST context | +0.11 / **+0.22 [+0.10, +0.34], 8 of 10 people, without P01 +0.17** / +0.18 | positive with an interval above 0 in the design context, but **below the bar** (≥ +0.30 in both contexts) |

**Reading.** The teacher pipeline is rebuilt and works on Leonardo end to end: corpora → pretraining → heads → map, and
the control reproduces the stored teacher. The cognitive teacher on Shin2017B alone adds nothing to the shipped chain
and +0.2 inside the design input, where its interval stays above 0. It is not adopted by the rule. It is the first
teacher variant that gains anywhere, and its word-generation half (Shin2018WG, zips on Galvani) is still missing.
Next forms for the day:
- cog2 with Shin2018WG (needs the zips copied from Galvani, so the user's 2FA);
- the cognitive heads trained into the m7 teacher as one 4-way block (`pretrain.py --unified`, R17);
- more corpora in the m7 teacher (the map's "more corpora": the KEEP list's openly licensed MI corpora not yet
  downloaded).

### 149. Teacher line, next forms (written 13 Nov 06:23 UTC, before any job ran)

The rebuilt pipeline reproduces the stored teacher (§148), so the gap 2 forms that need retraining go out:
- **m7u**: the seven m7 corpora, with PhysionetMI's and Shin2017B's classes trained into one shared 4-way block
  (motor imagery / subtraction / word / rest; `pretrain.py --unified physionetmi,shin2017b`, R17). This puts the
  cognitive contrast inside the teacher instead of beside it.
- **m10**: the seven m7 corpora plus three openly licensed MI corpora from export_moabb's KEEP list (Zhou2020,
  Weibo2014, Liu2024), downloaded tonight. This is "more corpora in the teacher", the data-side route §146 says is
  worth its core-level size.
Both use 12 seeds × EEGNeX and IFNet, the pre-flight check, and heads into `runs/teach/`. They are scored with teach3's
arms (replacing the teacher, weight re-chosen 3 / 5 / 7) in the shipped and FT+OCU+REST contexts.

| Check (new, six-base mean, against the stored m7c teacher at weight 5) | Prediction (80 % range) |
|---|---|
| m7u replacing the teacher, best weight | +0.2 (−0.4 to +0.8) |
| m10 replacing the teacher, best weight | +0.3 (−0.4 to +1.0) |
| **Bar** | ≥ +0.30 in both contexts, with the gain without the best person ≥ +0.15 → a teacher candidate for the next package (only the map changes; trunks are traced as today's) |

### 150. OR: the stack without fine-tuning (written 13 Nov 07:37 UTC, before any result; `pkg.py` ARMS=FOR,FO,OR, Mac, seed set of PKG2)

Fine-tuning is the package's run-time cost (§107), and its own share of the stack is P10 (§135). But on sets B and C,
REST alone was about 0 while FOR gained, so FT may matter through the combination. OR (eye-net view + rest member,
no FT) has never been scored. Same replay and seeds as PKG2 (§140).

| Check (against T1, six-base mean) | Prediction (80 % range) |
|---|---|
| OR new / seen | +0.35 (0.00 to +0.70) / +0.40 (−0.30 to +1.00) |
| FOR − OR (what FT adds inside the stack), new | +0.70 (+0.30 to +1.10); without P10 +0.45 |
| **Reading** | if FOR − OR without P10 is ≥ +0.30, FT earns its run time inside the stack and the package carries it on never-routed sessions; if < +0.15, the package ships OR and spends no run time on FT |

| §150 item (13 Nov 07:45 UTC; `runs/pkg_or_13nov.log`, per person `runs/pkg_or_13nov.npz`; against T1, six bases) | Observed | Verdict |
|---|---|---|
| OR (eye-net view + rest member, no FT) | **new +0.61 [+0.22, +1.04], 8 of 10, without its best person +0.39; seen +0.61 (P04 +0.16, P10 +1.07); J +0.61** | Hit (+0.35 / +0.40) |
| FOR − OR (what FT adds inside the stack) | new +0.47 [−0.07, +1.02], 6 of 10 people; **without P10 +0.17** (P10 +3.2); seen 0.00 | between the two readings (+0.15 / +0.30): **undecided**. FT adds +0.17 for the nine other people, and its run time is the package's largest cost |

**Reading for the design.** OR carries the stack's returning-people gain in full (FT never reaches a routed
session) and most of its new-people gain except P10's, at no run-time cost. FT is an add-on. It is worth carrying only
if its cheap form fits the run-time budget, and its fresh-seed share is no better than +0.17 for the nine. OR
should therefore be scored on the three seed sets as FOR was, when Leonardo is back (OCU and REST seeds of sets B and C
are there).

### 151. V6: v5.1b plus the eye-net view and the rest member, on the shipped members (written 13 Nov 07:46 UTC, before any result; `pkg.py` ARMS=V6,V6b, Mac)

Every stack arm so far (§140–§150) swapped the SessionBN slots for study twins and ran the stage at 0.15, so its
numbers are deltas between study systems. The package question is simpler: what do the eye-net view (OCU) and the
rest member (REST) add to v5.1b as uploaded? Shipped members everywhere, REST as an extra member at weight 1, OCU in
SessionEM's view, stage at 0.10 (V6) or 0.15 (V6b); against P0 = v5.1b.

| Check (against P0, six-base mean) | Prediction (80 % range) |
|---|---|
| V6 new / seen / J | +0.45 (+0.05 to +0.85) / +0.45 (−0.30 to +1.10) / +0.45 (+0.05 to +0.85) |
| V6b − V6 (stage 0.15 without personal nets) | seen −0.1 (−0.5 to +0.3) |
| **Bar** | V6 is the next package's design if its new gain without the best person is ≥ +0.25 with the 80 % interval above 0, and seen does not fall for both P04 and P10 |

| §151 item (13 Nov 07:49 UTC; `runs/pkg_v6_13nov.log`, per person `runs/pkg_v6_13nov.npz`; against P0 = v5.1b, six bases) | Observed | Verdict |
|---|---|---|
| **V6** (shipped members + OCU + REST, stage 0.10) | **new +0.34 [−0.04, +0.72], 7 of 10, without the best +0.14; seen −0.24 (P04 −1.39, P10 +0.91; bases −1.21 to +1.43); J +0.05** | **Miss, bar not met** (prediction +0.45 / +0.45 / +0.45) |
| V6b (the same at stage 0.15) | new +0.34; seen −0.06 (P04 −1.03, P10 +0.90); J +0.14 | Miss |

**Reading. This changes what §140–§150 mean.** Those arms were measured against T1: study twins in the SessionBN
slots, personal nets in the routed tail, stage 0.15. T1 sits about 0.7 below v5.1b on returning people (T0 against P0
−0.26 seen; T1 against T0 −0.45). Most of the stack's "seen" gain (+0.6 to +0.9) was the rest member and eye-net view
repairing that weaker base, not a gain over the uploaded system. On v5.1b's own members:
- OCU and REST add +0.34 for new people, +0.14 without the best person;
- they cost P04 1.0–1.4 and help P10 by about 0.9;
- J is +0.05 to +0.14.

The new-people part is also smaller on the shipped members than on twins (OR against the twin base gave about +0.53
for new people). **v6 as designed (no FT) is not a package by its own bar.** What is not measured: FT on the shipped
members (FT outputs exist only for study nets). FT carries most of the new-people gain, P10 included, and P10's +3 as a
new person is real in dev. Measuring it needs fine-tuning runs that start from the shipped SessionBN nets, a GPU job
for when Leonardo is back.

**§149 incidents and resubmission (13 Nov 07:55 UTC, before any §149 score).**
- **m7u.** All 24 pretraining tasks completed, but the head export failed. With `--unified`, `pretrain.py` keeps only
  the shared block's 3-class head (MI / SUB / WORD) and drops the per-corpus heads. `export_heads.py` now reads that
  head directly as the teacher's input ("unified:MI/SUB/WORD", 3 log-probabilities per trunk) and asserts `--unified`
  in the meta; the export is resubmitted.
- **m10.** Zhou2020 and Liu2024 failed in `export_moabb.py`: their channel names do not match the montage. Only
  Weibo2014 arrived, so the pre-flight stopped m10, as designed. The arm is resubmitted under its true corpus set as
  **m8w** (the seven m7 corpora + Weibo2014), with the same prediction and bar as m10.

**§150/§151 replication queued (13 Nov 07:56 UTC, before any result):** OR and V6 on seed sets B and C (Leonardo CPU
jobs 64357979 / 64357982, `runs/night13/v6_set{B,C}.txt`, per person `.npz`). Question: are V6's new +0.34 and its P04
cost (−1.39) seed-specific? Prediction for V6 on each set: new +0.3 (0.0 to +0.6), seen P04 −0.8 (−1.6 to 0.0).

### 152. FTSHIP: fine-tuning applied to the shipped members, scored against v5.1b (written 13 Nov 07:57 UTC, before any result)

**The reference from here on is v5.1b as uploaded** (shipped members, shipped gate, stage 0.10). The study nets of
seed K reproduce dev base B2_sK's own SessionBN slots (fb and fbl exactly, sinc to 0.016, raw fbcnet to 0.149), so
fine-tuning becomes a change to the shipped members. On rows of never-routed sessions only, each base's slots 5–8 get
the mean fine-tuning change of that base's own seeds (B2_s0: seed 0, …, B2_s012: seeds 0–2, B2_s345: 3–5, B2_pool:
0–5); slot 9 (fb at k 5) stays frozen, as in dev; every other window keeps the shipped outputs exactly. Draws: ttt6's
cum10 (draw 0) and three fresh augmentation draws on the same nets (`studies/13nov/ttt13.py`, DRAW 1–3, Leonardo; ttt6
gained a DRAW offset that leaves draw 0 bit-identical). Arms against P0 (`pkg.py`): SF = v5.1b + FT; V6F = v5.1b + FT +
OCU + REST.

| Check (against v5.1b, six-base mean; per draw and over draws) | Prediction (80 % range) |
|---|---|
| SF new / without P10 | +0.40 (0.00 to +0.80) / +0.10 (−0.20 to +0.40); seen 0.00 |
| V6F new / J | +0.70 (+0.20 to +1.20) / +0.25 (−0.30 to +0.70) |
| **Bar** | A package design needs, on every draw, J ≥ +0.25 against v5.1b with the new gain without P10 ≥ +0.20 and seen not falling for both P04 and P10. Short of that, nothing in the stack is a package |

**§152 additions (13 Nov 08:00 UTC, before any result).**
- cum10 draws 1–3: scorings chained on Leonardo against v5.1b (`ftship_d{1,2,3}`, jobs 64358189–64358191).
- The cheap schedule gC5 (§113) on the shipped members, read the same way:
  - draw 0 from the stored TTT7 outputs (`pkg.py FT_VAR=gC5`, Leonardo CPU job);
  - fresh draws 1–3 from `studies/13nov/ttt14.py` (ttt7 gained the same DRAW offset; draw 0 stays bit-identical);
  - scorings `ftship_gc5_d{0,1,2,3}`.
- Same prediction and bar as cum10. gC5 costs a quarter of fixA's run time.

| §152 item, draw 0 (13 Nov 08:03 UTC; `runs/pkg_ftship_d0.log`, per person `.npz`; **against v5.1b as uploaded**, six bases) | New [80 % person] / without P10 | Seen (P04, P10) | J |
|---|---|---|---|
| SF: v5.1b + FT on never-routed sessions (each base's own seeds' change) | +0.23 [−0.44, +0.91] / −0.12 | 0.00 | +0.11 |
| V6: v5.1b + OCU + REST (§151) | +0.34 [−0.03, +0.71] / +0.37 | −0.24 (−1.39, +0.91) | +0.05 |
| **V6F: v5.1b + FT + OCU + REST** | **+0.95 [+0.49, +1.42] / +0.70** | −0.24 (−1.39, +0.91) | **+0.36** |

**Verdict on draw 0: the bar is met** (J ≥ +0.25, new without P10 ≥ +0.20, P04 and P10 do not both fall). The three
parts are strongly super-additive for new people: alone +0.23 and +0.34, together +0.95, and +0.70 without P10. The bar
asks for every draw; draws 1–3 (cum10) and 0–3 (gC5) are running.

**The seen side needs a design rule.** All of V6F's seen change (P04 −1.39, P10 +0.91) comes from OCU and REST on
routed sessions; FT never reaches them. Every chain stage works per session and the gate is fixed, so a package that
applies FT, OCU and REST only to sessions the gate has not recognised, and leaves recognised sessions exactly as v5.1b,
has new = V6F's new and seen = v5.1b's. **Dev reading of that form: J = +0.95 / 2 = +0.48 on draw 0, seen
exactly 0.** It is exact in the replay because no new-person session is routed in dev. A false accept on the sealed
set would send that session down v5.1b's routed path, which is what v5.1b does today.

### 153. MSTYLE (backlog, gap 7): MixStyle on the SessionBN nets (written 13 Nov 08:03 UTC, before any job ran; `netaug.py` tasks 500–523, Leonardo)

MixStyle was closed on 2 Nov on teacher pretraining, an earlier pipeline, and never tried on the SessionBN nets. Its
mechanism (mixing feature statistics across windows) is close to what SessionBN normalises away per session, so it
may be redundant there or complementary. Arms ms05 / ms10 (MixStyle on the first two convolutions with probability
0.5 / 1 per batch), four families, seeds 0–2. **Screen:** against the in-harness twins (`focal_swap.py ARMS=ms05,ms10`;
**reference: study twins, not v5.1b**). An arm passing the screen is then scored against v5.1b in `pkg.py` before any
claim.

| Check (new, six-base mean, against twins) | Prediction (80 % range) |
|---|---|
| ms05 / ms10 | 0.0 (−0.4 to +0.4) / −0.1 (−0.5 to +0.3) |
| **Screen** | ≥ +0.30 without the best person → scored against v5.1b |

### 154. HENT (fine-tuning, hard windows): keep the hard windows, drop their labels (written 13 Nov 08:04 UTC, before any job ran; `studies/13nov/ttt15.py`, Leonardo)

§142: the whole fine-tuning ceiling sits in the hard half's labels. §145 C: removing the hard half hurts. This form keeps
the hard windows as a training signal without trusting their labels: hard labels on the confident half of the prefix,
entropy minimisation on the low-confidence half. Same nets as ttt6 seeds 0–5, cum10 schedule. **Scored against
v5.1b as uploaded** (`pkg.py` SF / V6F with FT_VAR=hent), paired with §152's cum10 on the same draw.

| Check (against v5.1b, six bases) | Prediction (80 % range) |
|---|---|
| SF with hent / V6F with hent, new without P10 | +0.10 (−0.20 to +0.40) / +0.75 (+0.30 to +1.20) |
| **Bar** | replaces cum10 in the package design if V6F-hent is ≥ +0.15 above V6F-cum10 without P10, on the same draw |

| §149 m7u screen (13 Nov 08:09 UTC; `runs/teach3_m7u.log`; A.system v3 path, new; reference: the stored m7c teacher, **not v5.1b**) | Observed | Verdict |
|---|---|---|
| m7u alone (3-class unified head of both trunks, mapped) | 42.3 against 50.5 for the stored teacher | far weaker: the shared block's 3-class head alone carries less than the 19 corpus heads |
| m7u replacing the teacher, weight 3 / 5 / 7 | −1.62 / −1.56 / −1.64 (shipped); −1.13 / −1.18 / −1.19 (FT+OCU+REST) | **Refuted** in this form; screen not passed, so not scored against v5.1b |

**Reading.** Saving only the shared block's 3-class head drops the per-corpus heads that carry the stored teacher's
signal. A unified teacher would need its corpus heads kept as well (a `pretrain.py` change that saves the full
final layer). That is recorded as the next form, not queued yet. m8w (eight corpora, standard heads) is pretraining.

**Incident (13 Nov 08:12 UTC).** `pkg.py` printed `FT_VAR` before defining it, after my 08:00 edit, so the chained
scorings `ftship_d1` and `ftship_gc5_d0` failed loudly (NameError) on Leonardo. The line is moved and both are
resubmitted (64359839, 64359876); the others had not started.

### 155. Inside the combination (V6F): part strengths, an oracle view, and retests on top (written 13 Nov 08:17 UTC, after a one-base smoke, before the six-base run; `pkg.py`, against v5.1b as uploaded, paired with V6F on the same draw)

The user: push the super-additive combination, measure its ceiling, retest earlier methods on top of it. Arms, all
V6F plus one change:
- **Part strengths:** ft05 / ft15 (FT change ×0.5 / ×1.5); r05 / r2 (rest member weight 0.5 / 2); o05 / o2 (eye-net
  view ×0.5 / ×2).
- **Ceiling:** orc, an ORACLE view (true labels in place of the eye-net posteriors in SessionEM's view, never
  shippable): what a perfect view of this kind would give.
- **Retests on top:** cog (the cognitive teacher, §148, as a member at weight 1); pls (§74's PLS-DA view, a second TSEM
  pass on never-routed new rows); mix (member mix to the power 0.9); weak (FT only on sessions below the
  out-of-person median of early confidence, §133).
- **Reading:** new people only. The design applies the combination to unrecognised sessions, so seen is v5.1b's by
  construction. Seen is still printed.

Smoke, B2_pool, new against v5.1b: V6F +0.99; orc **+4.89**; cog +1.22; pls +1.00; mix +1.19.

| Check (new, six-base mean, paired against V6F on the same draw; draw 0 on the Mac, draws 1–3 on Leonardo) | Prediction (80 % range) |
|---|---|
| orc − V6F (ceiling of the view) | +3.5 (+2.0 to +5.0) |
| cog, mix − V6F | +0.15 (−0.15 to +0.45) each |
| pls − V6F | 0.0 (−0.3 to +0.3) |
| the best of the part strengths − V6F | +0.10 (−0.10 to +0.30) |
| **Bar** | a change is kept if it is ≥ +0.15 over V6F without the best person, on draw 0 and on at least two of draws 1–3 |

| §155 item, draw 0 (13 Nov 08:25 UTC; `runs/combo_d0.log`, per person `.npz`; new, paired against V6F; V6F against v5.1b +0.95, without P10 +0.70) | Against V6F [80 % person] / without P10 / without the best person | Verdict |
|---|---|---|
| **orc** (ORACLE view: true labels in SessionEM's view) | **+3.76 [+3.26, +4.27] / +3.78 / +3.60** (against v5.1b +4.71) | Hit (+3.5). **The ceiling of this kind of view is about +3.8 over the combination** |
| cog (cognitive teacher at weight 1) | +0.07 [−0.02, +0.17] / +0.08 / +0.02 | below the bar |
| mix (0.9) / pls / weak gate | −0.07 / −0.18 / −0.11; without P10 −0.07 / −0.21 / +0.03 | no gain on top |
| ft05 / ft15 (FT change ×0.5 / ×1.5) | −0.07 / **−0.35** | the shipped FT strength is best of the three |
| r05 / r2 (rest weight 0.5 / 2) | −0.19 / −0.15 | weight 1 is best |
| o05 / o2 (eye-net view ×0.5 / ×2) | **exactly 0.00** | **Structural no-op**: SessionEM standardises every view column by its session's spread (`replay/gm.py` EMg, `A.std(0)`), so a column's scale cannot matter. The arms are withdrawn; view strength needs another form (more columns, or a different weight in the likelihood) |

**Reading (draw 0; draws 1–3 decide).** V6F's defaults are a local optimum for the knobs tried. None of the retests
adds on top (cog +0.07 is the closest). The headroom is in the eye-net view's content: a perfect view of this kind gives
+3.8 more. The super-additivity looks like the view letting SessionEM sort sessions by what the fine-tuned members get
wrong. So the next lever is better views of the same kind: more eye-net seeds and families, views from other
label-free signals, and the SessionBN nets' own embeddings (§72–73, closed on the v3 path) scored inside V6F.

### 156. VIEWS: better eye-net views inside the combination (written 13 Nov 08:26 UTC, before any result; `pkg.py`, against v5.1b, paired with V6F, Mac, draw 0)

§155: a perfect view of this kind adds +3.8. Cheap steps towards it from stored outputs:
- (a) ocm, a second eye-net family (the Mixup arm, seeds 0–2) appended to the view.
- (b) V6F with the nomix eye net averaged over nine seeds (0–8) instead of three: the same view, less noise.

| Check (new, paired against V6F with the 3-seed nomix view) | Prediction (80 % range) |
|---|---|
| (a) ocm | +0.10 (−0.20 to +0.40) |
| (b) nine seeds | +0.10 (−0.15 to +0.35) |
| **Bar** | ≥ +0.15 without the best person on draw 0, then draws 1–3 on Leonardo (mix seeds 3–11 are there) |

### 157. VIEWS2: other members' posteriors as SessionEM view columns inside V6F (written 13 Nov 08:26 UTC, before any result; `pkg.py`, Mac, draw 0)

If the eye-net view works because its errors are independent of the mix, other members' posteriors in the view (rather
than, or as well as, in the mix) may act the same way: vc1 (core1), vte (the teacher), vre (the rest member). Paired
against V6F, new. Prediction for each: +0.05 (−0.30 to +0.40). Bar: ≥ +0.15 without the best person on draw 0, then
Leonardo draws.

| §157 item (13 Nov 08:30 UTC; `runs/views2.log`, `.npz`; new, paired against V6F, draw 0) | Against V6F / without P10 / without the best | Verdict |
|---|---|---|
| vc1 (core1 posteriors in the view) | −0.59 / −0.50 / −0.72 | **Refuted** |
| vte (teacher posteriors in the view) | −0.87 / −0.98 / −1.04 | **Refuted** |
| vre (rest member posteriors in the view) | −0.37 / −0.43 / −0.49 | **Refuted** |

**Reading.** Members that already sit in the mix hurt when they are added to SessionEM's view: their errors are not
independent of the mix the view is meant to correct. The eye net is the only view of this kind that helps, and it sees
a different signal (ocular traces) through a different model. Better views must come from signals or models that are
not in the mix: more eye-net families and seeds (§156, running), and new label-free signals.

### 158. VIEWNET (explore, high ceiling): view nets on label-free signals the mix does not use (written 13 Nov 08:31 UTC, before any job ran; `studies/13nov/viewnet.py`, Leonardo, 9 seeds each)

**Ceiling:** §155's oracle view adds +3.8 over the combination; the eye net is the only real view that helps, and mix
members as view columns hurt (§157). New views from other signals, through the eye net's architecture and causal
standardisation:
- emg: muscle activity, log 30–45 Hz envelopes at temporal and frontal sites;
- scp: slow cortical potentials below 1 Hz at the midline and C3/C4.

Scored inside V6F against v5.1b (`pkg.py` V6F_vemg, V6F_vscp, V6F_vboth), paired with V6F.

| Check (new, paired against V6F) | Prediction (80 % range) |
|---|---|
| vemg / vscp / both | +0.10 (−0.30 to +0.50) each; both +0.15 |
| **Bar** | ≥ +0.15 without the best person on draw 0 and on two of draws 1–3 → a view for the package |

| §156 item (13 Nov 08:32 UTC; `runs/views_{a,b}.log`, `.npz`; new, paired against V6F with the 3-seed nomix view, draw 0) | Against V6F / without P10 / without the best | Verdict |
|---|---|---|
| (a) ocm: the Mixup eye-net family appended | −0.07 / −0.09 / −0.13 | Miss (+0.10 predicted): no gain |
| (b) the nomix eye net averaged over nine seeds | −0.11 / −0.12 / −0.16; against v5.1b +0.84 | Miss: more seeds of the same view do not help, they cost a little |

**Reading.** The eye-net view's gain does not grow with more seeds or a second family of the same net: the three-seed
view is not noise-limited. The ceiling (+3.8) has to come from different information, which is what §158's new signals
test.

### 159. VIEWNET wide (explore, the largest label-free-reachable ceiling): eight more view nets (written 13 Nov 08:33 UTC, before any job ran; `viewnet.py` KIND=…, Leonardo, 9 seeds each)

**Ceiling:** +3.8 over the combination for a perfect view (§155); only signals the mix does not carry help (§157);
more seeds or a second family of the eye net do not (§156). §158's emg and scp are running. Eight more kinds, each 7
derived traces through the eye net's architecture and causal per-session standardisation:
- ocuslow (ocular derivations below 1 Hz) and ocufast (1–8 Hz): the eye net split by time scale;
- ocuh1 / ocuh2: the first or second half of the window only;
- asym: left-right asymmetries;
- alpha (parietal-occipital 8–13 Hz envelopes), beta (sensorimotor 13–30 Hz), theta (frontal midline 4–8 Hz).

Each is scored as an added view column inside V6F against v5.1b (`pkg.py VN_KINDS=…`), alone and all ten together
(V6F_vall), paired with V6F. Share of the ceiling captured = (arm − V6F) / 3.76.

| Check (new, paired against V6F, draw 0) | Prediction (80 % range) |
|---|---|
| the best single kind | +0.30 (0.00 to +0.70) |
| all ten together | +0.40 (−0.30 to +1.00) |
| **Bar** | a kind is kept if ≥ +0.15 without the best person on draw 0 and on two of draws 1–3; a kind of the ocular family that helps alone, while the eye net already sits in V6F, would say the eye net's architecture leaves information unused |

Kinds that do not help and kinds that do are compared by what they carry (time scale, region, independence from the
mix, measured as their posteriors' correlation with the mix's) to say what distinguishes a useful view.

| §152 FTSHIP, all draws (13 Nov 08:34 UTC; `runs/night13/ftship_*.npz`, draw 0 `runs/pkg_ftship_d0.npz`; **against v5.1b as uploaded**, six bases) | New [80 % person] / without P10 | J (as scored) / J with recognised sessions left as v5.1b | Verdict |
|---|---|---|---|
| V6F, cum10, draw 0 / 1 / 2 / 3 | +0.95 / +0.93 / +0.84 / +0.86; without P10 **+0.70 / +0.65 / +0.58 / +0.59**; 80 % intervals all above 0 (lowest +0.36) | +0.36 / +0.35 / +0.30 / +0.31; **+0.48 / +0.47 / +0.42 / +0.43** | **Bar met on every draw** (J ≥ +0.25, new without P10 ≥ +0.20, P04 and P10 never both down) |
| SF (FT alone), cum10, draws 0–3 | +0.23 / +0.09 / +0.00 / +0.21; without P10 −0.12 to −0.38 | +0.11 / +0.05 / 0.00 / +0.11 | FT alone is P10 |
| V6F, gC5, draw 0 / 1 / 2 / 3 | +0.70 / +0.56 / +0.71 / +0.58; without P10 +0.48 / +0.33 / +0.47 / +0.33 | +0.23 / +0.16 / +0.23 / +0.17; recognised as v5.1b +0.35 / +0.28 / +0.36 / +0.29 | **Bar missed on every draw** (J < +0.25). gC5 is about 0.25 below cum10 inside the combination; the cheap schedule costs gain |
| §154 HENT inside V6F (draw 0) | +0.09, without P10 −0.07 | −0.08 | **Refuted**: entropy on the hard windows destroys the combination's gain |

**Reading. The combination (cum10 FT + eye-net view + rest member, on sessions the gate does not recognise) is the
first design that meets its bar against v5.1b on every one of four augmentation draws.** New people gain +0.84 to
+0.95 (without P10 +0.58 to +0.70); with recognised sessions left exactly as v5.1b, dev J is +0.42 to +0.48. The
cheaper gC5 schedule keeps about two thirds of it. Its run time is the next fact a build needs: cum10 on 4 families ×
never-routed sessions, §107's bench at half fixA's steps.

| §155 draw 1 (13 Nov 08:34 UTC; `runs/night13/combo_d1.npz`; paired against V6F) | Against V6F / without the best |
|---|---|
| orc (oracle view) | **+3.71 / +3.50** (draw 0: +3.76) |
| cog | +0.13 / +0.09 (draw 0: +0.07 / +0.02) |
| mix, pls, weak, ft05, ft15, r05, r2 | −0.03 to −0.29; without the best −0.08 to −0.42 |
| o05, o2 | 0.00 (structural no-ops, §155) |

Draws 2 and 3 are scoring. Nothing has passed the +0.15 bar on two draws; cog is the closest at +0.07 to +0.13.

**Incident:** `pkg.py` loaded the FBCNet rest net for every run, and its seeds stop at 8, so the set B/C replications
(§150/§151, `v6_set{B,C}`) failed loudly on the missing seed-9 files. It now loads only for the arms that use it, and
both are resubmitted (64361598, 64361599).

| §153 MSTYLE screen (13 Nov 08:38 UTC; `focal_swap.py ARMS=ms05,ms10`, A.system; **reference: study twins, not v5.1b**) | New [80 % person], people ≥ 0 | Verdict |
|---|---|---|
| ms05 (MixStyle, probability 0.5) | −0.22 [−0.58, +0.10], 4 of 10; seen (v3 path) −0.38 | **Refuted** (prediction 0.0) |
| ms10 (probability 1) | −0.55 [−0.90, −0.18], 1 of 10 | **Refuted** |

**Reading.** On the SessionBN nets MixStyle hurts. Mixing feature statistics across windows works against SessionBN's
own per-session normalisation. The screen is not passed, so it is not scored against v5.1b. The backlog item (gap 7,
"MixStyle on SessionBN nets") is closed in this form on a valid test.

| §149 m8w screen (13 Nov 08:50 UTC; `runs/teach3_m8w.log`; A.system v3 path, new; reference: the stored m7c teacher, **not v5.1b**) | Observed | Verdict |
|---|---|---|
| m8w alone (seven m7 corpora + Weibo2014) | 49.97 against 50.46 | no stronger alone |
| m8w replacing the teacher, weight 3 / 5 / 7 | −0.85 / −0.66 / −0.88 (shipped); −0.19 / −0.17 / −0.49 (FT+OCU+REST) | **Refuted**; screen not passed |

**Reading.** An eighth MI corpus does not make the teacher stronger, and in the chain it costs. Together with m8s
(Scherer, §18 and §132) and m7u (§149), every teacher change tried since 11 Nov loses in the chain except the
cognitive member cog1 (+0.07 to +0.22, below every bar). The teacher line's ceiling argument (an independent 70 %
member) is not reached by adding corpora. It would need a member that is accurate on our windows, not one with more
outside data. Not queued further now.

**§120 addendum 2: FT run time per never-routed session, warm, paired (13 Nov 08:56 UTC; Leonardo A100, debug job
64361778; `studies/13nov/ft_cost_session.py`).** Eager dev nets carrying the shipped SessionBN weights and alignment
(equal to the traces, `scripts/sbn_trace_base.py`); GPU-drawn augmentation; warm start; prediction pass over the prefix
after each refit; four families (fb at k 2, slot 9 frozen). v5.1b itself, streaming the same sessions in the same job:
15.5 s (190 windows) and 31.6 s (378). Added by FT, in seconds per session:

| Schedule, form (nets) | 190 windows | 378 windows | Sealed, 57 % / 100 % of sessions unrouted (190 / 378 windows) |
|---|---|---|---|
| cum10, all24 (24) | +60.7 (3.9×) | +239.7 (7.6×) | 69 / 105 min; 114 / 184 min |
| cum10, delta2 (8) | +20.2 | +79.4 | 37 / 49; 52 / 75 |
| cum10, delta1 (4) | +10.1 | +39.7 | 29 / 35; 37 / 48 |
| gC5, all24 (24) | +31.1 | +122.2 | 46 / 64; 69 / 104 |
| gC5, delta2 (8) | +10.4 | +40.7 | 30 / 36; 37 / 49 |
| gC5, delta1 (4) | +5.2 | +20.4 | 26 / 29; 29 / 35 |

Sealed = 1,287 s × (1 + unrouted share × FT/v5.1b ratio), with v5.1b's real-worker 1,287 s; it does not include the
second routed tail, the eye net or the rest member.

Reading:
- FT cost grows with the square of session length (refits × prefix): 378 windows cost 4× what 190 cost.
- Against the 45-min worst case (every session unrouted), only gC5 delta1 fits at both lengths (29 / 35 min). cum10
  delta1 and gC5 delta2 fit only if sessions are near 190 windows (35 / 36 min).
- cum10 as scored in §152 (each base's own seeds: one to six per family) does not fit as all24.
- A window cap (fine-tune on the last W windows only) would make the cost linear. It is unscored and would need its own
  arm before use.

| §155 draws 2 and 3 (13 Nov 08:58 UTC; `runs/night13/combo_d{2,3}.npz`; paired against V6F) | Draw 0 / 1 / 2 / 3 against V6F (without the best) | Verdict over four draws |
|---|---|---|
| orc (ORACLE view) | +3.76 / +3.71 / +3.82 / +3.87 (+3.50 to +3.69) | **the view ceiling is +3.7 to +3.9, stable** |
| cog (cognitive teacher) | +0.07 / +0.13 / +0.09 / +0.10 (+0.02 to +0.09) | consistent but below the bar (+0.15 without the best): **not kept** |
| mix, pls, weak, ft05, ft15, r05, r2 | negative on all four draws | **not kept**: V6F's defaults stand |

| §150/§151 on seed sets B and C (13 Nov 08:58 UTC; `runs/night13/v6_set{B,C}.npz`; against v5.1b) | New / without P10 | Seen (P04, P10) |
|---|---|---|
| V6 (shipped + OCU + REST), set A / B / C | +0.34 / −0.07 / +0.23; without P10 +0.37 / −0.09 / +0.29 | −0.24 (−1.39, +0.91) / **+0.45 (+0.66, +0.24)** / **+0.28 (+0.44, +0.12)** |
| OR (study base), sets B / C | −0.19 / +0.30 | +0.65 / +0.09 |

**Reading.** On seed sets B and C, OCU+REST on v5.1b's members *helps* P04 (+0.66, +0.44) instead of costing it
(−1.39 on set A). §151's P04 cost is a property of set A's eye-net and rest seeds, not of the lever. New-people gains of
OCU+REST alone vary from −0.07 to +0.34 by seed set. The combination with FT (V6F) has so far been read on set A's
OCU/REST seeds only (with four FT draws). Its next check is V6F on OCU/REST seed sets B and C, queued with the view
wave.

**Incident:** both view-net scorings failed loudly. `pkg.py`'s default fine-tuning path is the Mac's
(`runs/ttt/probs6`; on Leonardo it is `runs/diag/ttt/probs6`). They are resubmitted with explicit paths, with draws 0–3
(64363611–64363615). All ten view nets trained (90 tasks).

**§152 on other OCU/REST seeds (queued 13 Nov 09:00 UTC, before any result):** V6F (and SF, V6) with eye-net and
rest seeds from sets B and C (OCU 18–26 / 27–35, REST 9–14 / 15–20) × FT draws 0 and 1, against v5.1b
(`v6f_set{B,C}_d{0,1}`, jobs 64363637–64363641). The bar of §152 is read on these four too: the design is confirmed if
it holds on the other seed sets of the two non-FT members, not only on new FT draws.

### 160. FITFORM: the combination with fine-tuning forms that fit the run time (written 13 Nov 09:23 UTC, before any result; `pkg.py FT_FORM=delta1|delta2`, Leonardo CPU jobs)

§152 confirmed V6F with cum10 on each base's own seeds. §120 addendum 2: that form costs 69–105 min sealed at 57 %
unrouted sessions (114–184 min at 100 %), over the limit. What fits:
- gC5 delta1: 26 / 29 min at 57 %, 29 / 35 at 100 % (190 / 378 windows);
- cum10 delta1: 29 / 35 and 37 / 48;
- gC5 delta2: 30 / 36 and 37 / 49.

These figures leave out the eye net and the rest member. Here, V6F is scored against v5.1b with FT in the delta1 and
delta2 forms (one or two fine-tuned nets per family, their change applied to the whole family), for cum10 and gC5,
each on its four draws. On the single-seed bases (B2_s0, s1, s2) the forms coincide with §152's. Reported per the
fresh-eyes rule (`docs/FRESH_EYES_13NOV.md` item 9): new people, and returning people without P04 (P10) separately.

| Check (V6F against v5.1b, new, mean of four draws) | Prediction (80 % range) |
|---|---|
| cum10 delta1 / delta2 | +0.80 (+0.55 to +1.00) / +0.85 (+0.60 to +1.05); §152 (all seeds) +0.90 |
| gC5 delta1 / delta2 | +0.60 (+0.35 to +0.85) / +0.62 |
| **Bar (a package form)** | new without P10 ≥ +0.40 on every draw, with the 80 % person interval above 0, and a sealed worst case ≤ 45 min at 100 % unrouted once the eye net and rest member are timed |

### 161. FTCAP: gC5 with a window cap, linear cost (written 13 Nov 09:24 UTC, before any job ran; `studies/13nov/ttt16.py`, Leonardo)

§120 addendum 2: FT cost grows with the square of session length, because each refit trains on the whole prefix. A
cap that trains on the last CAP windows only makes it linear (the prediction pass still uses the whole prefix, for the
SessionBN state). Arms: gC5 with CAP 60 and CAP 100, draws 0–3, same nets as ttt7 seeds 0–5. Scored inside V6F
against v5.1b, all and delta1 forms (`pkg.py FT_DIR=runs/diag/ttt/probs16 FT_PFX=ttt16c<CAP>d<D> FT_VAR=gC5`).

| Check (V6F against v5.1b, new, mean of four draws) | Prediction (80 % range) |
|---|---|
| gC5 CAP 100 / CAP 60, against uncapped gC5 | −0.05 (−0.25 to +0.10) / −0.10 (−0.30 to +0.10) |
| **Bar** | a cap that loses ≤ 0.10 against uncapped gC5 on the mean of four draws makes the cost linear; then it is timed |

| §158/§159 VIEWNET, four draws (13 Nov 09:28 UTC; `runs/night13/viewnet_wide*.npz`; new, paired against V6F; V6F against v5.1b +0.95 on draw 0) | Against V6F, draws 0 / 1 / 2 / 3 | Verdict |
|---|---|---|
| emg / ocufast / ocuslow / beta | about 0: −0.01 to +0.10 / −0.11 to 0.00 / −0.07 to +0.09 / −0.05 to −0.01 | no gain |
| alpha / asym / ocuh1 / ocuh2 / theta | −0.08 to −0.32 on every draw | cost |
| scp | −0.27 to −0.36 | cost |
| all ten together | −0.34 to −0.47 | cost |

**Verdict: refuted, every kind.** None captures any part of the +3.8 ceiling. The eye net's view stays the only real view
that helps inside the combination. Before reading this as "only ocular information helps", a null is needed: if
any three added columns cost SessionEM something (its covariance in a higher dimension, with fixed shrinkage), real
views are judged against a handicap. **§162 null (written 13 Nov 09:28 UTC, before the run):** V6F + 3 columns of
seeded Gaussian noise, on the Mac, draw 0. Prediction −0.10 (−0.30 to +0.05). If the null costs about as much as the
weak views (−0.1 to −0.2), the view test needs a form without that handicap (views that replace columns, or a
dimension-aware SessionEM). If it costs about 0, the views themselves carry nothing useful.

**§120 addendum 3: the whole v6 candidate path on an unrecognised session, warm, paired (13 Nov 09:35 UTC; Leonardo A100,
debug job 64366211; `studies/13nov/v6_cost.py`).** Same node, same sessions:
- v5.1b streaming the session: 15.2 s (190 windows) and 30.7 s (378).
- Added per unrecognised session: the second routed tail (rebalancing, SessionEM, TSEM, timed as v5.1b's own routed-tail
  stages) 0.47 / 1.09 s; the eye net (7 derived traces with running standardisation, three seeds at batch 1) 1.16 / 2.31
  s; the rest member (a shallow StreamBN trace at batch 1) 0.20 / 0.43 s; fine-tuning with one net per family (delta1),
  gC5 5.2 / 20.4 s and cum10 10.0 / 41.0 s.
- The eye-net and rest weights were untrained (timing only); SessionEM's three extra view columns are not timed (a 51-
  instead of a 48-column view).

| Path on unrecognised sessions | Added, 190 / 378 windows | Sealed at 57 % unrecognised (190 / 378) | at 100 % (190 / 378) |
|---|---|---|---|
| tail + eye net + rest, no FT | +1.8 s / +3.8 s (×0.12) | 22.9 / 23.0 min | 24.0 / 24.1 min |
| + FT gC5 delta1 | +7.0 / +24.2 s (×0.46 / ×0.79) | 27.1 / 31.1 | 31.4 / 38.4 |
| + FT cum10 delta1 | +11.8 / +44.8 s (×0.78 / ×1.46) | 31.0 / 39.3 | 38.1 / **52.8** |

Sealed = 1,287 s × (1 + unrecognised share × added / v5.1b), with v5.1b's real-worker 1,287 s; recognised sessions add
nothing (bit-identical to v5.1b by design).

Reading: the eye net, the rest member and the extra tail cost about 12 % on unrecognised sessions; fine-tuning dominates.
Against a 45-min worst case (every session unrecognised), gC5 delta1 fits at both lengths (31 / 38 min). cum10 delta1
fits at 190 windows (38 min) but not at 378 (53 min).

| §162 null (13 Nov 09:36 UTC; `runs/null_d0.log`; draw 0) | Against V6F / without the best |
|---|---|
| 3 noise columns, seed 1 / seed 2 | −0.09 / −0.15; −0.16 / −0.23 |

**Reading.** Added columns cost SessionEM about −0.12 regardless of content. Against that handicap, emg, ocufast,
ocuslow and beta sit about +0.1 above noise (some information). scp, alpha, theta and asym fall below noise (their
content hurts). **§163 (written 09:36 UTC, before the run):** put the information in without adding dimensions, by
averaging the view nets' log posteriors with the eye net's in its own 3 columns: avg4 (emg, ocuslow, ocufast, beta) and
avgslow (ocuslow only). Prediction against V6F, new: +0.05 (−0.15 to +0.25) each; bar ≥ +0.15 without the best person
on draw 0, then draws 1–3.

| §152 on OCU/REST seed sets B and C (13 Nov 09:37 UTC; `runs/night13/v6f_set{B,C}_d{0,1}.npz`; against v5.1b) | V6F new [80 % person] / without P10 | J, recognised sessions as v5.1b | V6 (no FT) new / seen P04, P10 |
|---|---|---|---|
| set B, FT draw 0 / 1 | +0.63 [+0.20, +1.07] / +0.44; +0.55 [+0.05, +1.03] / +0.32 | +0.32 / +0.27 | −0.07; +0.66, +0.24 |
| set C, FT draw 0 / 1 | +0.63 [+0.24, +1.03] / +0.41; +0.70 [+0.24, +1.17] / +0.45 | +0.32 / +0.35 | +0.23; +0.44, +0.12 |

**Verdict: the bar holds on the other seed sets of the eye net and rest member** (new without P10 ≥ +0.20, 80 %
intervals above 0, J with recognised sessions as v5.1b ≥ +0.25 on all four), at about two thirds of set A's size.
Over three OCU/REST seed sets and four FT draws, V6F's new-people gain against v5.1b is **+0.55 to +0.95 (without P10
+0.32 to +0.70)**. The design-level expectation is about +0.7 new, J +0.35 with recognised sessions untouched, in the
form scored here (cum10, each base's own seeds). Whether it survives the run-time forms is §160 (FITFORM), due shortly.

### 164. SHUF: the chain under non-chronological delivery (written 13 Nov 09:39 UTC, after a one-permutation smoke, before the full run; `studies/13nov/shuffle.py`, Mac)

**Organisers, 13 Nov (STATE §2, verbatim there):** the sealed stream is not in recording order and has not been since
the rescoring of 10 Nov. §62's conclusion ("the sealed order is still close to acquisition order") is **withdrawn**: a
failed fingerprint meant we did not find the order, not that there is none. Every dev number in this ledger was
measured in recording order. From here the primary protocol is non-chronological, under several schemes with fixed
permutations; recording order is a secondary column.

**Method.** Every per-window array of the replay (members, views, labels, cells, sessions; three stores and `A.C1`) is
permuted within each session and the stages are rerun. Schemes: full (uniform within the session), runs (runs in a
random order, recording order inside), within (runs in order, windows shuffled inside each run); five permutations
each, six bases, new people, stage by stage (mix → rebalancing → gated SessionEM → gated TSEM, i.e. gm.pipe_g, the
tail every new-person window takes). Checks (asserted): the identity permutation reproduces the chain (B2_pool
80.18), and the per-window mix scores the same under every permutation. **Limit:** members stay as computed in
recording order. Their own causal session statistics (SessionBN, core1's running recentring, the eye net's
standardisation, fine-tuning's schedule) are not re-streamed, so this measures the stages' order dependence only;
member re-streaming needs GPU runs and comes next.

Smoke (B2_pool, one full permutation), new against recording order: rebalancing +0.06, SessionEM −1.40, TSEM (the
whole chain) **−3.14**.

| Check (new, against recording order, six-base mean over five permutations) | Prediction (80 % range) |
|---|---|
| full: whole chain | −3.0 (−4.0 to −2.0) |
| runs: whole chain | −1.0 (−2.0 to 0.0) |
| within: whole chain | −2.5 (−3.5 to −1.5) |
| **Use** | the stage that loses most under each scheme is the first target of the order-robust re-tune (§165, next) |

| §163 item (13 Nov 09:42 UTC; `runs/avg_d0.npz`; new, paired against V6F, draw 0) | Against V6F / without the best | Verdict |
|---|---|---|
| avg4 (eye net averaged with emg, ocuslow, ocufast, beta in its own 3 columns) | −0.10 / −0.15 | **Refuted** |
| avgslow (eye net averaged with ocuslow) | −0.10 / −0.15 (equal six-base means by coincidence: the per-base values differ, up to 2.6 for single people; checked) | **Refuted** |

**Reading (view line, recording order).** No view built tonight adds to the eye net's, alone, together, or averaged into its
columns. The +3.8 oracle stays out of reach with the signals tried. This line is paused until it is re-scored under
non-chronological delivery (§164): the view nets' causal standardisation and SessionEM's use of the view are both
order-dependent.

### 165. RESTREAM: the SessionBN members and fine-tuning re-streamed in a non-chronological order (written 13 Nov 09:41 UTC, before any job ran; `studies/13nov/ttt17.py`, Leonardo)

§164 measures the stages with members fixed in recording order. The SessionBN nets (causal session statistics),
their causal alignment, and the fine-tuning schedule are order-dependent too. TTT17 streams each session in the exact
delivery orders of §164 (`runs/shuffle_perms.npz`, schemes full and within, permutation 0), with seeds 0–5 nets:
frozen ("base", the members under the new order) and cum10. Pseudo-labels still come from the v3 chain in recording
order (an approximation). Scoring: §164's replay with slots 5–8 replaced by these outputs, so that members and stages
are both in the new order. Checks against v5.1b under the same order.

| Check (new, full scheme, permutation 0) | Prediction (80 % range) |
|---|---|
| members re-streamed vs fixed (on top of §164's stage loss) | −0.5 (−1.5 to +0.2) |
| V6F's FT share under the new order | +0.2 (−0.3 to +0.6), against about +0.6 inside V6F in recording order |

| §164 SHUF (13 Nov 09:42 UTC; `runs/shuffle_13nov.log`, `.npz`; new, against recording order, six bases × five permutations; members fixed) | mix / rebalancing / SessionEM / whole chain | Whole chain: 80 % person interval, people losing |
|---|---|---|
| **full** (uniform within the session) | 0.00 / −0.61 / −2.17 / **−3.75** (permutations −4.39 to −3.07) | [−4.72, −2.85], 10 of 10 |
| **runs** (runs in random order, recording order inside) | 0.00 / +0.05 / −0.86 / **−2.11** (−2.54 to −1.66) | [−2.75, −1.47], 9 of 10 |
| **within** (runs in order, shuffled inside) | 0.00 / −0.10 / −0.38 / **−1.11** (−1.27 to −0.74) | [−1.74, −0.52], 7 of 10 |

Predictions (−3.0 / −1.0 / −2.5): full and runs worse than predicted, within better. **Reading.** The stages, not the
30-window rebalancing, carry the order dependence. SessionEM and TSEM build running class templates that track drift
across the session, and a full shuffle takes away 40 % of the stages' +9.4 over the mix. Even run order alone matters
(−2.1): the templates follow slow change between runs. With the members' own order dependence still to add (§165), v5.1b
on new people loses 1 to 4 points under the schemes the organisers may use. **The re-tune of the stages under a
non-chronological protocol (§166, next) is now the largest lever on record.** Sealed check: the rescoring cost the
packages 0.72 to 1.17 J, which fits a mild scheme or a mixture better than a full shuffle (seen people not yet
measured here).

### 166. RETUNE: the stage constants re-chosen under non-chronological delivery (written 13 Nov 09:43 UTC, before any result; `studies/13nov/retune.py`, 12 Leonardo CPU jobs; selection `select_retune.py`)

§164: v5.1b's stages lose 1.1 to 3.8 for new people under shuffled delivery. Under a shuffle, a session's earlier
windows are an unbiased sample of the whole session, which argues for longer memory and stronger session templates.

**Grid (144 configurations):**
- rebalancing window 30 / 90 / cumulative, strength 1.15 / 1.5;
- SessionEM λ 0.5 / 0.8, n0 60 / 20;
- TSEM λ 0.15 / 0.30 / 0.45, n0 60 / 20.

Each configuration is scored on new people in recording order and under full / runs / within (two permutations each),
on the exploration bases B2_pool and B2_s345. The shipped configuration reproduces gm.pipe_g, which is asserted; the
smoke gave rec 80.18 and full 77.04.

**Selection:** nested over people. Choose on five people (mean over the three shuffled schemes), score on the other five,
both ways. The chosen configuration is then scored on the four fresh bases, with members re-streamed (§165) when those
land.

| Check (new, mean of the three shuffled schemes, nested) | Prediction (80 % range) |
|---|---|
| chosen against shipped constants, under shuffle | +1.0 (+0.3 to +2.0) |
| the same configuration in recording order | −0.3 (−1.0 to +0.2) |
| **Bar** | ≥ +0.5 under shuffle on both halves and on the fresh bases, with recording order ≥ −0.5. Then it is the stage set of the next package (the stream is not chronological, so shuffled numbers decide) |

| §160 FITFORM (13 Nov 09:45 UTC; `runs/night13/fit_*.npz`; V6F against v5.1b, new, four draws; **recording order**) | New [80 % person] per draw | Without P10 per draw | Bar (without P10 ≥ +0.40, interval above 0, every draw) |
|---|---|---|---|
| **cum10 delta1** (4 fine-tuned nets) | +1.00 / +0.88 / +0.77 / +0.75 (lowest interval +0.22) | +0.76 / +0.61 / +0.50 / +0.47 | **met 4/4** |
| cum10 delta2 (8 nets) | +0.91 / +0.87 / +0.71 / +0.77 | +0.66 / +0.58 / +0.44 / +0.49 | met 4/4 |
| gC5 delta1 (4 nets, cheapest) | +0.81 / +0.55 / +0.68 / +0.69 | +0.60 / +0.32 / +0.43 / +0.47 | missed on draw 1 |
| gC5 delta2 | +0.73 / +0.55 / +0.68 / +0.63 | +0.52 / +0.31 / +0.43 / +0.38 | missed |

**Reading.** In recording order, one fine-tuned net per family (cum10 delta1) keeps the combination's full gain: new
+0.75 to +1.00, against +0.84 to +0.95 with every seed. Its cost (§120 addendum 2, without the eye net and rest member)
is 29 / 35 min sealed at 57 % unrouted and 37 / 48 min at 100 % (190 / 378 windows). The worst case at 378 windows is
over 45 min, but inside 60. **All of this is recording order.** Fine-tuning's schedule is position-based and its
pseudo-labels come from order-dependent stages, so the design is decided only after its shuffled-protocol score
(§165 re-streaming with cum10, §166 stage re-tune).

### 167. OCUSHUF and more re-streaming (written 13 Nov 10:06 UTC, before any job ran)

The user restarted Claude at about 10:00 UTC, and the §166 watcher died with it. RETUNE (12 jobs) and RESTREAM
(full0, within0) completed at about 09:55 UTC and are scored next. Queued now so that the GPUs keep running:
- **RESTREAM, more orders:** ttt17 with runs perm 0, full perm 1 and within perm 1 (arrays 64370563–64370565).
- **OCUSHUF:** the eye net with its causal trace standardisation done in the delivery order, for training and test
  sessions (`studies/13nov/ocushuf.py`; traces differ from recording order by up to 6.2 sd), full and within perm 0,
  nine seeds each.

Together with §165 these give the order dependence of all three parts of the combination except the rest member,
which is next.

Scoring: `shufcombo.py` (written next). New people, with members and stages in the same delivery order:
- v5.1b with re-streamed members, against recording order;
- the combination (cum10 delta1 + OCU + REST) under the same order, against v5.1b under the same order.

Prediction: v5.1b, full, members re-streamed on top of §164's stage loss: −0.5 (−1.5 to +0.2). The combination's gain
over v5.1b under full shuffle: +0.4 (−0.2 to +0.9), against +0.75 to +1.00 in recording order.

| §166 RETUNE (13 Nov 10:06 UTC; `runs/night13/retune_*.npz`, `select_retune.py`; new, B2_pool and B2_s345; shuffled = mean of full, runs and within, two permutations each) | Held-out half: shuffled / recording order | Verdict |
|---|---|---|
| half 0 chooses #8 (SessionEM n0 20, TSEM λ 0.30) | −0.02 / −0.02 | no gain |
| half 1 chooses #97 (cumulative rebalancing, TSEM n0 20) | +0.16 / **−0.91** | no gain under shuffle, a cost in recording order |
| best on all ten (#8, not nested) | +0.44 / +0.07; per permutation −0.49 to +1.15 | selection on the same people; the spread across permutations is as large as the gain |

**Verdict: refuted (predicted +1.0, bar +0.5 nested).** Re-choosing the constants of the existing stages (rebalancing
window and strength, SessionEM and TSEM λ and n0) does not recover the shuffle loss of §164. The loss is structural:
the stages' running templates profit from local continuity in recording order, and no constant in this grid replaces
it. **Also closed by this:** "cumulative rebalancing" and "longer memory" as the fix (half 1 picked cumulative
rebalancing; it gains +0.16 held out and costs −0.91 in recording order). Next: forms built for an unordered session
(whole-session iterative estimates instead of running ones, e.g. EM over all delivered windows at each step; an
order-free session embedding), scored under the shuffled protocol from the start.

| §165 RESTREAM, six bases (13 Nov 10:10 UTC; `runs/shufcombo_{full0,within0}.log`, `.npz`; new [80 % person] / without P10; members re-streamed: the SessionBN slots and fine-tuning; eye net and rest member still as recorded) | full, permutation 0 | within, permutation 0 |
|---|---|---|
| v5.1b, stages shuffled, against recording order | −3.07 [−4.30, −1.91] / −3.06 | −1.28 [−2.07, −0.51] / −1.17 |
| v5.1b, stages and SessionBN members re-streamed | −3.10 / −3.12 | −1.29 / −1.23 |
| **the combination (cum10 delta1 + eye net + rest member) over v5.1b, both shuffled** | **+1.47 [+0.68, +2.30] / +1.10** | **+1.65 [+1.19, +2.12] / +1.39** |
| the combination over v5.1b, recording order (check: FITFORM draw 0 gave +1.00) | +1.00 [+0.55, +1.47] / +0.76 | the same |

**Reading.** Re-streaming the SessionBN members in the delivery order adds no loss beyond the stages' (−3.10 vs −3.07):
v5.1b's order loss is in the stages. **The combination gains more under shuffled delivery than in recording order**
(+1.47 / +1.65 against +1.00, without P10 +1.10 / +1.39). Fine-tuning and the eye-net view partly replace what the
stages lose without chronology. Prediction (+0.4) missed, high. Still to add: the eye net and rest member re-streamed
(§167, running); more orders (runs, second permutations); returning people under shuffle. Under the organisers' stream
this is the strongest result on record: dev new +1.1 to +1.7 over v5.1b under the same order, the measure that
decides now.

### 168. RUNORACLE (explore; the coordinator's suggestion; ceiling first): do the stages lose under shuffle because they mix runs? (written 13 Nov 10:10 UTC after a one-base smoke, before the six-base run; `studies/13nov/runoracle.py`)

The oracle (never shippable) gives the stages the true run of each delivered window, and each run is processed as
its own sub-session. No cue-order prior and no labels: the run id is its only extra input. Form B (latent runs from
slow EEG state) would be the shippable version.

Smoke (B2_pool, full perm 0): the run split costs **−3.23 even in recording order** and −4.22 under full shuffle
(shipped stages −3.14). Each run has only about 32 windows, too few for its own templates.

Prediction for the six-base run: the oracle split below the shipped stages under every scheme. **Reading rule:** if so,
hard run-split forms (form B as specified) are closed. The loss is not "templates mix runs" but the amount of local
data. Form A (similarity-weighted templates over the whole delivered set, no hard split) remains the open form.

### 169. SHUFPKG: v5.1b as uploaded under non-chronological delivery, returning people included (written 13 Nov 10:12 UTC after a one-base smoke, before the six-base run; `studies/13nov/shufpkg.py`)

Routing v4.1 plus the own-space stage, with the gate's calibration fixed from v5.1b in recording order. Every store is
permuted, each gate test session is delivered in the same order, and session-keyed caches are cleared per order.
Members are as recorded (§165: re-streaming them adds no loss). The check reproduces recording order exactly (B2_pool new
80.18, seen 82.89).

Smoke (B2_pool, full perm 0), against recording order: new −3.14, **seen −4.66 (P04 −6.67, P10 −2.64), J −3.90**; all
four returning sessions are still routed.

| Check (six bases, full / runs / within, two permutations each) | Prediction (80 % range) |
|---|---|
| seen, full | −4 (−6 to −2) |
| J, full / runs / within | −3.5 / −2.0 / −1.0 |
| **Reading** | the sealed rescoring cost 0.7–1.2 J. A scheme whose J loss lands near that range is the more plausible sealed scheme. Returning people's order loss sizes the next re-design (the own-space stage and the routed tail built for unordered sessions) |

| §168 RUNORACLE (13 Nov 10:12 UTC; `runs/runoracle.log`; new, six bases, two permutations; against the shipped stages in recording order) | Shipped stages | ORACLE run split |
|---|---|---|
| recording order | 0.00 | **−3.48** |
| full / runs / within | −3.10 / −1.95 / −0.97 | −4.61 / −3.48 / −4.10 |

**Verdict:** as predicted, the hard run split loses under every order, including recording order. Runs (about 32
windows) are too short for their own templates. **Form B as specified (latent runs with per-run templates) is
closed.** The shuffle loss is not that templates mix runs. Form A (similarity-weighted templates over the whole
delivered set, no hard split) stays open; its oracle would weight by recording-time distance instead of splitting.

| §169 SHUFPKG (13 Nov 10:24 UTC; `runs/shufpkg.log`, `.npz`; v5.1b as uploaded against itself in recording order; six bases × two permutations) | New | Seen (P04, P10) | J |
|---|---|---|---|
| full | −3.10 | −3.95 (−5.64, −2.25) | −3.52 |
| runs | −1.95 | −3.41 (−6.12, −0.70) | −2.68 |
| **within** | −0.97 | −0.52 (−1.32, +0.29) | **−0.74** |

**Reading.** Returning people lose as much as new people under full and runs; P04 loses most (−5.6 to −6.1). **Sealed
check:** the rescoring of 10 Nov cost the uploaded packages 0.72 to 1.17 J. Only *within* (runs kept in order, windows
shuffled inside each run) lands in that range; full (−3.5) and runs (−2.7) are far outside. **The sealed scheme is most
plausibly a within-run shuffle**, or something as mild. From here, *within* is the primary column and full / runs are
stress columns. This also explains why the combination's sealed value was hard to predict from recording-order
numbers: under *within*, the combination's gain over v5.1b was +1.65 (§165), against +1.00 in recording order.

**Correction to this reading (14 Nov 00:00 UTC, coordinator and s205): the rescoring does not identify the scheme, and
if anything leans away from within.**
- **Transfer:** the reading above compared dev losses with sealed drops 1:1. Dev-to-sealed transfer on the last two
  uploads was 0.41–0.46. At that rate v5.1b would drop about 0.3–0.35 sealed under within and 1.4–1.6 under full. The
  observed 0.7–1.2 sits between the two.
- **Per package:** the rescored packages' dev order losses come from the stage-wise shuffle (runs/shuffle_13nov.log, new
  people, members fixed). cand_nxif = R19e members + SessionEM, so the cumulative loss to SessionEM: full −2.17, runs −0.86,
  within −0.38. cand_rec_ts2_tsem adds TSEM: −3.75 / −2.11 / −1.11. Sealed drops: −0.72 and −1.17, a ratio of 1.63.
  - Dev ratios: full 1.73, runs 2.45, within 2.92.
  - Proportional slope (drop / loss, no offset): full 0.33 and 0.31 (consistent); runs 0.84 / 0.55; within 1.89 / 1.05
    (inconsistent).
  - With a uniform offset c (drop = c + k·loss, fitted exactly through the two packages): full k 0.28, c +0.10; runs k 0.36,
    c +0.41; within k 0.62, c +0.49. Within fits only if about half a point of the drop was a backend effect unrelated to
    order.
- **Noise:** each package saw one sealed order. At ±0.15 per drop the sealed ratio ranges 1.17–2.32, which holds full and
  runs but not within. At ±0.25 it ranges 0.95–3.02, which holds all three.
- **Limits:**
  - two packages carry the fit (route3geb's −1.05 adds returning people, whose pre-routing loss is not measured);
  - nxif's SessionEM was an older form;
  - members' own order dependence is not in the stage-wise loss;
  - the sealed set also holds returning sessions.
- **Consequence:** no scheme is primary. The morning table gives every candidate per scheme without a preference, plus the
  equal-weight and within-only averages. The scheme can be settled only by the organisers (the coordinator suggests that
  the user ask how windows are ordered within a session).

**§170 TORACLE smoke (B2_pool, full perm 0; pre-registration below):** ORACLE stages weighting delivered windows by
true recording distance (H = 30). Full shuffle, against v5.1b in recording order: new −2.97 (shipped −3.14), seen
−4.80 (shipped −4.66). Recording order: new −0.82, seen −2.26. The oracle recovers about 5 % of the new-people loss and
none of the seen loss.

### 170. TORACLE: stages with delivered windows weighted by true recording distance (form A's ceiling) (written 13 Nov 10:24 UTC after the one-base smoke above, before the two-base run; `timeoracle.py`, `toracle.py`)

The H = ∞ check reproduces the shipped stages exactly. Bar, stated before the two-base run: form A (slow-state
similarity in place of true time) gets several variants only if the oracle recovers ≥ 50 % of the shuffle loss under
*within* or *full* for new or returning people, at some H. The smoke recovers about 5 % and 0 %. Running B2_pool and
B2_s345, H 30 / 90, all three schemes.

**Caveat for every shuffled number so far:** the stored stage views (GE for SessionEM, pc128 for TSEM) were standardised
causally in recording order. Under a real shuffled stream their standardisation would follow the delivery order. The
SessionBN members re-streamed without extra loss (§165), and the eye net is re-streamed now (§167). GE and pc128
cannot be re-streamed from stored outputs, so the shuffled losses may be partly an artefact of that mismatch.

### 171. GC5SHUF: the shippable fine-tuning form under shuffled delivery (written 13 Nov 10:24 UTC, before any job ran; `studies/13nov/ttt18.py`, scored with `shufcombo.py FT_V=gC5`)

The build session timed the whole v6 path on an A100 (§120 addendum 3, `docs/V6_DESIGN.md`). Without fine-tuning the
worst case is about 24 min. **gC5 delta1 is 31–38 min and fits the 45-min worst case**; cum10 delta1 reaches 53 min at
378-window sessions. So gC5 delta1 is the package form, and it has no shuffled number yet. TTT18 re-streams gC5 in the
delivery orders within 0, within 1, full 0 and runs 0, with seeds 0–5 nets.

Scoring: the combination with gC5 delta1, eye net re-streamed where available, over v5.1b under the same order. New
people; *within* is the primary column (§169).

| Check (new, over v5.1b under the same order) | Prediction (80 % range) |
|---|---|
| within | +1.20 (+0.70 to +1.70); cum10 gave +1.65 |
| full | +1.00 (+0.40 to +1.60) |
| **Bar (package form)** | without P10 ≥ +0.40 with the 80 % person interval above 0, under *within* on both permutations |

**Plan for an upload decision this evening (13 Nov, written 10:26 UTC).** The user expects a decision at about
19:00 UTC (20:00 CET), earlier than the research window they set. The upload itself still needs their go-ahead by
name. Plan:
- **Provisional package form,** confirmed by about 13:30 UTC on GC5SHUF (§171): v5.1b plus, on unrecognised
  sessions only, gC5 delta1 fine-tuning, the eye-net view and the rest member. Recognised sessions stay bit-identical
  to v5.1b.
- **Build:** v52-build started on the lean tree, with final weights for the eye net and rest net trained on all ten
  dev people.
- **Checks before the decision:** the package equals the dev arm per window; warm, paired A100 run time against
  v5.1b at worst-case session length, with the margin to 60 min; the official local replay; and the list of what is not
  validated (the worker-image replay needs Galvani 2FA).
- **Decision table for the user at about 18:30 UTC:** dev gain per delivery scheme (recording, within, full, runs) for
  new and returning people and without P10; expected run time; what was not checked. If the evidence says "not yet",
  the table says so.

| §167 OCUSHUF in the combination (13 Nov 10:30 UTC; `runs/shufcombo_{full0,within0}_ocu.log`; eye net trained and standardised in the delivery order; new, over v5.1b under the same order) | full 0 | within 0 |
|---|---|---|
| combination (cum10 delta1 + eye net + rest member), both shuffled | **+1.69 [+0.94, +2.46], without P10 +1.36** | **+1.51 [+0.90, +2.10], without P10 +1.20** |
| the same, eye net from recording-order standardisation (§165) | +1.47 / +1.10 | +1.65 / +1.39 |
| the shuffled-standardisation eye net scored in recording order | +0.85 / +0.62 | +0.94 / +0.70 |

**Reading.** With the eye net trained and standardised in the delivery order, as a package would run it, the
combination's gain holds under both schemes (+1.5 to +1.7 new, +1.2 to +1.4 without P10). The prediction (+0.4) missed
high. The eye net is not order-fragile.

| §170 TORACLE (13 Nov 10:30 UTC; `runs/toracle.log`; B2_pool and B2_s345; against v5.1b in recording order) | Shipped stages: new / seen / J | Oracle H 30 | Oracle H 90 |
|---|---|---|---|
| recording | 0 / 0 / 0 | −0.87 / −2.12 / −1.49 | −0.15 / −1.01 / −0.58 |
| full | −3.05 / −4.26 / −3.66 | −2.95 / −4.73 / −3.84 | −2.99 / −4.26 / −3.62 |
| runs | −1.72 / −4.52 / −3.12 | −2.06 / −4.43 / −3.25 | −1.69 / −3.86 / −2.78 |
| within | −0.70 / −1.47 / −1.09 | −0.62 / −1.21 / −0.91 | −0.83 / −1.68 / −1.26 |

**Verdict: refuted** (bar: ≥ 50 % of the loss recovered under within or full). The best case recovers about 17 % of J
(within, H 30) and 11 % under runs (H 90); full is about 0. Weighting the delivered windows by true recording time does
not restore what the stages lose, so **form A (slow-state similarity in place of time) is not given several variants**.
The loss is not about which windows the templates weight. It may be the views' own recording-order standardisation
(the §170 caveat), which this oracle cannot remove.

| §165 more orders (13 Nov 10:31 UTC; `runs/shufcombo_{runs0,full1,within1}.log`; cum10 delta1, eye net as recorded; new) | v5.1b shuffled (members re-streamed) against recording order | **combination over v5.1b, both shuffled** [80 %], without P10 |
|---|---|---|
| runs 0 | −2.25 | **+1.87 [+1.14, +2.58], +1.54** |
| full 1 | −2.90 | **+1.54 [+0.94, +2.18], +1.36** |
| within 1 | −1.26 | **+1.65 [+1.10, +2.24], +1.35** |

**Over six delivery orders** (full 0/1, within 0/1, runs 0; plus full 0 and within 0 with the eye net re-streamed), the
combination gains **+1.47 to +1.87 for new people over v5.1b under the same order (without P10 +1.10 to +1.54), with
every 80 % person interval above +0.68**, against +1.00 in recording order. The fine-tuning form here is cum10 delta1;
gC5 delta1 (the run-time form) is §171, due about 11:20 UTC.

### 172. GC5SHUF draws on Galvani (written 13 Nov 10:33 UTC, before any job ran; `studies/13nov/ttt18.py` with DRAW, `scripts/cloud/galv_array.sh`)

Galvani is reachable again (the user's EDU VPN). New waves move there step by step, and Leonardo keeps a smaller share.
The first Galvani wave: gC5 re-streamed under *within* (the plausible sealed scheme), permutations 0 and 1 × fresh
augmentation draws 1 and 2 (96 tasks; smoke 3096130, arrays 3096131–3096134). Code and delivery orders were synced
first. Galvani tasks skip finished outputs and requeue when preempted. Scoring as §171, against v5.1b under the same
order. Together with §171's draw 0 this gives the shippable form three draws × two permutations under *within* before
the decision; same prediction and bar as §171.

### 173. The no-FT fallback under shuffled delivery (written 13 Nov 10:34 UTC, before any result; `shufcombo.py` arm V6r)

The build session reports that fine-tuning through the shipped TorchScript traces costs 3–4× eager on an A100 (about
60 min worst case for gC5 delta1). If neither the executor fix nor eager rebuilds work, the package would ship the eye
net and rest member without FT. That form has only a recording-order number on the shipped members (§151: new +0.34,
J +0.05). V6r is scored under within 0/1 and full 0 (eye net re-streamed where available), new, against v5.1b in the
same order. Prediction: +0.6 (+0.1 to +1.1), since under shuffle the eye-net view may carry more than in recording
order. **Bar for the fallback to be uploadable on its own:** without P10 ≥ +0.40 with the 80 % interval above 0 under
*within* on both permutations.

| §173 fallback without FT (13 Nov 10:36 UTC; `runs/fallback_*.log`; eye net + rest member on unrecognised sessions; new, over v5.1b under the same order) | New [80 % person] | Without P10 | Bar |
|---|---|---|---|
| within 0 (eye net re-streamed) | **+0.73 [+0.40, +1.09]** | +0.78 | met |
| within 1 (eye net as recorded) | **+0.76 [+0.43, +1.10]** | +0.84 | met |
| full 0 (eye net re-streamed) | +0.68 [+0.42, +0.93] | +0.64 | — |

**Verdict: the fallback meets its own bar** under *within* on both permutations (prediction +0.6, hit). Its gain is not
P10's (without P10 it is a little larger), unlike FT's. On v5.1b's shipped members and under the plausible sealed
order, the eye-net view plus the rest member alone are worth about +0.7 for new people (J about +0.35 with recognised
sessions untouched). FT adds about +0.8 to +0.9 more (§165). **Package decision rule from here:** if FT runs inside
the run-time budget, ship the full combination; otherwise ship the fallback. Both pass their bars under shuffled
delivery.

### 174. Next forms for the stage loss under shuffle (written 13 Nov 10:56 UTC)

Refuted so far: RETUNE (stage constants, §166), the hard run split (§168), and the time-weighted oracle (§170). What
comes next, in order:

1. **Measurement validity first.** Core1's member output and its components, which are TSEM's view (pc128) and part of
   the gate's input, use causal references built along the session's row order (`experiments/ts_core.py`,
   `causal_refs`). Every shuffled number so far kept them in recording order. Recomputing core1 in the delivery order
   (a ts_core run with a delivery-order option, like ttt6's PERM) tells how much of the −1 to −3.5 J stage loss is
   real and how much is a recording-order artefact of the views. The same applies to the gaze and event view of
   SessionEM. Until this runs, the size of the stage loss is uncertain in both directions.
2. **Order-free session estimates.** At each delivered window, run SessionEM and TSEM as a fixed point over all
   delivered windows (EM to convergence), instead of the running, prefix-built template. That is the natural estimator
   when the delivered set is an exchangeable sample. Prediction: it recovers part of the loss under full and runs; a
   ceiling is measured first by giving it the whole session (non-causal).
3. **None beyond these** are specified yet. Slow-state similarity (form A) is not pursued after §170.

Queue: GC5SHUF draws are duplicated on Leonardo (64374729–64374732) next to Galvani, and the first copy to land is
used. Item 1 is built next, on Galvani or Leonardo CPU, behind the evening path.

### 175. CORESHUF: TSEM's view recomputed in the delivery order (written 13 Nov 10:57 UTC, before any result; `studies/13nov/coreshuf.py`, Leonardo CPU job; §174 item 1)

Core1's causal references are rebuilt along the delivery order (within 0 and full 0) and projected to the fold's 128
PCs, the TSEMK recipe of §54, with a harness check of the recording-order rebuild against the stored view. The view
then replaces pc128 in §164's replay, permuted with everything else, to give v5.1b's stage loss with TSEM's view built
as a shuffled stream would build it. Core1 as a member (slot c1) and the gaze and event view stay as recorded, so
this is a partial correction.

| Check (new, B2_pool and B2_s345, v5.1b against recording order) | Prediction (80 % range) |
|---|---|
| within, TSEM view in delivery order against §164 (−0.97) | −0.8 (−1.5 to −0.2) |
| full, against §164 (−3.10) | −2.5 (−3.5 to −1.5) |
| **Reading** | if the loss shrinks by more than a third, part of every shuffled number so far is a view artefact, and the package expectation for v5.1b under the sealed order is better than measured |

| §171 GC5SHUF (13 Nov 10:58 UTC; `runs/gc5shuf_*.log`; gC5 delta1 + eye net + rest member on unrecognised sessions; new, over v5.1b under the same order) | Combination, shuffled [80 %] / without P10 | Recording order | No-FT fallback, shuffled |
|---|---|---|---|
| **within 0** (eye net re-streamed) | **+1.47 [+0.91, +2.02] / +1.18** | +0.85 | +0.72 |
| **within 1** | **+1.38 [+0.89, +1.90] / +1.11** | +0.81 | +0.78 |
| full 0 (eye net re-streamed) | +1.60 [+0.97, +2.27] / +1.31 | +0.70 | +0.69 |
| runs 0 | +1.67 [+1.03, +2.32] / +1.37 | +0.81 | +0.53 |

**Verdict: the bar is met** (without P10 ≥ +0.40 with the interval above 0, *within* on both permutations; prediction
+1.20 within, hit high). gC5 delta1 keeps nearly all of cum10's gain under shuffle (+1.38 to +1.67 against +1.47 to
+1.87). **The package form is confirmed:** v5.1b, plus gC5 delta1 fine-tuning, the eye-net view and the rest member on
sessions the gate has not recognised; recognised sessions stay bit-identical to v5.1b. Dev expectation under the
plausible sealed order (*within*): new +1.4 (without P10 +1.1), returning 0, J about +0.7. Fresh FT draws (§172) are
still to come.

### 176. EVERYTHING IN DELIVERY ORDER: the remaining inputs re-streamed (written 13 Nov 11:01 UTC, before any job ran; Leonardo)

The coordinator asks that tonight's table score v5.1b and both package forms with every order-dependent input in the
delivery order, and list any input still in recording order. Done so far: the SessionBN members and fine-tuning (§165,
§171), the eye net (§167), and TSEM's view (§175, running). Added now:
- **COREMEMBER** (`coremember.py`, CPU job): core1 as a member (slot c1) streamed in the delivery orders within 0/1,
  full 0 and runs 0 through `deliver.py`, a generic hook that reorders each session inside `protocol.predict_streams`.
  The head is chosen by reproducing the stored member in recording order (asserted correlation ≥ 0.99).
- **RESTSHUF** (`restshuf.py`): the rest net (sh, r1, seeds 0–2) streamed in the same four orders.
- **More orders for the package form:** gC5 via ttt18 for within 2 and 3, full 1, runs 1; eye net via ocushuf for
  within 1 and runs 0.
- **Follow-up wave queued behind:** gC5 draw 3 under within 0 and 1.

**Still in recording order after this:** the gaze and event view of SessionEM (GE: 48 hand-built features; its
causal standardisation would need its own export), and the Hilbert core and transformer members (their causal
statistics). These go into the table's "not checked" line unless a later job removes them.
Scoring when all land: `shufcombo.py` with core1 (member and view), the rest net and the eye net all from their
delivery-order files.

**Scoring form and bar (written 13 Nov 11:07 UTC, before any delivery-order core1 or rest-net result).** `shufcombo.py`
now takes each re-streamed input as a change against its own recording-order rebuild, added to the stored input, so
recording order stays bit-identical to v5.1b. Core1 the member: c1 plus the change, with Ps[0] moved by half of it so
that the Hilbert member (hb = 2 Ps[0] − c1) stays as stored. pc128: the column-sign-aligned change. Rest net: the
re-streamed seeds replace the recorded ones in V6Fr and V6r. All of these go into P0r, V6Fr and V6r; P0rec, P0s and
V6Frec stay as stored. `STRICT=1` fails when any of them is missing for the order. The header prints what is still in
recording order.
No-op check before scoring: zero-change stand-ins must reproduce `runs/shufcombo_within0_ocu_gC5.npz`, and perturbed
stand-ins must change P0r, V6Fr and V6r but not P0rec, P0s or V6Frec.
Prediction: re-streaming core1 costs v5.1b 0 to 0.5 more under within (P0r against P0rec). The gC5 combination's gain over
v5.1b, both fully re-streamed, stays within 0.3 of §171 (+1.38 to +1.67). Refuted if it falls below +0.8 new on the mean
of within 0 and 1. That is still a gain, but then the table quotes the lower figure and calls §171 optimistic. Bar for the
table: ≥ +0.5 new and > 0 without P10 on both within orders, otherwise "not yet". The fallback (V6r) is under the same
bar at ≥ +0.3.
No-op check passed (13 Nov 11:10 UTC). The zero-change stand-ins reproduce the earlier within 0 run exactly (max
difference 0.0). The perturbed stand-ins move P0r, V6Fr and V6r (up to 3.1 points per person) and leave P0rec, P0s and
V6Frec at 0.0. CORESHUF covered only within 0 and full 0, so pc128 for within 1 (PSEED 1) and runs 0 is queued
(Leonardo 64376116, 64376119; `coreshuf_p<seed>_<schemes>.npz`). Scored per order when complete:
`runs/allorder_<order>.log`.

### 177. AFTERNOON WAVE: four more orders with every input in delivery order; upper-bound forms on Galvani (written 13 Nov 11:50 UTC, before any job ran)

The coordinator (11:46 UTC) asks for both clusters to be filled, with a second wave chained behind each.
- **Leonardo, wave 1 (exploit):** within 2, within 3, full 1 and runs 1 are covered for gC5 (ttt18, home). Add for the
  same four orders: core1 member (`coremember.py`, ORDERS env, `runs/coremember_<orders>.npz`), pc128 (`coreshuf.py`, one CPU
  job per order), rest net (`restshuf.py`, seeds 0–2), eye net (`ocushuf.py`, seeds 0–8).
  **Wave 2 (afterany wave 1):** gC5 draw 3 for full 0 and runs 0, draw 4 for within 0 and 1.
- **Galvani, explore (upper bound under shuffle):** cum10, the heavier fine-tuning form, re-streamed (ttt17) for within 2
  and 3. The eye net with nine more seeds (9–17) for within 0 and 1. Smoke first, arrays afterok.
  **Wave 2:** ttt17 for full 1 and runs 1.

Predictions. (a) The gC5 combination over v5.1b with all inputs in delivery order, mean over the eight orders: +1.2
(+0.8 to +1.6) new; the fallback +0.6 (+0.3 to +0.8). Refuted if the mean is < +0.8, which would mean §171's four orders
were the lucky ones. (b) cum10 against gC5 under the same order: +0.15 (0 to +0.3). Within noise this is "undecided",
and gC5 stays (it fits the time budget). (c) 18 against 9 eye-net seeds: +0.05. Kept only if > +0.3.
**Explore items not yet queued (they need code first, after the evening table):** the order-free EM fixed point
(§174 item 2), and a stage for returning people under shuffle (they lose most to shuffling, and the package gives them 0).

**§171 draws 1 and 2 (13 Nov 11:52 UTC; `runs/gc5shuf_within{0,1}d{1,2}.log`; Galvani; core1, pc128 and rest net as recorded,
eye net re-streamed for within 0 only).** gC5 combination over v5.1b, both under the same order, new [80 %], without P10:

| Order | draw 0 (§171) | draw 1 | draw 2 | fallback (no FT), draws 1/2 |
|---|---|---|---|---|
| within 0 | +1.47 | +1.20 [+0.66, +1.73], +0.95 | +1.29 [+0.74, +1.84], +1.02 | +0.68 / +0.76 |
| within 1 | +1.38 | +1.44 [+0.97, +1.93], +1.17 | +1.28 [+0.86, +1.73], +1.00 | +0.79 / +0.77 |

All six pass the bar (≥ +0.5 new, > 0 without P10). The spread between draws (0.27) is inside the measure's noise.
**Worker image (Galvani 3096279, CUDA, image torch 2.14.1+cu130):** v5.1b own, own_b, lean_v5, cand_v6f_gc5 and
cand_v6f_noft all give 0.99328 on the local replay sample (files unchanged, about 100 s each). This ran on a build from
before the executor fix, so it is "not on the final ZIP". Equal scores here do not show that the new-person path ran;
v52-build's no-op check (recognition off) covers that.

### 178. PACKAGE-FAITHFUL FINE-TUNING LABELS AND GATE under shuffle (written 13 Nov 12:16 UTC, before any job ran)

v52-build's list of what was not validated (12:00 UTC) names two ways that dev's shuffle scores differ from the package:
- **Labels.** Under shuffle, ttt18 fine-tunes on `chain_final.npz`, which is v5.1b's chain in recording order (checked:
  equal to `gm.pipe_g` on B2_pool in recording order, agreement 1.0000, max |Δp| 0). The package fine-tunes on v5.1b's
  outputs as emitted in the delivery order. `shufcombo.py LABEL_OUT` writes those: `studies/13nov/chain_v51b_<order>.npz`,
  with pc128 and the SessionBN members re-streamed and core1 as recorded. Label accuracy, new people:
  within 0 0.7518 and within 1 0.7533, full 0 0.7348, runs 0 0.7405, against 0.7657 in recording order
  (agreement 0.91, 0.92, 0.86, 0.88). ttt18 now takes `CHAIN_NPZ` (only this order's file, asserted; tag suffix L).
  Queued on Galvani: gC5 with these labels for the four orders.
- **Gate.** In dev the combination's gate is computed on the combination's own chain; the package computes it on its
  inner v5.1b. `shufcombo.py GATE_P0=1` takes the gate from v5.1b's chain under the same order.

Prediction: labels cost the gC5 combination 0.1 (0 to 0.3) under within shuffle and 0.2 (0 to 0.4) under full and runs.
That is from the true-label ceiling (+1.9 for about 23 points of label accuracy, so 1.3 to 3 points are worth 0.1 to
0.25). The package gate costs 0 to 0.15. Refuted if either costs more than 0.3; then §171 was optimistic by that much.
Either way the table quotes the package-faithful numbers (labels L, GATE_P0) where they exist and marks the rest.

**§178 gate result (13 Nov 12:19 UTC; `runs/gatep0_within{0,1}.log`; gC5 draw 0, eye net re-streamed for within 0, other inputs as in
§171).** With the package's gate (computed on v5.1b's chain), against the gate on the combination's own chain:

| Order | gC5 combination, new [80 %], without P10 | own gate (§171) | fallback, package gate / own |
|---|---|---|---|
| within 0 | +1.41 [+0.85, +1.98], +1.12 | +1.47 | +0.71 / +0.72 |
| within 1 | +1.32 [+0.91, +1.74], +1.11 | +1.38 | +0.69 / +0.78 |

The package gate costs 0.06 on both orders, within the prediction (0 to 0.15) and inside the noise. Recording order
gives +0.89 and +0.71, against +0.85 and +0.81 with the own gate. Not refuted. The table uses GATE_P0 from now on.

**§176 RESULT: every input re-streamed in delivery order (13 Nov 12:38 UTC; `runs/allorder_<order>_gp0.log`, STRICT, package gate).**
- In delivery order: SessionBN members, core1 member, pc128, rest net, eye net, and gC5 fine-tuning (draw 0, labels still
  from recording order; §178 replaces them).
- Still in recording order: the GE view, the Hilbert member, the transformer and shallow members, and the teacher.
- Before scoring, P0s (untouched by the new inputs) was checked against the earlier runs: equal on all four orders.
- New people, against v5.1b under the same order [80 %], without P10:

| Order | v5.1b re-streamed vs recording order | gC5 combination over v5.1b | fallback (no FT) over v5.1b |
|---|---|---|---|
| within 0 | −1.20 | +1.20 [+0.57, +1.83], +0.86 | +0.74 [+0.42, +1.06], +0.71 |
| within 1 | −1.62 | +1.58 [+1.17, +1.98], +1.37 | +1.23 [+0.90, +1.57], +1.33 |
| full 0 | −3.32 | +1.58 [+0.89, +2.28], +1.34 | +0.63 [+0.39, +0.87], +0.65 |
| runs 0 | −2.31 | +1.78 [+1.00, +2.53], +1.54 | +0.36 [−0.02, +0.71], +0.54 |

Prediction (a) of the scoring form: within mean +1.39, within 0.3 of §171; not refuted. The table bar is met on both within
orders for gC5 (≥ +0.5, > 0 without P10) and for the fallback (≥ +0.3).
Note: within 0 (+1.20) is the lowest of the four orders.

### 179. LEONARDO WAVE: label-faithful gC5 copies, four more orders, more draws (written 13 Nov 12:38 UTC; jobs were queued 12:44 UTC, before any of them ran or was scored)

The coordinator (12:36) asks that the label-faithful number not depend on Galvani's queue (4 of 96 tasks running).
- **Copies:** t18L on Leonardo for within 0 and 1, full 0, runs 0 (64385496–500). The first copy to land is used, as
  with the draws.
- **Four more orders:** labels job 64385503 (afterok coremember2) writes chain_v51b_<order> for within 2 and 3,
  full 1 and runs 1. This is fully re-streamed (core1 member and pc128 included, STRICT). Then t18L arrays
  64385506–14 run afterok.
- **Chained:** t18L draw 1 for within 0 and 1 (64385516/20, afterany the copies).
- Scoring: `score_L` loop, STRICT, GATE_P0, FT tag …L. All eight orders → `runs/allorder_<order>_gp0_L.log`.
Predictions as §178: labels cost the combination 0.1 (0 to 0.3) under within and 0.2 under full and runs. The
label-faithful mean over eight orders is +1.2 (+0.9 to +1.5) new. Refuted if it is < +0.8; then the table says so.
Explore code still to write: the order-free EM fixed point and a stage for returning people under shuffle. Due by
16:00 UTC, smoke-tested and then queued.

### 180. BUG: stages and re-streamed members used different delivery orders for within and runs (found 13 Nov 12:45 UTC; fixed)

`shuffle.perm_for` seeded its RNG with the scheme's position in the SCHEMES env. `shufcombo.py` sets SCHEMES to one
scheme (index 0), while `runs/shuffle_perms.npz` was dumped with the default `full,runs,within` (full 0, runs 1, within 2).
Every re-streamed member follows the dumped orders: ttt17/ttt18, ocushuf, restshuf, coremember, coreshuf.
For **full** the two agree. For **within** and **runs**, the stages ran in another random permutation of the same kind
than the members were streamed in. A window early in the stage order could carry a member output (fine-tuning,
SessionBN, standardisation) adapted on windows that come later in the stage order. The direction is likely optimistic
for the adaptive parts, but it was not measured.
- **Fix:** `SCHEME_ID = {full: 0, runs: 1, within: 2}`, independent of the env. Checked for within, runs and full, seeds
  0–3, both protocols: the stage order equals the dumped order. `shufcombo.py` now asserts this and stops otherwise.
  Callers of shuffle.py with the default SCHEMES keep the same permutations: §164, §166 RETUNE, `DUMP_PERMS`.
- **Affected (within and runs only):** §165, §167 (within), §171 and its draws, §173, the §176 result, and §178 (the gate
  check on within; the label files for within 0/1 and runs 0, built in the stage order). Their full-scheme numbers stand.
- **Withdrawn and redone:** the t18L runs for within 0/1 and runs 0 were cancelled (Galvani 3096653/54/56, Leonardo
  64385496/97/500 and draws 64385516/20). 28 wrong-label outputs were deleted on Galvani. The labels were rebuilt in the
  dumped order with every input re-streamed (STRICT): accuracy 0.7552 / 0.7589 / 0.7533. They were resubmitted on both
  clusters (Leonardo 64386161/64/66, draws 64386167/69; Galvani 3096837–39). Hashes match on both clusters.
  The labels2 job (within 2/3, full 1, runs 1) had not started and uses the fixed code.
- **Rescoring without new GPU work** (`runs/allorder_*_gp0.log`; old logs kept as `*_WRONGORDER.log`): within 0/1 and runs 0
  with all inputs and the package gate, plus the within draws 1 and 2.
- **Also found:** 349b1bf left `shufcombo.py` with a syntax error (a comment inside the savez call). Fixed before any job
  used it; Leonardo had the earlier version.

### 181. EMCEIL: where the stage loss under shuffle comes from (§174 item 2; written 13 Nov 12:45 UTC, after a one-base smoke in recording order only)

`studies/13nov/emceil.py`. Item 2's premise is mostly already true. SessionEM refits an EM fixed point over every
delivered window at each step, and TSEM rebuilds its template from all earlier windows. What carries the order is:
- the rebalancer's running window (W 30);
- the prefix itself;
- TSEM labels fixed at emission.
Arms, members as recorded, six bases: C0 (shipped, asserted equal to pipe_g), Winf (rebalancer over the whole prefix:
causal, order-free), EMc and TSc (non-causal ceilings fitted on the whole session), ALLc (Winf + EMc + TSc).
Smoke (B2_s0, recording order): Winf −1.24, EMc +0.24, TSc +0.37, ALLc −0.44 against C0. So the W 30 window is worth
1.2 in recording order, consistent with it exploiting temporal class blocks. Whole-session knowledge adds little.
Prediction under within, full and runs: Winf ≥ C0 (the local window loses its edge when there are no blocks), and the
ceilings recover ≤ 0.5 of the 1.2 to 3.3 loss. Then the loss is mostly temporal structure that recording order offers
and a shuffled stream does not; no order-free estimator can win it back, and item 2 closes as "no ceiling".
Refuted, and an arm becomes a package candidate, if Winf beats C0 by ≥ 0.3 on both within and full (it is causal and cheap).
**§180 scope, item by item.**
- **Withdrawn for within and runs (rescore running):**
  - v5.1b's own loss with re-streamed members (P0r: §165, §176 and the first column of the 12:50 table);
  - the combination and the fallback (§165, §167, §171 and its draws, §173, §176);
  - the package-gate cost (§178, both arms had the same mismatch).
- **Stand:**
  - every full-scheme number;
  - P0s (stages only, members as recorded): consistent within itself, only a different order sample;
  - §164;
  - §169 returning people (`shufpkg.py`, stages only, default SCHEMES, so its permutations are unchanged);
  - RETUNE §166, RUNORACLE §168, TORACLE §170 (stages only);
  - the pc128 and core1 change sizes (`coreshuf.py` and `coremember.py` read the dumped orders directly).
- **Order file:** `runs/shuffle_perms.npz` is byte-identical on the Mac, Leonardo and Galvani (md5 f9466522…).
  shufcombo now pins its SHA-256 (f9209851…) and asserts that the stage order equals the dumped order.
- **Next guard (new jobs):** re-streamed outputs should carry the hash of the order they streamed. Not yet in the GPU
  scripts. Until then all of them read the pinned file by (scheme, seed) key.

**§181 RESULT (13 Nov 12:47 UTC; `runs/emceil.log`; six bases, corrected orders, members as recorded).** New people; each arm's
change against C0 under the same order, with C0's own loss against recording order in brackets:

| Order | C0 loss | Winf | EMc (ceiling) | TSc (ceiling) | ALLc |
|---|---|---|---|---|---|
| recording | 0 | −1.04 | +0.07 | −0.05 | −0.71 |
| within 0 | (−0.74) | −0.24 | −0.02 | +0.05 | +0.10 |
| full 0 | (−3.07) | +0.21 | +0.93 | +0.77 | +1.43 |
| runs 0 | (−1.66) | −0.18 | +0.18 | +0.92 | +0.34 |

- **Winf is refuted as a candidate.** It costs 1.0 in recording order and 0.2 under within and runs, and gains only 0.2
  under full.
- **Within** (the scheme that matches the sealed drop) has no ceiling: whole-session knowledge recovers +0.10 of
  −0.74. There, item 2 is closed.
- **Full and runs** have a non-causal ceiling of about half the loss: full +1.43 (ALLc), runs +0.92 (TSc). A causal
  form can only get part of it. Recorded as "undecided": worth a form only if later evidence points to a full or runs
  stream.
- **Prediction:** confirmed for within. For full and runs the ceilings sit at the stated limit (≤ 0.5 of the loss).

**§180 RESCORE (13 Nov 12:49 UTC; corrected orders; `runs/allorder_*_gp0.log`).** Every input is re-streamed except GE, Hilbert,
transformer/shallow and teacher; package gate; STRICT; FT labels still from recording order (§178/§179 replace them).
New people, against v5.1b under the same order [80 %], without P10:

| Order | v5.1b re-streamed vs recording order | gC5 combination | fallback (no FT) |
|---|---|---|---|
| within 0 | −1.25 | +1.97 [+1.51, +2.41], +1.83 | +0.96 [+0.55, +1.38], +1.09 |
| within 0, draws 1 and 2 | | +2.14, +2.12 (without P10 +2.04, +2.02) | +1.01, +0.93 |
| within 1 | −1.31 | +1.55 [+0.89, +2.24], +1.37 | +0.76 [+0.48, +1.02], +0.81 |
| within 1, draws 1 and 2 | | +1.56, +1.40 (without P10 +1.38, +1.20) | +0.73, +0.76 |
| full 0 (unaffected) | −3.32 | +1.58 [+0.89, +2.28], +1.34 | +0.63 [+0.39, +0.87], +0.65 |
| runs 0 | −1.70 | +1.33 [+0.76, +1.93], +1.24 | +0.94 [+0.47, +1.45], +1.05 |

The mismatch was not uniformly optimistic: within 0 rose by 0.77, within 1 stayed level and runs 0 fell by 0.45.
All rows meet the table bar.

**§178 gate cost restated in corrected orders (13 Nov 12:51 UTC; `runs/allorder_*_owngate.log` against `*_gp0.log`).** Package gate
minus the combination's own gate:
- gC5: within 0 −0.04 (+1.97 against +2.01), within 1 +0.07 (+1.55 against +1.48), runs 0 +0.01.
- Fallback: +0.01, +0.05, −0.03.
The gate form makes no difference (|Δ| ≤ 0.07). Not refuted; the table keeps the package gate.

**§171/§179 draws 3 and 4 (13 Nov 13:08 UTC; corrected orders, every input re-streamed, package gate, labels from recording order;
`runs/allorder_<order>d<draw>_gp0.log`).** gC5 combination over v5.1b, new (without P10):
- within 0: draw 3 +2.18 (+2.08), draw 4 +1.99 (+1.86).
- within 1: draw 3 +1.56 (+1.38), draw 4 +1.52 (+1.32).
- full 0: draw 3 +1.54 (+1.29).
- runs 0: draw 3 +1.30 (+1.20).
Across draws 0–4: within 0 +1.97 to +2.18 (5 draws), within 1 +1.40 to +1.56 (5), full 0 +1.54 to +1.58 (2), runs 0 +1.30
to +1.33 (2). The spread between draws is ≤ 0.21, inside the noise; the order matters more than the draw. The fallback
does not depend on the draw: +0.63 to +0.96.

### 182. UPPER BOUND OF THE FT FORM WITH PACKAGE LABELS; more rest seeds (written 13 Nov 13:32 UTC, before any job ran)

- The label-faithful gC5 copies landed on Leonardo first. All 144 tasks read the rebuilt labels (the hashes in the logs
  equal the local files), and the local files are identical to Leonardo's. The Galvani duplicates were cancelled and
  their partial outputs deleted.
- **cum10 with package labels** (`ttt17.py` now takes CHAIN_NPZ, tag suffix L): within 0/1, full 0 and runs 0
  (64389580–83, behind smoke 64389579). Chained: within 2/3 after labels2 (64389588/89). Scored with `shufcombo`
  FT_V=cum10, delta1, every input re-streamed, package gate.
  Prediction: cum10 over gC5 under the same order and labels +0.15 (0 to +0.35), the §171 recording-order gap.
  If ≤ +0.3 it is "undecided", and gC5 stays, because it fits the run time (cum10 delta1 needs up to 53 min at 378
  windows). Only if > +0.3 is a form between the two, fitting 50 min, worth building.
- **Rest seeds 3–8** for within 0/1 (64389585/87). Prediction: 9 seeds against 3 add ≤ +0.1; kept only if > +0.3.
- Eye-net seeds 9–17 (Galvani, §177) are scored in the same way when home.

**§178/§179 RESULT, PACKAGE-FAITHFUL (13 Nov 13:32 UTC; `runs/allorder_<order>[d1]_gp0_L.log`).** Corrected orders; every input re-streamed
except GE, Hilbert, transformer/shallow and teacher; package gate; gC5 fine-tuned on v5.1b's outputs as emitted in that
order (Leonardo, label hashes verified). New people, over v5.1b under the same order [80 %], without P10, and the change
against the same run with recording-order labels:

| Order | gC5, package labels | labels' cost | fallback (no FT) |
|---|---|---|---|
| within 0, draw 0 | +1.70 [+1.28, +2.14], +1.68 | −0.27 | +0.96 [+0.55, +1.38], +1.09 |
| within 0, draw 1 | +1.63 [+1.31, +1.95], +1.62 | −0.51 | (same) |
| within 1, draw 0 | +1.33 [+0.79, +1.86], +1.22 | −0.22 | +0.75 [+0.47, +1.02], +0.80 |
| within 1, draw 1 | +1.33 [+0.86, +1.81], +1.21 | −0.23 | (same) |
| full 0 | +1.18 [+0.74, +1.64], +1.17 | −0.40 | +0.63 [+0.39, +0.87], +0.65 |
| runs 0 | +1.51 [+1.01, +2.02], +1.44 | +0.18 | +0.93 [+0.47, +1.44], +1.04 |

- **Prediction (§178):** labels cost 0.1 (0 to 0.3) under within and 0.2 (0 to 0.4) under full and runs.
  - Within: cost 0.22 to 0.27 on draw 0. Inside the range, not refuted.
  - Full: cost 0.40, at the edge.
  - Within 0 draw 1: cost 0.51.
  - The refutation rule (> 0.3) holds for full 0 and for within 0 draw 1, so §171 with recording-order labels was
    optimistic by up to 0.5 there.
- **The package-faithful gain** is +1.18 to +1.70 new and +1.17 to +1.68 without P10. Every row meets the bar.
- Without P10 is almost as large as all people: the gain is spread over people, not carried by P10.
- The mean over the four draw-0 orders is +1.43 new, against +1.6 with recording-order labels.

### 183. RETURNING PEOPLE THROUGH THE NEW PATH under shuffle (written 13 Nov 13:33 UTC, before any result)

The coordinator asks what the new path would give returning people, who lose most under shuffle and get 0 from the
package by design. `shufcombo.py PROTO=seen_session` runs the same arms on the returning-people protocol with
recognition forced off: every window goes through the unrouted tail (`gm.pipe_g`). Inputs are re-streamed as for new
people. FT labels are chain_final's seen part (recording order; no delivery-order labels exist for seen rows yet).
Package gate, STRICT. Orders within 0 and 1. A regression check first confirms the new-people path gives the logged
within 0 numbers unchanged.
**Limit:** the comparison is against v5.1b's unrouted tail, not against the package's routed path for returning people
(personal heads, §169 J). It shows what the v6 additions do on these sessions, not whether routing them through v6
would beat routing.
Prediction: +0.8 (+0.3 to +1.3) over unrouted v5.1b under within. That is smaller than for new people, because the
members already fit these people and fine-tuning has less to correct. If it is ≥ +1.0, routing returning people through
v6 is worth a direct comparison against the routed path (§169 machinery) after the table.

**§183 RESULT (13 Nov 13:35 UTC; `runs/returning_within{0,1}.log`; regression check OK, the new-people path is unchanged).** Returning people
(P04, P10) with recognition forced off. Each line is against v5.1b's unrouted tail under the same order: mean | P04 | P10.

| Order | v5.1b re-streamed vs recording | gC5 combination | fallback (no FT) |
|---|---|---|---|
| within 0 | −0.46 (P04 −0.67, P10 −0.25) | +0.98 (+1.95, +0.02) | +1.05 (+1.51, +0.59) |
| within 1 | +0.48 (P04 +1.73, P10 −0.77) | +1.25 (+2.07, +0.44) | +0.79 (+1.29, +0.29) |

In recording order the combination gives +1.27 / +1.31.
- **Prediction** +0.8 (+0.3 to +1.3): inside; not refuted.
- The gain sits on P04. P10 gets +0.0 to +0.6.
- With two people this is not an estimate. It shows the v6 additions do not hurt returning people on the unrouted path.
- **Limit (unchanged):** the package routes returning people through personal heads, and this does not compare against
  that path. The gain is ≥ +1.0 on one order, so the direct comparison (v6 path against the routed path, §169
  machinery) is the next step; it is not for tonight's upload.

### 184. THE LAST RECORDING-ORDER INPUTS: GE event rates, Hilbert, teacher (written 13 Nov 13:42 UTC)

An agent traced the five inputs still in recording order and recomputed slots 1–4, 10 and GE from their sources, exactly.
- **GE:** the gaze half is per window; the event rates (16 columns, a running median per session) depend on order.
- **Hilbert:** `ts_core2` `car_fb7_hfs2_t0.1_le_k2`, head lr3 (C = 3); causal log-Euclidean reference and slow centring.
- **Transformer and shallow:** only the `ea` slots (causal EA, k 5) depend on order; `raw` uses a fixed scale with BN in
  eval mode.
- **Teacher:** depends on order through the causal EA of its inputs.
- **Head naming in coremember.py:** heads are C values (`fit_head`: C = float(h[2:])). The core1 re-stream used lr0.3
  (C 0.3) against a stored lr10. It passed the 0.99 harness (0.9976), and only its change enters.

**GE (`geshuf.py`, done; scored before this entry was written, so not pre-registered; stated here).** The recording-order
rebuild equals the stored event rates exactly. In delivery order they change by 1.4 % of an sd under within and 2.4–4 %
under full and runs. `shufcombo` adds the change to GE columns 32–47 (asserted to be the event rates). The regression
check with GE and Hilbert off reproduces the logged package-faithful within 0 line exactly.
Package-faithful plus GE (`runs/allorder_<order>_gp0_L_ge.log`). gC5 over v5.1b, new (without P10), and the change from GE:
- within 0: +1.70 (+1.67), change 0.00.
- within 1: +1.39 (+1.30), change +0.06.
- full 0: +0.93 (+0.85), change −0.25.
- runs 0: +1.34 (+1.30), change −0.17.
- Fallback: +1.03, +0.91, +0.59, +0.73.
- All meet the bar. Full 0 is now the lowest order at +0.93.

**Queued:**
- **Hilbert:** `coremember.py MEMBER=hilbert CFG=car_fb7_hfs2_t0.1_le_k2 HEADS=lr3` (Leonardo 64389930 for within 0/1, full 0,
  runs 0; 64389931 for the other four orders). `shufcombo` moves Ps[0] by half the change.
- **Teacher:** `teachshuf.py` (Leonardo 64390300/02). Causal EA in the delivery order; the map is fitted on training
  people's recording-order features and applied to the test rows' delivery-order features. The harness asserts that the
  rebuilt teacher equals slot 10. The 24 checkpoints were copied from Galvani. Locally, `runs/pretrain_a100all` holds symlinks from a Galvani pull whose targets never existed on the Mac; nothing was deleted (coordinator, 15:00).
- **Still left:** the `ea` transformer and shallow slots (weights 0.5 and 1 of 15). They need GPU retraining per fold
  (`run_cv.py` with deliver.py), so not before the table. Their effect is not measured; together they carry 1.5 of the
  mix's 15 weight.
- **Prediction** (Hilbert and teacher together): v5.1b loses 0 to 0.5 more under within, and the combination's gain
  changes by ±0.3. The teacher has weight 5, so if it moves more than that, the table carries the teacher-included number.

**§177 RESULT, four further orders (13 Nov 13:58 UTC; `runs/allorder_{within2,within3,full1,runs1}_gp0.log`; corrected orders; core1
member, pc128, GE event rates, rest net, eye net, SessionBN and gC5 in delivery order; package gate; labels still from
recording order; Hilbert, teacher and the ea slots as recorded).** New people, over v5.1b under the same order [80 %], without P10:

| Order | v5.1b re-streamed vs recording | gC5 combination | fallback (no FT) |
|---|---|---|---|
| within 2 | −1.04 | +0.81 [+0.43, +1.17], +0.66 | **+0.17** [−0.10, +0.45], +0.23 |
| within 3 | −0.95 | +1.03 [+0.64, +1.44], +0.97 | +0.74 [+0.44, +1.05], +0.83 |
| full 1 | −3.44 | +1.28 [+0.81, +1.76], +1.05 | +0.59 [+0.20, +0.97], +0.68 |
| runs 1 | −2.04 | +0.82 [+0.18, +1.45], +0.49 | +0.40 [−0.00, +0.78], +0.46 |

- gC5 passes the bar on all four.
- **The fallback misses its +0.3 bar on within 2 (+0.17).** It is the first order where it fails.
- Mean of gC5 over all eight orders (recording-order labels): +1.30, against the §177(a) prediction of +1.2 (+0.8 to
  +1.6). Not refuted.
- The package-faithful (label L) versions of these four are on Leonardo (t18L within 2/3, full 1, runs 1).

**§177(b,c) and §182 RESULT, upper bounds (13 Nov 14:07 UTC; `runs/upper_*.log`; corrected orders, package gate, GE event rates in delivery
order).** Each comparison uses the same order, inputs and labels on both sides. gC5 combination over v5.1b, new:

| Order | labels | gC5 | cum10 delta1 | cum10 − gC5 |
|---|---|---|---|---|
| within 0 | package (L) | +1.70 | +1.66 | −0.04 |
| within 1 | package (L) | +1.39 | +1.68 | +0.29 |
| full 0 | package (L) | +0.93 | +0.97 | +0.04 |
| runs 0 | package (L) | +1.34 | +1.35 | +0.01 |
| within 2 | recording | +0.81 | +1.26 | +0.45 |
| within 3 | recording | +1.03 | +1.22 | +0.19 |

- **cum10 against gC5:** mean +0.16 over six orders, +0.08 over the four with package labels. Prediction +0.15 (0 to +0.35):
  confirmed. That is ≤ +0.3, so "undecided", and gC5 stays (it fits the run time; cum10 delta1 needs up to 53 min).
  A form between the two, fitting 50 min, is not worth building on this evidence. cum10 helps most on the weakest
  orders (within 2 +0.45, within 1 +0.29), which is worth a look after the upload if more time is found.
- **18 against 9 eye-net seeds** (gC5, package labels): within 0 +1.63 against +1.70, within 1 +1.23 against +1.39, so
  −0.07 and −0.16. No gain; prediction (+0.05, kept only if > +0.3) confirmed. Not kept.

### 185. ORDER SPREAD: twelve more delivery orders, package-faithful (written 13 Nov 14:23 UTC, before any job ran)

The coordinator (14:17 UTC): the spread between delivery orders (+0.81 to +1.70 under within) is now the largest
uncertainty in tonight's number.
- **Order file:** `runs/shuffle_perms.npz` is extended to seeds 0–11 per scheme. The 30 old keys are unchanged (checked);
  the old file is kept as `runs/shuffle_perms_5seeds.npz`. The new SHA-256 is pinned in shufcombo (6046bf5b…). The file is
  identical on the Mac, Leonardo and Galvani (md5 590a19e5…). GE event rates for all 36 orders: `runs/geshuf.npz`.
- **New orders:** within 4–11, full 2 and 3, runs 2 and 3.
  - Leonardo: within 4–9, every stage. Stage 1: ttt18 (base and gC5), rest net, eye net, pc128, core1 member, Hilbert,
    teacher. Stage 2: labels. Stage 3: ttt18 L.
  - Galvani: GPU stage 1 for within 10 and 11, full 2 and 3, runs 2 and 3. Their pc128, core1, Hilbert and teacher run on
    Leonardo. Labels and the L runs follow once the base files are relayed.
- **Scored** package-faithful (labels L, package gate, every input in delivery order except the ea slots), gC5 and fallback.
- **Prediction:** gC5's mean over all within orders is +1.2 (+0.9 to +1.5), with ≥ 80 % of within orders ≥ +0.5. The
  fallback misses +0.3 on 10–30 % of within orders.
- **Refuted** if gC5's within mean is < +0.8 or more than 20 % of within orders fall below +0.5. Then the table says "not
  robust over orders".

**PACKAGE-FAITHFUL, all eight orders so far (13 Nov 14:26 UTC; labels L, package gate, GE event rates in delivery order;
Hilbert, teacher and the ea slots as recorded; `runs/allorder_<order>_gp0_L_ge.log`).** Over v5.1b under the same order,
new (without P10):

| Scheme | Order | gC5 | fallback | FT's own part (gC5 − fallback) |
|---|---|---|---|---|
| within | 0 | +1.70 (+1.67) | +1.03 | +0.67 |
| within | 1 | +1.39 (+1.30) | +0.91 | +0.48 |
| within | 2 | **+0.41** (+0.38) | **+0.18** | +0.23 |
| within | 3 | +0.99 (+0.99) | +0.73 | +0.26 |
| full | 0 | +0.93 (+0.85) | +0.59 | +0.34 |
| full | 1 | +0.99 (+0.82) | +0.59 | +0.40 |
| runs | 0 | +1.34 (+1.30) | +0.73 | +0.61 |
| runs | 1 | **+0.42** (+0.09) | +0.40 | +0.02 |

Per scheme, mean [range]:
- within: gC5 +1.12 [+0.41, +1.70], without P10 +1.09; fallback +0.71 [+0.18, +1.03].
- full: gC5 +0.96 [+0.93, +0.99]; fallback +0.59.
- runs: gC5 +0.88 [+0.42, +1.34]; fallback +0.57 [+0.40, +0.73].
**Two of eight orders miss gC5's bar** (≥ +0.5 new): within 2 and runs 1. Runs 1 is also +0.09 without P10. The fallback
misses its bar on within 2. Every order is positive for both forms. The order is now the largest source of spread
(within: 1.3 points), and §185's twelve further orders measure it.

**PACKAGE-FAITHFUL + TEACHER IN DELIVERY ORDER, eight orders (13 Nov 14:31 UTC; `runs/allorder_<order>_gp0_L_ge_ts.log`).** Still in
recording order: the Hilbert member (1 of 15; re-stream running) and the ea transformer and shallow slots (1.5 of 15).
Over v5.1b under the same order, new (without P10):

| Scheme | Order | v5.1b re-streamed vs recording | gC5 | fallback |
|---|---|---|---|---|
| within | 0 | −1.06 | +1.47 (+1.47) | +0.96 |
| within | 1 | −1.38 | +1.37 (+1.29) | +0.71 |
| within | 2 | −0.97 | **+0.42** (+0.35) | **+0.14** |
| within | 3 | −0.89 | +1.11 (+1.14) | +0.93 |
| full | 0 | −3.20 | +1.18 (+1.17) | +0.42 |
| full | 1 | −3.22 | +0.89 (+0.88) | +0.34 |
| runs | 0 | −1.31 | +1.20 (+1.24) | +0.45 |
| runs | 1 | −2.21 | +0.82 (+0.53) | +0.30 |

Per scheme, mean [range]:
- gC5: within +1.09 [+0.42, +1.47]; full +1.04 [+0.89, +1.18]; runs +1.01 [+0.82, +1.20].
- Fallback: within +0.69 [+0.14, +0.96]; full +0.38; runs +0.38.

The teacher in delivery order raises runs 1 (+0.42 → +0.82) and full 0 (+0.93 → +1.18), lowers within 0 (+1.70 → +1.47), and
cuts the fallback on full and runs by about 0.2–0.3. **gC5 now misses its bar only on within 2** (+0.42). The fallback misses
on within 2 (+0.14) and sits at its bar on runs 1 (+0.30). FT's own part over the fallback is +0.28 to +0.85; under full and
runs it is larger than under within.

**MOST COMPLETE FORM, eight orders (13 Nov 15:58 UTC; `runs/allorder_<order>_gp0_L_ge_ts_hb.log`).** Every input in delivery order except the
ea transformer and shallow slots (1.5 of 15). The Hilbert re-stream reproduces the stored member (correlation 1.0000).
Package labels, package gate. Over v5.1b under the same order, new (without P10):

| Scheme | Order | v5.1b re-streamed vs recording | gC5 | fallback |
|---|---|---|---|---|
| within | 0 | −1.08 | +1.39 (+1.37) | +0.76 |
| within | 1 | −1.51 | +1.49 (+1.46) | +0.84 |
| within | 2 | −1.29 | **+0.32** [−0.03, +0.65] (+0.22) | **−0.05** |
| within | 3 | −1.04 | +1.02 (+1.05) | +0.69 |
| full | 0 | −3.30 | +1.27 (+1.18) | +0.47 |
| full | 1 | −3.02 | +0.62 (+0.56) | **+0.21** |
| runs | 0 | −1.38 | +1.35 (+1.37) | +0.70 |
| runs | 1 | −2.28 | +0.91 (+0.63) | +0.43 |

Per scheme, mean [range]:
- gC5: within +1.06 [+0.32, +1.49]; full +0.95 [+0.62, +1.27]; runs +1.13 [+0.91, +1.35].
- Fallback: within +0.56 [−0.05, +0.84]; full +0.34; runs +0.57.

gC5 misses its bar on within 2 only, where its interval reaches −0.03. The fallback misses on within 2 (where it is
negative) and on full 1. FT's own part over the fallback is +0.37 to +0.80. This form scores the twelve §185 orders.

**§183 follow-up: the new path against the routed path for returning people (13 Nov 16:38 UTC; `runs/shufpkg.npz` within 0/1 against
`runs/shufcombo_seen_session_within{0,1}_*.npz`; same orders, which the fixed scheme index reproduces).** Mean over six bases, P04 and P10:
- within 0: routed v5.1b 71.4 / 91.3; new path (gC5) 65.0 / 90.4; new path − routed **−6.4 / −0.9 (mean −3.63)**.
- within 1: routed 73.5 / 91.5; new path 67.5 / 90.3; **−5.9 / −1.2 (mean −3.55)**.
- Recording order, routed: 73.8 / 91.1. Routed under shuffle loses 2.4 / −0.2 (within 0) and 0.3 / −0.4 (within 1).
Routing returning people through v6 would cost about 3.6. **Closed.**
Limit: shufpkg keeps members as recorded, while the new path has them re-streamed. That costs the new path at most
the ~0.5 v5.1b loses from re-streaming, far below 3.6.

### 186. RANKED EXPLORE DIRECTIONS by measured or bounded ceiling on the sealed mix (written 13 Nov 16:38 UTC)

The sealed mix is about 57 % new and 43 % returning sessions. A new-people ceiling counts at about 0.57 of its size
before transfer (transfer 0.4 to 1.0). Third place by team is +1.1 sealed above us (coordinator, 16:40).

| Rank | Direction | Measured ceiling | Sealed bound | Status |
|---|---|---|---|---|
| 1 | Eye-view quality (oracle eye view, §155) | +3.8 new over V6F (recording order) | ≤ +2.2 | Five view forms refuted (§156–§163); still the largest measured ceiling, so it needs other forms (learned eye features, more data). |
| 2 | FT label quality (true-label FT, §142) | +1.9 against chain labels; package labels cost 0.2–0.5 more | ≤ +0.75 | Refuted: stack/indep/agree/confhalf labels (§145). Untried: labels from the v6 chain itself in the package. |
| 3 | Stage loss under full and runs (§181) | non-causal +1.4 (full), +0.9 (runs) | ≤ +0.8, and ≈ 0 if the stream is within-like | Only a causal approximation is possible. Undecided. |
| 4 | v6 additions on top of the routed path for returning people | unmeasured; the additions gave +1.0 to +1.25 over the unrouted tail (§183) | ≤ +0.5 | Untested. The routed path keeps personal heads; whether the eye/rest/FT additions add on top is open. |
| 5 | More training people (PCURVE, §146) | +4.7 per doubling (chain) | large, but needs data | The teacher route mostly refuted (§148–§150); external-corpus transfer is a longer project. |
| 6 | cum10 instead of gC5 | +0.16 mean, +0.45 on the weakest order | ≤ +0.1 | Undecided; does not fit the run time as is. |
| — | Returning people through the v6 path | −3.6 against routed | — | Closed (above). |
| — | Order-free stages under within | +0.10 | — | Closed (§181). |

### 187. FT LABELS FROM THE V6 CHAIN (§186 item 2; written 13 Nov 16:40 UTC, before any FT job ran)

The package fine-tunes on its inner v5.1b's outputs. Here the labels instead come from the v6 chain without FT (V6r: eye
view and rest member in, package gate), as emitted in the delivery order (`shufcombo LABEL_OUT LABEL_ARM=V6r` →
`chain_v6r_<order>.npz`; ttt18 tag suffix R).
Label accuracy, new people, against v5.1b's labels (L):
- within 0: 0.7642 against 0.7552; within 1: 0.7646 against 0.7589;
- runs 0: 0.7653 against 0.7533;
- within 2/3: 0.7537 / 0.7605; full 0/1: 0.7379 / 0.7405; runs 1: 0.7461.
- Recording-order chain_final: 0.7657.
Queued on Galvani: gC5 with R labels on the eight scored orders. Scored in the most complete form against the same run
with L labels.
Prediction: +0.10 (0 to +0.3) over L. True-label FT gains about +1.0 per 23 points of label accuracy (§142), and R gains
0.4 to 1.2 points. Kept as a package change only if > +0.3 on the mean over eight orders and ≥ 0 on each within order.
It also needs no run-time change (the v6 chain is already computed when FT runs; the build session must confirm).

### 188. EYE VIEW, three new forms (§186 item 1; written 13 Nov 16:42 UTC, before any result)

The oracle eye view (§155) is the largest measured ceiling (+3.8 new over V6F; ≤ +2.2 sealed). The five refuted forms
(§156–§163) changed which features feed SessionEM's view. These differ in kind; each gets its ceiling test first.
- **A. The eye view in TSEM as well as SessionEM** (`shufcombo OCU_TS=k`: k copies of the eye log-posteriors appended to
  pc128; copies, not scaling, because TSEM standardises columns). Ceiling: `OCU_ORACLE=1` (eye view = smoothed true label)
  with k = 0 against k = 4. Form: k = 1, 4, 16 with the real eye net. Most complete inputs, labels L, within 0/1.
  Prediction: ceiling +1 or more over SessionEM-only; the real form +0.1 (0 to +0.3). Kept if ≥ +0.3 on both orders.
- **B. The eye net fine-tuned per session on the delivered windows** (TTT of the eye net with chain pseudo-labels,
  like gC5 for the SessionBN nets). Ceiling: the same with true labels. Needs an ocunet TTT script (GPU).
  Smoke tonight, wave overnight.
- **C. Learned eye-trace features pretrained on public corpora** (self-supervised encoder on frontal-channel traces of
  the permitted corpora). Ceiling: a LOSO linear probe on the embedding with true labels against the eye net's own
  probe. Corpora with frontal channels must be checked first; this is a longer project.
Order: A tonight on CPU, B as the overnight GPU explore wave, C after B's ceiling.

**§188 A, first numbers and what they mean (13 Nov 16:46 UTC; `runs/formA_*.log`).**
- **What the oracle contains:** `OCU_ORACLE` replaces the eye view with 0.6·one-hot(true label) + 0.4/3, a perfect and
  confident view. Its gains are the ceiling of **any** perfect extra view, not of eye signals. They mostly say how much
  SessionEM and TSEM can use such a view.
- **Oracle results:** within 0, gC5 over v5.1b +6.05 in SessionEM only, +12.09 with 4 copies in TSEM too.
- **Real eye net:** 1 copy in TSEM gives +1.43 against +1.39 without.

**Accuracy curve (coordinator, about 16:50; pre-registered here, before results):** `OCU_SYN=d` shuffles the eye net's
posteriors within person × true class. That keeps its window accuracy and confusion but removes any correlation with
the EEG members' errors (and with time). It then raises accuracy by d points, swapping the top class with the true
class on random wrong windows. d = 0, 5, 10, 20; k = 0 (SessionEM only) and 4 (TSEM too); within 0/1; most complete
inputs; labels L.
**Reading:**
- If the synthetic view at d = 0 gains and the real eye net does not, the real eye view's limit is correlation with the
  EEG errors, and forms B/C are judged on error decorrelation.
- The slope over d says how accurate an eye-derived view must be for +2 new.
- Prediction: d = 0 synthetic +0.2 to +0.6 over the real eye net; slope about +0.1 per accuracy point in SessionEM only,
  higher with TSEM.

**§185 RESULT: all twenty orders in the most complete form (13 Nov 16:47 UTC; `runs/allorder_<order>_gp0_L_ge_ts_hb.log`).** Every input in
delivery order except the ea transformer and shallow slots (1.5 of 15); package labels L; package gate; FT draw 0.
New people, over v5.1b under the same order:

| Scheme | Orders | gC5 mean [range], median | without P10 | gC5 < +0.5 | fallback mean [range] | fallback < +0.3 | FT's part | v5.1b's loss |
|---|---|---|---|---|---|---|---|---|
| within | 12 | +1.24 [+0.32, +1.86], +1.35 | +1.18 | 1 (within 2) | +0.69 [−0.05, +1.02] | 2 (within 2, 4) | +0.55 | −1.46 |
| full | 4 | +0.87 [+0.62, +1.27], +0.79 | +0.81 | 0 | +0.42 [+0.21, +0.67] | 1 (full 1) | +0.45 | −3.82 |
| runs | 4 | +1.00 [+0.78, +1.35], +0.94 | +0.88 | 0 | +0.42 [+0.20, +0.70] | 1 (runs 3) | +0.58 | −1.88 |
| all | 20 | +1.12 [+0.32, +1.86] | +1.05 | 1 | +0.58 | 4 | | |

Within orders 4–11, gC5 (fallback): +0.91 (+0.26), +1.36 (+1.02), +1.72 (+0.99), +1.76 (+0.69), +1.86 (+1.00), +1.08 (+0.75),
+0.63 (+0.67), +1.33 (+0.62). Full 2/3: +0.89, +0.70. Runs 2/3: +0.78, +0.97.
- **Prediction:** within mean +1.2 (+0.9 to +1.5) with ≥ 80 % of within orders ≥ +0.5. Result: +1.24, 11 of 12.
  **Not refuted.**
- The fallback misses on 2 of 12 within orders (predicted 10–30 %).
- FT's own part is positive on 19 of 20 orders. The exception is within 10 (−0.04).
- gC5 is above v5.1b on all 20 orders. Within 2 is the only one whose 80 % interval reaches below 0.

**§186 item 4 RESULT: v6 additions on the routed path for returning people (`studies/13nov/retv6.py`, an agent; `runs/retv6.log`).
Scored before a prediction was written** (the coordinator asked to start at once; stated here). The routed v5.1b arm
reproduces `runs/shufpkg.npz` to 1e-9 on all six bases. Change against routed v5.1b under the same order, mean (P04, P10):
- Recording order: eye +0.27, rest +0.18, FT −0.41, all three +0.49 (P04 +0.37, P10 +0.61).
- Within mean: eye +0.14, rest +0.16, FT +0.52, all three **+0.61 (P04 +1.24, P10 −0.03)**.
- The routed set never changes. Only P04|2, which routes late through the v3 gate, moves its first routed window.
- **Reading:** under shuffle the gain sits on P04 and comes mostly from FT. P10 is unchanged (inside ±0.3). The package's
  ship form switches FT off on routed sessions, so as built the package gives returning people 0.
- Turning eye + rest on for routed sessions would be +0.2 to +0.3 on P04 and about 0 on P10. Two people, four sessions:
  not an estimate. **"Undecided"; not for this upload.**

**§188 form A RESULT (13 Nov 16:52 UTC; `runs/formA_*.log`; most complete inputs, labels L).** The real eye net appended to TSEM in
k copies. gC5 over v5.1b, new; k = 0 is the baseline:

| Order | k = 0 | k = 1 | k = 4 | k = 16 |
|---|---|---|---|---|
| within 0 | +1.39 | +1.43 | +1.21 | +0.96 |
| within 1 | +1.49 | +1.52 | +1.43 | +1.13 |

- The fallback falls with k (within 1: +0.88, +0.60, +0.09 for k = 1/4/16).
- **Form A is refuted:** +0.04 at best, worse with more copies.
- The perfect-view ceiling (oracle) is +6.05 in SessionEM and +12.1 with TSEM, so TSEM can use a perfect view. The real
  eye net is far from one: its window accuracy for new people (LOSO, within 0) is 0.514.

**§188 form B queued (13 Nov 16:55 UTC).** `studies/13nov/ocuttt.py`.
- **Smoke:** the second smoke checks that pseudo-label and true-label outputs differ (max 0.81) and that FT moves outputs
  (max 0.68); the job stops otherwise.
- **Jobs (Galvani):** 3098591–94 for within 0/1, full 0, runs 0, nine seeds each. 3098595–98 for within 2/3, full 1,
  runs 1, afterany. Pseudo-labels from the v6 chain (chain_v6r).
- **Scoring:** a local loop fetches from Galvani only (Leonardo's certificate may lapse tonight) and scores base / pl /
  true per order → `runs/formB_summary.txt`.
- **Prediction:** the true-label ceiling is +0.5 to +1.5 over the current eye net; pl is +0.1 (0 to +0.3).
- **Kept** if pl ≥ +0.3 on the mean over orders and ≥ 0 on each within order. If the ceiling stays below +0.5, form B is
  closed with it.

### 189. MORE TRAINING PEOPLE (§186 item 5, the largest measured ceiling; written 13 Nov 16:58 UTC, before any result)

The accuracy curve makes the eye view a small lever in SessionEM (§188: +0.6 for 20 accuracy points). The largest
measured number left is the people slope: +4.7 per doubling at chain level (§146; the stages keep 0.95 of the core
slope). The teacher results (§148–§150) refuted the external corpora as a teacher, not as extra training people.
Three parts:
- **(a) Survey** (an agent, read-only): which permitted corpora have PROTEUS's three classes or a mappable subset with a
  compatible montage, and how many people each adds. Said plainly if empty.
- **(b) The people slope under shuffled delivery and with the eye view** (`peoplecurve.py` with SHUF and OCU_DIR; core1
  surrogate as in §146; five draws): recording order (regression check against §146), recording + eye, within 0 + eye,
  within 1 + eye, within 0. Prediction: the chain slope stays ≥ 0.7 × the core slope under within (+3.5 to +5 per
  doubling), and the eye view does not flatten it. Refuted if the within chain slope is < 0.4 × core.
- **(c) Pooled training** of the SessionBN families and the eye and rest nets with the corpus people as extra training
  people, scored through the chain under within shuffle. Designed after (a). Its size is bounded by (b): N extra
  people are worth about 4.7 × log2((9 + N·q)/9), where q is the corpus people's worth relative to a PROTEUS person.

**§189 (b) RESULT (13 Nov 17:00 UTC; `runs/pcurve2_*.log`; core1 surrogate as §146, five draws).** The recording-order run reproduces §146
exactly (chain +4.69 per doubling). Chain slope per doubling of training people (k 3–9; core +4.92):
- recording + eye view: +4.60;
- within 0: +5.03; within 0 + eye: +4.90; within 1 + eye: +4.60.
- Ratio to core: 0.93–1.02.
**Prediction (≥ 0.7 × core under within) confirmed.** Neither shuffled delivery nor the eye view flattens the people gain.
A doubling of effective training people is still worth about +4.7 at chain level, and one more PROTEUS-like person
about +0.7. (The surrogate chain at k = 9 loses nothing under within shuffle, 70.0 against 69.3. The stage loss
measured elsewhere comes with the full member set.) Lever #5 keeps its size; (a) decides whether data exist for it.

### 190. FT ON THE ROUTED PATH FOR RETURNING PEOPLE, package-style labels (§186 item 4 follow-up; written 13 Nov 17:00 UTC, before the FT jobs ran)

The agent's +0.61 (§186.4) fine-tuned returning rows on chain_final's seen part (recording order). Package-style:
returning rows labelled by routed v5.1b as emitted in the delivery order (`retv6.py LABEL_RT` →
`runs/labels/chain_rt_<order>.npz`; new-people rows as chain_v51b). Label accuracy on returning rows: within 0 0.815,
within 1 0.827, against chain_final 0.799. gC5 with these labels (ttt18 tag T): Leonardo 64404681/83 and Galvani
3098657/58 (first copy wins). Scored with `retv6.py FT_SFX=T` (arms F and ERF) against routed v5.1b.
- **Run time:** FT on every routed session means FT on 100 % of sessions. That is v52-build's worst case (44.2 min sealed,
  margin 15.8, gc5 at 378 windows), not the 57 % case.
- **Prediction:** ERF within mean +0.6 (+0.3 to +0.9), P10 0 (±0.3).
- **Kept for a v6.1 build** if ERF ≥ +0.3 on the mean of within 0/1 and P10 ≥ −0.3 on both.

**§189 (a) RESULT: the corpus survey (13 Nov 17:01 UTC; an agent, read-only).** PROTEUS classes: 0 kinesthetic motor imagery of the
non-dominant hand, 1 mental subtraction, 2 word generation. 41 channels, with no frontal-polar, occipital or T7/T8.
- **All three classes:** only Scherer 2015 (BNCI 004-2015): 9 stroke or spinal-cord-injury patients × 2 days, 2,130
  windows; 19 of 41 channels real; CC BY-NC-ND, cleared by the organisers (11 Nov). Exported at
  `data/scherer2015/windows.npz` (Mac, Galvani).
- **Two classes** (motor imagery, subtraction; no word generation): Shin 2017 A+B (29 people; class equals recording
  block) and ds007554 CMx7-MM (24 exported; tasks in separate recordings).
- **One class:** Shin 2018WG (26, word), Zyma 2019 (36, subtraction), and a few hundred motor-imagery-only people.
- **Earlier pooling as extra training people:**
  - Scherer in the SessionBN FBCNet and shallow nets (§7): members +1 to +3 alone, chain −0.23 nested.
  - Scherer in core1's head (§21): −0.9 to −1.1.
  - Scherer on its 19 real channels (§53): −3.2 to −3.7.
  - Shin 2017 A+B (R13): every family lost.
  - Dreyer: collapsed.
  - The eye and rest nets never saw external data.
- **Reading:** (a) is close to empty. 9 three-class people (patients) plus 53 two-class people with a block confound. The
  worth per person q measured ≤ 0 each time it was tried. Still, §7's −0.23 in the chain is inside the noise (±0.3),
  so by the rules it is "undecided", not closed. It was measured in recording order, on an older chain, without FT,
  the eye view or the rest member.
- **(c) form, if run:** Scherer pooled into the SessionBN families (run_cv `@scherer`), scored through today's chain
  under within shuffle against the same families without Scherer. The ceiling from (b): 9 people at q = 1 would be
  +4.7 × log2(18/9) = +4.7. At q = 0.2 it is +0.9. The expected value is low, given three negative measurements.

### 191. SCHERER POOLED INTO THE SESSIONBN FAMILIES, retested on today's chain (§189 c; written 13 Nov 17:04 UTC, before any job ran)

The coordinator asked for this on Leonardo's idle GPUs, behind §187 and §190 in priority. The user's line from this
morning applies: "throw even existing or already tried methods again on top and see whether they work now". §7's −0.23
was inside the noise, in recording order, on an older chain.
- **Pooled arm (X):** `ttt18.py EXT=scherer2015~m`. The four SessionBN families (fbraw, fb, sinc, fbl; six seeds each)
  are retrained per fold with Scherer's 9 people (2,130 windows; `~m` flags their missing band below 4 Hz) as extra
  training people. They are re-streamed in the delivery order and fine-tuned (gC5, labels L).
- **Control (C):** the same run without Scherer (`CTRL=1`), retrained in the same wave (paired, warm).
- **Orders:** within 0 and 1 (most complete inputs, package gate) and recording order (`SCHEME=rec`).
- **Scoring:** chained on Leonardo (shufcombo per arm, then `poolcmp.py` X − C), so no fetch is needed overnight.
- **Bar, in the q framing:** 9 people at worth q are worth about 4.7 × log2(1 + q).
  - Act (a v6.x candidate) if the combination's X − C is ≥ +0.5 on the mean of within 0 and 1 (q ≈ 0.08 or more), each
    within order ≥ +0.2, and recording order ≥ 0.
  - Between −0.3 and +0.3: "undecided" again.
  - Below −0.3 on both within orders: refuted on today's chain.
- **Prediction:** −0.1 (−0.6 to +0.4), from §7 and §21. The members gain alone (§7: +1 to +3), and the chain gives most of
  it back.
- **Not run:** the two-class corpora with a masked loss. The SessionBN training has no per-sample class mask, and
  building one is not overnight work.

**§190 RESULT (13 Nov 17:15 UTC; `runs/retv6_T.log`; FT with package-style labels on returning rows: routed v5.1b as emitted).**
Change against routed v5.1b, six-base mean (P04, P10):

| Arm | within 0 | within 1 | shuffled mean |
|---|---|---|---|
| eye | +0.21 | +0.07 | +0.14 (+0.38, −0.10) |
| rest | +0.28 | +0.04 | +0.16 (+0.37, −0.06) |
| FT | +0.69 | −0.05 | +0.32 (+0.86, −0.22) |
| all three (ERF) | +0.36 | +0.08 | **+0.22 (+0.54, −0.09)** |

**The bar (ERF ≥ +0.3 on the within mean, P10 ≥ −0.3) is missed:** +0.22. P10 passes (−0.07, −0.12). The §186.4 figure of
+0.61 came from recording-order labels; with the labels the package would have, it is about a third of that, carried by
P04, and within 1 is flat. Recorded as "undecided" (inside the noise). No v6.1 build is requested on this. FT alone is
+0.32 but goes negative on P10 (−0.22) and on within 1 (−0.05).
Supporting fix found on the way (§191): `protocol.load`'s `~m` (nolow) option is not read by deep.py or session_bn.py.
It is a no-op for every SessionBN run; ttt18's EXT now refuses `~` options and states why it clears the weighting tag.

**§191 guard and audit (13 Nov 17:18 UTC; coordinator's request).**
- **(a) Guard:** `protocol.require_honoured(d, idx, honours, who)` fails when the training rows carry an option the model
  does not apply. Every fit calls it:
  - deep.AlignedNet (and so SessionNet): honours `~w` only;
  - models.TangentSpace: `~w` and `~m`;
  - MeanLinear and ReveLinear: neither.
  - `~n` is applied inside `protocol.load` itself.
  - Checked: `scherer2015~m` is refused for a deep-like fit and accepted for TangentSpace; `replay/check.py` prints OK.
- **(b) Which results passed an option the model did not apply:**
  - Round 5 (19 Oct, `docs/archive/OPTIMIZATION_R5.md`): `~m` was used only for v2 (models.TangentSpace), which applies
    it; the deep members ran `~n~w0.1`. Stands.
  - **§7 (SCH, 11 Nov):** `scherer2015~n~w0.1 / ~n~w0.25 / ~n~w0.25` with HAND at weight 0, trained with a scratch copy of
    session_bn.py that supported weighting (`runs/archive_scratch_11nov/schr/galv/`). No `~m`. Stands as a test of
    what it ran.
  - **§191 as first written:** `scherer2015~m` was meant to flag Scherer's missing band below 4 Hz in the SessionBN nets,
    and deep/session_bn never read it. Caught by the smoke before any training. The run uses `scherer2015` at weight 1
    and says so.
  - Nothing else found with a `~` option passed to a deep net.
- **(c) Shipped packages:** no member family in the uploaded or staged packages was trained with external rows through
  `protocol.load`. The SessionBN, TF and shallow members are PROTEUS-only; the teacher was pretrained on external
  corpora through `experiments/pretrain.py`, not through these options. No description error for the audit.
- **Note on the §191 arm:** §7 used Scherer at loss weight 0.1/0.25; §191 uses weight 1 (q-test, class-balanced). The
  smoke passed (2,130 windows, 9 people); arrays and scoring are released.

**§187 RESULT (13 Nov 17:23 UTC; `runs/allorder_<order>_R.log`; most complete form, package gate).** gC5 fine-tuned on labels from the v6
chain (R) against package labels (L), over v5.1b, new:
- within 0: +1.51 against +1.39 (+0.12); within 1: +1.67 against +1.49 (+0.18);
- within 2: +0.51 against +0.32 (+0.19); within 3: +1.12 against +1.02 (+0.10);
- full 0: +0.97 against +1.27 (−0.30); full 1: +0.51 against +0.62 (−0.11);
- runs 0: +1.32 against +1.35 (−0.03); runs 1: +0.93 against +0.91 (+0.02).
- Mean over eight orders: **+0.02**; within mean +0.15 (all four ≥ 0); full −0.21.
- **The keep rule (> +0.3 on the eight-order mean) is missed:** "undecided", not a package change. Prediction +0.10 (0 to +0.3):
  inside the range.
- Worth noting: under within (the scheme that matches the sealed drop), R lifts every order, including the weak within 2
  (+0.32 → +0.51, above the +0.5 bar). Under full it costs. With a within-like sealed stream, R is a small positive
  (+0.15); a second draw per within order would decide it.

### 192. §187 SECOND DRAW under within (written 13 Nov 17:24 UTC, before any job ran; coordinator's request)

gC5 draw 1 with R labels (v6 chain) and with L labels (package) on within 0–3, at low priority behind §191 and form B.
Draw 1 with L exists for within 0/1 (§179). New: R d1 for within 0–3 (Galvani), L d1 for within 2/3 (Leonardo).
**Reading, within mean only:** R − L over the four within orders, draw 1, and pooled with draw 0 (+0.15).
- At +0.15 new, it is worth about +0.05 sealed: a tie-breaker for a later build, not a reason for one.
- It counts as "R preferred" if the pooled within mean is ≥ +0.10 with draws 0 and 1 the same sign.
- Otherwise it stays undecided. Full and runs are not re-read.

**§191 RESULT (13 Nov 17:49 UTC; Leonardo `runs/night13/poolcmp_{within0,within1,rec0}.txt`; six bases; Scherer at weight 1, retrained
control in the same wave).** Pooled − control, new [80 %], without P10:

| Order | members only (no FT, eye, rest) | combination (gC5 + eye + rest) | fallback (eye + rest) |
|---|---|---|---|
| within 0 | +0.16 [−0.79, +1.01], +0.03 | −0.23 [−0.64, +0.17], −0.34 | +0.08, −0.00 |
| within 1 | +0.02 [−0.60, +0.63], −0.01 | −0.51 [−1.02, −0.04], −0.62 | −0.31, −0.38 |
| recording | −0.42 [−1.15, +0.22], −0.53 | −0.49 [−0.89, −0.08], −0.51 | −0.54, −0.52 |

- Per person, the combination's loss is carried by P05 (−2.0 to −3.6) and P01 (−1.7 to −1.8).
- **The act bar (≥ +0.5 within mean) is clearly missed: the within mean is −0.37.**
- By the letter of the pre-registered refutation rule (below −0.3 on both within orders) it is not refuted, since within 0
  is −0.23. But all three orders are negative for the combination, and two of their 80 % intervals exclude 0.
- **Reading:** the pooled members are about neutral on their own under within (+0.16, +0.02). In the chain they cost about
  0.4 to 0.5, as in §7 (−0.23) and §21 (−0.9). Prediction −0.1 (−0.6 to +0.4): inside.
- No further Scherer-pooling forms are planned. The remaining door would be a lower loss weight (§7 used 0.1/0.25), which
  is §7's own result, already undecided at −0.23.

### 193. PER-SESSION SELECTOR between v5.1b's tail, the fallback and gC5 (coordinator's suggestion; written 13 Nov 17:53 UTC, before any result)

The order spread (within +0.32 to +1.86) is the largest avoidable loss left. `studies/13nov/selector.py` reads per-window
outputs (`shufcombo SAVE_OUT`, all 20 orders, most complete form; each re-run checked against its logged number).
- **Ceiling first:** per base and new-person session, the arm (P0r, V6r, V6Fr) with the best session balanced accuracy,
  against always-gC5. **Stop if the within ceiling is < +0.3.**
- **Otherwise, label-free causal criteria** read on a session's first N = 30 or 60 delivered windows (gC5 is emitted for
  them): agreement between P0r and V6Fr, and the mean margin difference V6Fr − P0r. Threshold and alternative arm are
  chosen on five people and scored on the other five (all 126 splits).
- **Keep bar:** held-out gain ≥ +0.3 over always-gC5 under within, with the 10th percentile over splits ≥ 0.
- Also: per person on within 2, eye + rest (V6r − P0r) and FT (V6Fr − V6r).
- **Prediction:** ceiling +0.8 (+0.4 to +1.5; any oracle over three arms gains from noise alone); held-out criteria +0.05
  (−0.1 to +0.2).

### 194. FINE-TUNING FORM SWEEP under package conditions (coordinator's suggestion; written 13 Nov 17:53 UTC, before any job ran)

The true-label ceiling of FT is +1.9 (§142); the package form (gC5) sits far under it. Labels were varied (§145, §187), but
the form was not, under package conditions.
- **Sweep:** `ttt18 SWEEP=1` (tag LS), 16 run-time-neutral forms side by side in one run. All keep 5 steps per refit,
  every 10 windows, warm start, one view. gC5 itself is the in-run control.
  - learning rate: 2.5e-4, 1e-3, 2e-3;
  - head-only (SessionBN affine and the output layer) at 5e-4 and 2e-3;
  - consistency weight: 0, 0.5, 2, 4;
  - soft labels at 5e-4 and 1e-3;
  - start window: 20, 40;
  - two combinations.
  - Not in the sweep: an EMA of the weights between refits (not implemented in ttt7; it needs code).
- **Wave 1:** within 0–3, package labels (L), Leonardo. **Wave 2 (afterany):** within 4–7.
- **Scoring and nested selection** run on Leonardo: shufcombo per form, most complete form; then `sweepsel.py`, which picks
  a form on five people and scores it on five (126 splits).
- **Keep bar:** held-out gain ≥ +0.3 over gC5 on wave 1, confirmed (≥ +0.15) on wave 2's orders.
- **Prediction:** held-out +0.1 (−0.1 to +0.3). The best single form looks better in-sample (+0.3 to +0.5), which is why
  only the nested number counts.

### 195. BUG: within 1 read within 10's pc128 view after 15:02 UTC (found 13 Nov 17:56 UTC; fixed)

`shufcombo` found TSEM's delivery-order pc128 file with `glob("coreshuf_p{PSEED}*.npz")`. For PSEED = 1 that also matched
`coreshuf_p10_within.npz` and `coreshuf_p11_within.npz`, which arrived at 15:02 UTC with §185. `p10` sorts first and
holds the key `within|all_loso`, so every within 1 score after 15:02 UTC used within 10's pc128 view. Other seeds were not
affected (no p0x, p2x or p3x files; full and runs keys are not in the p10/p11 files).
- **Found by:** the per-window re-run for §193 did not reproduce the logged within 1 number (+1.53 against +1.49).
- **Fix:** exact patterns (`coreshuf_p<seed>.npz` and `coreshuf_p<seed>_*.npz`), plus a check that exactly one file
  holds the key, for pc128, core1, Hilbert and teacher alike (`one_with`). The fixed scorer reproduces the logged within 1
  line exactly (+1.49, fallback +0.84). Synced to Leonardo before the §194 scoring runs.
- **Unaffected (scored before 15:02 UTC):** the most complete table (§185), the package gate (§178), labels L, §176, §180,
  §183, the upper bounds.
- **Affected and re-run now:**
  - form A within 1 and the accuracy curve's within 1 half (§188);
  - §187 R within 1;
  - the §193 per-window outputs;
  - §191 within 1, rescored on Leonardo (64408767). Both arms there had the same wrong view, so its pooled-minus-control
    difference was paired but not on the right input.
  - Old logs are overwritten in place; this entry is the record.

**§195 re-runs, within 1 with the right pc128 (13 Nov 18:14 UTC).**
- Form A: k = 1/4/16 give +1.37/+1.20/+1.02 against the baseline +1.49 (−0.12/−0.29/−0.47). Still refuted.
- §187 R within 1: +1.71 against L +1.49 (+0.22, was +0.18). §187's within mean is now +0.16; the eight-order mean is +0.03.
  Undecided, unchanged.

**§193 RESULT, per-session selector (`runs/selector.log`; 20 orders, 6 bases).**
- **Oracle** (the best of P0r / V6r / V6Fr per session) over always-gC5: within +0.41, full +0.49, runs +0.40.
  Always-fallback: full −0.45, runs −0.57.
- **Label-free causal criteria**, chosen on five people and scored on five (126 splits): agreement on the first 30/60
  windows −0.02 / −0.06; margin difference −0.04 / −0.02. **Refuted:** no criterion recovers any of the +0.41 ceiling.
  The ceiling is within the size an oracle over three noisy arms finds by chance (prediction +0.4 to +1.5).
- **Within 2, who and what** (per person P01..P10):
  - eye + rest: −0.8 +0.1 −0.3 −0.1 −0.0 −0.2 +1.0 +0.5 −0.7 +0.1;
  - FT: +2.1 −0.2 −0.6 −0.9 +0.1 +1.3 −0.4 +0.7 +0.5 +1.1.
  - Its low total comes from the eye/rest part, which is small and mixed (P01 −0.8, P09 −0.7), and FT being negative for
    P03 and P04. No single part or person carries it.

**§188 accuracy curve RESULT (`runs/syn_*`; within 0 and 1, right pc128).** gC5 over v5.1b, new; the real eye net beside the
synthetic view at the same window accuracy (0.514):

| | real eye net | synthetic, same accuracy | +5 | +10 | +20 (0.71) |
|---|---|---|---|---|---|
| SessionEM only (k = 0), within 0 / 1 | +1.39 / +1.49 | +1.57 / +1.76 | +1.55 / +1.94 | +1.63 / +1.93 | +2.01 / +2.22 |
| SessionEM + TSEM (k = 4), within 0 / 1 | +1.21 / +1.20 | **+2.36 / +2.14** | +2.57 / +2.38 | +2.64 / +2.42 | +3.06 / +2.95 |

- **Reading:** at the same accuracy, the synthetic view (errors shuffled within person × class, so independent of the EEG
  members' errors and of time) beats the real eye net by +0.2 to +0.3 in SessionEM and **by about +1.0 in TSEM** (k = 4).
- Accuracy itself is a weak lever: +20 points adds about +0.5 to +0.7.
- **So the eye view's limit is error correlation, not accuracy.** Its errors are correlated over time within a session and
  with the EEG members. TSEM's templates average independent errors away, not correlated ones.
- **Consequence for forms B and C:** judge them on decorrelation. A session-drift-free eye view at today's accuracy would be
  worth up to about +1.0 new in TSEM. Form A failed because the real view's errors are correlated.
- **Caveat:** the shuffle also removes temporal structure that might be partly real signal. The +1.0 is an upper bound for
  "decorrelation", not a promise.

### 196. EYE-VIEW ERROR CORRELATION, split and removed (coordinator's D1/D2; written 13 Nov 18:15 UTC, before any result)

§188: at unchanged accuracy, decorrelated eye errors are worth about +1.0 new in TSEM. `shufcombo OCU_TX`:
- **D1, diagnostics (non-causal):**
  - shufrun: shuffle within person × class × run; keeps each run's bias, removes time structure inside runs.
  - demeanrun: subtract each run's mean eye log-posterior, add the session mean back.
  - demeansess: subtract each session's mean.
- **D2, causal and package-ready** (a few lines; running mean over the session's earlier delivered windows, subtracted
  before SessionEM/TSEM):
  - causal: whole prefix;
  - causal30: the last 30 windows.
- **Scope:** within 0–3, k = 0 and k = 4, most complete inputs, labels L; against the real eye net at the same k.
- **Reading:** if demeanrun or demeansess recovers most of the synthetic gain, the harmful part is a slow per-run or
  per-session bias, and D2 causal centring should take it. If only shufrun helps, it is within-run time structure. If none
  helps, it is correlation with the EEG members' errors, and D2-residualisation (eye on the member mix, map fitted on
  training people) is next.
- **Keep bar (D2):** ≥ +0.3 over the real eye net on the within 0–3 mean at the better k, ≥ 0 on each order. If it passes,
  it is a package-level change of a few lines.
- **Prediction:** demeansess / demeanrun +0.3 to +0.8 at k = 4; causal +0.2 to +0.6 at k = 4; about +0.1 at k = 0.
- **§196 note (13 Nov 18:20 UTC, after the first within 0 k = 0 block):** demeansess is **structurally a no-op**. SessionEM
  (`gm.EMg._adapt`) and TSEM (`tsem_lik`) both standardise each view column on the session's earlier windows, so
  subtracting a per-session constant cancels exactly. Observed: +1.39 / +0.76, identical to the real eye net. A per-session
  bias of the eye net is therefore already removed by the stages. demeansess is withdrawn as a diagnostic (its runs
  continue and are reported as the no-op check). The causal centring is not a constant per session and is not affected.
  First within 0 k = 0 numbers: real +1.39, causal +1.36, causal30 +1.30, demeanrun +1.32, shufrun **+1.54**. Only removing the
  time structure inside runs helps so far.

**§196 RESULT (13 Nov 19:19 UTC; `runs/s196_summary.txt`; within 0–3, most complete inputs, labels L).** Mean gC5 over v5.1b, new; change
against the real eye net at the same k:

| Transform | k = 0 | k = 4 |
|---|---|---|
| real eye net | +1.06 | +0.82 |
| D2 causal (prefix mean) | +1.07 (+0.02) | +0.85 (+0.03) |
| D2 causal30 | +1.10 (+0.05) | +0.94 (+0.12) |
| demeanrun (diagnostic) | +1.11 (+0.05) | +0.98 (+0.16) |
| shufrun (diagnostic) | **+1.28 (+0.22)** | **+1.49 (+0.67)** |
| demeansess (structural no-op) | +1.06 | +0.82 |

- **D2 fails its bar.** The best real setting is k = 0 (+1.06); causal centring adds at most +0.05 there, and k = 4 stays
  below k = 0 even with centring.
- **D1 reading:** removing per-run bias (demeanrun) is worth +0.05/+0.16. Removing the time structure inside runs
  (shufrun) is worth +0.22/+0.67. So the harmful part is the eye net's error structure over time within a run
  (autocorrelated errors), not a slow per-run or per-session bias (a per-session bias is removed by the stages anyway).
- No causal fix for within-run autocorrelation is in hand. Forms B and C are judged on whether they reduce it.

**§194 wave 1 RESULT (Leonardo `runs/night13/sweepsel_w1.txt`).**
- **Nested held-out gain over gC5: +0.11 [−0.11, +0.29]. Misses the +0.3 bar.**
- In-sample, per form: s03 (lr 2e-3) +0.29, s14 (lr 1e-3, consistency 2) +0.27, s15 (lr 1e-3, start 20) +0.24, s02 (lr 1e-3)
  +0.23, s11 (soft, lr 1e-3) +0.19.
- Losing forms: lower learning rate (s01 −0.17), head-only (s04 −0.36, s05 −0.18), consistency 4 (s09 −0.18).
- Selection chose s03 in 95 of 126 splits.
- **Reading:** a monotone learning-rate effect (2.5e-4 < 5e-4 < 1e-3 < 2e-3), not yet at a plateau on that axis. Wave 2
  (within 4–7) is a clean out-of-sample check for s03 because it was not used to choose. Rule: s03 − gC5 ≥ +0.15 on the
  within 4–7 mean confirms the direction.

**§191 within 1 rescored with the right pc128 (Leonardo 64408767).** Combination pooled − control −0.81 [−1.39, −0.27] (was
−0.51); members −0.27; fallback −0.30. Verdict: within 0 −0.23, within 1 −0.81, recording −0.49, within mean −0.52. By the
letter of the rule still not "both below −0.3"; in substance negative. No further Scherer forms.

**§192 so far** (draw 1, R − L): within 1 −0.07, within 2 +0.22, within 3 +0.15; within 0 pending (Galvani). With draw 0 (+0.12,
+0.22, +0.19, +0.10), the pooled within mean so far is +0.13.

**Form B:** about half its Galvani tasks died on "CUDA-capable device busy or unavailable" (shared nodes, not the code).
The missing seeds were resubmitted on a100-galvani only (3099147–54). Scoring waits for 9 seeds per order.

### 197. SWEEP 2: further along the learning-rate axis (written 13 Nov 19:19 UTC, before any job ran)

`ttt18 SWEEP=2` (tag LS2), run-time neutral as before; gC5 and s03 included as in-run references.
- lr 3e-3, 4e-3, 6e-3, 8e-3;
- lr 2e-3 / 4e-3 with consistency 2;
- lr 2e-3 / 4e-3 starting at window 20;
- lr 2e-3 soft;
- lr 2e-3 without consistency;
- lr 3e-3 with consistency 2 and start 20;
- lr 2e-3 with consistency 4.
Wave 1: within 0–3; wave 2 (afterany): within 4–7. Nested selection on Leonardo (`sweepsel.py TAG=LS2`).
**Keep bar:** the held-out gain over gC5 ≥ +0.3 on wave 1, and the selected form ≥ +0.15 on wave 2's orders. Then it is a
package change (constants only, same run time), and a v6.1 build would be proposed.
**Prediction:** the learning-rate curve peaks between 2e-3 and 4e-3. Held-out +0.2 (0 to +0.4).

### 198. THE STACK OF TONIGHT'S SMALL POSITIVES, and pushing each further (written 13 Nov 19:24 UTC, before any job ran)

**The user (via the coordinator, about 19:40 UTC):** "Don't treat the 0.3 as a hard threshold but we should push further and
keep iterating". From here, +0.3 bars are reading aids, not gates. Small positives that keep their sign are pushed and
stacked. **Reading rule for §198:** sign consistency across orders and people, the size of the stack, and its interval,
all stated together.
- **Stack run** (`ttt18 FORMS=gC5,s03,s14,s15,c01,e02`, labels K = `chain_stk_<order>`, tag KF), within 0–7.
  - New people get labels from the v6 chain.
  - Returning people get routed v5.1b as emitted where it exists (within 0/1), otherwise chain_final.
  - c01 combines the three axes that helped (lr 2e-3, consistency 2, start 20). e02 is s03 with an EMA (0.8) of the
    weights across refits, newly coded in ttt7 (`EMA`; existing variants unchanged).
  - Scored on Leonardo by `stackcmp.py`, per order and pooled: stack = <form> under K − gC5 under L (= v6f_gc5), split into
    labels (gC5 K − gC5 L) and form (<form> K − gC5 K).
  - Returning people: `retv6` with the K run (within 0/1).
- **Push further:**
  - (a) `SWEEP=3` (tag LS3): lr 1.5e-3 and 2.5e-3, consistency 1.5 and 3 at 2e-3, start 15 and 25, EMA 0.8 and 0.9, c01,
    e03; s03 and gC5 as references. Within 0–3, then within 4–7.
  - (b) More draws of R against L, paired by draw: draw 1 on within 4–7, draw 2 on within 0–7.
- **Infrastructure:** the 2080 partition is out of the Galvani default (it caused the CUDA-unavailable failures). The
  collector now ends with a per-job count of failed tasks and exits 3. The §192 within 0 R draw 1 (12 of 24 tasks lost)
  goes to Leonardo.
- **Prediction for the stack:** +0.3 (0 to +0.6) over v6f_gc5 on new people under within, positive on ≥ 6 of 8 orders.

**13 Nov 20:12 UTC: Leonardo access lost** (the step certificate expired; only the user renews it).
- Leonardo's queued jobs keep running, and their chained scoring writes to its runs/night13/.
- Local fetches from Leonardo are gated behind `runs/.leo_ok`, which is created only after a renewal and a single
  successful test; there is no login loop.
- The stack run is duplicated on Galvani (within 0/1 now; within 2–7 are submitted by a loop as Galvani's submit limit
  frees slots) and scored on the Mac (`runs/s198_stack_summary.txt`), including returning people via `retv6 FT_SFX=KF
  FT_FORM`.

**§192 / §198(b) so far, R − L by draw** (gC5 over v5.1b, new; R = labels from the v6 chain, L = package labels):

| | within 0 | within 1 | within 2 | within 3 |
|---|---|---|---|---|
| draw 0 | +0.12 | +0.22 | +0.19 | +0.10 |
| draw 1 | +0.01 | −0.07 | +0.22 | +0.15 |
| draw 2 | (pending) | +0.15 | +0.25 | +0.18 |

10 of 11 pairs are positive; the mean is +0.14. Read by sign consistency (§198), R is a small, consistent positive and stays
in the stack.

**20:17 UTC: Leonardo access back** (the user renewed the certificate; a single test passed). `runs/.leo_ok` is set and the
Leonardo fetches resume. The Galvani stack duplicates for within 2–7 were never placed (the submit limit), and that loop
is stopped so it does not crowd form B or SWEEP 3. The Leonardo stack copy covers within 0–7; Galvani keeps within 0/1.

**§194 wave 2 RESULT (Leonardo `runs/night13/sweepsel_w2.txt`; within 4–7, orders not used to choose).**
- Form minus gC5:
  - learning rate: s03 (lr 2e-3) −0.00, s02 (1e-3) −0.01;
  - s14 (1e-3, consistency 2) −0.05;
  - **s15 (1e-3, start 20) +0.13** (without P10 +0.15);
  - head-only s04 −0.39, s05 −0.28; low lr s01 −0.12.
- Nested held-out on wave 2: −0.07 [−0.36, +0.14].
- **The confirmation rule (s03 ≥ +0.15 on within 4–7) is missed: −0.00.** Wave 1's monotone learning-rate effect does not
  replicate on fresh orders; it was selection on four orders.
- What does replicate: head-only and low learning rates lose on both waves. **s15 is the only form positive on both
  waves (+0.24, +0.13).**
- Consequences:
  - SWEEP 2 (§197, lr 3e-3 to 8e-3) was premised on the learning-rate effect; it is read with that in mind.
  - The stack's form part is judged on within 4–7 as well as 0–3. s15 is the form to carry, s03 is not.

**§198 STACK RESULT (13 Nov 20:28 UTC; Leonardo `runs/night13/stackcmp.txt`; within 0–7, package conditions; new people).** The stack
(labels K from the v6 chain plus a form) against v6f_gc5 (gC5 with package labels L). Combination over v5.1b, six bases:

| form | stack [80 %] | without P10 | form part [80 %] | orders + | people + |
|---|---|---|---|---|---|
| **s15** (lr 1e-3, start 20) | **+0.24 [+0.11, +0.38]** | +0.28 | +0.15 [+0.05, +0.26] | 7/8 | 7/10 |
| c01 (lr 2e-3, cons 2, start 20) | +0.21 [−0.00, +0.43] | +0.25 | +0.12 [−0.06, +0.31] | 7/8 | 6/10 |
| s03 (lr 2e-3) | +0.19 [−0.01, +0.40] | +0.22 | +0.10 [−0.08, +0.30] | 7/8 | 5/10 |
| s14 (lr 1e-3, cons 2) | +0.18 [+0.07, +0.29] | +0.20 | +0.09 [+0.00, +0.18] | 7/8 | 8/10 |
| e02 (s03 + EMA 0.8) | +0.18 [+0.04, +0.33] | +0.21 | +0.09 [−0.04, +0.23] | 7/8 | 6/10 |
| gC5 (labels only) | +0.09 [+0.04, +0.14] | +0.11 | 0 | 6/8 | 7/10 |

- **Labels part (v6 chain against package):** +0.09 [+0.03, +0.14], positive on 6 of 8 orders and 7 of 10 people. It agrees
  with §192 (10 of 11 draw pairs positive).
- **s15's form part by wave:** within 0–3 +0.36 / +0.18 / +0.30 / −0.05 (mean +0.20); within 4–7 −0.06 / +0.59 / +0.28 / −0.36
  (mean +0.11). Positive on both waves, as in §194.
- **Within 7** is negative for every form (−0.4 to −0.6), from both its labels part (−0.26) and its form part.
- **Prediction** +0.3 (0 to +0.6), ≥ 6/8 orders: +0.24 with 7/8. Inside the range.
- **Reading by the §198 rule:** labels and the s15 form keep their sign across orders and people. The stack is small but
  consistent: about +0.24 new, so about +0.06 to +0.14 sealed (57 % × transfer 0.4 to 1.0).
- **Package cost:** s15 is constants only (same run time). The labels need the v6 chain's own outputs without FT
  (eye and rest in), as emitted; the package already computes the parts of that chain, but the build session must
  confirm the run time.
- **Returning-people part:** pending (the Galvani stack copy, within 0/1).

**§198 within 2 and within 7 (13 Nov 20:29 UTC; Leonardo npz, six-base means per person P01..P10).**
- **within 2** (v6f weak, +0.32):
  - v6f over v5.1b: +1.3 −0.1 −0.9 −1.1 +0.1 +1.1 +0.6 +1.2 −0.2 +1.2;
  - labels R: −0.3 +0.3 −0.0 +0.7 +0.5 +0.0 +0.4 +0.3 +0.5 −0.5;
  - s15 form: +0.3 +0.5 +0.6 +0.5 −0.4 +0.3 +0.0 −0.1 +1.6 −0.5.
  - The stack helps (+0.49).
- **within 7** (v6f strong, +1.76; positive for all ten people):
  - eye + rest: +2.2 +1.7 +2.0 +0.6 +1.4 −2.7 +0.9 +0.2 +1.6 −1.0;
  - FT: +1.8 −0.1 +1.1 +1.1 +0.1 +4.3 −0.8 +0.0 +0.3 +2.8;
  - labels R: −0.9 −0.1 −0.2 −0.6 +0.3 −0.6 −0.1 −0.3 +0.3 −0.4;
  - s15 form: −0.4 −0.6 −0.3 +0.2 −1.2 +0.2 +0.4 −0.9 −0.1 −0.8.
  - The stack hurts (−0.62).
- **Reading:** the two weak orders are weak for opposite reasons. Within 2 is weak for v6f itself, and the stack recovers some
  of it. Within 7 is one of v6f's strongest orders (its FT gains +1.06), and there the stack gives some back. The overlap is
  only in people: P05 and P10 lose on the s15 form part in both, and P01 on the labels part in both. The parts differ.
- **v6.1 build path:** requested from v52-build at 20:35 UTC (new package directory; the staged ZIP stays the default). The
  check list is as for v6f, plus a dev row in the exact package form. The coordinator is told at once if the worst case
  passes about 50 min.

**§198 v6.1 dev row in the package's form (13 Nov 20:47 UTC).** v52-build's label chain (lean/src61/v6chain.py) is the same computation as dev's V6r
under the package gate:
- (15/16)·mix_v51b + (1/16)·rest, with the eye log-posteriors in GE;
- v5.1b's gated rebalancing, SessionEM and TSEM, with no FT and the gate never seeing it;
- labels = argmax and max-probability weights, causal in the delivery order;
- FT s15 delta1, one net per family.
So the stack's s15 arm is the package-form dev row. New people, over v5.1b under the same order (Leonardo npz, six bases):

| order | v6f | v6.1 |
|---|---|---|
| within 0 | +1.39 | +1.85 |
| within 1 | +1.49 | +1.89 |
| within 2 | +0.32 | +0.80 |
| within 3 | +1.02 | +1.08 |
| within 4 | +0.94 | +1.16 |
| within 5 | +1.36 | +2.02 |
| within 6 | +1.72 | +1.98 |
| within 7 | +1.76 | +1.14 |
| **mean [80 %]** | **+1.25 [+0.94, +1.59]**, without P10 +1.20 | **+1.49 [+1.14, +1.84]**, without P10 +1.49 |

- v6.1 has 8 of 8 orders ≥ +0.5 (v6f: 7 of 8).
- Within 4's v6f +0.94 differs from the local table's +0.91: Leonardo's rest-net seed set differs; same code.
- The v6.1 ZIP is cand_v61_s15 (sha256 2ca06dc0…). Official local replay 0.9932767697. Worst case at 100 % unrecognised
  about 45.8 min sealed with warm-up (v6f about 45.3). The other checks are running (v52-build).

**§198(b) R against L, all paired draws so far (13 Nov 20:51 UTC).** gC5 over v5.1b, new; R − L:
- draw 0, within 0–3: +0.12 +0.22 +0.19 +0.10;
- draw 1: within 0 +0.01, within 1 −0.07, within 2 +0.22, within 3 +0.15, within 4 +0.02, within 5 +0.27, within 6 +0.12,
  within 7 −0.07;
- draw 2: within 0 +0.13, within 1 +0.15, within 2 +0.25, within 3 +0.18, within 4 +0.04, within 5 +0.31, within 6 +0.09
  (within 7 waits for Galvani).
- **17 of 19 pairs positive, mean +0.13.** By the sign rule, the v6-chain labels are a small positive that holds over
  orders and draws. It is part of v6.1.

**About 20:50 UTC: Galvani's SSH master dropped** ("Permission denied (keyboard-interactive)"); it needs the user's 2FA.
Local Galvani fetches are gated behind `runs/.galv_ok` (the four loops were restarted gated; no login loop). Waiting on
Galvani: form B's resubmits, SWEEP 2 within 8–11, SWEEP 3 within 0–3, the stack's returning-people copy (within 0/1), L draw 2
within 7, and v6.1's worker-image FT row (v52-build's staged script).

**§197 SWEEP 2 RESULT (Leonardo `runs/night13/sweepsel2_w{1,2}.txt`).** Nested held-out over gC5: within 0–3 **+0.10 [−0.26, +0.41]**;
within 4–7 **−0.02 [−0.42, +0.25]**. In-sample on wave 1, every higher-lr form is +0.17 to +0.33 (the s03 pattern). On wave 2 the
picks (t10, t09) do not hold. Null, as predicted after §194's wave 2: raising the learning rate further adds nothing.

**21:03 UTC, work moved to Leonardo** (Galvani: about 295 tasks pending, none running; Leonardo idle).
- Form B's missing seeds per order (64419694–707; first copy wins against Galvani; the local loop fetches from both).
- L draw 2 within 7 (64419709).
- SWEEP 3 within 0–3 with scoring and nested selection chained (64419711–19 → sweepsel3_w1.txt).
- Galvani duplicates of SWEEP 3 and L d2 within 7 cancelled; the Galvani stack copy cancelled too, because Leonardo's stack
  run already has within 0/1's returning-people outputs (scored locally via `retv6 FT_SFX=KF`).
- **Follow-up wave:** the v6.1 stack (FORMS=gC5,s15, labels K) on within 8–11, full 0/1 and runs 0/1, with scoring and
  stackcmp chained (64419720–37 → stackcmp2_within8-11.txt and stackcmp2_fullruns.txt). It gives v6.1's dev row on twelve
  more orders, including full and runs.
- v6.1's worker-image FT check stays on Galvani (v52-build), ahead of my lowered sweeps.

**§198 returning people from the stack run (13 Nov 21:12 UTC; `runs/retv6_KF_*.log`; Leonardo's within 0/1 outputs).** The three v6
additions on the routed path (eye view, rest member, FT with labels K: routed v5.1b as emitted). Against routed v5.1b,
shuffled mean of within 0/1, mean (P04, P10):

| form | FT alone | all three (ERF) |
|---|---|---|
| gC5 | +0.33 (+0.88, −0.22) | +0.20 (+0.49, −0.09) |
| **s15** | +0.00 (+0.41, −0.41) | **+0.29 (+0.33, +0.24)** |
| s03 | +0.03 (+0.21, −0.15) | +0.29 (+0.57, +0.00) |
| c01 | −0.19 (+0.10, −0.47) | +0.19 (+0.25, +0.13) |
| e02 | +0.12 (+0.22, +0.01) | +0.14 (+0.05, +0.24) |

- With s15, all three additions on the routed path are positive for **both** returning people (+0.33, +0.24). It is the
  first routed form with P10 above zero. §190's gC5 form gave +0.22 with P10 at −0.09.
- Two people and two orders: not an estimate, but the sign holds for both people.
- **Not in v6.1**, which fine-tunes only unrecognised sessions.
- A v6.2 option would be "v6.1 + eye/rest/FT s15 also on routed sessions". Its run time is FT on every session, which is
  the 100 %-unrecognised worst case already measured for v6.1 (about 45.8 min). Its gain for the 43 % returning share would
  be about +0.29 × 0.43 ≈ +0.12 sealed before transfer.

### 199. v6.2 OPTION: the v6 additions on the routed path, firmed up (coordinator's request; written 13 Nov 21:15 UTC, before any job ran)

§198 found eye + rest + FT s15 on the routed path at +0.29 for returning people (P04 +0.33, P10 +0.24), within 0/1 only. v6.2
would change recognised sessions, which v6f and v6.1 leave bit-identical to v5.1b, so the sign is firmed up first.
- **Labels:** routed v5.1b as emitted (`retv6 LABEL_RT`), now also for within 2–7, full 0/1 and runs 0/1. Accuracy on
  returning rows: within 0.797–0.827, full and runs 0.784–0.797, against chain_final's 0.799.
- **Files:** `chain_rts_<order>` (new people from the v6 chain, returning people routed); ttt18 tag Q.
- **GPU runs (Leonardo):** FORMS=gC5,s15 on within 0–7, full 0/1 and runs 0/1, plus draws 1 and 2 on within 0–3.
- **Scoring:** `retv6 FT_SFX=QF FT_FORM=s15|gC5`, with arms E, R, **ER (eye + rest, no FT; new)**, F and ERF, per order, P04 and
  P10 separately.
- **Reading rule (an aid, not a gate):** v6.2 = ERF s15 on routed sessions is worth proposing if P10 is positive on ≥ 6 of 8
  within orders, no order has P10 below −1.0, and the within mean (P04 + P10) is ≥ +0.15. The safer v6.2 (ER only, no
  routed FT, same run time as v6f at 57 %) is preferred if ER carries ≥ 2/3 of ERF's mean. gC5 is reported alongside as
  the control.
- **Prediction:** ERF s15 within mean +0.2 (0 to +0.4), P10 positive on 5–7 of 8; ER about half of ERF.

**§198 extended (13 Nov 21:56 UTC; Leonardo stackcmp2_*).** The v6.1 stack (labels K + s15) against v6f_gc5:
- **within 8–11:** stack +0.13 [−0.04, +0.30], 4/4 orders, 6/10 people. Labels +0.03 [−0.04, +0.09], form +0.10 [−0.06, +0.25].
  Per order +0.01 / +0.01 / +0.37 / +0.11.
- **full 0/1 and runs 0/1:** stack −0.07 [−0.26, +0.12], 1/4 orders. Labels −0.09 [−0.20, +0.01], form +0.02 [−0.14, +0.19].
  Per order full 0 −0.07, full 1 −0.37, runs 0 +0.18, runs 1 −0.04.
- **v6.1's dev row per scheme** (v6.1 = v6f + the stack, per order; full 2/3 and runs 2/3 have no stack run):
  - within (12 orders): v6f +1.24, stack +0.20 (11/12 orders ≥ 0) → **v6.1 about +1.44**;
  - full 0/1: v6f +0.95, stack −0.22 → about +0.73;
  - runs 0/1: v6f +1.13, stack +0.07 → about +1.20.
- **Reading:** v6.1's edge over v6f is about +0.2 under within (the scheme that matches the sealed drop) and about zero to
  slightly negative under full and runs. The labels part is what turns: positive under within (+0.09, +0.03), negative
  under full (−0.28, −0.11).

**§198 SWEEP 3 wave 1 (Leonardo sweepsel3_w1.txt; within 0–3).** Nested held-out +0.21 [−0.05, +0.44]. e03 (lr 2e-3, consistency 2, start
20, EMA 0.8) was chosen in 87 of 126 splits, then u02 and u05. e03 is the first form positive for every person but one
(P05 −0.1), with P10 +0.4. **The clean check:** e03, u02, u05 against s15 and gC5 on the fresh orders within 4–11
(Leonardo 64422100–70 → sweepsel_fresh.txt).

**§188 form B RESULT** (eye net fine-tuned per session on v6-chain labels; `runs/formB_summary.txt`; 8 orders, 9 seeds).
gC5 over v5.1b, new; current eye net → form B (pseudo-labels) → true-label ceiling:
- within 0 +1.39 → +1.51 → +1.70; within 1 +1.49 → +1.68 → +1.84;
- within 2 +0.32 → +0.42 → +0.58; within 3 +1.02 → +1.18 → +1.31;
- full 0 +1.27 → +1.24 → +1.35; full 1 +0.62 → +0.69 → +0.82;
- runs 0 +1.35 → +1.47 → +1.44; runs 1 +0.91 → +0.94 → +1.06.
- **Form B against the current eye net: +0.14 within, +0.09 over all eight; positive on 7 of 8** (full 0 −0.03).
- The true-label ceiling is +0.30 within: form B reaches about half its own ceiling. That is far below the +1.0 the
  decorrelation curve allows, so per-session eye adaptation reduces little of the eye net's error autocorrelation.
- By the sign rule, a small stackable positive. Combined with v6.1: being scored now (CPU). Form B on within 4–11 is queued
  (Leonardo).

**On making the label chain scheme-robust (coordinator's question):** the labels part is +0.09 / +0.03 under within (two
order sets) and −0.09 under full and runs (labels −0.28 and −0.11 on full). A robust form would blend the two label sources
(the mean of the v5.1b and v6-chain outputs) or take the v6 chain only where both agree. Both are cheap to test on stored
outputs plus one FT wave. Recommendation: test the blend on within 0–3 and full 0/1 before choosing. Until then v6.1 stays
as built, because the within-like reading of the sealed stream (§169) favours it.

### 200. SCHEME-ROBUST LABELS: the blend of the v5.1b and v6-chain outputs (written 13 Nov 21:58 UTC, before any job ran)

The v6.1 labels part is positive under within (+0.09 / +0.03) and negative under full and runs (−0.09). Blend labels =
0.5 × (v5.1b's outputs + the v6 chain's outputs) as emitted (`shufcombo LABEL_ARM=BLEND` → `chain_blend_<order>`; ttt18 tag B).
Accuracy within 0–3: 0.752–0.762; full 0/1: 0.736 / 0.740; runs 0/1: 0.760 / 0.742.
- **Paired wave:** FORMS=s15 with labels L, R and B on within 0–3, full 0/1 and runs 0/1 (tags LF, RF, BF).
- **Scoring:** most complete form; the combination under s15 with each label source, against v6f (L, gC5).
- **Reading:** the blend is worth taking if, against R, it loses ≤ 0.05 under within and gains under full/runs, with the sign
  holding across those orders. Then v6.1 would use blend labels.
- **Prediction:** blend between L and R everywhere: about +0.05 against L under within, about 0 under full.

**§199 RESULT (13 Nov 22:10 UTC; `runs/s199_*`; returning people P04 and P10, routed path, labels = routed v5.1b as emitted).**
Change against routed v5.1b:

| arm (form) | within 0–7: mean, P04, P10 | P10 > 0 | min P10 | orders > 0 | full 0/1 + runs 0/1: mean, P10 > 0 | draws 1–2 within 0–3 |
|---|---|---|---|---|---|---|
| **ERF (s15)** | **+0.86**, +1.09, +0.62 | **8/8** | +0.10 | 8/8 | +1.17, 3/4 (full 0 −0.28) | +0.74, P10 > 0 8/8 |
| ERF (gC5) | +0.87, +1.13, +0.62 | 6/8 | −0.12 | 8/8 | +1.15, 4/4 | +0.67, 6/8 |
| **ER (eye + rest, no FT)** | **+0.53**, +0.61, +0.45 | 6/8 | −0.15 | 8/8 | **+0.98, 4/4** (all orders > 0) | +0.52, 4/8 |
| F (s15) alone | +0.34, +0.74, −0.07 | 3/8 | −0.57 | 6/8 | −0.00 | +0.22 |

- **Check:** routed v5.1b on within 2–7 has no shufpkg reference. Its absolute values sit in the range of the checked orders
  (P04 69–74, P10 89.6–91.9). Full 0/1 and runs 0/1 do have references, and they match.
- Within 0/1 (the first measured, §198) are the lowest orders. Across twelve orders the routed gain is larger.
- **Rule** (P10 > 0 on ≥ 6/8 within orders, no P10 < −1.0, within mean ≥ +0.15):
  - **ERF s15 passes clearly** (8/8, min +0.10, +0.86);
  - ER passes too (6/8, min −0.15, +0.53).
- ER's share of ERF is 0.62, under the 2/3 needed to prefer it by the rule. But ER is the more robust form: positive on all
  twelve orders, including full 0 where ERF's FT costs −1.06 (P04 −1.95). It also needs no FT on recognised sessions, so its run
  time stays at v6f's 57 % figure. ERF's FT needs the 100 % worst case (about 45.8 min).
- The FT form on the routed path hardly matters (ERF s15 +0.86 against gC5 +0.87). FT alone is weak for P10 (3/8 positive).
- **Size for the sealed mix** (43 % returning, before transfer): ERF about +0.37, ER about +0.23. That is larger than v6.1's
  new-people edge (about +0.1).
- **Limit:** P04 and P10 are the only returning people in dev (two people, four sessions); every order reuses them.
  Sealed returning sessions may be other people.

**v6.1 + form B against v6.1** (8 orders; `runs/v61b_summary.txt`): +0.07 (6 of 8 positive; within +0.06). Per order: within 0 −0.06,
within 1 +0.08, within 2 +0.11, within 3 +0.12, full 0 −0.05, full 1 +0.04, runs 0 +0.16, runs 1 +0.15. A small, mostly-positive
addition.

**§199 follow-ups (13 Nov 22:12 UTC).**
- **v6.2 build requested** from v52-build (coordinator), one package with two switches:
  - v6.2 = v6.1 + eye, rest and FT s15 on recognised sessions;
  - v6.2-safe = eye and rest only.
  - The exact dev insertion points are handed over (rest and FT before the personal head; eye in shadow and tail; the shadow,
    and so the v3 gate, sees the change).
  - Added checks: switches off ≡ v6.1 bitwise on recognised sessions; references for the routed path on CPU and A100;
    a package-form dev row per person; run time for both variants; the worker image.
- **Reference gap:** `shufpkg.py SCHEMES=within PERMS=8 SHUFPKG_OUT=shufpkg_within8.npz` computes routed v5.1b independently for
  within 0–7. Within 0/1 are checked against the old reference, within 2–7 against retv6's base (`runs/shufpkg8_check.txt`).
- **Silent mislabel fixed:** retv6 printed "(= shufpkg)" next to the base even on orders without a reference (within 2–7 in
  `runs/s199_*`). Those lines were not checked; the label is now conditional.
- **Robustness (coordinator):** P10 under full and runs.
  - ERF: full 0 +0.33, full 1 +0.82, runs 0 +1.25, runs 1 −0.08.
  - ER: +0.69, +1.15, +1.36, +0.41.
  - P04 is the one hurt: full 0 ERF −0.89 (FT −1.95).
  - So if the sealed stream were not within-like, ER would still help both people on all four orders. ERF would help P10
    (3/4 positive, worst −0.08) but can cost P04 about 1 point on a full-shuffle order through the routed FT.

### 201. v6.2 INSERTION: the gate sees v5.1b only (coordinator's request; written 13 Nov 22:14 UTC, before any result)

In v6f and v6.1 the gates never see the second chain, so recognition is exactly v5.1b's. In §199's measured form the shadow
(the v3 gate's input) carries the additions, so recognition can move. A wrongly routed new person is the expensive error.
- **(1) Gate-unchanged form** (`retv6 GATE_V51B=1`): the shadow, both gate input and pre-routing output, comes from v5.1b's
  own mix and view; the additions enter only the routed tail. Arms ER and ERF (s15, labels Q), returning people, the same
  twelve orders as §199, next to the measured form.
- **(2) Measured form on new people** (`retv6 RT_PROTO=all_loso`): for every order and base, whether any window is routed that
  v5.1b does not route (the routed set, first window and person), plus how returning people's first routed window moves
  (from the §199 logs).
- **Rule for v52-build:** build the gate-unchanged form unless (1) loses a clear part of the gain (≥ 1/3 of ERF's within mean,
  or P10 positive on < 6/8). In that case, the measured form gets an explicit check that the routed set equals v5.1b's for
  all LOSO people under the shuffled orders.
- **Prediction:** (1) keeps ≥ 80 % of the gain (the bio gate decides most routing, and three of four returning sessions route
  at window 10 either way); (2) shows no new routed new-person session.

**§199/§201 reference gap closed (13 Nov 22:23 UTC; `runs/shufpkg8_check.txt`).** `shufpkg.py SCHEMES=within PERMS=8` recomputed routed v5.1b
independently for within 0–7.
- within 0/1 reproduce the old reference exactly (max |diff| 0).
- On within 2–7, retv6's routed v5.1b matches the independent reference to ≤ 0.005 over all bases and both people (the logs
  round to 0.01).
- All eight within orders of §199 are now cross-checked. The new keys are merged into `runs/shufpkg.npz` (shared keys
  identical).

**22:23 UTC, Galvani becomes the primary cluster (the user: "queue on galvani bc ultimatly I wanna switch away from leaonardo
all to galvani").**
- **Limits** (sacctmgr, QOS normal): 300 submitted jobs per user (array tasks count), 36 running jobs and 36 GPUs per user.
- **Galvani lacked the replay environment** (no `replay/`, no `runs/replay_data`). A bundle of about 1.4 GB is being
  copied from the Mac, which holds the union of both clusters' outputs: replay code, replay_data (59 MB), stored and
  delivery-order member outputs (probs6/7/17/18 together about 1 GB, ocushuf 320 MB, restshuf, rest4, ocunet), core1, pc128,
  Hilbert and teacher files, labels, form B outputs and the pinned order file.
- **`scripts/cloud/galv_cpu.sh`** chains scoring on Galvani's CPU partition.
- **Duplicated on Galvani:** the fresh-order sweep check (within 4–11; 3099592–645), with scoring and selection chained on
  Galvani's CPU partition (3099880–88 → runs/diag/cpu/sweepsel_fresh.txt); the pending part of the blend wave (runs 0/1;
  3099646–49).
- **22:25 UTC, Galvani's submit limit binds.** v52-build could not resubmit v6.1's worker-image check (QOSMaxSubmitJobPerUserLimit at
  302 of 300). I cancelled the four blend-wave duplicates on Galvani (3099646–49; Leonardo keeps its copies), leaving 206 queued
  and room for both worker jobs.
- The first v6.1 worker run failed on node galvani-cn228 with a hardware fault ("CUDA error: Invalid access of peer GPU memory over
  nvlink or a hardware error", sticky), not our code. Its 190-window session completed in the worker image at ×1.721, 17
  refits, errors 0.
- **Lesson for a Galvani-only workflow:** with 300 slots, a 24-task array per order allows about 12 orders queued at once. Waves
  have to be chained (afterany) rather than all submitted up front.

**§201 RESULT (13 Nov 22:26 UTC; `runs/s201_*`).**
- **(1) Gate-unchanged form** (the gates and the shadow see only v5.1b; the additions reach the routed tail). Returning people,
  against routed v5.1b:
  - within 0–7 ERF s15: +0.33 / +0.19 / +1.23 / +1.18 / +0.90 / +0.84 / +1.22 / +1.24, **mean +0.89** (measured form +0.86);
    P10 > 0 on **8/8**, min +0.10.
  - ER: mean **+0.65** (measured +0.53); P10 > 0 6/8, min −0.15.
  - full 0/1, runs 0/1: as in the measured form (ERF −0.28 / +1.91 / +2.10 / +1.06; ER +0.40 / +1.78 / +0.90 / +0.83).
  - **It loses nothing, so it is built** (rule met). Recognition stays exactly v5.1b's by construction.
- **(2) Measured form on new people:** no arm changes routing on any order (within 0–3 checked) or base, and no new person is
  routed by v5.1b or by any arm. For returning people the routed set never changes; only P04|2's late first routed window
  moves by 1 to 4 (§199 logs).
- **v52-build told (22:30 UTC)** to build the gate-unchanged insertion (dev reference `retv6 GATE_V51B=1`). Check (a) becomes
  "routed set and first routed window bitwise equal to v5.1b's for every session", including new people.

**§202 PRE-REGISTRATION (13 Nov 22:37 UTC): v6.2's worker-image check on Leonardo.** Galvani's queue estimates the v6.1
worker job (3099908) for 14 Nov 21:56 UTC, after the 06:30 decision. Leonardo cannot build the image (12 Nov: no fakeroot), but
SingularityPRO 4.3.1 runs a prebuilt one. Galvani's `runs/cloud/ni26_worker.sif` (md5 d5e978f4…) was copied there.
`scripts/cloud/leo_worker_replay.sbatch` checks the md5, CUDA in the image and the pip freeze. It then runs
`scripts/replay_package.py` on the exact ZIPs inside the image, in the order v5.1b, v6.2 ZIPs, v5.1b.
- **Mechanism:** the warm-up replay sessions are recognised at window 10. Under v6.2 the eye view, the rest member and (r2)
  fine-tuning act on the routed tail, so unlike v6f and v6.1 this replay exercises the new path in the worker's torch.
- **Prediction:** errors 0, files unchanged, CUDA visible. Score within 0.01 of v5.1b's 0.9932767697 (the warm-up set is near
  ceiling). Ingestion time against the paired v5.1b at most ×1.3 (FT on about 180 of 190 windows per replay session, short
  sessions).
- **Refuted if:** any error or traceback, files changed, CUDA not visible in the image, or a score below 0.983. A time ratio above
  ×1.3 is reported and sent to the worst-case run-time row; it does not fail the check.
- **Also (s202):** `retv6.py` wrote every run to `runs/retv6.npz`, so runs with other settings overwrote each other (gate a/b).
  It now writes `runs/retv6/<gate>_<form>_<sfx>_<proto>_<orders>.npz`. Smoke: within 7 gate-unchanged must reproduce s201's row.
- **s202 (22:50 UTC): table builder** `studies/13nov/morning_table.py` (new people from runs/leo_npz, v6f L and v6.1 KF from the
  same machine; returning people from runs/retv6/g1_s15_QF_*).
  - Pairing found: v5.1b's P0r is not identical across FT runs of one order. P0r takes the no-FT re-stream from each FT job's
    own base outputs (FR17), and GPU float noise moves it by up to 0.10 on one person (runs 0, within 1/7/8/10). Not an input
    error: each file reproduces exactly on both machines. Each variant is now taken against its own P0r.
  - Earlier cross-file comparisons (s200 subtracted v6f's V6Fr from the s15 file's) carry up to ±0.1 of that noise per person.
  - New people (v6.2 = v6.1 here, no new person routed): within 0–11 v6f +1.24 [+1.07, +1.41], v6.1 +1.44 [+1.28, +1.60];
    full 0/1 +0.94 vs +0.72 (P10 +1.64 vs +0.19); runs 0/1 +1.13 vs +1.21.
- **Galvani worker jobs** 3099908/09 estimate 14 Nov 21:56 / 22:27 UTC. The cluster is full (5 of our GPU jobs running), and
  `scontrol top` is not available there. The Leonardo image upload (8 chunks, about 0.6 MB/s) lands about 00:30 UTC.
- **§202 amended (22:55 UTC, v52-build's three changes):**
  1. ZIPs only, each checked against its folder's SHA256SUMS before any GPU time. Order: v5.1b, v62_r0, r1, r2, v6.1, v5.1b.
  2. The FT path inside the image: `v6_time.py cuda lean_v5_pkg final_unz_61/c`, then `… cand_v62_r2` with and without
     `--gates-on`. Replays never reach FT, because their sessions are recognised at window 10.
  3. A CUDA error fails the job and names the node for `--exclude`.
  - A watcher submits the job when the 8 chunks (3,472,891,904 bytes) and the three v62 ZIPs listed in SHA256SUMS are on
    Leonardo.
  - A truncated v62_r0.zip (23.5 MB) that my stopped rsync had left under the real name was deleted. The SHA256SUMS check
    would also have caught it.
  - FT-path prediction: errors 0, run-time ratios within 10 % of the Galvani image run (v6.1 ×1.72 at 190 windows) and
    of the venv A100 rows (×1.81 / ×2.08).

**§200 RESULT (13 Nov 22:59 UTC; runs/s200_*; paired, each file against its own P0r).** Blend labels against R
(v6.1), new people, s15: within 0–3 −0.05, full 0/1 +0.11, runs 0/1 −0.14 (orders up 2/4, 2/2, 0/2). The blend lowers the full
loss a little but costs within and runs. Inside the noise with mixed sign: **undecided, not added**; R stays in v6.1/v6.2. L
labels against R: within −0.10, full +0.21, runs −0.03.

### 203. FULL AND RUNS ORDERS 2/3 FOR THE MORNING TABLE (written 13 Nov 22:59 UTC, before any job ran)
- **Why:** the organisers removed chronology "to avoid tricks" without EEG. Of our schemes, full (a uniform permutation of
  the session's windows) matches that best. Yet full has two orders for v6.1/v6.2, and it is where v6.1 trails v6f on new
  people (−0.22 Leonardo files, −0.35 local, P10 driving it).
- **What:** for full 2/3 and runs 2/3 (all inputs exist):
  - chain_v51b and chain_v6r labels (shufcombo LABEL_OUT);
  - chain_rt (retv6 LABEL_RT);
  - chain_rts (v6r on new people, rt on returning, as for full 0);
  - ttt18 FORMS=s15 on the Q labels (Leonardo, 24 tasks per order), then shufcombo FT_V=s15 (new people, v6.1 = v6.2) and
    retv6 GATE_V51B=1 FT_SFX=QF FT_FORM=s15 (returning people, v6.2 and v6.2-safe).
  - The Q run's new-people labels equal v6.1's R labels (v6r), checked on the files.
- **Prediction:** new people v6.1 − v6f on full 2/3 −0.2 (−0.6 to +0.2), runs 2/3 +0.05. Returning people v6.2 on full 2/3
  +0.8 (−0.3 to +1.9), v6.2-safe +0.9.
- **Reading rule (stated now):**
  - If v6.1 − v6f on full over four orders is ≤ −0.2 with ≥ 3/4 orders below zero, the coordinator is told: on the scheme
    closest to the organisers' statement, v6f's FT (gC5, v5.1b labels) beats v6.1's on new people. A v6.2 on v6f's FT would
    be the alternative, the user's choice.
  - Otherwise the four-order full rows go into the table as they are.

**23:15 UTC, after the app restart (about 23:02 UTC):** watchers, loops and the image upload died with the app.
- Leonardo's login01 timed out on port 22, so scripts now use login02 (`-o HostName=login02-ext…`; ssh config unchanged).
- The image upload resumes per chunk by appending the missing bytes, with md5 per chunk (macOS openrsync has no
  `--append-verify`). Detached with nohup; landing about 00:10 UTC.
- s203 labels are built (full 2/3, runs 2/3). The Q-label FT wave is going to Leonardo.
- **Silent no-op fixed:** shufcombo's STRICT checked only the core members. A missing GE, Hilbert or teacher input fell back to
  recording order silently (only the file name changed). STRICT now requires all of them. Every scored file so far carries
  `_ge_hb_ts`, so no past result is affected.
- **Fresh-order sweep (§194/SWEEP 3, within 4–11, Leonardo sweepsel_fresh.txt), recorded:** over gC5, s15 +0.14 (without
  P10 +0.17), e03 +0.03, u05 +0.04, u02 −0.12. Nested held-out −0.00 [−0.23, +0.18]; s15 chosen in 137 of 252 picks. s15 is
  confirmed as the form, and SWEEP 3's winner adds nothing.

### 204. FULL ORDERS 4–7 FROM SCRATCH (written 14 Nov 23:15 UTC, before any job ran)
- **Why:** full (a uniform permutation of each session's windows) is the scheme closest to the organisers' statement. Even
  with §203 it has four orders, and v6.1 trails v6f there on new people.
- **What, Leonardo, per order:**
  - members re-streamed: core1, Hilbert, teacher, pc128;
  - base FT (gC5) with the eye and rest re-streams;
  - labels chain_v51b and chain_v6r;
  - then t18L (gC5 on v5.1b labels = v6f) and t18R (s15 on v6r labels = v6.1, new people);
  - scoring on Leonardo by shufcombo for both; fetched to runs/leo_npz.
  - Returning people need chain_rt, which retv6 builds on the Mac. That relay follows once the labels exist.
- **Prediction:** new people, v6.1 − v6f on full 4–7: −0.15 (−0.5 to +0.2).
- **Reading rule:** as §203, now over up to eight full orders. With ≥ 6/8 orders below zero and a mean ≤ −0.2, the coordinator
  is told that v6.1's FT form or labels cost on full, and the alternative (v6.2 on v6f's FT) is named.
- **Timing:** members take 2–5 h, so this lands after the 06:00 UTC table. It serves the remaining uploads (deadline 21 Nov).
- **Form B within 4–11 (23:26 UTC; runs/formB411_*, local, paired per file):** the eye net fine-tuned on the session (ocuttt
  pl) against v6.1's eye view, both inside v6.1 (s15, K labels): **+0.03 [+0.01, +0.05]**, 6/8 orders up, 5/10 people up,
  without P10 −0.00. Together with within 0–3 (+0.07 on top of v6.1), form B is a real but negligible gain that costs run
  time. **Not added.**

**§206 (14 Nov 00:10 UTC, coordinator's question): a scheme-neutral form of v6.2.** No new runs were needed.
- **Candidate:** "v6f + routed additions", meaning v6f's new-people path (gC5, v5.1b labels L) plus v6.2's returning-people
  additions. Returning people in v6.2 are fine-tuned on v5.1b's routed output, so their gain does not depend on the
  new-people label source. morning_table.py now prints this candidate as v6f+ER and v6f+ERF.
- **Result (full over 4 orders; J = 0.5 new + 0.5 returning):**

| Candidate | Full: new / returning / J | Equal weights J | Within-only J |
|---|---|---|---|
| v6.2 | +0.73 / +1.18 / +0.95 | +1.17 | +1.17 |
| v6f + ERF | +0.87 / +1.18 / +1.03 | +1.15 | +1.07 |

  Level under equal weights (−0.02), +0.07 under full, −0.10 under within.
- **Where v6.1's scheme dependence sits** (new people, paired per file):
  - The s15 form alone (s15 with L labels − v6f) is about neutral everywhere: within 0–3 +0.01; full 0/1 −0.15 locally and
    −0.02 on Leonardo (stackcmp2, K labels); runs −0.07 / +0.07.
  - The label change (R − L) gains within (+0.10 on 0–3, +0.03 on 8–11) and costs full (−0.21 locally, −0.20 on Leonardo),
    with runs about 0.
  - So the bet on the scheme is the label source, not the form.
- **Package question (coordinator → v52-build):** can v6.2g emit this by a switch (label source L instead of the v6 chain
  on unrecognised sessions) without a new code path?

**§206 build (14 Nov 00:06 UTC, v52-build; written before any check of these packages): v6.2gL, the L switch on v6.2g.**
- **No new code path:** v6.2g's bundle already takes the label source as a key (`v6_ft_labels`: 1 = R, the v6 chain
  without FT as in v6.1; 0 = L, the inner v5.1b's output, which is v6f's label). Built `cand_v62_r2gL` (v6.2 routed FT) and
  `cand_v62_r1gL` (v6.2-safe) with `build6.py --v62g --labels v51b --t0 20 --lr 1e-3 --budget 2700 --routed 2|1`.
  The schedule is s15 on both paths, the guards are unchanged (cap 400, budget 2700 s) and the routed tail is unchanged.
- **Identity with v6.2g at R (shown at build time):** every weights.npz array except `v6_ft_labels` (1 → 0) and every
  other file (`submission.py` 37da0689…) are byte-identical to `cand_v62_r2g` / `r1g`. So the guard checks carry over,
  and v6.2gL at R *is* v6.2g.
- **Predictions:**
  - official local replay: files unchanged, errors 0, score within ±0.002 of v6.2g's 0.99458;
  - `check_v6.py --skip-recognised` (gates forced off; the v6f label path, FT reference with the inner v5.1b's outputs as
    labels): eye/rest/ft/mix6/tail within tolerance, on the CPU and on the A100 with package shapes;
  - `check_v62.py` (recognised sessions) passes as for v6.2g, because the labels are the same after recognition.
- **Refutation rule:** any tolerance miss or error in the replay or the checks fails the package for the morning table.

**§203 RESULT (14 Nov 00:10 UTC; runs/s203_*, morning_table.py → runs/morning_table_0007.txt).** Full and runs now have 4
orders each, for new and returning people.
- **New people, v6.1 − v6f:** full −0.14 (per order −0.07, −0.37, +0.16, −0.29; 3/4 below zero); runs +0.14; within +0.20
  (12 orders, 10/12 up). The pre-registered flag (full mean ≤ −0.2 with ≥ 3/4 below zero) is not met, so the reading is
  "undecided". The prediction (−0.2, runs +0.05) held for full and was low for runs.
- **Returning people:** v6.2 (ERF) full +1.18 (3/4 orders up, P10 min +0.33), runs +0.82 (3/4 up, P10 min −0.08), within
  +0.89 (8/8 up). v6.2-safe (ER): full +1.05, runs +0.86, within +0.65. Prediction (v6.2 +0.8, safe +0.9 on full 2/3): ERF
  higher (+1.78 / +0.36 on full 2 / 3), ER as predicted.
- **J (0.5 new + 0.5 returning), scheme-averaged:**

| Candidate | Equal weights | Within only | Full only |
|---|---|---|---|
| v6.2 | +1.03 | +1.17 | +0.95 |
| v6f + ERF (§206) | +1.00 | +1.07 | +1.03 |
| v6.2-safe | +0.97 | +1.05 | +0.89 |
| v6f + ER | +0.94 | +0.95 | +0.96 |
| v6.1 | +0.55 | +0.72 | +0.37 |
| v6f | +0.52 | +0.62 | +0.44 |

- **Reading:** both routed forms carry most of the gain. v6.2 and v6f + ERF differ by +0.03 under equal weights, which is
  inside noise. The choice between them is the bet on the scheme: v6.2 wins under within, v6f + ERF under full.

### 207. WEIGHT-SPACE EDITS RETESTED ON THE CURRENT CHAIN (written 14 Nov 00:25 UTC, before any result)
**The user (via the coordinator, about 00:15 UTC):** "now on the recent baseline, best performe I would like to give weight
space programmin another shot on top of that". §13 tested this on 11 Nov's chain in recording order, on core1 only: ten
hand edits, block gains, a 60-step Weco search. None passed. Old code is in 30060d5^ (experiments/wsp_*.py,
.weco/wsp-core1/); the old run's step history was deleted, and §13 plus evidence/wsp_bio1_11nov keep what it found.
- **Baseline:** v6f + routed additions (§206) and v6.2. Scored package-style under shuffled delivery (within, full, runs),
  new and returning people, P10 separately.
- **Stage 1, leverage (tonight, CPU, Mac):** where can a weight edit move the chain at all? shufcombo `SYN_T/SYN_D`
  (oracle diagnostic, never shippable) adds +5 accuracy points to one component, in recording layout before any
  permutation: top and true classes swapped on random wrong windows, the same windows for every base and arm.
  - Components: member slots s0–s15 (s0 = core1 + Hilbert pair; s5–s8 the fine-tuned SessionBN families; s10 teacher),
    core1 alone (c1), the rest net, the eye view.
  - Read: the combination V6Fr minus the unedited file, per person, on within 0, full 0 and runs 0, for v6f (gC5, L).
    Leverage = chain points per component point.
  - Cost: one run is about 2.5 min (6 bases × 3 arms). 19 components × 3 orders, two at a time: about 1.2 h, landing
    about 02:00 UTC. A SYN_D=0 smoke must reproduce the regular file exactly.
- **Prediction:** leverage per point: core1 (c1 / s0) about 0.15; the four fine-tuned SessionBN slots about 0.05 each; eye
  and rest about 0.03 (§188: eye +0.6 for 20 points); other slots ≤ 0.03.
- **Rule:**
  - Components with leverage ≥ 0.05 per point under at least two schemes go to stage 2.
  - Below 0.03 everywhere: "no lever on this baseline" for that component (output-level accuracy only; this does not
    close edits that change error correlation).
  - In between: undecided.
- **Stage 2 (after stage 1):** for the lever components, the gain weight edits can reach on the component alone.
  - core1: head edits from 30060d5^, restored to studies/14nov.
  - SessionBN nets: final-layer and per-block gains, GPU inference.
  - Measured in recording order on the component (member accuracy is nearly order-free), with edits fitted on inner
    people only.
  - Expected chain gain = leverage × reachable points.
- **Stage 3 (only if stage 2 shows ≥ +0.3 expected):** a nested search (selection on five people, scored on the other five,
  all 126 splits) whose evaluation is the shuffled chain over several orders. Weco may be the optimiser, with the cost
  per step stated. Any survivor must be expressible as changed weights only, riding on the checked package with a
  weights-file change and the identity checks.
- **§207 amended (00:28 UTC, coordinator: the user asked for an actual attempt; bars are reading aids).** Stage 2
  runs on the two components with the highest leverage whatever stage 1 shows, unless every component is at zero within
  noise. Reachable points are reported next to leverage, so the product is visible.
  - **Edit family 1 (exact, CPU):** final-layer edits. W' = A·W, b' = A·b + c with a 3×3 A whose rows sum equally, so the
    edit acts on the output logits as z' = A·z + c. It is computable from the outputs already re-streamed in delivery
    order: core1's head, a SessionBN family's classifier layer, the eye and rest nets' output layers. A and c are fitted
    leave-one-person-out (the held-out person never enters its own edit), objective: the component's own accuracy.
  - **Edit family 2 (what §188/§196 found to matter):** the same final-layer form with the objective changed to the
    error correlation with the rest of the mix and the within-run error streaks.
  - **Edit family 3, input-layer drift edit, eye net:** down-weight first-layer inputs whose features drift within
    sessions on training people. It needs the eye net's weights and a CPU re-stream.
  - Families 2 and 3 run for the eye net and one SessionBN family.
  - If the best product is small, it is reported with its number; the user decides whether a search is worth its cost.

**§207 STAGE 1 RESULT (14 Nov 00:41 UTC; runs/s207_leverage.txt; +5 points per component, within 0 / full 0 / runs 0, v6f).**
Chain points per component point (V6Fr; v5.1b's chain P0r in brackets), mean of the three orders:

| Component | Leverage, mean (within / full / runs) | v5.1b chain | Rule |
|---|---|---|---|
| s0, core1 + Hilbert pair | **0.181** (0.268 / 0.125 / 0.150) | 0.25 | stage 2 |
| c1, core1 alone | **0.113** (0.125 / 0.112 / 0.101) | 0.20 | stage 2 |
| s10, teacher | **0.078** (0.095 / 0.049 / 0.092) | 0.13 | stage 2 |
| rest net | **0.062** (0.081 / 0.045 / 0.061) | – | stage 2 |
| s3, s4, s9, s7 | 0.03–0.04 | 0.05–0.07 | undecided |
| s1, s2, s5 (SessionBN fb raw) | 0.01–0.02 | 0.04–0.08 | undecided |
| s6, s8 (fine-tuned SessionBN fb, fbl) | 0.017, 0.012 | 0.04 | no lever |
| eye view | 0.006 | – | no lever |

- s11–s15 failed loudly: the chain has 11 member slots (0–10), so there is nothing to measure there.
- **Reading:**
  - The prediction held for core1 (0.15 predicted, 0.11–0.18 measured) and for the eye (0.03 predicted, 0.006). It was
    high for the fine-tuned SessionBN families (0.05 predicted, 0.01–0.03) and low for the teacher.
  - The v6 additions dilute single-member levers: each member's leverage on the combination is about half its leverage on
    v5.1b's chain.
  - Core1 keeps its lever under all three schemes (0.10–0.13), and so does the teacher (0.05–0.09).
  - The eye view's accuracy is no lever. Its limit is error correlation (§188), which family 2 targets.
- **Stage 2 (started 00:45 UTC, runs/s207_stage2.txt, about 01:50 UTC):** families 1 and 2 on c1, s10, s0 and rest; family 2 on
  the eye view and s5. Each is chain-scored on within 0, full 0 and runs 0, and reported as reachable points next to the
  chain change.

**§204 RESULT (14 Nov 00:55 UTC; Leonardo, all of full 4–7 in about 1 h: members 30–40 min, not the 2–5 h estimated).** New
people over 8 full orders: v6f +0.94 [+0.76, +1.12], v6.1 +0.87 [+0.66, +1.07]. So v6.1 − v6f = −0.07; without P10 0.00 (v6f
+0.93, v6.1 +0.93). Per order (morning_table_0050.txt): see the line below. The rule (≥ 6/8 below zero and mean ≤ −0.2) is
not met (per order 0..7: −0.07 −0.37 +0.16 −0.29 +0.16 −0.12 +0.02 −0.04; 5/8 below zero). **The full-scheme cost of v6.1's labels is P10's alone** (P10 +0.99 → +0.26, min −1.60). J scheme-averaged with equal
weights: v6.2 +1.06, v6f + ERF +1.01, v6.2-safe +1.00; within only 1.17 / 1.07 / 1.05.
- **§207 in-net edit 1, core1 block gains (written 14 Nov 00:51 UTC, before any result).**
  - **Why core1:** its leverage is 0.11 per point, and 0.18 with the Hilbert member.
  - **Mechanism:** core1's head is StandardScaler + logistic regression over 52 feature blocks (17 bands × 3 windows + the slow
    block). A gain on a block's raw features would be undone by the scaler, so the edit is a gain on its head coefficients:
    z = Σₖ gₖ zₖ + b. `studies/14nov/coreblocks.py` re-streams core1 and saves every block's logit contribution in recording
    order and in delivery order (within 0/1, full 0/1, runs 0/1; Leonardo 64432182–87, about 40 min each). It asserts that
    the blocks reproduce the head to 1e-6.
  - **Fitting:** `coregain.py` fits 52 gains per held-out person on the other nine (ridge towards 1). Objective acc is the
    log loss; corr adds the error correlation with the other members and the within-run autocorrelation. It writes the
    edited member in delivery order for shufcombo (`CM_NPZ`).
  - **Shipping:** a survivor ships as rescaled head coefficients in core1's weights file, with no new code path.
  - **Scoring:** v6f's combination on the six orders, new people.
  - **Prediction:** §13 found learned block gains neutral (and the slow block raised about 1.5×). Here: held-out core1 accuracy
    +0.0 to +0.5 points, chain +0.0 to +0.1 (leverage 0.11 × reachable points).
  - **Rule:** report the product. A chain gain ≥ +0.1 with ≥ 4 of 6 orders up is followed by nested selection of REG and the
    objective over people.
- **Follow-up behind it (penultimate-layer gains):** the rest net (leverage 0.06) and the teacher (0.08). A re-stream saves
  each net's penultimate embedding, so per-unit gains (z = W·diag(g)·h + b) are exact on CPU. This needs restshuf.py and
  teachshuf.py to save h: 3 + 24 GPU tasks per order.
- **Eye-net drift edit (correlation-targeted, input layer):** the per-fold dev nets are not stored, so this needs LOSO refits
  with down-weighted drifting input channels: 9 GPU tasks per order plus the ocushuf re-stream. It is queued after the
  penultimate gains.

**§207 STAGE 2 RESULT, final-layer edits (14 Nov 01:00 UTC; runs/s207_stage2.txt; within 0 / full 0 / runs 0, v6f, new).**
Edit z' = t(I+B)z + c on a component's logits, fitted per held-out person on the other nine.

| Edit | Reachable (held-out accuracy, points) | Chain change (mean of 3 orders) |
|---|---|---|
| core1, accuracy form | −0.50 | −0.22 (+0.13 / −0.29 / −0.51) |
| teacher (s10), accuracy | −0.58 | −0.10 |
| rest net, accuracy | −0.37 | −0.14 |
| core1+Hilbert pair (s0), accuracy | −0.40 | −0.40 |
| eye view, accuracy | −0.76 | −0.07 |
| correlation forms, λ = 1 (c1, s10, s0, rest, eye) | −25 to −60 | −0.27 to −1.33 |
| SessionBN fb raw (s5), correlation, λ = 1 | −51 | +0.02 |

- **Reading:**
  - Fitted on nine people, an output-layer edit lowers the held-out person's accuracy for every component. The fitted heads
    are already at their best, and the fitted temperatures (0.25–0.4, overconfident nets) lower the chain when applied: the
    stages use the members' scale.
  - λ = 1 bought decorrelation by destroying the component, which is no test of the idea. It is rerun at λ = 0.1 (stage 2b,
    c1 / s10 / eye / s5).
  - **Product of leverage and reachable points for this family: negative.** Same verdict as §13 for core1's head, now on
    every lever component and on the shuffled chain.
- In-net edits (core1 block gains, rest-net filter gains) are running; they are a different family (per-block and
  per-filter directions, not one 3×3 map).

**§202 outcome (14 Nov 04:55 UTC).** The worker image cannot use Leonardo's GPUs. Leonardo's driver is CUDA 12.2 and the
image's torch is 2.14.1+cu130 ("NVIDIA driver too old", 64443706).
- The first attempt (64432181, 01:50) failed on a missing .md5 file (the chunk upload did not include it), and the Leonardo
  waiter did not report it. Three hours were lost.
- The job script now has DEVICE=cpu. The image's software is checked on CPU on the exact ZIPs (64443845, then the FT path
  64443846/47).
- GPU runs of the image need Galvani (v52-build's 3100010 / 3100013, readable after the user's 2FA).
- Morning table: side-session notes `morning_table_14nov.md`.
- **§207 stage 2b (05:05 UTC):** the correlation form at λ = 0.1. Held-out component accuracy: c1 −0.77, eye −1.18, s10 −1.30,
  s5 −0.62 points. Chain: c1 −0.25, eye −0.07, s10 −0.08, s5 −0.12. **Final-layer edits lose in both objectives on every
  component tried: the product of leverage and reachable points is negative.**
- Rest-net filter gains: the smoke found the final convolution slides over time (26 steps, then reduced). restblocks.py
  now decomposes per filter and per time step and checks which linear reduction reproduces the net. Smoke 2: 64443913.

**§202 FINDING (14 Nov 05:15 UTC): the v6.2 step has no exception guard, which blocks every current v6.2 ZIP.**
- **How it showed:** worker image on CPU, Leonardo 64443845. The GPU was visible but its driver too old. v5.1b passed
  (0.99328, files unchanged, 279 s). v6.2g crashed the official ingestion: torch 2.14's Adam health check
  (`_accelerator_graph_capture_health_check`) initialised CUDA in the first refit, and the RuntimeError reached benchopt.
- **The bug:** `step62` (submission.py line 1225 in v62_r2g.zip) runs eye, rest, tail6r, ft.step, mix6 and routeB after
  v5.1b's output with no try/except. v6.1's step (1283–1298) wraps the same calls and keeps v5.1b's output for the session.
  So on the worker, any rare error in FT or the routed tail fails the sealed run instead of degrading to v5.1b. The
  "errors 0" counter covered only the v6.1 path, so this was a silent no-op guard for v6.2.
- **Sent to v52-build:** guard step62 after out_v5 (count the error, return out_v5, stop v6.2 for the session, keep recognition
  bookkeeping), rebuild, and add a forced-exception check.
- The CPU rerun with the GPU hidden (no --nv, CUDA_VISIBLE_DEVICES="") is 64444264, then the FT path 64444265/66.
- My Leonardo waiter had run its ssh string in zsh without word splitting, so it never saw the 01:51 failure. Waiters now run
  under bash.

**§206 build addendum (14 Nov 05:08 UTC, v52-build; written before any check of these packages): v6.2h, the step62
exception guard.**
- **Finding** (ni26-push, Leonardo 64443845, worker image on the CPU with a GPU whose driver is too old): v6.2g crashed the
  official ingestion inside a refit (Adam's CUDA graph-capture check). `step62` had no exception guard, while v6.1's
  `step61` has one. This holds for every v6.2 / v6.2g / v6.2gL ZIP.
- **Fix** (`lean/src62h` → `submissions/lean_v62h`, `ccf16a94…`): everything after v5.1b's output and the recognition
  bookkeeping runs under try/except. An exception is counted in the [v6] line's errors and prints one stderr line, and
  that window and the rest of the session emit v5.1b's output. A non-finite output now also counts as an error. With
  no error, the code computes what v6.2g computes.
- **Packages:** `cand_v62_r2hL`, `r2h`, `r1hL`, `r1h`; every data file is byte-identical to the g / gL counterpart.
- **Predictions:**
  1. bitwise equal to v6.2g / v6.2gL on all 14 sessions (gates as shipped and off for r2, off for r1);
  2. forced exception at window 30 (ft.step or eye.step) on P01–P03: the run completes with errors 1, windows 0–29
     bitwise as the unforced run, and windows 30 on bitwise equal to the inner v5.1b and to the v5.1b package;
  3. official local replay at 0.99458 with files unchanged.
- **Refutation rule:** any miss fails the h package; the g ZIPs stay unsafe either way (no guard).

**§208 SILENT OVERWRITE FOUND AND CORRECTED (14 Nov 05:35 UTC).** shufcombo's output name did not carry OCU_DIR. Last night's
form B runs (OCU_DIR=runs/ocuttt/<order>/{base,pl,true}, gC5, tag L) therefore wrote over the regular v6f files of within 1–3,
full 0/1 and runs 0/1 on the Mac.
- **Found by** §207's no-op control: core1 gains fixed at 1 changed V6Fr by up to 3 points per person, with P0r unchanged.
- **Fix:** the name now carries any non-default OCU_DIR. The seven files are recomputed; they reproduce the original logs
  exactly, and the gains-1 run now equals them (max diff 0.0). The overwritten copies are in runs/overwritten_formB/.
- **Unaffected:** the morning table (it reads Leonardo's paired files for those orders; local files only for full 2/3 and runs
  2/3); every label comparison (LF / RF / BF / KF files); v6.2gL vs v6.2g.
- **Corrected** (runs/s208_recomputed.txt, s207_*_fixed.txt):
  - **§206, s15 form alone (s15 with L labels − v6f):** within **+0.24** (+0.29, +0.27, +0.32, +0.08; was +0.01); full −0.01
    (was −0.15); runs +0.05 (was −0.07). So the s15 form gains within and is neutral under full and runs. The label change
    (R − L) is unchanged: within +0.10, full −0.21, runs +0.03.
  - **v6.2gL's new-people path (s15 + L):** equal to v6f under full (+0.94 / +0.94), +0.24 under within; against v6.2g: within
    −0.10, full +0.21, runs −0.03. J on the 8 orders is unchanged: v6.2gL +1.09, v6.2g +1.08 (v6f +0.52, was +0.60).
  - **§200 blend − R:** unchanged (no v6f file used).
  - **§207 stage 1 (leverage), full 0 / runs 0 corrected:** s0 0.193, c1 0.125, s10 0.090, rest 0.074, s5 0.035, s6 0.028,
    eye 0.017. The rule verdicts are unchanged except s6 (now undecided).
  - **§207 stage 2 (final-layer edits), corrected means:**
    - c1 acc −0.17, c1 corr0.1 −0.20, s0 acc −0.35, rest acc −0.08;
    - s10 acc −0.04, s10 corr0.1 −0.02, s5 corr0.1 −0.06, eye −0.01.
    - The verdict holds: none positive.
  - **§207 core1 block gains** (6 orders, against the clean files): REG 0.1 **+0.04** (4/6 up; within 1 +0.17, runs 0 +0.09);
    REG 1 +0.00; corr λ 0.1 +0.03 (4/6). Held-out core1 accuracy −0.14 / +0.19 / −0.25 points. The pre-registered bar
    (+0.1 with ≥ 4/6 up) is not met, so the reading is "undecided, small".

**§202 worker image on CPU (14 Nov 06:07 UTC; Leonardo, GPU hidden; official ingestion and scoring; torch 2.14.1+cu130,
braindecode 1.8.1).**

| ZIP | Result |
|---|---|
| v6.2g ERF 90a62470… (pre-guard) | 0.994579, files unchanged, 2,786 s (FT on CPU) |
| v6.2hL ER 2f53c3f9… | 0.994579, files unchanged, [v6] errors 0, 288 s |
| v6.2h ER 9bf9f6e4… | 0.994579, files unchanged, [v6] errors 0, 268 s |
| v6.2hL ERF e22f67e7… | 0.994579, files unchanged, 2,786 s; [v6] 17 refits / 340 steps, errors 0; one session frozen 48 by the FT wall-clock budget on CPU (guard as designed) |
| v6.2h ERF 3b13ff96… | 0.994579, files unchanged, 2,759 s; [v6] 17 refits / 340 steps, errors 0; one session frozen 38 by the CPU budget (guard) |

GPU runs of the image need Galvani (3100010 / 3100013).
- **§207 in-net edit 2, rest-net filter gains (06:46 UTC; runs/s207e_restgain.txt).** 40 per-filter gains on the final layer, per
  seed, leave-one-person-out. Each edit is scored against the gains-1 member of the same re-stream.
  - Chain, 6 orders: acc −0.05 (2/6 up), corr λ 0.1 −0.05 (2/6 up).
  - Held-out rest-net accuracy: −0.3 to −1.1 points.
  - Negative, like the final-layer edits.
- **§207 summary so far (corrected per §208):**
  - final-layer edits on every lever component: ≤ 0;
  - core1 block gains: +0.04 (4/6; below the +0.1 bar, undecided);
  - rest-net filter gains: −0.05.
  - On the current chain, edits fitted on nine people do not transfer to the tenth. The fitted weights are at their transfer
    optimum, and only core1's block gains come out slightly positive. Remaining untried: teacher per-unit gains and the
    eye-net drift edit (GPU refits).

### 209. v6.2hL's dev row on every order (written 14 Nov 06:49 UTC, before any job ran)
- **What:** new people under v6.2hL = s15 fine-tuned on v5.1b's labels (tag LF), paired against the in-run gC5 control on the
  same labels (= v6f's path), on the same machine.
  - within 4–11: these exist from the fresh-order sweep on Leonardo (sweepsel_fresh).
  - full 2–7 and runs 2–3: ttt18 FORMS=gC5,s15 on chain_v51b (24 tasks per order), scoring chained on Leonardo.
  - Returning people: as already scored (ERF).
- **Prediction:** s15 − gC5 under L labels: within +0.15 (fresh sweep: +0.14), full 0.0, runs +0.05. So v6.2hL ≈ v6f + ERF
  + 0.15 under within, and level under full.
- **§207 last two edits (written 06:53 UTC, before any result; coordinator: run them to a number):**
  - **Teacher input gains (Leonardo features 64448904, then local).** The teacher is a logistic regression on 6 features
    (seed-mean head log-probs of two families × 3 classes), refitted exactly per held-out person. Six gains on its input
    columns are fitted leave-one-person-out (REG 0.1 and 1) and scored against the gains-1 refit on 6 orders.
    Prediction: −0.05 to +0.05.
  - **Eye-net drift edit (Galvani, input layer, correlation-targeted).** After each fold's training, the first-layer kernels
    of every trace channel are scaled by g_c = clip(1 − α · drift_c / max drift, 0.3, 1); drift_c is the within-session
    lag-1 autocorrelation of window means on the fold's training people. α 0 (paired baseline), 0.5 and 1; 3 seeds × 6
    orders. Smoke on Galvani 3100145. Prediction: about 0 (eye leverage 0.017 per point; §188 found the correlation lever
    in TSEM).

### 210. RUN OFFSETS IN THE TEMPLATE VIEWS, ORACLE (written 14 Nov 06:55 UTC, before any result; panel BCI §2.1)
- **Claim tested:** within a session the run explains 11–31 % of the log-covariance variance, and every public run is
  class-balanced. So subtracting a shrunk run offset from each view before SessionEM and TSEM should sharpen the templates.
  Panel: +3.0 with true run ids and true templates in a simplified setting.
- **Test:** shufcombo `RUNOFF=κ RUNOFF_V=pc|both` with true run ids (session + task|run), causal in delivery order:
  v − j/(j+κ)·(run prefix mean − session prefix mean). Templates and covariance stay pooled over the session. κ = 5;
  views pc128 (TSEM) alone and pc128 + GE (SessionEM too); within 0/1, full 0/1, runs 0/1. Read v6f's combination (V6Fr)
  and v5.1b's chain (P0r) against the regular files, new people.
- **Prediction (panel):** +0.3 to +1.0 if runs are contiguous (within, runs), 0 to +0.4 under full.
- **Kill (panel's rule, adopted):** below +0.4 with true run ids, on the mean over the six orders. Then a label-free run
  recovery is not built.

### 211. MEMBER RANKING: POST-BALANCE vs RAW ACCURACY (written 14 Nov 06:57 UTC, script written, result not yet read; ML panel §3)
- **Claim tested:** "member accuracy barely moves the chain" is an artefact of raw accuracy, which mixes session bias
  (removed for free by rebalancing) with single-trial skill. The panel measures the nets' mix at 67.2 raw → 72.0
  post-balance.
- **Data:** the fine-tuning form sweeps (LS, LS3, LF, KF): each form changes the four fine-tuned SessionBN members, with a
  chain delta against the in-run gC5 control already scored (V6Fr, new people). Per (order, form), the members'
  raw and post-balance accuracy against gC5's (`studies/14nov/member_rank.py`).
- **Rule:** if corr(Δchain, Δpost-balance) exceeds corr(Δchain, Δraw) by ≥ 0.2 with order means removed, member variants
  are ranked on post-balance accuracy from here and the closed member variants are re-opened under it. Otherwise
  "not supported on this data".
- **Limit:** these variants differ in fine-tuning, not in member architecture or training; the claim about trained member
  families is tested by the episodic-training experiment (GPU), not here.
- **§211 RESULT (06:58 UTC; runs/s211_member_rank.txt; 240 (order, form) pairs from the LS / LS3 / LF / KF sweeps).**
  - Order means removed: corr(Δchain, Δraw) **+0.52**, corr(Δchain, Δpost-balance) **+0.55**. Pooled: +0.37 / +0.47;
    Spearman +0.30 / +0.43.
  - The gap (0.03) is far below the pre-registered 0.2: **not supported on this data.** Post-balance accuracy predicts the
    chain marginally better, not decisively.
  - Side finding: fine-tuning forms cost more raw accuracy (−0.80 points mean) than post-balance accuracy (−0.19). They
    shift session biases, which the rebalancing removes.
  - Chain per post-balance point: +0.11, consistent with §207's leverage of the fine-tuned slots.
  - The claim about member families trained inside their adaptation stays open for the episodic-training test.

### 213. CLASS-SEQUENCE-PRESERVING SHUFFLE, DIAGNOSTIC (written 14 Nov 06:58 UTC, before any result; solution-space review C1)
- **Question:** is v5.1b's stage loss under full (−3.75, stages only, members fixed; runs/shuffle_13nov.log) lost sequence
  structure (the class sequence) or EEG non-stationarity (temporally adjacent windows look alike)?
- **Test:** shuffle.py scheme classseq. Within each session, windows of one class trade places among that class's positions,
  so the class sequence stays and the temporal neighbourhood of the EEG is destroyed. 3 permutations, six bases, new people.
- **Reading:** a loss near full's −3.75 means the loss is EEG non-stationarity, which a stage that tracks drift by content
  could recover. A loss near 0 means the stages lean on the class sequence, which a real shuffle removes for good.

**§210 RESULT (14 Nov 07:10 UTC; runs/s210_runoff.txt; true run ids, κ 5, new people, change against the regular file).**

| Views | within 0 / 1 | full 0 / 1 | runs 0 / 1 | mean v6f (v5.1b) | without P10 | people up |
|---|---|---|---|---|---|---|
| pc128 + GE | +0.27 / +0.22 | +0.87 / +1.25 | −0.02 / +0.58 | **+0.53** (+0.67) | +0.47 | 9/10 |
| pc128 only | +0.20 / +0.10 | +0.69 / +0.80 | +0.00 / +0.33 | +0.35 (+0.45) | +0.31 | 8/10 |

- **Passes the kill rule** (+0.4) for both views: the oracle lever is real in the real stages.
- The gain is largest under full, where true run ids carry the most information the stages otherwise lack. But under full,
  run membership is not recoverable without labels (panel: purity 0.36–0.52). The reachable part is within and runs,
  +0.2 to +0.6.
- Next: the 2 × 2 with the run-level count filter (panel round 2), then inferred boundaries.

**§213 RESULT (07:10 UTC; runs/s213_classseq.log; stages only, members fixed, 3 permutations).** Under classseq, v5.1b's
stages lose −3.08 (rebal −0.16, SessionEM −1.64, TSEM −3.08), against −3.75 under full. **About 80 % of the full-shuffle loss
is EEG non-stationarity** (temporally adjacent windows resemble each other), which a content-based stage could in principle
track. About 0.7 is the class sequence itself, gone for good under a real shuffle.

**Returning-people rows marked (bugs_scoring.md P0-a / P0-b, 07:10 UTC).** Every returning-people number with the eye view
under shuffle (retv6 arms E / ER / ERF; §183, §190, §198, §199, §201, §203, §206, the v6.2 / v6.2-safe / v6.2hL rows of the
morning table) is **"eye order mismatched, members as recorded"** until re-scored:
- (a) ocushuf.py standardised every session in the all_loso permutation and wrote those outputs for seen_session too,
  while the stages used seen_session's own permutation.
- (b) retv6 / shufpkg keep core1 as member, the frozen SessionBN part, Hilbert, teacher, GE event rates, pc128 and the gate
  and personal-head inputs as recorded for returning people. Arm F also lacks the frozen-net re-stream change that
  shufcombo applies for new people.
- New-people numbers are unaffected.

### 214. RETURNING PEOPLE RE-SCORED WITH BOTH P0 FIXES (written 14 Nov 07:07 UTC, before any result)
- **P0-a:** ocushuf.py uses each protocol's own delivery order. Windows outside it keep all_loso's, so the trained net is
  unchanged. Every output stores `order_sha`; retv6 requires it to match seen_session's permutation, and shufcombo checks
  it where present. Local smoke: hash matches.
  - Re-stream: seen_session only, 9 seeds, OUT_SUB=ocushuf_ss. Galvani for full 0–3 and runs 0–3; Leonardo for within 0–7.
- **P0-b:** retv6 `RT_MEMBERS=1` adds, after the permutation, the seen_session re-stream changes of core1 (as member and
  behind the gate), Hilbert, teacher, pc128, GE event rates and the frozen SessionBN part (ttt18 base − ttt7 base, per
  base's seeds), as shufcombo does for new people. Each file holding an order must exist (no silent recording order).
- **Re-score:** retv6 GATE_V51B=1 FT_SFX=QF FT_FORM=s15 on within 0–7, full 0–3, runs 0–3, giving arms ER / ERF (v6.2-safe
  / v6.2 / v6.2hL returning rows).
- **Prediction:** the eye fix moves ER / ERF by up to ±0.3 under full and runs (the orders differed most there), little under
  within (rank correlation +0.97). The members' re-stream costs routed v5.1b about 0.6 beyond the stages (as for new
  people), but ER / ERF are differences against the same re-streamed base, so they move less.

### 215. THE 2 × 2: RUN OFFSETS × RUN-LEVEL COUNT FILTER, TRUE RUN IDS (written 14 Nov 07:09 UTC, before any result; panel round 2)
- **Arms:** run-offset removal in the stage views (§210: κ 5, pc128 + GE; off / on) × a run-level count filter on the final
  posteriors (off / on). The filter (`studies/14nov/runcount.py`) treats each run as an urn of 10/10/10 or ≤ 12 per class.
  It runs an exact causal forward filter over the class counts in delivery order, with posterior^τ as the likelihood
  (τ 2) and a soft floor of 0.5 per class. It acts on the emitted output only; gates and FT labels are untouched.
- Within 0/1, runs 0/1, full 0/1; v5.1b (P0r) and v6f (V6Fr); new people first (returning after §214). The error
  run-clustering ratio is reported for every cell.
- **Panel prediction:** filter with true runs on v3's stored posteriors +0.30 (recording) to +0.68 (full, within).
  Together, 0.7–1.3 × the larger gain, not the sum.
- **Kill (panel):** below +0.3 on both contiguous schemes with inferred boundaries, or below −0.1 under full with the
  detector in the loop. This oracle (true runs) sizes the ceiling; the inferred-boundary version decides.
- **§207 teacher input gains (07:11 UTC; runs/s207f_teachgain.txt).** 38 gains on the teacher's input columns, fitted
  leave-one-person-out, against the gains-1 refit on 6 orders: REG 0.1 −0.00 (3/6 up), REG 1 −0.00 (3/6). **Null.** The
  eye-net drift edit (Galvani) is the last open family.

### 218. RUN-AWARE VIEW WITHOUT RUN IDS: THE BETWEEN-RUN SUBSPACE PROJECTED OUT (written 14 Nov 07:15 UTC, before any result; coordinator after §210)
- **Why:** §210's oracle is largest under full (+0.87 / +1.25), where run ids cannot be recovered. A form needing no run id
  and no order: estimate on the training people the subspace spanned by between-run offsets (run mean minus session mean,
  classes centred within session) and project it out of the views the template stages get. It is a fixed matrix (in the
  package, one from all ten people).
- **Test:** shufcombo `NUIS=k NUIS_V=pc|both`, projector per held-out person fitted on the other nine. k 4, 8, 16; pc128
  and pc128 + GE (GE's own columns). Within 0/1, full 0/1, runs 0/1, v6f. Read the share of §210's oracle it recovers
  per scheme.
- **Prediction:** k 8, both views: +0.2 to +0.4 under full (30–50 % of the oracle), +0.1 under within.
- **Rule:** keep it for a package form if it reaches ≥ +0.3 mean over the six orders with people up ≥ 7/10 and is not
  negative under any scheme mean. Undecided between +0.1 and +0.3.

### 217. RUN DETECTOR AND INFERRED BOUNDARIES (written 14 Nov 07:17 UTC, before any result)
- **Question:** how much of §215's true-run gain (count filter +0.43 to +0.74 within; offsets +0.2 to +0.3) survives
  with run labels inferred label-free in delivery order. Code `studies/14nov/rundetect/`, labels `runs/runids/`,
  diagnostics `runs/rundetect/`. CPU only.
- **Nuisance view:** per window, log-variance per channel in 6 bands (4–8, 8–13, 13–20, 20–30, 30–45, 45–58 Hz; 41 ch).
  Projection learned leave-one-person-out on the other people's TRUE runs: session-centred features, 12 generalized
  eigenvectors of between-run vs within-run scatter (within-run whitened). Run lengths prior from the other people.
- **Detector:** per session, at window W of the delivery, ratio = mean distance of consecutive windows / mean distance of
  all pairs among the first W (projected). Threshold chosen on the other people (balanced error, full vs contiguous).
  Reported per scheme (recording, runs, within, full; seeds 0–11) at W 40/60/100/150, latched decision at W 40.
- **Boundaries (contiguous):** online Bayesian change point (run-length posterior, hazard from the length prior,
  conjugate Gaussian run means around the causal session mean); a window's label is fixed when it arrives (causal file
  `bocpd`). Offline length-constrained DP segmentation as a non-causal reference only (`dpoff`, never a gate).
- **Full shuffle:** online sequential mixture, at most 6 blocks, capacity-weighted, conjugate means (`mix`).
- **Detector in the loop (`loop`):** before W 40 the BOCPD labels (fixed default); from W 40 BOCPD if the latched
  detector says contiguous, else singletons (filter inert). Every order uses this one file.
- **Prediction:** detector error ≤ 5 % at W 40 on every scheme; online purity ≥ 0.85 (ARI ≥ 0.75) within and runs;
  mixture purity under full ≈ 0.5 (ARI ≈ 0.2). Count filter (runcount.py, τ 2) with `bocpd`/`loop` labels: +0.25 to
  +0.5 on within and runs (half to two thirds of the true-run gain); `loop` under full −0.05 to +0.05; `mix` under full
  −0.2 to +0.1.
- **Kill (panel's rule, adopted):** below +0.3 on both contiguous schemes with inferred boundaries, or below −0.1 under
  full with the detector in the loop. Between +0.0 and +0.3 on one contiguous scheme: undecided (noise ±0.3).
- **Known limit, stated now:** with inferred groups runcount.py uses the 12/12/12 cap for every group (it cannot know
  the 10/10/10 online design), so part of a gap to §215 is the cap, not the boundaries.

### 219. TSEM'S SHRINKAGE TARGET = POPULATION WITHIN-CLASS COVARIANCE (written 14 Nov 07:16 UTC, before any result; panel BCI §2.2)
- **Claim:** TSEM shrinks its seed covariance 0.7 towards scaled identity, so run offsets enter as isotropic noise. A target
  that contains the population's within-class, within-session covariance (with the between-run offsets inside it) should
  sharpen the templates under any order.
- **Test:** `TSEM_TGT=β` (studies/14nov/stagehooks.py). Target = (1−β)·identity + β·Σ_pop, where Σ_pop comes from the
  session-standardised pc128 of the other nine people, class-centred per session, scaled to the identity's trace. β 0.5
  and 1. Six orders, v6f, new people. 256 components are not tested here: they would need a re-streamed pc256 view.
- **Kill (panel):** below +0.3 with the person interval above 0; between 0 and +0.3, undecided.

**§215 RESULT (14 Nov 07:20 UTC; runs/s215_runcount_tau{2,1}.txt; true run ids, new people, change against offsets off /
filter off).** τ 2 (τ 1 similar). Means over six orders:

| | filter only | offsets only | both |
|---|---|---|---|
| v6f (V6Fr) | +0.49 | +0.53 | **+0.70** |
| v5.1b (P0r) | +0.64 | +0.67 | +0.79 |

- **Per order, v6f filter / offsets / both:** within 0 +0.43 / +0.27 / +0.68; within 1 +0.59 / +0.22 / +0.37; runs 0 −0.33 / −0.02 /
  −0.03; runs 1 +0.52 / +0.58 / +0.67; full 0 +0.78 / +0.87 / +1.21; full 1 +0.92 / +1.25 / +1.30.
- **Reading:**
  - Both levers are real with true run ids, and they overlap: together about 1.3 × the larger alone (the panel expected
    0.7–1.3 ×).
  - Under within the filter carries more; under full the offsets do.
  - runs 0 is the exception: the filter costs v6f −0.33, also without P10. It needs a look before any package form.
  - The error run-clustering ratio stays about 4 in every cell.
- **What decides:** the same with inferred boundaries (§217, running) under the contiguous schemes, and under full with the
  detector in the loop (the filter and hard offsets must then be inert).
- Corrected returning people (§214): within 0 ER +0.61, ERF +0.84 (both people up); the other 15 orders are running.

### 216. EXTERNAL HARNESS v1 (written 14 Nov 07:22 UTC, before any result; ML panel §2.2)
- **Question:** a second instrument for ranking stage changes the ten dev people cannot rank: the chain's adaptation
  stages on public corpora with many people. Code `studies/14nov/extharness/` (corpus.py, harness.py), results
  `runs/extharness/`. CPU, Mac.
- **Corpora (3 balanced classes; the exports of `scripts/export_moabb.py`, copied from Leonardo `$WORK/ni26/data/` into
  gitignored `data/`):** PhysionetMI, left hand / right hand / feet (109 people, one session, about 22 per class; its
  runs are class-pure: l/r runs and feet runs, unlike PROTEUS' 10/10/10 runs). Schirrmeister2017 (HGD), left / right /
  feet (14 people, one session of two runs, classes mixed inside runs, about 700 windows per person). Scherer2015 (the
  PROTEUS tasks, 9 people, two sessions, no run ids: within = full and runs = rec there) as a small third column.
- **Member (stand-in for core1):** CAR, bands 4–8/8–13/13–20/20–30 Hz, covariance shrunk 0.05, log-Euclidean tangent
  vectors; training sessions centred by their own mean, test sessions by a causal running mean in delivery order (prior
  5 windows at 0); standardised; multinomial logistic regression, C 0.05 fixed a priori. Leave-one-person-out.
- **Stages:** `gm.M.rebal` → `gm.gate` → `gm.em_g` → `gm.tsem_g` from replay/gm.py, called unchanged through a new store
  key `ext`. Substitutes: TSEM view (pc128) = 128 PCs of the member's standardised tangent features (PCA on the fold's
  training people); SessionEM view (GE: gaze + ocular events) = the first 48 of those PCs (no gaze part). Asserts: stages
  equal `gm.pipe_g` at W 30; W 15/60 change the rebalanced output; orders are permutations.
- **Orders:** rec, and full / runs / within (shuffle.py's definitions) × 3 seeds per test session. Person-equal balanced
  accuracy (per person × session cell, then the mean); 80 % intervals from 2000 person resamples.
- **Predictions:** member alone (rec) PhysionetMI 50–60, Schirrmeister 60–75, Scherer 50–65. Rebalancing − member
  +1 to +4 on every corpus and scheme. SessionEM − rebalancing −0.5 to +1.5; TSEM − SessionEM 0 to +2 (generic views
  are weaker than GE / pc128). Rebalancing window: W 30 within 0.3 of the best of {15, 30, 60} on Schirrmeister in
  every scheme; on PhysionetMI W 15 ahead under rec / runs / within (class-pure runs) and W 30–60 under full.
- **Rule (W check):** "W 30 not near optimal there" only if W 15 or W 60 beats W 30 on the final output by more than
  0.3 with the 80 % person interval above 0, on both large corpora (PhysionetMI, Schirrmeister), under at least 2 of
  the 3 shuffle schemes. One corpus only, or inside ±0.3: undecided / near optimal. Either way it is a pointer for a dev
  check in the chain, not an adoption. A corpus where the member is below 45 is reported but not used for the rule.

### 222. SUB-8 HZ VIEW (written 14 Nov 07:23 UTC, before any result; panel BCI §2.3)
- **Claim:** class variance sits mostly below 8 Hz, but a 41 × 41 covariance of a 3–4 Hz wide band over 4 s has only about
  25–30 degrees of freedom, so its log map mostly sees the shrinkage floor. Reducing each sub-8 band to 12–16 spatial
  components (fit on the training people) before the log, optionally with an ERP-prototype augmented covariance
  (population class-average slow waveforms stacked on the window), gives a well-conditioned slow view for TSEM.
- **Test (CPU, studies/14nov/views/, runs/views/):**
  1. Per band (1–4, 4–8, 8–13, 13–20, 20–30, 30–45 Hz): CAR, band-pass, 41 × 41 trace-shrunk covariance (0.1), log-Euclidean
     tangent at the session reference (training sessions whole-session, test causal with the fold prior at k 2, as core1;
     recording order), scaler + LR C 0.3; all_loso (10 people) and seen_session; cell-mean BA per person.
  2. The view: bands 1–4 and 4–8, m ∈ {12, 16} components per band (multiclass CSP in the session-whitened space, PCA as the
     alternative), m × m covariance, tangent; with and without the ERP-prototype augmentation. Two reference modes: causal
     session recentring (order-dependent, needs delivery-order re-streams) and none (order-free).
  3. As a template view: the replay's tsem_lik on the view with (a) one-hot true labels (template separability ceiling) and
     (b) the chain's TSEM input; and the recording-order chain (A.system, check.py OK) with pc128 replaced and with the
     view appended to pc128. Reference: pc128 under the same measures; core1 member alone 69.3 new.
- **Prediction:** 1–4 and 4–8 Hz are the two best of the six bands, each 50–62 %; the bands above 8 Hz 42–55 %. The reduced
  sub-8 view alone 55–63 % new (below core1); reduced ≥ full-41 sub-8 by 0 to +3; ERP augmentation 0 to +2. Recording-order
  chain: replacing pc128 −1.0 to 0; appended to pc128 −0.2 to +0.3.
- **Kill (panel):** view alone below 55 % new → closed as a view. Streak correlation with the eye net: the phi of the two
  correctness vectors above 0.5 is flagged as the panel's risk. **Chain-test rule:** the view goes to the parent's shuffled
  chain (shufcombo hook) only if it is at least 55 % alone and either its template ceiling (3a) beats pc128's or the
  appended recording-order chain gains ≥ +0.3 J; 0 to +0.3: undecided (noise ±0.3), stated as such.

### 223. BAGGED MEMBERS (written 14 Nov 07:23 UTC, before any result; statistics panel proposal 5)
- **Claim:** training a member on random subsets of the training people and averaging puts the members' diversity where
  the error is (people). For core1 (convex LR on fixed features) the only diversity is the person subset.
- **Test (CPU, studies/14nov/bagging/, runs/bagging/):** core1 features (car_ov17_fs2_t0.1_le_k2, causal test references with
  the full fold's priors), head lr0.3, all_loso and seen_session. Per fold: the full fit (must reproduce the stored member,
  correlation ≥ 0.99), 8–9 leave-one-person-out bags and 8 random leave-two-out bags, averaged (arithmetic). Member alone,
  then the recording-order chain with core1 replaced (Hilbert slot hb kept as stored).
- **Prediction:** member alone bagged − full −0.5 to +0.5 new (each bag loses about 0.5–1 point from one person fewer;
  averaging wins it back, not more); chain J −0.2 to +0.2.
- **Kill (statistician's):** the bagged member not above the full fit at equal member count → closed for core1 on CPU.
  Within ±0.3: undecided. This does not close the net version (different model class, seed variance is the
  baseline there); the GPU version is proposed with its cost either way.
- **§217 addendum (14 Nov 07:24 UTC, labels frozen, no chain score read yet).** Changes against the entry, all decided on
  label-side numbers only (runs/rundetect/detect.txt, segment.txt): projection 6 directions, ridge 1.0 (best held-out
  same/different-run distance ratio, 0.73, of 8 settings); the plain consecutive ratio cannot see 'within' before the
  first run ends (50 % error at W 40), so the loop uses a second statistic, 'lag' (pairs ≤ 3 apart / pairs ≥ 15 apart):
  error at W 40 recording 7 %, runs 4 %, within 26 %, full 11 %; at W 60 0 / 1 / 7 / 3 %. BOCPD labels: a new label when
  the posterior mass on a later run start exceeds 0.5. Added arms: loop60 (latch at W 60) and loopM40/loopM60 (mix labels
  instead of singletons when the detector says full). Pre-registered loop = loop40; the others are reported as extra
  forms, not swapped in. dpoff stays a non-causal reference. Note §215's true-run filter gains +0.6 to +1.2 under full,
  so the full-scheme ceiling is large; the 6-direction view separates runs poorly when shuffled (nearest TRUE run mean
  purity 0.56), so mix is expected near zero.
- **§215 detail (07:24 UTC; runs/s215_detail_tau2.txt / .npz; 80 % person intervals).**
  - The offsets do not change the posteriors' sharpness: mean top posterior 0.608 → 0.610 (v5.1b), 0.626 → 0.628 (v6f) on every
    order. The filter's temperature stays valid with offsets on.
  - The apparent negative interaction for v5.1b under within (filter +0.74 [+0.44, +1.04], both +0.38 [−0.21, +0.96]) lies inside
    the person interval: noise, not an interaction.
  - Single-tool cells exclude 0 on most orders. Exception: runs 0, where for v6f the filter is −0.33 [−0.73, +0.05] and
    offsets −0.02.
- **§214 corrected returning people, first orders:** within 0 ER +0.61 / ERF +0.84; within 1 +0.30 / +0.29; within 2 +0.47 / +0.78.
  Before the fixes: within 1 ERF +0.19, within 2 ERF +1.23.

### 220. EPISODIC SESSIONBN FAMILY (written 14 Nov 07:31 UTC, before any result; ML panel 14 Nov §2.1)
- **Mechanism:** the nets carry a per-session class bias that the chain's rebalancing removes anyway, so they spend
  capacity on something free at test time. Train one SessionBN family inside the adaptation it is deployed with:
  each batch is one episode of ONE training session (21 rows per class, 63 rows, about the stock 64; episodes of all
  sessions in random order), SessionBN normalises with the episode's own statistics (chunk = episode), the logits get a
  per-class bias b (sum 0, two free parameters) from 8 unrolled fixed-point steps b ← b − log mean softmax(z + b), so
  the episode's mean prediction is uniform, then the recipe's cross-entropy on z + b. Epochs, lr, augmentation, k 20 and
  the test-time forward are the stock ones: drop-in weights, no run-time cost. Class-stratified episodes make the uniform
  target exact (the sessions are 10/10/10 and 32/32/32 by design); stock training is unstratified.
- **Family:** fbraw (slot 5, fbcnet_raw_sbn), seeds 0–5, all LOSO folds and seen_session. Arm E (episodic) and arm E0
  (the stock recipe through the same runner: paired retraining control and no-op check, its recording-order output
  must equal ttt7_fbraw_base of the same seed up to GPU nondeterminism). Code `studies/14nov/episodic/`
  (episodic.py trains and streams, wave.sh packs smoke + 12 tasks into one Galvani job, alone.py, chain.py).
  Each fold's net is streamed frozen (causal SessionBN) in recording order and in within 0/1, full 0/1, runs 0/1;
  files `ttt18<order>{E,E0}_fbraw_base-<proto>-s<seed>.npz`, scored by shufcombo with FT_V=base (fb, sinc, fbl
  linked to the stock base of the same order), package gate, eye net re-streamed, against the stock tag ttt18<order>
  (recording order: ttt7, the shipped slot).
- **Readouts:** (1) family alone, seed-mean posteriors, person-equal cell BA after label-free whole-session balancing,
  new people, recording order and the mean of the six delivery orders; (2) chain, new people, P0r (v5.1b with slot 5
  replaced) minus stock P0r, mean over the six delivery orders, on seed sets B2_s012 and B2_s345, with and without P10;
  V6r (eye view + rest, no FT) as a second column; E0 − stock as the retraining-noise reference.
- **Prediction:** family alone after balancing E − stock +0.5 (−1 to +2) new; raw (unbalanced) E − stock lower than
  that, possibly negative (the net no longer learns the population bias). Chain P0r E − stock +0.1 (−0.3 to +0.4) per
  seed set; E0 − stock within ±0.2.
- **Kill (panel's, adopted):** the family alone after balancing rises less than +1 point, or the chain gains less than
  +0.3 on two seed sets without P10 → not adopted in this form. Per AGENTS a chain delta inside ±0.3 is recorded as
  "undecided" for the direction (prototype step, other families, both stratified and unstratified episodes remain open);
  it is closed for this form only if the chain is ≤ −0.3 or the balanced family-alone change is ≤ 0. If E0 − stock is
  as large as E − stock, the difference is read as retraining noise, not as the mechanism.

### 221. FT PREFIX RELABELLING AND EVEN SAMPLING (written 14 Nov 07:33 UTC, before any result; performance panel 14 Nov P3)
- **Mechanism:** the s15 refits train on the whole prefix with each window's label as emitted when it arrived, and full
  batches visit early windows far more often (window 0 about 85 steps, window 175 about 5 in a 190-window session), so
  the oldest, worst labels weigh most. (a) Relabel: at the refit at t, every window 0..t-1 takes the label chain's
  output under its state at t. (b) Even visits: every window in the prefix has the same cumulative visits.
- **What the label files allow:** `runs/labels/chain_*_<order>.npz` hold one output per window, as emitted when it
  arrived (shufcombo LAB = stage output in delivery order). The chain's output for an old window at a later state is not
  in them. `studies/14nov/ftrelabel/relabel.py` reruns the label chain (shufcombo LABEL_OUT, LABEL_ARM=V6r, GATE_P0 =
  chain_v6r = v6.1's labels K for new people) with hooks on gm.em_g and gm.tsem_g: at each s15 refit time (t = 20, 30,
  …) it applies EM's and TSEM's state at t (fitted on windows < t exactly as the stock stage, gate of the window at t)
  to all earlier windows; the stage inputs (rebalanced mix) stay as emitted. Causal. Checks: the state applied to window
  t reproduces the stock output for window t (1e-9); the as-emitted labels agree with chain_v6r (argmax 0.984 within 1,
  ≥ 0.998 elsewhere: today's chain carries s214's eye-order fix, so every arm takes its labels from the new file).
  Label accuracy, relabelled against as emitted over the same windows, refits at t = 60–180: about +0.5 to +1.5 points
  on most orders, −0.6 to +0.8 on within 1 and 3. Files `runs/labels/relab_v6r_{within0-3,full0,full1}.npz`.
- **Arms** (`studies/14nov/ftrelabel/ftrel.py`, s15 hyperparameters, new people only, tag `ttt18<order>RP`, in-run
  control s15 on the same labels; one Galvani array of 24 tasks per order): r15 relabel; e15 even visits = loss
  weights water-filled so that cumulative weights are level after every refit, budget t per refit (today's batch size,
  in expectation; first refit uniform); b15 both (the proposal); q15 even visits at a fixed budget of 20 (a fixed batch
  of 20 in expectation; drifts to the newest windows). Weights replace drawn batches (same expected gradient; SessionBN
  statistics, KL anchor and cost unchanged). Smoke: control s15 must equal ttt7.stream's s15 (≤ 1e-4).
- **Scoring:** shufcombo (OCU_DIR=runs/ocushuf/<order>, GATE_P0=1, FT_TAG=ttt18<order>RP, FT_V=<arm>, REC_PFX=ttt7,
  T6_DIR=runs/ttt/probs7, T17_DIR=runs/ttt/probs18), line "combination over v5.1b, both shuffled", new people, each arm
  minus the in-run s15 on within 0–3 and full 0/1; RF/KF s15 rows of §198/§200 alongside as references.
- **Prediction:** b15 − s15: within mean +0.15 (−0.1 to +0.4), full mean +0.1 (−0.2 to +0.4). r15 about +0.1, e15 about
  0 (±0.2), q15 −0.2 (−0.6 to +0.1; recency, cf. §161's cap −0.05/−0.10).
- **Kill (panel's, adopted) for b15:** under +0.15 over s15 within (mean of 0–3), or negative on full (mean of 0/1) → not
  adopted. Per AGENTS a delta inside ±0.3 is recorded "undecided" for the direction (other forms: soft relabelled
  labels, relabel with the FT chain itself, sampled batches), not closed; closed for these forms only if ≤ −0.3.
- **§218 RESULT (07:40 UTC; runs/s218_nuis.txt).** Projecting the between-run subspace out of the template views loses
  everywhere, and more with more directions. Both views: k 4 **−0.47**, k 8 −1.10, k 16 −1.65; pc128 only, k 8: −0.33. People up
  1–3/10. The true-run oracle on the same orders: +0.53. **Refuted:** the between-run directions carry class signal for the
  held-out person, so a fixed projection removes signal with the offsets. A form without run ids needs a softer mechanism
  (§219's covariance target, running on Galvani; soft run assignment, §217).

**§209 RESULT (14 Nov 07:42 UTC; runs/leo_npz_lf; morning_table.py row v6.2hL).** v6.2hL's new-people path (s15 on v5.1b
labels) on every order, each file against its own v5.1b:

| New people | within (12) | full (8) | runs (4) |
|---|---|---|---|
| v6f | +1.24 [+1.07, +1.41] | +0.94 | +1.00 |
| v6.2 / v6.2g / v6.2h (s15, v6-chain labels) | +1.44 | +0.87 (P10 min −1.60) | +1.14 |
| **v6.2hL** | **+1.41** [+1.25, +1.57] | **+0.93** (P10 min −0.82) | **+1.17** |

- **Reading:** v6.2hL takes the best of both: v6.2's within gain (the s15 form) without its full-scheme P10 loss (the label
  change). The prediction (within +0.15 over v6f, level under full) held: +0.17 and −0.01.
- Returning rows are identical across the routed variants (ERF), still the uncorrected ones until §214 lands.

### 224. cum10 ON PACKAGE LABELS (written 14 Nov 07:40 UTC, before any result; coordinator: content the run-time work would unlock)
- **What:** ttt18 FORMS=cum10 (warm start: each refit continues from the last, 10 steps) on chain_v51b labels, tag LF.
  Within 0/1, full 0/1, runs 0/1; Leonardo 64451354–365, scoring chained (shufcombo FT_V=cum10, FT_TAG=ttt18<order>LF).
- **Read:** cum10 against s15 on the same labels (both tag LF, same machine), new people, per scheme.
- **Prediction:** cum10 − s15 between −0.2 and +0.3 (no earlier package-form measurement).
- **Rule:** ≥ +0.3 mean with ≥ 4/6 orders up goes to a run-time check (its cost was ×2.4 in the 13 Nov timing).
- **Also queued on Leonardo (07:42 UTC):** copies of Galvani's pending §214 eye re-streams (full 1–3, runs 0–3: 64451330–345,
  first copy wins), and §221's six arrays (64451216–221). All become readable once the certificate is renewed.
- **§217 RESULT (14 Nov 07:43 UTC; runs/rundetect/{detect,segment,summary}.txt, runcount_<labels>.txt; code 65ac552+).**
  - Detector ('lag', thresholds from the other people), error at W 40 / 60 / 100 / 150: recording 7 / 0 / 0 / 0 %, runs
    4 / 1 / 0 / 0 %, within 26 / 7 / 2 / 4 %, full 11 / 3 / 0 / 0 %. The pre-registered consecutive ratio: within 50 / 31 /
    12 / 5 %, full 8 / 5 / 3 / 2 % (blind to 'within' until the first run ends). Prediction (≤ 5 % at W 40) missed.
  - Boundaries, purity / ARI (labels per session, true 6): bocpd (causal) recording 0.92 / 0.68 (11.9), runs 0.94 / 0.73
    (11.9), within 0.96 / 0.80 (10.3); length prior alone 0.95 / 0.69–0.71 (12.2); dpoff (non-causal) runs 0.93 / 0.86,
    within 0.99 / 0.98. Over-segmented, pure. mix under full 0.45 / 0.19 (offline GMM 0.50 / 0.24, nearest TRUE run mean
    0.56 / 0.28: the view is the limit).
  - Count filter (τ 2, offsets off, mean over P0r / V6Fr × 2 orders; true runs §215: within +0.57, runs +0.25, full +0.87;
    true runs with the 12 cap everywhere +0.51 / +0.23 / +0.78, so the cap costs ≤ 0.09):
    loop40 (pre-registered) within +0.25, runs −0.04, full −0.12 | loop60 +0.33 / −0.04 / −0.08 | bocpd +0.33 / −0.04 /
    −0.16 | loopM60 +0.34 / −0.04 / −0.10 [−0.40, +0.12] | mix full +0.07 [−0.34, +0.31] (−0.14 within, −0.21 runs) |
    dpoff (non-causal) +0.48 / +0.14 / −0.24.
  - **Verdict by the panel's rule: loop40 is killed** (both contiguous schemes below +0.3; full −0.12 < −0.1). Past the
    rule, stated plainly: loop60 / bocpd reach +0.33 on within (≈ 60 % of the oracle) but ≈ 0 on runs, where even the
    oracle is +0.25 with a range −0.33 to +0.74 and offline boundaries give +0.14; all within-scheme bars are inside the
    ±0.3 noise. The "offsets on" cells in these files use TRUE-run offsets (s215_*_on chainouts) and are not a label-free
    result. Under full, the oracle's +0.87 is unreachable with this 6-direction view (mix +0.07, wide). Open forms with a
    measured ceiling: a better run view for full shuffle (the ceiling is the largest of the three); online runs' 10/10/10
    cap from an inferred run type. runcount.py's header still says "true run ids" when RUN_IDS is set (cosmetic).

### 225. RUN OFFSETS WITH INFERRED LABELS (written 14 Nov 07:45 UTC, jobs started 2 min earlier, no result read)
- §217 killed the count filter with inferred boundaries. Offsets may tolerate over-segmented but pure labels better than an
  urn: a split run only shrinks each offset's estimate.
- **Test:** shufcombo RUNOFF 5, both views, RUN_IDS = runs/runids/{loop60, bocpd}.npz (§217, causal; purity 0.92–0.96, 10–12
  groups per session). Within 0/1, runs 0/1, against the regular files and the true-run oracle (§210: within +0.27 / +0.22,
  runs −0.02 / +0.58).
- **Prediction:** 50–80 % of the oracle on within (+0.12 to +0.20), about 0 on runs 0, +0.2 to +0.4 on runs 1.
- **Rule:** a package form needs ≥ +0.3 mean over the four orders with inferred labels, with the detector routing it off
  under full. Otherwise undecided or closed by the numbers.
- runcount.py's header now says which labels the filter used (subagent caveat).

### 226. STRUCTURED RUN BOUNDARIES (written 14 Nov 07:47 UTC, before any result; coordinator after §217)
- **Design (description, runs/rundetect/run_design.txt):** every public session has its two acquisition runs first
  (Graz or BrainHero first, 10 of 14 BH first), then four online runs. Graz acquisition always 36 (12/12/12); BH
  acquisition 30–36 (33 in 7 of 14; 9–12 per class); online runs 30 (10/10/10) except two BH online runs of 36
  (12/12/12, P10's third session). So the only uncertainty is the acquisition order and the BH acquisition length.
- **Forms (studies/14nov/rundetect/structured.py, labels runs/runids/s226_*.npz; causal: window t's label is fixed at t
  from windows ≤ t):** a hypothesis is a sequence of block lengths; its prior comes from the other people's design
  (LOPO: BH acquisition length 30–36, smoothed; acquisition order). Evidence = the conjugate segment marginal (§217's
  6-direction view) of the windows so far under the hypothesis' boundaries. Label of t = the block index with the largest
  posterior mass at t.
  pos   within: fixed 36 / 36 / 30 × 4 (the ML expert's position-only form, no evidence).
  wprior within: acquisition-first hypotheses (2 orders × L 30–36), prior only.
  wcp   within: the same with change-point evidence.
  rcp   runs: all distinct orders of {36, L, 30, 30, 30, 30}, L 30–36, uniform order prior, change-point evidence.
- **Joint filter (jointcount.py, step 3):** the same hypotheses, each with its own urn filter (cap from the block type:
  30-window block 10, else 12); hypothesis weights updated by the urn's predictive evidence for the chain posterior
  (J-urn) or by urn × feature evidence (J-both); output = mixture over hypotheses. Validated first: one hypothesis =
  true runs must reproduce runcount.py's true-run numbers exactly.
- **Scored:** within 0/1, runs 0/1, offsets off, P0r and V6Fr, new people, against true runs (§215: within +0.57, runs
  +0.25), with person-bootstrap 80 % intervals. runcount.py with RUN_IDS for the label forms.
- **Prediction:** wcp purity ≥ 0.97 with ≤ 7 groups per session, filter within +0.40 to +0.55 (70–95 % of the oracle);
  pos +0.35 to +0.45; rcp purity ≥ 0.9, filter runs +0.1 to +0.25; J-both ≥ wcp by +0.05 on within.
- **Rule:** a structured form is kept for the package if within (mean of P0r and V6Fr, two orders) ≥ +0.3 and ≥ half the
  oracle, with no person-bootstrap lower bound below −0.2 on runs. Below half the oracle on within: said plainly.
- **Runs 0 diagnosis (step 5):** per person and per run, filter Δ with true runs for V6Fr vs P0r; capacity check (true
  counts vs cap); τ 1 vs 2 (s215 τ 1 file: V6Fr −0.30, so not τ); calibration of V6Fr on runs 0 vs other orders.

**§206 build addendum 2 (14 Nov 07:55 UTC, v52-build; written before any check of these packages): v6.2iu, the time
limit off.**
- **Why:** finalists are audited for reproducibility, and an audit rerun on other hardware must give the same labels.
  v6.2i rev2's per-session time limit (120 s) makes outputs depend on machine speed: it binds on CPUs, about 10× slower
  refits. It also protects nothing the 400-window cap does not already bound (capped refits total 27.5 s at A100
  speed).
- **Change:** key only. `v6_ft_session_s` = 1e9 s (range check: > 0), same code path and cap. `cand_v62_r2iLu`, `r2iu`,
  `r1iLu`, `r1iu`; every other file and array is identical to rev2.
- **Predictions:**
  1. bitwise equal to rev2 with rev2's limit lifted on all 14 sessions (r2 both gate modes, r1 off);
  2. official local replay at 0.99458 with files unchanged and the full schedule on CPU (ingestion about 860–890 s for
     the routed-FT variants, as v6.2h);
  3. GPU worker image passes.
- **Refutation rule:** any miss fails the u package.

**§214 RESULT (14 Nov 07:58 UTC; runs/retv6/p0_*; morning_table → runs/morning_table_0800.txt).** Returning people (P04,
P10) with both P0 fixes (eye net in seen_session's own order, members re-streamed), change over routed v5.1b; before the
fixes in brackets:

| | within (8) | full (4) | runs (4) |
|---|---|---|---|
| **ER** (eye + rest, no routed FT) | **+0.59** (+0.65); 8/8 up; P04 min +0.08, P10 min +0.08 | **+1.46** (+1.05); 4/4; P04 min +1.07 | **+0.68** (+0.83); 4/4; P10 min −0.34 |
| ERF (+ routed FT s15) | +0.68 (+0.89); P10 min −0.42 | +0.88 (+1.18); P04 min −1.40 | +0.46 (+0.82); runs 3 −0.25 |

- **Reading:** with the fixes, ER is the robust returning-people form: positive on all 16 orders and larger than ERF under
  full and runs. Routed FT adds only under within (+0.09), costs under full and runs, and is P04-fragile. The prediction
  (ER / ERF move ≤ ±0.3) was right for within and wrong for full: ER +0.41, ERF −0.30.
- **Sealed-weighted J** (0.566 new + 0.434 × (0.2 P04 + 0.8 P10)), 80 % interval over people and orders:

| | within | full | runs | equal-weight J (dev) |
|---|---|---|---|---|
| **v6.2hL-safe** (s15 + L labels; ER) | +1.04 [+0.80, +1.28] | **+1.10** [+0.81, +1.41] | **+0.92** | **+1.04** |
| v6.2hL (ERF) | +1.03 | +1.04 | +0.87 | +0.92 |
| v6.2-safe | +1.06 | +1.06 | +0.90 | +1.03 |
| v6f + ER | +0.94 | +1.10 | +0.83 | +0.98 |
| v6f | +0.70 | +0.53 | +0.57 | +0.53 |

- **Consequence:** v6.2hL-safe (ER) is the scheme-robust candidate. It leads or ties under every scheme, with no
  routed-FT fragility. The "eye order mismatched, members as recorded" mark on the returning rows is lifted for these
  corrected rows.

**§222 RESULT (14 Nov 08:01 UTC; runs/views/{bands,sub8_*,tview_*}.log; check.py OK; recording order).** Refuted: the sub-8 view
fails the panel's kill in every form and loses in the chain.
- **Per band, member alone (LOPO, new / seen):** 1–4 40.5 / 45.4; 4–8 42.0 / 42.5; 8–13 **44.8** / 42.0; 13–20 42.4 / 44.3;
  20–30 36.9 / 37.6; 30–45 34.3 / 33.6. 1–4 + 4–8 full 41 × 41: 45.0 / 49.5. All six bands concatenated: 58.1 new (core1 with
  17 bands, halves and the slow block: 69.3).
- **Per band, template ceiling** (TSEM with true labels, new / seen; pc128 71.7 / 66.0): 1–4 54.4 / 53.4; 4–8 57.2 / 56.3;
  8–13 **66.6** / 53.6; 13–20 54.4 / 52.8; 20–30 51.5 / 40.7; 30–45 47.7 / 46.2. The panel's variance shares are not
  discriminability: 8–13 Hz carries the most class information both across people and within a session. The
  prediction (sub-8 bands the two best) is refuted.
- **The view** (member new / seen; template ceiling new; TSEM-alone new; chain dJ replacing pc128 / appended):

| form | dim | member | ceiling | TSEM | replace | append |
|---|---|---|---|---|---|---|
| full 41 × 41, 2 bands | 1722 | 45.0 / 49.5 | 60.1 | – | – | – |
| CSP 12 | 156 | 40.4 / 40.5 | 54.0 | 52.6 | −2.96 | −1.40 |
| CSP 16 | 272 | 38.6 / 38.5 | 53.2 | 50.9 | −2.77 | −1.22 |
| CSP 12, no recentring (order-free) | 156 | 36.2 / 36.0 | 56.4 | 53.2 | −2.31 | −0.95 |
| PCA 12, recentred after projection | 156 | **51.0** / 50.5 | **63.1** | 61.0 | −1.50 | −0.78 |
| CSP 12, 4 sub-bands | 312 | 39.7 / 40.1 | 54.6 | 53.6 | −3.63 | −1.60 |
| CSP 12 + ERP-prototype covariance | 292 | 41.1 / 39.2 | 53.2 | 51.6 | −3.53 | −1.96 |

  pc128 for reference: TSEM alone 69.8 new. Phi of correctness with the eye net 0.10–0.17 (the panel's streak risk is
  absent, but there is nothing to add).
- **Reading:** spatial reduction fit on other people hurts (CSP below the full 41 × 41 in every measure); PCA with
  in-session recentring is the best form and still below 55 % alone and 9 points below pc128 as a template view. The ERP
  augmentation adds nothing (−0.5 to −1). Every append loses (−0.8 to −2.0), far outside ±0.3. **Closed as a view**
  (kill met: best member 51.0 < 55; no form reaches pc128's ceiling). No shuffled chain test is warranted.

**§225 RESULT (08:07 UTC; runs/s225_runoff_inferred.txt).** Run offsets with inferred labels, v6f (v5.1b), against the regular
files:
- true runs (§210): +0.26 (+0.44);
- loop60: −0.02 (+0.15);
- bocpd: −0.00 (+0.17).
Per order, bocpd: within 0 +0.24, within 1 −0.11, runs 0 −0.24, runs 1 +0.10. **Below the +0.3 rule; closed for the package.**
Over-segmented labels lose most of what true runs give. v5.1b keeps about +0.16, but v6f's eye view and rest member appear to
absorb what is left.

**§219 RESULT (08:07 UTC; Galvani CPU, same-machine baseline).** TSEM's shrinkage target from the population within-class
covariance:
- β 0.5: −0.10 [−0.14, −0.06], 1/10 people up;
- β 1: −0.23 [−0.33, −0.13], 2/10.
**Refuted** by the panel's own kill rule. The identity target is not what limits TSEM at 128 components.

**§207 eye-net drift edit (Galvani, 54 tasks):** α 0.5 +0.05 (3/6 up), α 1 +0.04 (3/6). **Null.**

**§207 CLOSED: weight-space edits on the current chain, every family tried to a number:**

| Family | Result |
|---|---|
| final-layer edits (accuracy and correlation forms) | ≤ 0 on every lever component |
| core1 block gains | +0.04 |
| rest-net filter gains | −0.05 |
| teacher input gains | 0.00 |
| eye-net input drift edit | +0.05 |

Fitted on nine people, no edit transfers to the tenth beyond noise. The trained weights are at their transfer optimum for
these edit families.

### 216 result. EXTERNAL HARNESS v1 (14 Nov 08:15 UTC; `runs/extharness/report.md`, res_<corpus>.npz; Mac CPU, ~1 min per corpus)
- **Ran:** PhysionetMI (109 people, l/r/feet, ~67 windows per session), Schirrmeister2017 (14, l/r/feet, 360–790
  windows), Scherer2015 (9, the PROTEUS tasks). rec + full / runs / within × 3 seeds. The parity asserts held (stages =
  gm.pipe_g at W 30; W 15/60 change the rebalancing). Chain stages at W 30, person-equal BA (80 % person intervals):

  | corpus, scheme | member | rebal | SessionEM | TSEM | chain − member |
  |---|---|---|---|---|---|
  | Schirrmeister rec | 74.8 | 75.5 | 76.8 | 77.3 | +2.56 [+2.12, +3.00] |
  | Schirrmeister full | 74.6 | 74.9 | 76.1 | 76.7 | +2.10 [+1.67, +2.50] |
  | Schirrmeister runs | 74.7 | 75.5 | 76.8 | 77.3 | +2.55 [+2.14, +2.96] |
  | Schirrmeister within | 74.8 | 75.1 | 76.3 | 76.8 | +1.98 [+1.60, +2.34] |
  | PhysionetMI rec | 49.8 | 49.2 | 49.5 | 49.5 | −0.25 [−0.59, +0.06] |
  | PhysionetMI full | 45.8 | 46.4 | 46.5 | 46.5 | +0.70 [+0.48, +0.91] |
  | PhysionetMI runs | 46.0 | 46.1 | 46.2 | 46.2 | +0.20 [−0.02, +0.42] |
  | PhysionetMI within | 49.4 | 48.9 | 49.3 | 49.4 | −0.09 [−0.38, +0.18] |
  | Scherer rec / full / within | 45.0 / 45.7 / 45.2 | | | 45.8 / 45.8 / 45.8 | +0.87 / +0.08 / +0.53 |

- **Predictions:** member Schirrmeister 74.8 (in 60–75); PhysionetMI 49.8 rec, 45.8 full (miss, below 50–60); Scherer 45.0
  (miss). Rebalancing − member +1 to +4: **missed everywhere** (Schirrmeister +0.23 to +0.76, PhysionetMI −0.62 to
  +0.56, Scherer −0.12 to +0.39). SessionEM − rebalancing (Schirrmeister +1.14 to +1.36) and TSEM − SessionEM (+0.50 to
  +0.63) inside their ranges; on PhysionetMI both are about 0 (sessions of 67 windows: SessionEM starts at 20, TSEM 15).
- **Order:** on Schirrmeister the shuffles inside runs (full, within) cost the chain 0.5 against rec / runs, nearly all in
  rebalancing (+0.7 → +0.2/+0.3); the template stages hold. Same direction as on PROTEUS.
- **W check (final output, W − W 30):** Schirrmeister W 15 −0.09 to −0.16, W 60 −0.00 to +0.13 (all inside ±0.3).
  PhysionetMI W 15 −0.13 to 0.00; W 60 +0.98 rec and +0.83 within [+0.68, +0.99], +0.03 full, +0.04 runs. **Rule: W 30
  is near optimal** (the only > 0.3 gains are on one corpus, one shuffle scheme plus rec). The PhysionetMI W 60 gain is
  its class-pure runs (MOABB order: three l/r runs, then three feet runs): a 30-window balance target is wrong inside
  such blocks; PROTEUS runs are 10/10/10, so it does not transfer. My mechanism prediction (W 15 ahead there) was wrong.
- **Limits:** the member is a cheap stand-in (core1 is far stronger on PROTEUS); PhysionetMI's member is near 46–50, so
  its stage numbers are weak evidence; the gate thresholds (0.20/0.35 margin) were tuned on PROTEUS posteriors; the
  SessionEM view is not separate from the classifier (PCs 1–48 of the same tangent features), unlike GE; PhysionetMI's
  "rec" is MOABB run order, not chronology; one session per person, so the returning path is not covered.

**§223 RESULT (14 Nov 08:16 UTC; runs/bagging/bagscore.log; Galvani CPU job 3100585, 218 LR fits at 11.5 s each, 25 min on 8 cores;
check.py OK; recording order).** Refuted for core1: bagging over training people loses alone and in the chain.
- The full refit reproduces the stored member (correlation 0.998 / 0.999; 68.8 new against the stored 69.3), so every
  arm is compared with the refit, not the stored member.
- **Member alone (new / seen):** full 68.77 / 66.47; leave-one-person-out bags averaged 68.30 / 66.60 (−0.47); leave-two-out
  (8 bags) 67.33 / 65.69 (−1.44); both 68.04 / 66.35 (−0.73); geometric means the same within 0.1. A single bag 66.1 new on
  average (range 64.4–67.7): one or two people fewer cost about 2.7 points, and averaging wins back only 0.5–2.
- **Chain (B2_pool, core1 replaced, Hilbert slot as stored), dJ against the refit:** l1 −0.21, l2 −0.21, l1+l2 −0.37,
  l1 geometric −0.18 (new −0.1 to −0.4).
- **Kill met** (bagged member not above the full fit): closed for core1 on CPU. As stated beforehand, this does not close
  the net version, but it removes its main support: the person-subset diversity did not repay the lost people for a
  member whose error is mostly between people.
- **GPU version, sized, not submitted:** leave-two-out bags (8 per fold) for sbnfb (3 nets) and sbnsinc (1 net) = 8 member
  tasks per net at 2.5–6.5 GPU min each (STATE §7) → 1.3–3.5 GPU h, plus an equal-count seed-mix control of the same
  size (2.7–7 GPU h in all), plus delivery-order re-streams of the bagged nets for the shuffled chain (inference only if
  checkpoints are kept, else ×6 orders). The Galvani GPU queue is at its per-user job limit now. Expected value from
  this CPU result: below the statistician's −0.3 to +0.8; I would run it only on otherwise idle GPUs.

### 227. EXTERNAL HARNESS v2: RETURNING PATH (written 14 Nov 08:19 UTC, before any result; coordinator)
- **Code:** `studies/14nov/extharness/ret.py` (returning path), `kstages.py` (K-generic stages), `ret_report.py`; results
  `runs/extharness/ret_<corpus>.pkl`. Item 3 (K-generic stages) is done first because the multi-session corpora on hand
  are two-class. `python kstages.py` prints PARITY OK: equal to gm to 1e-10 on B2_pool and B2_s0, both protocols,
  recording order and a full shuffle (rebal, gate, SessionEM, TSEM). `replay/check.py` OK.
- **Corpora:** Lee2019_MI (54 people, 2 sessions on different days, l/r, 2 runs each); Shin2017A (29, 3 sessions, l/r MI);
  Shin2017B (29, 3 sessions, mental arithmetic vs rest); Scherer2015 (9, 2 sessions, 3 classes; reported, not used for
  the rules). Stieger2021 is not exported anywhere reachable (not on the Mac or Leonardo). Shin's sessions were recorded
  on one day, so its "gap 2" is within-day: a weak test of decay over days.
- **Stand-in for the routed path:** v1 member (LOPO population, causal recentring) and views (PCs 1–48 / 1–128); personal
  heads = logistic regression (C 0.1) on session-1 PCs of the person and 9 gallery people; identity gate = bio_id.TwoCov
  (psd view) fitted on the other people, 10-person galleries of session-1 signatures, routed from window 10 when the top
  score ≥ τ (99th percentile of impostor maxima at n 20 among the other people), wrong top person = wrong head;
  blend = (1 − s r)·member + s r·head, then the chain. s ∈ {0, .05, .10, .15, .25, .40, .60, .80, 1}. (The package's head
  share at full ramp is G3·a / (1 − a + G3·a) ≈ 0.18.) Test sessions in orders rec, full × 2, within × 2; per-person
  numbers are means over orders. LORO: on session 1, head trained on the other runs (no run ids: 4 blocks), blended with
  the population member on the held-out run (member level), BA per s; s_cv = argmax; advantage = head alone − member.
- **Use 1 (person-specific strength, panel proposal 10, §79).** Policies at gap 1: common s chosen leave-one-person-out
  on the other people's test curves (`common`), per-person s_cv (`pp`), per-person oracle (`oracle`).
  - Predictions: member Lee2019 60–70, Shin2017A 55–65, Shin2017B 65–75. Best common s 0.25–0.60. pp − common 0.0 to
    +0.4; oracle − common +1 to +3; Spearman(s_cv, best s) +0.2 to +0.5. Identity right on ≥ 90 % of windows after 10.
  - **CONFIRMS the per-person strength** if on ≥ 2 of the 3 large corpora pp − common ≥ +0.3 with the 80 % person
    interval above 0 and Spearman(s_cv, best s) ≥ +0.3 with its interval above 0. **KILLS it** if pp − common ≤ +0.1 on
    all three, or oracle − common < 0.5 on all three (nothing on the table). Otherwise undecided. A confirmation makes it
    a candidate for a PROTEUS dev test (P04/P10 LORO on their training sessions), not an adoption.
- **Use 2 (decay with sessions between training and test).** Gain(g) = chain BA at s − at s 0, for g = 1, 2 on Shin2017A/B,
  at the common s and at s 0.25. Prediction: gain(2) / gain(1) 0.7–1.0 (same day). **Decay confirmed** if gain(2) <
  0.5·gain(1) with the interval of gain(2) − gain(1) below 0 on both Shin corpora; **no decay** if that interval contains
  0 and the ratio ≥ 0.8 on both; otherwise undecided. Either way the package cannot see the gap (no metadata); a
  confirmed decay lowers the expected sealed gain of the returning path and argues for a label-free trust weight.
- **Item 2 (core-like member)** follows as its own pre-registered row once these land.

### 228. EXTERNAL HARNESS v2, ITEM 2: CORE-LIKE MEMBER (written 14 Nov 08:21 UTC, before any result)
- **Member:** `MEMBER=core` (corpus.py FSETS): CAR, core1's 17 overlapping bands (ov17), full window, shrunk covariance,
  log-Euclidean tangent, plus core1's slow block (1 Hz low-pass waveform, 8 samples per channel); 14,965 features;
  causal recentring and logistic regression (C 0.05) as v1. Differences from core1, stated now: no affine whitening at the
  session reference, no half windows, no per-session standardisation flag. Views (PCs 1–48 / 1–128) follow the member.
- **Runs:** the §216 harness on Schirrmeister2017 and PhysionetMI; §227's returning path on every corpus it runs.
- **Prediction:** member +2 to +6 over v1 on Schirrmeister and PhysionetMI (rec); the stage gains of §216 within ±0.5 of v1;
  §227's verdicts unchanged.
- **Rule:** a §216 / §227 verdict stands only if it is the same with both members; where they differ, it is "undecided"
  and the core member's number is the one quoted.
- **§221 scoring (08:35 UTC):** score.sh scored without STRICT (a missing input would have fallen back to recording order
  silently). Rescored with STRICT=1, two at a time, outside run_capped: this is scoring, not training. The lock was held by
  v52-build's replays until about 09:35. All 42 cells (6 orders × 7 arms) are scored the same way.

**§220 RESULT (14 Nov 08:36 UTC; Galvani job 3100517, commit 06df3ab code; evidence/episodic_14nov/).** fbraw retrained episodically
(arm E), seeds 0–5, all folds; E0 (stock recipe through the same runner) reproduces ttt7_fbraw_base to 2e-16 on every
seed and protocol, and its chain delta is exactly 0 in every order (runner and chain path are no-ops when they should be;
GPU training is deterministic, so E0 gives no retraining-noise estimate: the seed sets are the noise reference).
- **Family alone, new people, seed mean (stock → E):** raw 63.8 → 56.4 rec (−7.4), −6.1 over the six delivery orders
  (the bias the net no longer learns, as predicted). **After whole-session balancing: 64.9 → 63.0 rec (−1.9), 64.1 → 63.6
  over six orders (−0.5, without P10 −0.2, 6/10 up).** Seed sets 0–2 / 3–5: −0.6 / −1.4 rec, −0.3 / −0.4 six orders.
  Returning (seen_session) after balancing −3.3 to −4.1 (P04 loses; P10 about 0).
- **Chain, new people, P0r (v5.1b, slot 5 = E) − stock, mean of six orders:** B2_s012 +0.16 (without P10 +0.11),
  B2_s345 −0.02 (−0.03), B2_pool −0.10 (−0.13); orders range −0.30 to +0.42; recording order +0.03 / +0.18 / +0.15.
  V6r (eye + rest, no FT): +0.08 / +0.09 / −0.03.
- **Against the rule:** family alone after balancing rises less than +1 (it falls: −0.5 / −1.9) and the chain gains
  less than +0.3 on both seed sets → **killed by the panel's rule; closed for this form** (stratified episodes, batch
  marginal shift, fbraw), since the balanced family-alone change is ≤ 0. The chain itself is inside ±0.3 (−0.10 to
  +0.16): the direction stays **undecided** for other forms (prototype refinement step, unstratified episodes, a mixed
  loss keeping part of plain CE, other families). The prediction (alone +0.5, chain +0.1) missed on the alone part.
- **Reading:** removing the session bias from the training loss did not buy accuracy after the bias is removed at test
  time; it cost about 0.5–2 points of balanced accuracy, most in recording order, where the causal stream starts inside
  one run (a net that never learned a population prior is worse before balancing has data).
- **§226 RESULT (14 Nov 08:47 UTC; runs/rundetect/{run_design,structured,jointcount_s226,summary_s226,diag_runs0}.txt,
  runcount_s226_*.txt; code 0f7df43).** Filter change, offsets off, mean of P0r / V6Fr × orders 0/1 (person 80 % bands in
  the files). Oracle (true runs): within +0.57, runs +0.25. jointcount.py reproduces runcount.py exactly (true runs and
  every label file).
  - Labels (purity / groups per session; filter within | runs): pos 0.92 / 6.1, +0.29 | +0.11; wprior 0.97 / 6.3, +0.43 |
    +0.08; wcp 0.99 / 6.1, +0.43 | −0.07; rcp (runs purity 0.95 / 6.2) +0.41 | −0.01; §217 bocpd +0.33 | −0.04.
  - Joint filter (hypotheses × urn, mixture output): J-wprior/urn +0.51 | +0.08 (89 % of the oracle on within; wcp/urn
    is the same hypothesis set, identical); J-wcp/both +0.49 | −0.11; J-rcp/urn +0.38 | +0.15; J-rcp/both +0.48 | +0.06;
    J-pos/urn +0.31 | +0.11. Type caps (10 for online) against 12 everywhere: +0.02 (J-pos vs pos labels), so the joint
    gain over labels (+0.08) is the marginalisation, not the cap.
  - Rule: within ≥ +0.3 and ≥ half the oracle: met by wprior, wcp, rcp, all J forms (pos at +0.29 just short). The runs
    clause (no lower band below −0.2) fails for every form, the oracle included (V6Fr runs 0: true −0.33 [−0.72, +0.03]):
    not kept for the package by the letter; the clause cannot be met by any run filter on this order. Features add
    nothing on within beyond the design prior; on runs the change-point evidence lowers the filter (wcp/rcp/both < urn).
  - Runs 0 diagnosis: no capacity mismatch (no true run exceeds its cap; four BH acquisition runs sit one below 12); not
    τ (τ 1: −0.30); not order-specific calibration (V6Fr conf 0.63 / acc 0.77 / ECE 0.14 on runs 0, 1 and within 0
    alike; the filter brings ECE to 0.02 everywhere). The loss is P01 −1.9, P05 −1.8, P06 −0.9, P07 −0.7 and falls on
    acquisition runs (net corrected windows: BH acquisition −7.2, Graz acquisition −2.8, online +4.2); BH acquisition runs
    lose on runs 1 too (−3.5). Reading: the 12/12/12 urn is loose for BH acquisition (9–12 per class) and those runs
    carry the noise; an order-level draw, not a mechanism specific to runs.

**§224 RESULT (14 Nov 11:50 UTC; Leonardo; each variant against v6f on its own machine).** cum10 on v5.1b labels against s15 on
the same labels: within 0 −0.21, within 1 −0.04, full 0 −0.05, full 1 −0.08, runs 0 −0.18, runs 1 +0.12. **Mean −0.07, 1/6 up.**
Closed: no gain, at about twice the run time.
- Note: Leonardo's older shufcombo omits the form from the output name when it is cum10 (files
  shufcombo_<o>_ocu_all_..._ttt18<o>LF.npz). The current shufcombo names every form.

### 230. ALL-SEED FINE-TUNING, CPU SCORING VARIANT (written 14 Nov 11:44 UTC, before any result)
- The package fine-tunes one net per family (delta1: the first base seed's FT change applied to the seed-mean slot). All-seed
  applies the mean of every base seed's FT change. Run time is × seeds; this is the content a run-time saving would buy.
- **Test:** shufcombo ALLSEED=1 on v6.2hL's new-people form (FT_V=s15, FT_TAG=ttt18<order>LF). Within 0/1, full 0/1, runs 0/1
  against the same files without ALLSEED.
- **Prediction:** +0.1 to +0.3 (averaging seeds' FT changes cuts FT noise).
- **Rule:** ≥ +0.3 mean with ≥ 4/6 up → size the run-time cost; between 0 and +0.3, undecided.

### 231. PER-SESSION EYE-VIEW WEIGHT, ORACLE FIRST (written 14 Nov 11:45 UTC, before any result; panel neuroscience P3 iii)
- The eye code is bimodal across people (own-label accuracy 0.38–0.50 or 0.76–0.84). Ceiling first: shufcombo EYEW=w scales the
  eye-view columns. w ∈ {0, 0.5, 1, 2}, v6f, six orders, per-window outputs. The oracle picks per session the w with the best
  session accuracy (labels: ceiling only).
- **Rule:** oracle gain ≥ +0.5 over w = 1 → build a label-free per-session rule (e.g. agreement of the eye view with the member
  mix over the first 40 windows), chosen on five people and scored on five. Below +0.3: closed. In between: undecided.

### 229. r15 ON v5.1b LABELS INSIDE v6.2hL (written 14 Nov 11:45 UTC, before any result)
- **Why:** §221's r15 (labels refreshed at every refit) gave within +0.12, full +0.22, five of six orders ≥ 0 on v6.1's
  labels. The upload candidate v6.2hL-safe fine-tunes new people on v5.1b's labels (tag LF, chain_v51b_<order>), so r15
  is retested on those.
- **Labels:** `relabel.py KIND=v51b` (LABEL_ARM=P0r): v5.1b's own SessionEM / TSEM state at each s15 refit time applied
  to every earlier window, causal, the same hooks and checks as §221 (state at t reproduces the stock output for
  window t to 1e-9; as-emitted labels against chain_v51b_<order>, argmax agreement > 0.98). Orders whose inputs are all
  local: within 0–11, full 0–3, runs 0–3 (20); full 4–7 lack local eye/core re-streams. Orders without a local ttt17
  base take the frozen-net base of tag ttt18<order> (FT_TAG/T17_DIR; the label arm reads only base outputs).
  Files `runs/labels/relab_v51b_<order>.npz`.
- **Arms:** `ftrel.py KIND=v51b ARMS=s15,r15`, tag `ttt18<order>LP`, s15 the in-run control on the same as-emitted
  labels; 24 tasks per order; smoke first.
- **Scoring:** `score229.sh`: shufcombo as score.sh plus STRICT=1 (OCU_DIR=runs/ocushuf/<order>, GATE_P0=1,
  REC_PFX=ttt7, T6_DIR=runs/ttt/probs7, T17_DIR=runs/ttt/probs18, FT_TAG=ttt18<order>LP, FT_V=s15|r15), line
  "combination over v5.1b, both shuffled", new people, r15 − s15 per order.
- **Prediction:** r15 − s15 mean over the 20 orders +0.10 (−0.05 to +0.25), 60–75 % of orders ≥ 0, full mean +0.1.
- **Rule (coordinator):** kept as a candidate addition to v6.2hL if r15 − s15 ≥ +0.1 on the mean over the orders, with
  ≥ 60 % of orders ≥ 0 and the full-scheme mean ≥ 0. Otherwise "undecided" (mean inside ±0.3) or closed by the numbers
  (mean ≤ −0.3). Not closed on P10 or P04 alone.

### 228. COUNT FILTER, PACKAGE-TRUE (written 14 Nov 11:45 UTC, before any result; coordinator after §226)
- **Chain:** the per-window outputs that exist: v5.1b (P0r) and v6f (V6Fr), runs/chainout/<order>.npz (identical to the
  s215 *_off files where both exist), new people, within 0–11, runs 0–3, full 0–3. v6.2hL's new-people per-window
  outputs (FT_TAG ttt18<order>LF, FT_V s15, SAVE_OUT) do not exist yet; the script takes any SAVE_OUT file and is rerun on
  them when they are produced. Returning people: no per-window outputs exist; not scored.
- **(a) End to end (studies/14nov/rundetect/package.py):** per session, the 'lag' detector (thresholds fitted on the other
  people, runs/rundetect/decisions.npz) is checked at W 40, 50, 60, 80, 100, 150; it latches at the first pass and stays
  latched. Emitted output = chain posterior before the latch window, J-wprior/urn (jointcount.py) from it on. The urn
  and hypothesis state are built from every window ≤ t (causal); only the emission switches. Reported per scheme with
  person-bootstrap 80 % bands; under full also the latched sessions and their cost.
- **(b) Online-only urn:** the same with the urn on online blocks only (cap 10 per 30-window block; acquisition blocks and
  windows past the six blocks get a uniform urn, i.e. unfiltered); with true runs also the true-run online-only filter
  (true caps, so the two 12/12/12 BrainHero online runs included).
- **(c) Gain by delivery position:** the always-on filter (true runs; J-wprior/urn) applied only to windows at session
  positions 0–40, 40–80, 80–120, 120–160, 160+; per-bin change, within orders.
- **(d)** Code audit of where the filter sits (dev scorers and the package path).
- **Prediction:** (a) within +0.35 to +0.45 (most latches at W 40–50), runs −0.1 to +0.05 (the wprior hypotheses assume
  acquisition first), full −0.05 to 0 (false-positive latches ≈ 15 % of sessions). (b) within ≈ +0.40, runs 0's V6Fr loss
  ≥ −0.1. (c) windows 0–40 carry ≤ 25 % of the gain.
- **Rule:** (a) goes to the user as a package candidate if within ≥ +0.3 on both arms (mean over within 0–11) with the
  lower band ≥ 0, full ≥ −0.05 and runs ≥ −0.1 (means over orders and arms); (b) replaces (a) if it is not lower on within
  by more than 0.05 and higher on runs.
- **Numbering note:** "228" was already taken (external harness v2, item 2, 08:21 UTC); this entry is cited as §228c
  (count filter) from here.

**§230 RESULT (11:50 UTC; runs/s230_allseed.txt).** All-seed FT on v6.2hL's new-people form, against delta1: within 0 −0.03,
within 1 +0.04, full 0 +0.12, full 1 +0.12, runs 0 −0.13, runs 1 +0.21. **Mean +0.05 [+0.01, +0.10]**, 4/6 orders up, 7/10 people
up. Undecided and small: below the +0.3 rule. It costs × 6 FT run time, so it is not worth a package change.

**§232 hooks (11:45 UTC):** `EXTRA_VIEW` (extra SessionEM view columns, recording or delivery-order rows), `REST_W`, and
`EM_LAM` / `TS_LAM` / `BTS` (studies/14nov/stagehooks.py). No-op smoke: the stage constants at their shipped values reproduce
the regular file exactly (max diff 0.0). Synced to Galvani and Leonardo. Helpers launched on them: §233 external pairwise
experts as views, §234 embedding views, §235 upper-bound sweeps of the kept components, §236 session-type conditioning.

### PV1. PERSONAL VIEWS FOR RECOGNISED RETURNING PEOPLE: CEILINGS (pview track; written 14 Nov, before any result)
A separate session's track (sections named PV1, PV2, ... so they cannot collide with the main numbering). Code
`studies/14nov/pview/` (data.py, ceil.py), outputs `runs/pview/`.
- **Question:** for a recognised returning person, is a view trained with labels on that person's public data a better
  classifier of their later data than the population versions the chain uses? An independent review found 32 ocular
  features at 0.66 with a session's own labels against 0.47 across people (P10 0.81–0.84 vs 0.35).
- **Arms (standalone, CPU, public data):** views gaze32 (SessionEM's 32 slow-potential columns), GE48 (SessionEM's full
  view), pc128 (TSEM's view, core1 PCs in the person's LOSO basis); methods LR (logistic regression, C 0.05) and TPL
  (SessionEM-style Gaussian class templates: class means, shared covariance shrunk 0.7); source personal (the person's
  train rows) or population (the other nine people). Plus the stored population eye net.
- **Protocols:** XS across sessions for P04 and P10 (s1→s2, s1→s3, s1+s2→s3; the real case); R42 inside each session
  (train four runs, test one Graz + one BrainHero run, 3 folds); IF inside each session (train one interface, test the
  other). R42 and IF are the proxy for the eight single-session people.
- **Standardisation of the test part:** causal (windows < t plus the train prior as 10 pseudo-windows), in recording
  order and under shuffled delivery (within 0–3, full 0/1, runs 0/1 from runs/shuffle_perms.npz); a non-causal
  'whole' column for reference. Headline = mean over the shuffled orders. GE48's event rates and pc128's recentring are
  as stored (causal in recording order); the eye net is as stored.
- **Prediction:** XS gaze32-LR personal P10 ≈ 75 (65–85) vs population ≈ 40 and eye net ≈ 45; P04 personal ≈ 50
  (40–60). R42 personal gaze32-LR mean over ten ≈ 60 (55–66), population ≈ 47. pc128-TPL personal XS ≈ 60, R42 ≈ 65,
  population pc128-TPL ≈ 55. GE48 between gaze32 and pc128.
- **Rule for step 2 (chain):** go on if, under XS, the best personal view beats the best population view (including the
  eye net) by ≥ +5 points on P10 and is not below it on P04 by more than 5, and under R42 the personal ocular view beats
  the population ocular view by ≥ +5 on the ten-person mean. Otherwise stop after step 1 and record the ceilings.

### 236. SESSION-TYPE CONDITIONING (written 14 Nov 11:52 UTC, before any result)
- **Idea (panel neuroscience, P3 ii/iii):** the class code is a personal strategy (strong eye code in four people, none in
  two; half without an alpha/mu peak). Choose the eye-view weight and the stage constants per session from a label-free
  session descriptor. Code `studies/14nov/sesstype/`; no edits to studies/13nov, replay, submissions.
- **Settings (10, one knob moved at a time from v6f as shipped):** shipped (EYEW 1, EM_LAM 0.5, TS_LAM 0.15, BTS 4);
  EYEW 0 / 0.5 / 2 (§231 files `runs/chainout/s231_<order>_w<w>.npz`); EM_LAM 0.25 / 1.0; TS_LAM 0.075 / 0.3; BTS 2 / 8
  (new files `runs/chainout/s236_<order>_<knob><v>.npz`, shufcombo SAVE_OUT, STRICT=1, GATE_P0=1, FT_V=gC5, REC_PFX=ttt7,
  FT_TAG=ttt18<order>L, OCU_DIR=runs/ocushuf/<order>). Arm V6Fr, new people, orders within 0/1, full 0/1, runs 0/1.
  Reference = §231's w1 file (the in-run control); a stage-constant file at shipped values must equal it (no-op check).
- **Oracle ceiling (labels; ceiling only):** per order and session, the setting with the best session BA (mean over the six
  bases) is taken for that session; gain = person score (cell-mean BA, mean over bases) − shipped, new people, per scheme
  (mean of its two orders), person-bootstrap 80 % band. Reported for all 10, for EYEW only, for stage constants only.
- **Descriptors (label-free, from the first K = 40 windows delivered in that order):** (e1) agreement of the eye view's
  argmax (ocushuf posterior) with v5.1b's chain (P0r, no eye view) argmax; (e2) mean max posterior of the eye view;
  (a) alpha/mu peak: 8–13 Hz log power minus the mean of 4–7 and 14–20 Hz, median over channels, from
  data/proteus_windows.npz X; (s) slow band: 1–4 Hz power as a share of 1–40 Hz, median over channels.
- **Rule family:** for descriptor d, setting s ≠ shipped, direction ±, threshold θ (midpoints of the training sessions'
  values): use s on a session when d is beyond θ, else shipped. Causal emission: windows at delivery positions < K emit the
  shipped output; from K on, the chosen setting's own output (each setting's state is built from every window ≤ t, as a
  package running the cheap stages side by side would; only the emission switches).
  The rule (d, s, ±, θ) is chosen on five people's sessions pooled over the six orders and scored on the other five;
  all 126 partitions, both directions (252 ordered splits). Controls in the same family: "always s" (a global setting
  change, no descriptor). Reported: held-out gain per scheme, mean over splits, 10–90 % range over splits, and the
  per-person held-out mean (over the splits holding that person out) with a person-bootstrap 80 % band.
  Primary = descriptor chosen inside the fold (fully nested). Per-descriptor nested numbers are secondary (picking
  the best of them afterwards is optimistic). K = 20 and 80 secondary.
- **Prediction:** ceiling, all 10 settings: +1.5 to +3 (EYEW alone +1 to +2, as the earlier per-session oracle); the
  oracle on 14 sessions also fits noise. Nested held-out gain of the best rule: −0.2 to +0.4, most likely about +0.1;
  e1 is the most likely descriptor to carry anything.
- **Rule:** kill below +0.3 nested held-out (primary, mean over the six orders); the ceiling is reported regardless. At
  ≥ +0.3 with the person band above 0 and no scheme below −0.1: candidate for a package-form test (not an adoption).
  Between 0 and +0.3: inside the dev noise, recorded as undecided for other descriptor forms. Not closed on P10 or P04 alone.

### 235. UPPER-BOUND SWEEPS OF THE KEPT COMPONENTS (written 14 Nov 11:52 UTC, before any result)
- **System:** v6.2hL-safe, new people: s15 fine-tuning on v5.1b labels (FT_TAG=ttt18<order>LF, FT_V=s15), eye-net view,
  rest member; shufcombo STRICT=1 GATE_P0=1, members re-streamed (Leonardo inputs runs/diag/{ocushuf,restshuf,ttt}).
  Orders within 0/1, full 0/1, runs 0/1. Read: V6Fr per person (mean over the scorer's bases) minus V6Fr of the same
  order without knobs, both run on the same machine with the same code (shufcombo d80f97d, unedited).
- **Knobs (one at a time, wave 1):** EYEW 0.5 / 1.5 / 2; REST_W 0.5 / 2 / 3; EM_LAM 0.3 / 0.7 / 1.0; TS_LAM 0.1 / 0.2 /
  0.3; BTS 2 / 8 (shipped gm.BTS 4). No-op check first: EM_LAM 0.5 TS_LAM 0.15 BTS 4 must reproduce the no-knob file
  exactly (else the hook is broken and nothing is read). If a knob's best arm sits on a grid edge, the grid is extended
  one step past it in wave 2.
- **Wave 2 (joint):** the two knobs with the largest full-data best-arm gains, 3 × 3 around their best values (the
  shipped value kept as a corner where it is adjacent).
- **Nested rule (primary):** for each of the 126 splits of the ten people into two halves of five, and both directions,
  the arm (shipped value included, gain 0) with the best mean gain over the six orders on the selection half is chosen and
  scored on the other half, per scheme and overall; a split's number is the mean of its two directions. Reported: mean
  held-out gain over the 126 splits with the 10–90 % range across splits, per scheme, and how often each arm is chosen.
  The best arm's full-data gain is shown only as the optimistic ceiling, never as the result.
- **Prediction:** stage constants (EM_LAM, TS_LAM, BTS) held-out within ±0.15 (they sat at an in-chain optimum for v4.1,
  §31, §66, §69); EYEW and REST_W 0 to +0.2 (never tuned under shuffled orders with FT); joint grid 0 to +0.25; no knob
  reaches +0.3.
- **Rule:** a knob (or the joint pair) with held-out gain ≥ +0.3 overall and > 0 under every scheme is a package constant
  change candidate (stated plainly; a package-form check and the user's decision follow). Inside ±0.3: undecided, the
  shipped constant stays. ≤ −0.3 for every non-shipped value: the shipped value is confirmed for this chain. Not closed on
  P10 or P04 alone; P10's share is reported.

### 234. EMBEDDING VIEWS (written 14 Nov 11:53 UTC, before any result)
- **Why:** §188: a view whose errors are independent of the EEG members helps SessionEM (the eye net works that way). Test
  low-dimensional embeddings of components we already have as extra SessionEM view columns (`EXTRA_VIEW`, §232 hook).
- **Sources:** T, the teacher's head features (38 = eegnex_ea / ifnet_ea m7r × 19 head log-probs on causally EA-aligned
  windows; `runs/teachfeat_within0_within1_full0_full1_runs0_runs1.npz`, delivery-order versions for the six orders).
  F, ST-EEGFormer's middle-layer embedding of the per-window input (512, `runs/fmemb/steeg.npz` key `emb_mid_raw`, §71's
  best arm; order-free). E, the eye net's penultimate embedding (128 × 9 seeds concatenated, `emb` of
  `runs/ocushuf/<order>/oc_nomix-all_loso-s*.npz`, each order's own stream).
- **Reduction:** per held-out person, columns standardised on the other nine people's rows, then PCA or PLS (on the nine's
  one-hot labels), k ∈ {4, 8, 16}; the held-out person's rows (recording and each delivery order) go through that fold's map.
  Code `studies/14nov/embview/`; views `runs/views_emb/<name>.npz` ('rec' and '<scheme><seed>' rows, replay row order).
- **Ceiling (step 1):** LOSO logistic regression on the view (balanced accuracy, cell mean); error correlation with the
  member mix (A.mix B2_pool, shipped families) within person × session × run × class cells: correlation of true-class
  log-probabilities after removing cell means, and of the error indicators. Reference: the eye net's posteriors.
- **Selection (fixed now):** per source the form with the best LOSO probe accuracy goes to the chain (ties: smaller k).
- **Chain:** shufcombo with EXTRA_VIEW plus exactly `STRICT=1 GATE_P0=1 OCU_DIR=runs/ocushuf/<order> FT_V=gC5 REC_PFX=ttt7
  T6_DIR=runs/ttt/probs7 T17_DIR=runs/ttt/probs18 FT_TAG=ttt18<order>L`; V6Fr against
  `runs/shufcombo_<order>_ocu_gC5_all_ge_hb_ts_gp0_ttt18<order>L.npz`, new people, mean over the six bases; within 0/1,
  full 0/1, runs 0/1; 80 % person intervals.
- **Predictions:** probe accuracy T 0.55–0.68, F 0.40–0.46, E 0.48–0.55. Error correlation (log-prob) T 0.4–0.6 (already a
  member, weight 5), F 0.05–0.2, E ≤ 0.15. Chain: T 0.0 (−0.3 to +0.2), F 0.0 (−0.3 to +0.3), E +0.1 (−0.2 to +0.3).
- **Kill rule (coordinator):** a source is not kept if its gain is below +0.3 mean over the six orders or fewer than 7/10
  people are up (person means over orders). Inside ±0.3 the direction is recorded "undecided", not closed; ≤ −0.3 closed.
  Kept sources go to a package-form check (map fitted on all ten people, all 20 orders).

**PV1 RESULT (14 Nov; `studies/14nov/pview/ceil.py` → runs/pview/ceil_all.txt).** Cell-mean BA (%), personal / population,
causal standardisation, mean over the 8 shuffled orders (recording order and 'whole' within ±1 of it everywhere, so the
delivery scheme hardly matters for a fixed classifier); eye = stored population eye net.

| XS (real case) | gaze32 LR | GE48 LR | GE48 TPL | pc128 LR | pc128 TPL | eye net |
|---|---|---|---|---|---|---|
| P04 | 40.6 / 32.7 | 42.2 / 31.6 | 45.4 / 31.2 | 57.7 / 45.5 | 59.0 / 45.5 | 42.4 |
| P10 | 58.2 / 45.8 | 73.5 / 39.5 | 71.7 / 38.1 | 76.1 / 56.0 | 75.2 / 59.3 | 49.2 |

| proxy, ten-person mean | gaze32 LR | gaze32 TPL | GE48 LR | GE48 TPL | pc128 LR | pc128 TPL | eye net |
|---|---|---|---|---|---|---|---|
| R42 (train 4 runs, test 2) | 55.8 / 43.6 | 56.8 / 46.0 | 66.7 / 52.1 | 66.7 / 53.5 | 75.3 / 64.4 | 76.5 / 64.9 | 54.2 |
| IF (train one interface) | 53.4 / 44.0 | 53.9 / 46.2 | 63.3 / 52.5 | 63.1 / 53.6 | 71.9 / 64.7 | 73.0 / 64.9 | 54.1 |

- Per person (R42, GE48 LR personal / population): P01 55/53, P02 84/66, P03 77/60, P04 45/33, P05 62/45, P06 64/52,
  P07 70/57, P08 61/55, P09 72/62, P10 78/40. Personal ≥ population for all ten in GE48; in gaze32 P01 and P05 are
  lower personally (35–43, chance-like), and in pc128 P08 and P01 (IF) are better from the population.
- **Against the prediction:** P10 gaze32 58 (predicted 75): the 0.81–0.84 within-session figure does not carry across
  sessions in the 32 slow-potential columns alone; with the 16 event-rate columns (GE48) P10 is 72–74 across sessions.
  P04 personal is weak everywhere except pc128 (58–59). R42 gaze32 personal 56 (predicted 60), population 44 (47).
  pc128 personal XS 67 (predicted 60), R42 77 (65): higher than predicted. Population pc128 TPL 52 XS, 65 R42.
- **Rule met → step 2:** under XS the best personal view beats the best population view by +16.8 on P10 (pc128 LR 76.1
  vs pc128 TPL 59.3) and +13.5 on P04 (59.0 vs 45.5); R42 ocular personal − population +10.8 (gaze32 TPL) to +14.6
  (GE48 LR).
- **What this does not say:** the chain already has a personal head on core1's PCs (the pc128 LR row is close to it) and
  the own-space stage; the new information for the chain is the personal ocular view (GE48: P10 73 against the eye
  net's 49) and template priors, whose worth is only measurable in the chain (PV2).

### 233. EXTERNAL PAIRWISE EXPERTS AS VIEWS (written 14 Nov 11:55 UTC, before any result)
- **Mechanism (§188):** the eye net failed as a member but works as a SessionEM view because its errors are independent
  of the EEG members'; a perfect extra view is worth +6 (SessionEM only). Experts trained only on public people outside
  PROTEUS cannot share the members' training-person errors either. Each expert is a two-class task classifier whose
  log posteriors are appended to SessionEM's view (shufcombo `EXTRA_VIEW`, s232 hook; no shufcombo edit).
- **Corpora on hand:** Shin2017A (29 people, MI left/right) and Shin2017B (the same 29, mental arithmetic vs rest; A and
  B sessions alternate on one day). Shin2018 (word generation vs rest) and ds007554 are on neither the Mac nor Leonardo
  `$WORK/ni26/data/` (NEMAR holds only Weibo, Liu, HGD, Zhou, Cho, Dreyer2023, Shin2017A/B, Lee2019); not downloaded
  (needs the user's go-ahead), so no WORD expert in this row.
- **Experts (`studies/14nov/extview/expert.py`):** `mivma` = Shin A MI (l + r) vs Shin B arithmetic → PROTEUS MI vs SUB;
  `mavr` = Shin B arithmetic vs rest → PROTEUS SUB vs the rest; `shin2` = both (4 columns). Montage: 35 of the 41 exported
  channels are interpolations (6 exact), so both corpora are CAR'd and projected on the top 20 eigenvectors of the pooled
  Shin covariance; bands 1–4/4–8/8–13/13–20/20–30 Hz, shrunk covariance, log-Euclidean tangent; session mean removed
  (Shin: pseudo-session A_k + B_k for mivma, B session for mavr; PROTEUS: causal running mean of the session's delivered
  windows, so the view file has 'rec' plus within/full/runs 0/1 keys computed in each order); L2 logistic regression,
  C by 5-fold person-grouped CV on Shin. No PROTEUS label is read.
- **Step 1 (ceiling, `ceiling.py`), predictions:** mivma MI-vs-SUB balanced accuracy on PROTEUS new people 55 (50–62),
  within-session AUC 0.53–0.62; mavr SUB-vs-MI and SUB-vs-WORD 52 (48–58). Error correlation with the member mix within
  class × run (margins) +0.0 to +0.2, about the eye net's.
- **Step 3 (chain):** V6Fr with `EXTRA_VIEW` against the regular `shufcombo_<o>_ocu_gC5_all_ge_hb_ts_gp0_ttt18<o>L.npz`,
  within/full/runs 0/1, env of `runs/s215_within0_off.log`, new people, mean over the six bases as shufcombo prints.
  Prediction: shin2 +0.1 (−0.2 to +0.4) mean over six orders; mivma and mavr alone 0 to +0.2 each.
- **Kill rule:** a view file is kept for a package test only if its mean over the six orders is ≥ +0.3 **and** ≥ 7/10
  people are up (person = mean over the six orders). Below that it is not kept. The direction (external task experts as
  views) is closed for this form only if the mean is ≤ −0.3 or the expert alone is at chance (pair BA < 53 and AUC
  < 0.53 averaged over orders); a mean inside ±0.3 with an above-chance expert is "undecided" (better corpora, a WORD
  expert, net experts are the next forms).

**§236 amendment (11:56 UTC, before any result was scored): EYEW is a silent no-op for every w > 0.** §231's files for within 0/1:
w 0.5 and w 2 equal w 1 to 0.0 on every base and arm; only w 0 differs. Cause: SessionEM z-scores each view column by the
session's own sd (replay/gm.py EMg._adapt), so scaling a column changes nothing; w 0 makes the columns constant, i.e.
removes the eye view. §231's oracle over {0, 0.5, 1, 2} is an oracle over {off, on}. For §236 the eye-weight axis is:
w 0 (§231 file), shipped, and weight 2 / 3 as one / two extra copies of the eye view appended through EXTRA_VIEW
(studies/14nov/sesstype/eyecopy.py, files s236_<order>_eyex2 / eyex3). oracle.py fails on any setting file equal to the
shipped one. Settings are now 9 (w0, eyex2, eyex3, six stage constants); the rest of §236 is unchanged.

### PV2. PERSONAL VIEWS IN THE ROUTED TAIL, P04 AND P10 (pview track; written 14 Nov, before any chain result)
PV1's rule was met (personal views +13 to +17 over population ones across sessions). Code
`studies/14nov/pview/chain.py` (retv6 exec'd up to its main loop, not edited; s214 form: GATE_V51B=1, members
re-streamed, FT_SFX=QF, FT_FORM s15) and `personal.py` (the offline personal models, fitted on the person's public
session only). Every addition acts in the routed tail only; rows the gate has not routed yet emit ER's output (checked:
the prior and member arms leave them bit-identical before the replacement). Gate and shadow are v5.1b's, so new people
are unchanged by construction (no new person is routed, s201). The routed person must equal the session's person
(asserted).
- **Arms against ER** (eye view + rest member, the v6.2-safe returning rows of s214; ER is checked against
  runs/retv6/p0_*):
  - **O:** the personal ocular model (LR on GE48, causal standardisation in delivery order with the public session's
    stats as 10 pseudo-windows) as 3 more SessionEM view columns.
  - **M:** the same model as a member in the routed mix, weight 1 of 16, times the gate's ramp.
  - **S:** SessionEM template prior: personal class means of the routed-tail view (cross-fitted on the public session),
    κ 20 pseudo-windows in the class means, so it fades as the session's own windows arrive.
  - **T:** TSEM template prior: route4's personal `pri`, κ 20, with κ also in the denominator (route4's KAPPA path
    lacked it: the over-scaling noted for s19/s57b).
  - **OST:** O + S + T.
  - **W:** O + M (w = 2s) + S and T (κ 20s), with a per-person strength s = clip((BA personal − BA eye net) / 20, 0, 1),
    from leave-one-run-out inside the public session.
- **Orders:** within 0–7, full 0–3, runs 0–3 (s214's 16), six bases. Reported per person and scheme, with 80 % intervals
  over orders (bootstrap) and the count of orders > 0.
- **Prediction:** O +0.4 (P10 +0.6, P04 +0.1); M +0.2; S +0.2; T 0 (±0.3); OST +0.5 (P10 +0.7, P04 +0.3); W within 0.1 of
  OST with P04 ≥ OST's P04.
- **Rule:** an arm is a package candidate if its 16-order mean (P04 and P10 equal) is ≥ +0.3, P10 > 0 on ≥ 12/16 orders,
  and no scheme mean is below −0.3 for either person. Between −0.3 and +0.3: "undecided". ≤ −0.3: this form closed. Not
  closed on one person alone.

### 227 / 228 result. RETURNING PATH ON EXTERNAL PEOPLE (14 Nov 12:20 UTC; `runs/extharness/ret_report{,_core}.md`, ret_<corpus>{,_core}.pkl)
- **Ran:** Lee2019_MI (54 people, 2 sessions of 100 windows), Shin2017A (29, 3 sessions of 40, l/r MI), Shin2017B (29, 3 × 40,
  arithmetic vs rest), each with the v1 member and the core-like member (§228); Scherer2015 v1 (9, not in the rules).
  Stieger2021 could not be used: it is on Galvani only, Galvani wants 2FA, and the copy failed after the first file.
  Copies from Leonardo; the feature cache key now holds the input file list (a 28-person partial Lee cache would otherwise
  have been reused silently; a 28-person Lee run of 09:40 is discarded: with 54 people its sign flipped).
  K-generic stages: PARITY OK against gm (§227). Identity: routed on 72–86 % of windows (from window 10), wrong ≤ 1 %.
- **Use 1, gap 1 (chain BA; pp = per-person s from LORO on session 1, common = s chosen LOPO; 80 % person intervals):**

  | corpus, member | s 0 | best common s | common | pp − common | oracle − common | Spearman(s_cv, best s) |
  |---|---|---|---|---|---|---|
  | Lee v1 | 72.4 | 0.40 | 74.0 | −1.76 [−2.40, −1.17] | +1.24 | −0.10 [−0.26, +0.06] |
  | Lee core | 77.7 | 0.40 | 78.9 | −1.24 [−1.97, −0.51] | +1.07 | +0.08 [−0.09, +0.25] |
  | Shin A v1 | 61.9 | 0.40 | 63.5 | −1.48 [−2.36, −0.64] | +2.14 | −0.11 [−0.34, +0.14] |
  | Shin A core | 64.5 | 0.25 | 65.3 | −1.78 [−3.03, −0.60] | +2.07 | −0.07 [−0.31, +0.18] |
  | Shin B v1 | 64.4 | 0.25 | 73.9 | −5.28 [−6.43, −4.07] | +1.12 | +0.14 [−0.07, +0.33] |
  | Shin B core | 69.4 | 0.25 | 75.9 | −4.17 [−5.48, −2.84] | +1.64 | +0.13 [−0.14, +0.39] |

  **Verdict: KILLED with both members** (pp − common ≤ +0.1 on all three; predicted 0 to +0.4). The headroom exists (oracle
  +1.1 to +2.1) but LORO inside the training session does not find it: s_cv sits at 0.6–1.0 for most people (a head
  tested on its own session looks better than on a later one), and it does not rank people (Spearman about 0). The best
  common s is 0.25–0.40 everywhere (the package's head share is about 0.18); the gain at s 0.25 is smaller under full than
  under rec (Shin B core +5.8 vs +8.2).
- **Use 2, gain at the common s by gap (Shin, same-day sessions):** v1: A gap 1 +1.79, gap 2 +0.81, difference −0.98
  [−2.03, −0.03]; B +9.48 / +7.95, −1.53 [−2.71, −0.41]. Core: A +0.81 / +0.79, −0.02 [−0.66, +0.60]; B +6.53 / +5.57,
  −0.97 [−2.12, +0.10]. Rule: v1 neither confirms decay (B ratio 0.84) nor clears it (A interval below 0); core clears
  it (ratios 0.98, 0.85, intervals contain 0). The members differ, so **undecided** (§228); core says: no decay within a
  day. Decay across days (the sealed returning sessions) is not measured: no ≥ 3-session multi-day corpus is reachable.
- **§228 member:** Schirrmeister 82.9 (v1 74.8), PhysionetMI 59.3 (v1 49.8) in rec: +8 to +9.5 (predicted +2 to +6). §216's
  verdicts hold with it (W 30 near optimal: Schirrmeister |W − W30| ≤ 0.12; PhysionetMI's W 60 gain only in rec/within);
  chain − member Schirrmeister +1.4 to +2.1.
- **§228c RESULT (14 Nov 11:58 UTC; runs/rundetect/package_{within,runsfull}.txt; code 594de76).** v5.1b (P0r) / v6f (V6Fr),
  new people, mean over orders [80 % person band]. Always-on true runs for scale.
  - (a) Detector-gated J-wprior/urn: within (12 orders) +0.54 [+0.39, +0.68] / +0.35 [+0.22, +0.49] (true runs +0.53 /
    +0.35; always-on J +0.51 / +0.31); runs (4) +0.27 [+0.11, +0.42] / +0.18 [+0.02, +0.34] (true +0.57 / +0.24); full (4)
    +0.03 [−0.00, +0.07] / −0.02 [−0.05, +0.00] (always-on J −0.33 / −0.30). Latches: every within and runs session at
    W 40; under full 7 of 56 sessions (12.5 %) latch falsely and cost ≈ 0 in sum. **Rule met: package candidate.**
  - (b) Online-only urn (gated): within +0.42 / +0.27, runs +0.25 / +0.18, full +0.05 / −0.02. Lower on within by 0.12 /
    0.08 and not higher on runs: does not replace (a). With true runs, online-only removes runs 0's V6Fr loss (−0.33 →
    −0.00) but costs on within and full; the BrainHero acquisition urn is a runs-0 draw, not a systematic loss.
  - (c) Gain by delivery position (true / J, within): windows 0–40 −0.05 / −0.04 (P0r) and −0.05 / −0.06 (V6Fr), i.e.
    the filter costs inside the first run; 40–80 +0.13 / +0.13, 80–120 +0.16 / +0.14, 120–160 +0.11 / +0.11, 160+ +0.18 /
    +0.17 (P0r). Starting at the W 40 latch loses nothing; it is why gated beats always-on.
  - (d) Placement: the dev scorers filter shufcombo's SAVE_OUT outputs after the run, so no gate, shadow or fine-tuning
    label can see them. In the package (lean_v62i) the gates, shadow and FT labels read out_v5 / tail6r / route state
    inside step(); only predict() takes step()'s return value. The filter belongs in predict(): argmax of
    filter.step(self.step(x)), with its own state (detector on the 6-direction band view, fitted on all ten public
    people; J-wprior/urn needs no features). Not yet built into a package.
  - Not done: v6.2hL's per-window outputs (SAVE_OUT with FT_TAG ttt18<order>LF, FT_V s15) do not exist; returning people
    have no per-window outputs.

### 238. COUNT FILTER: RETURNING PEOPLE AND ROBUSTNESS (written 14 Nov 12:02 UTC, before any result; coordinator after §228c)
- **Candidate:** §228c (a), the detector-gated J-wprior/urn filter, unchanged (τ 2, floor 0.5, checks W 40–150).
- **1. Numbers.** New people on v6.2hL's per-window outputs (runs/chainout/v62hL_<order>.npz, 24 orders; the
  coordinator's runs/s228c_v62hL.txt checked, rerun with bands if needed). Returning people: package.py generalised to
  the seen_session protocol (its sessions and delivery orders; detector thresholds and design prior still leave the
  person out), on retv6's per-window outputs (runs/chainout/ret_<order>.npz) for ER and ERF (base for reference), P04 and
  P10 separately, within 0–7, full 0–3, runs 0–3 as they arrive.
- **2. Robustness (studies/14nov/rundetect/robust.py).** Synthetic sessions from the 14 public sessions (new people),
  runs contiguous and shuffled inside (within-like), 3 seeds; the chain's per-window outputs reused from the within 0
  order (a simplification: the chain is not rerun in the new order, so this tests the filter's structure only). Violations:
  one online run removed; one online run duplicated; online runs of 36 (each online run + 2 windows per class from the
  next online run); acquisition runs absent; acquisition runs last; the session twice (second copy reshuffled); 10 % of
  windows dropped at random. Plus the real deviation: P10's third session (two online runs of 36). Metric: balanced
  accuracy per synthetic session, mean per person, gated filter − no filter, v6f and v6.2hL (v5.1b for reference).
- **Prediction:** violations −0.2 to +0.3 (the urn evidence moves the boundary hypotheses; removed / dropped windows
  shift every later boundary, the worst case); acquisition last and online 36 the worst (≈ −0.2).
- **Rule:** any violation below −0.2 on v6f or v6.2hL → add a self-check (the urn's predictive evidence against a uniform
  urn over the last windows; switch off for good when it falls below a threshold fixed on the normal public sessions'
  distribution) and report it on every violation and on the normal sessions (its cost there).
- **3.** A plain method statement, corrected against the code, in runs/rundetect/method_statement.md.

**§234 RESULT (14 Nov 12:42 UTC; studies/14nov/embview/, runs/views_emb/, runs/s234_summary.txt; check.py OK; local CPU runs, the
Galvani CPU queue had 168 jobs pending).**
- **Ceiling (LOSO probe BA, new; error correlation with the mix inside class × run cells, log-p / 0-1):** member mix 70.8; eye
  posterior 54.3, +0.17 / +0.12. T (teacher heads) PCA 46–48, PLS 48–50 (pls4 50.3, +0.40 / +0.20); the teacher member itself
  is 50.6 on new people, +0.56 / +0.25. F (ST-EEGFormer) PCA 37–38, PLS 32–33 (pca8 38.0, +0.02 / 0.00): near chance. E (eye
  emb) PCA 52–54 (pca4 53.8, +0.20 / +0.12), PLS 44–45 (PLS fails: each LOSO fold has its own eye net, the label directions do
  not transfer between fold nets; PCA's leading variance does). Predictions: T miss (0.55–0.68 predicted), F miss below
  (0.40–0.46), E hit; correlations T, F, E hits.
- **Chain (V6Fr over the same chain without the view, new, six bases):**

| View | within 0 | within 1 | full 0 | full 1 | runs 0 | runs 1 | mean [80 %] | people up |
|---|---|---|---|---|---|---|---|---|
| T_pls4 | −0.46 | −0.46 | −0.38 | −0.13 | −0.53 | −0.43 | **−0.40 [−0.61, −0.20]** | 3/10 |
| F_pca8 | −0.12 | −0.27 | −0.29 | −0.27 | −0.43 | −0.41 | **−0.30 [−0.49, −0.11]** | 3/10 |
| E_pca4 | +0.09 | −0.10 | +0.29 | +0.13 | +0.21 | +0.22 | **+0.14 [+0.07, +0.22]** | 8/10 |

- **Verdict:** T closed (≤ −0.3; a view whose errors follow the members' only adds noise columns). F closed at the boundary
  (−0.30, no class information, 0/6 orders up). E not kept (below +0.3) but consistent: 5/6 orders and 8/10 people up, interval
  above 0. "Undecided", small: it is a richer copy of the eye view (same accuracy and error correlation as the eye posterior).
  Predictions: T hit at the low edge of its range, F hit at its low edge, E hit (+0.1 predicted).
- **Stamp correction:** the §234 RESULT heading above says 12:42 UTC; `date -u` at writing was 12:03 UTC.

### PV3. COMMON, LARGER PERSONAL STRENGTHS ON THE RETURNING PATH (pview track; coordinator after §227; written 14 Nov, before any result)
- **What the package does (checked in `runs/upload/sealed_cand_v51b_own.zip` weights.npz and
  `submissions/ensemble_v5b/submission.py` l. 860):** routed mix = (mix − r a core1 + g r a h) / (1 − r a + g r a) with
  route_slot_g g = 3 and a = member_weights[0] / sum = 1/15 (route_slot_a 0.0667). The personal head's share at r = 1 is
  g a / (1 − a + g a) = 0.176 (v5.1b); with the rest member (a = 1/16, ER / v6.2-safe) 0.167. own_beta = 0.10.
- **§227 (external harness):** the best common strength was 0.25–0.40 on every corpus; a per-person strength from
  leave-one-run-out was killed (Spearman ≈ 0). PV2's arm W uses exactly that LOO strength; it is still run as
  registered but is not a candidate form whatever it scores.
- **Arms (in chain.py, same run as PV2, against ER):** H25 / H33 / H40 = the head share at r = 1 raised to 0.25 / 0.33 /
  0.40 (g = 5 / 7.5 / 10 at a = 1/16); B20 / B30 = own-space beta 0.20 / 0.30. Each alone, routed tail only (gate unchanged).
- **Prediction:** H33 +0.3 (P04 +0.5, P10 +0.1; the head is P04's main gain in the past); H25 +0.2; H40 +0.2 with P10
  ≤ 0; B20 +0.1, B30 0 (±0.3).
- **Rule:** as PV2 (16-order mean ≥ +0.3, P10 > 0 on ≥ 12/16, no scheme mean below −0.3 for either person → candidate;
  ±0.3 undecided).

**§231 RESULT (12:04 UTC; runs/s231_eyeoracle.txt; Galvani CPU; v6f, new people, against w = 1).**
- w 0.5 and w 2 equal w 1 exactly: SessionEM z-scores its columns, so the oracle reduces to eye on vs off per session.
- Eye off everywhere: within −0.22, full −0.42, runs −0.45.
- Per-session oracle (labels, ceiling): within +0.22 [+0.17, +0.27], full +0.16, runs +0.12.
- **Below the +0.3 rule even as a ceiling: closed.** A per-session eye on/off rule cannot be worth more than about +0.2. The eye
  view stays on in every session.

**§233 RESULT (14 Nov 12:05 UTC; code f7131c7 + the person-id fix below; evidence/extview_14nov/{cv_report,ceiling,compare}.txt;
Mac CPU, 45 s per chain run; `replay/check.py` OK).**
- **Fix before any PROTEUS number:** the first mivma fit grouped Shin2017A/k and Shin2017B/k as different people (58
  groups, CV NaN, person leakage across folds); ids now strip the corpus prefix. C grid extended to 1e-4 (the first
  pick sat on the grid edge). Within Shin, person-grouped CV: mivma 77.9, mavr 78.0 (C 3e-4).
- **Step 1, alone on PROTEUS (mean over rec + six orders):** mivma MI-vs-SUB BA **66.0** (people 53–87), within-session
  AUC 0.73 (prediction 55 [50–62]: beaten); pairwise accuracy 65 against the mix's 78.5 and the eye net's 67.3. Its margin
  correlation with the mix within class × run is **+0.23** (errors +0.12–0.14), higher than the eye net's +0.17 (+0.10);
  with the eye net +0.14. mavr SUB-vs-MI 53.2 (AUC 0.58), SUB-vs-WORD 53.4 (AUC 0.56), correlation with the mix +0.05–0.10.
- **Step 3, V6Fr with EXTRA_VIEW minus the regular file, new people, mean of six bases (80 % person intervals):**

  | view | within 0 | within 1 | full 0 | full 1 | runs 0 | runs 1 | mean 6 | people up | V6r mean |
  |---|---|---|---|---|---|---|---|---|---|
  | mivma | +0.06 | +0.12 | −0.28 | −0.11 | −0.48 | −0.25 | **−0.16** [−0.34, +0.03] | 4/10 | −0.06 |
  | mavr | −0.23 | −0.30 | +0.10 | −0.21 | −0.22 | −0.23 | **−0.18** [−0.34, −0.01] | 2/10 | −0.13 |
  | shin2 | −0.14 | −0.01 | −0.21 | −0.29 | −0.81 | −0.51 | **−0.33** [−0.55, −0.09] | 3/10 | −0.22 |

- **Against the rule:** none is kept (all below +0.3 and 7/10). shin2 ≤ −0.3: **closed for that form**. mivma and mavr are
  above chance alone and inside ±0.3 in the chain: **"undecided"** by the letter, but every mean is negative and 0–2 of 6
  orders are up. Chain prediction (+0.1 [−0.2, +0.4]) missed low.
- **Reading:** an expert as accurate as the eye net on its pair (66 vs 67) still costs in SessionEM: its errors are more
  correlated with the members' (+0.23 against +0.17) — a covariance-feature task classifier shares the members' failure
  windows even when trained on other people — and it carries no WORD information, so its two columns add noise for a
  third of the windows. External people alone do not buy independence; the eye view's independence comes from its
  input (slow frontal potentials), not from its training people. Next forms, if any: a WORD expert (Shin2018 / ds007554,
  not on hand; download needs the user's go-ahead), experts on non-covariance inputs, or residualising the expert on the
  mix before it enters the view.

**§235 wave 2 (written 14 Nov 12:05 UTC, after wave 1 was read, before any wave-2 job ran).**
- **Silent no-op found: EYEW > 0 does nothing.** EYEW 0.5 / 1.5 / 2 reproduce the no-knob file to 0.0 on all six orders
  (all bases, all arms). SessionEM standardises every column of its view ((A − m) / sd), so scaling the eye columns cancels;
  only EYEW 0 (constant columns) changes anything. §231's w 0.5 / 2 arms are no-ops for the same reason and cannot carry
  an oracle; its only live arms are w 0 and w 1. (shufcombo is not edited here; OCU_TS already notes the same for TSEM.)
- **Eye weight as copies instead:** EYEW 0 (eye view removed) and 1 or 2 extra copies through EXTRA_VIEW
  (studies/14nov/sweeps/eyecopies.py: the eye view exactly as shufcombo builds it, tiled; weight 2 and 3 under the 0.7
  shrinkage), each with REST_W 1 and 2.
- **Joint grid** (pre-registered rule: the two knobs with the largest full-data best non-shipped arm; every best arm was the
  shipped value; TS_LAM 0.2 −0.03 and BTS 8 −0.03 are the two least negative and act on the same exponent): TS_LAM {0.1,
  0.15, 0.2} × BTS {2, 4, 8}. No grid edge extension (no knob's best arm sits on an edge).
- **Prediction:** EYEW 0 −0.3 to −0.8 (the eye view carries part of v6's new-people gain); 1–2 extra copies −0.2 to +0.2;
  TS_LAM × BTS held-out within ±0.1. Same nested rule and thresholds as wave 1.

### 239. DESIGN DECODER: RUN POSTERIOR INTO TEMPLATES AND LABELS (written 14 Nov 12:06 UTC, before any result)
- **Why:** the offline equal-count assignment bound is about +1.1, the causal count filter (§228c) about +0.5. The filter
  cannot change emitted labels, but its belief about earlier windows of the same run can feed the stages that come later.
- **Mechanism (code studies/14nov/designdec/):** after each window the run urn (§215/§226: posterior^τ, τ 2, floor 0.5, cap
  10 per 30-window online block, else 12) is smoothed over the delivered prefix (forward-backward inside the current block
  of each boundary hypothesis; finished blocks stay as last smoothed), mixed over hypotheses with the weights of windows
  < t. Those posteriors for windows < t replace the stages' soft labels at window t only (one-way; nothing is re-emitted):
  SessionEM's past input posteriors B (its EM seeds) and TSEM's past seed labels and confidences. Forms:
  rep (smoothed posterior replaces both), ext (only the urn's extrinsic message m_s = smoothed_s / own likelihood_s,
  multiplied into B_s and into TSEM's input e_s), fwd control (the forward-filtered, i.e. emitted, filter posterior;
  no smoothing). Boundaries J-wprior (urn weights, §226) and true runs (ceiling). Active from the detector latch (§228c,
  W 40), as the gated filter. Emitted output: the gated J-wprior/urn filter (§228c a) on top of the changed chain.
- **Tests:** (i) self-contained replay (members as recorded, stages in the delivery order, B2_pool); (ii) package form:
  studies/13nov/shufcombo.py run unmodified through runpy with the stage installed (chainout env: GATE_P0, STRICT, gC5,
  ttt18<order>L), arms P0r and V6Fr, B2_pool, within 0/1 and runs 0/1. No-op check first: feedback off must reproduce
  runs/chainout/<order>.npz B2_pool exactly.
- **Prediction:** J rep on V6Fr: chain alone +0.15 to +0.35 on within; over the gated filter alone +0.05 to +0.2 on within
  (largely redundant with the output filter), runs 0 to +0.1. True runs 1.5–2 × J. ext ≥ rep by about 0.05; fwd about
  half of rep.
- **Kill:** J (best of rep / ext) below +0.3 over the gated filter alone on within (mean of within 0/1, V6Fr; P0r
  reported), or any collapse (an order mean down ≥ 1 point against the unmodified chain + filter, or one person down ≥ 5
  points on an order). A result inside ±0.3 is recorded as undecided as a direction, but not kept for the package.
- **(2) fine-tuning labels:** run only if (1)'s best J form is ≥ +0.15 over the gated filter on within, or the cheap
  proxy says so: balanced accuracy of the prefix-smoothed labels at s15's refit times ≥ the as-emitted labels + 2 points
  (new people, within). Form: ftrel.py r15 with RELAB_NPZ holding the smoothed labels at each refit, against r15 / s15.

### 240. EYE-EVENT VIEW (written 14 Nov 12:08 UTC, before any result; panel neuroscience proposal 4)
- **Mechanism:** a view helps SessionEM when it reads a different kind of signal (§188, §233, §234). Panel measurement: SUB
  and WORD differ in gaze direction with the same sign in all four interface × run-type cells, and the vertical difference
  flips sign about 2.0 s into the window, so window averages cancel. Signed, half-window and latency-coded eye events keep
  what averaging loses. Code `studies/14nov/eyeevents/`; views `runs/views_eye/<name>.npz` ('rec' + six '<scheme><seed>'
  keys, replay row order), appended through shufcombo's `EXTRA_VIEW` (no shufcombo edit).
- **Features (fixed now, 8 columns per window):** traces as `ocushuf.traces()` before standardisation: h = SG(F5/AF3/F3 −
  F6/AF4/F4), v = SG(AF3/AF4 − P1/P2/Pz) (Savitzky-Golay 25, 3); velocities dh, dv = gradient. Scale m = running session
  median of the per-window median |velocity| over the windows delivered so far (current included), blended with the
  all-window prior as 5 windows (as `EventRates`); hence order-dependent, one key per delivery order. Columns:
  (1, 2) signed horizontal event balance per half (0–2 s, 2–4 s): share of samples with dh > 4m minus share with dh < −4m;
  (3, 4) the same at 2m; (5) latency of the first blink, first sample with dv > 8m_v, in s, censored at 4.0;
  (6) vertical level late minus early (mean v over 2–4 s minus 0–2 s); (7) vertical event share (|dv| > 4m_v) late minus
  early; (8) horizontal level late minus early. Name `ev8`. No PROTEUS label is used to build the view.
- **Gate 1 (confusion):** share of v6f's emitted errors (`runs/chainout/s215_<order>_off.npz` `<base>|V6Fr`, new people,
  six bases, six orders) that are SUB ↔ WORD. Under 40 % → stop. Prediction 50 % (42–60 %).
- **Gate 2 (alone, LOSO logistic regression on the 8 columns causally z-scored per session, recording order and within 0):**
  SUB-vs-WORD balanced accuracy (person mean) 57 (52–63), within-session AUC 0.55–0.62; 3-class cell BA 42 (38–48). Error
  correlation with the member mix within class × run (margin, as `extview/ceiling.py`) +0.00 to +0.15 (eye net about +0.17).
  If SUB-vs-WORD BA < 53 and AUC < 0.53 (chance) → stop.
- **Chain:** V6Fr with `EXTRA_VIEW=runs/views_eye/ev8.npz` minus `runs/shufcombo_<order>_ocu_gC5_all_ge_hb_ts_gp0_ttt18<order>L.npz`,
  env exactly as §233/§234, new people, mean of six bases, within/full/runs 0/1, 80 % person intervals. Prediction +0.1
  (−0.2 to +0.4).
- **Kill rule:** kept for a package test only if mean over the six orders ≥ +0.3 **and** ≥ 7/10 people up (person means
  over orders). ≤ −0.3: closed for this form. In between: "undecided" (not closed, not kept). The panel's "with proposal 1"
  arm (eye regression out of the EEG) is not built and not part of this run.

**§240 gate 1 RESULT (14 Nov 12:11 UTC; `studies/14nov/eyeevents/confusion.py`).** SUB ↔ WORD share of v6f's emitted errors,
new people, six bases × six orders pooled: **39.9 %** (within 0/1 38.0 / 38.5, full 0/1 39.5 / 40.0, runs 0/1 41.8 / 41.8);
MI ↔ WORD 34.6, MI ↔ SUB 25.4. 80 % person interval 37.2–44.4; person-mean share 45.8 (P07 71, P08 75; P03 29, P10 31). By
cell: Graz acquisition 48.5, BrainHero online 41.9, BrainHero acquisition 39.3, Graz online 33.5. Prediction 50 (42–60):
missed low. **The gate fails by 0.06 points as written.** Deviation, stated before steps 2–3 ran: the bar sits inside its
own noise (interval 37–44) and SUB ↔ WORD is still the largest pair, so by AGENTS.md ("a bar inside the noise does not close a
direction") this is not a stop on substance; steps 2–3 are cheap CPU runs and go ahead under the same gates and kill rule,
flagged as past gate 1.

### 241. OFF-TASK WEIGHTING FROM THE REST NET (written 14 Nov 12:10 UTC, before any result; panel neuroscience 14 Nov P3 ii)

**Mechanism.** Lapsed or half-performed trials look like rest (no parietal alpha suppression). The rest member (shallow
SessionBN net, rest4 sh r1) already has a 4th output for PROTEUS's pre-cue rest; its p_rest was never saved. If p_rest marks
off-task windows, they should (i) be wrong more often in the chain's output and (ii) be bad teachers for the session's
templates. Weight each window's contribution to the template estimates (SessionEM responsibilities and residual covariance,
TSEM seed means and residual covariance) by w = 1 - p_rest; the window's own prediction is unchanged (learning only).
**Data.**  re-streams the rest net (3 seeds, each fold trained once) in recording order and
in within 0/1, full 0/1, runs 0/1 with a forward hook keeping all 4 logits; asserts the first three reproduce the member's
posterior (1e-5). p_rest = seed mean of the 4-way softmax's 4th entry, from the order's own re-stream.
**Step 1, tercile gate (CPU, existing v6f outputs).** Chain accuracy of v6f (runs/chainout/s215_<order>_off.npz,
<base>|V6Fr, new people, mean over six bases) by tercile of p_rest within each session (session-equal mean), averaged over
the six orders. Prediction: bottom-minus-top tercile gap +4 points (80 % range 0 to +9). **Gate: the top tercile at
least 3.0 points worse than the bottom one, else stop** (and record whether recording order agrees).
**Step 2a, template weighting (only if the gate passes).** Weighted SessionEM (gm.em_g) and TSEM (gm.L.G.tsem_lik)
installed by monkeypatching from , scored in the v6f chain (env of runs/s215_<order>_off.log)
against the same chain without the hook, paired, on within 0/1, full 0/1, runs 0/1, new people, 80 % person intervals.
Arms: W1 w = 1 - p_rest (primary), W3 w = (1 - p_rest)^3 (secondary, reported). Prediction for W1: new +0.2
(80 % range -0.1 to +0.5). **Kill rule:** mean over the six orders below +0.3, or fewer than 7/10 people up (mean over
orders) -> not adopted; inside ±0.3 is recorded as "undecided", not closed. Not closed on P10 or P04 alone.
**Step 2b, FT label weights:** sized only (GPU fine-tuning via ttt18 label weights); run only if 2a passes.
(Correction, same minute: two paths above were lost to shell quoting. The re-stream script is `studies/14nov/offtask/prest.py`;
the stage hooks are installed from `studies/14nov/offtask/`.)

**§240 gate 2 RESULT (14 Nov 12:11 UTC, before any chain result; `studies/14nov/eyeevents/{build,alone}.py`, `runs/views_eye/ev8.npz`).**
(Stamp correction: the gate 1 heading says 12:11; `date -u` at writing was 12:08.)
- **Alone, LOSO logistic regression, causal per-session z-scores (rec / within 0 / full 0 / runs 0):** 3-class cell BA 48.7–49.2
  (prediction 42 [38–48]: above); **SUB-vs-WORD BA 63.9–65.0, within-session AUC 0.66** (prediction 57 [52–63], 0.55–0.62:
  above); the eye net's SUB-vs-WORD BA is 65.7, the member mix's 79.2. Gate 2 passes.
- **Sign consistency (SUB − WORD by interface × run type):** 7 of 8 columns keep one sign in ≥ 3 of 4 cells; early horizontal
  balance (q 2) positive and late (q 2) negative in all four (the panel's flip), vertical late-minus-early level and event
  share positive in all four.
- **Error correlation with the mix within class × run (SUB/WORD margin):** **+0.35** (errors +0.14–0.15) against the eye net's
  +0.155 (+0.08); with the eye net +0.14. Prediction +0.00 to +0.15: **missed high**. The view is as accurate on the pair as
  the eye net but shares the members' margins about as much as §233's mivma (+0.23) or more, which §233 found costly in
  SessionEM. Chain prediction left unchanged as written (+0.1 [−0.2, +0.4]); on this correlation I expect its low end.

### 242. STACKED ARM ON v6.2hL-SAFE (written 14 Nov 12:11 UTC, before any result; coordinator 12:15 item 3)
Mechanism: the small parts that held their sign, together in one chain: fine-tuning labels refreshed at every refit (r15,
§229, tag ttt18<order>LP), the eye-net embedding view E_pca4 as SessionEM columns (§234, +0.14; rebuilt for all 20 orders
with eye nets by studies/14nov/stack/eview.py, checked identical on the six s234 keys), the detector-gated count filter
(§228/§238, studies/14nov/rundetect/package.py, emitted output only), and §235's stage constants once that sweep lands
(arm stk2; stk1 has the shipped constants). Control in the same run: s15 on the same LP labels, no view, no filter (ctl).
Orders: within0-11, full0-3, runs0-3 (the 20 with eye nets and LP fine-tuning); scored on Leonardo CPU with SAVE_OUT.
Prediction (new people, stk1 + filter against ctl, mean over orders, no scheme primary): +0.45 (within about +0.55,
runs about +0.35, full about +0.10). Returning people: the count filter on ER per §238; r15 and the view enter only if
seen-session fine-tuning / views exist, otherwise reported as absent, not as zero.
Report: new and returning, sealed-weighted J = 0.566 new + 0.434 (0.2 P04 + 0.8 P10), intervals resampling people and
orders. Refutation: stacked below the best single part on the same orders, or people up < 6/10 -> the parts do not stack;
inside ±0.3 -> undecided, not closed.

**§228c package (14 Nov 12:15 UTC, v52-build; written before any check beyond a one-session smoke): v6.2c, the
run-level count filter on v6.2iLu ER.**
- **Form:** `lean/src62c` → `submissions/lean_v62c` (`countfilter.py`). It is ni26-push's dev form at 3a10ba8: the
  'lag' detector checked at windows 40 / 50 / 60 / 80 / 100 / 150 that latches for good, then J-wprior/urn (14 design
  hypotheses, tau 2, floor 0.5) emitted from the latch window.
  - It sits in `predict()` only: label = argmax of `cf.step(x, step(x))`. Gates, shadow, routing and fine-tuning labels
    read the unfiltered output.
  - State on the model, reset per session; numpy on the CPU, no clock. Any error gives the unfiltered posterior for the
    rest of the session.
  - Key `v6_cf_on` (builder: required, no default). Packages `cand_v62c_r1iLu_off` / `_on`.
- **Ten-person fit** (`lean/tools/fit_cf.py`, `runs/countfilter/cf_params.npz` 33b29e20…), against dev's
  leave-one-person-out fit:
  - projection on 10 people, not 9, with its scales cross-fitted the same way;
  - thresholds as the balanced-error midpoint over every person's out-of-sample statistics. They equal the leave-one-out
    medians at 40 / 60 / 80 / 100 / 150; at 50, 0.9447 against 0.9433;
  - design prior over all 14 sessions: P(L = 30..36) = .086 .143 .086 .429 .143 .029 .086, P(BrainHero first) = 0.75.
  - On public sessions the projection has seen the person's runs, so latching there is in-sample. Dev's figures stay the
    estimate for new people.
- **Predictions:**
  1. with the switch off, bitwise equal to v6.2iLu ER (v6_compare, 14 sessions, both gate modes; record.py per case
     against v6.2iLu's recordings);
  2. the reference check against the pinned dev code with the same parameters, bitwise in recording, within 0–2 and
     full 0–2 (14 sessions each, gates off);
  3. forced exception at window 20 / 60: errors 1, the unfiltered output from there on;
  4. record.py stress cases complete with finite outputs;
  5. official local replay files unchanged; GPU worker image passes;
  6. filter cost: 4.4 ms per window on the Mac CPU (one-session smoke, P01: latched at 40 in both, bitwise, no label
     changed).
- **Refutation rule:** any miss fails the package.

**§235 wave 3 (written 14 Nov 12:12 UTC, after wave 2 was read, before any wave-3 job ran).** Extra eye copies' best arm (EYEC 2,
weight 3) sits on the grid edge, so by the edge rule EYEC 3 (weight 4) runs; EYEC 5 (weight 6) also runs to see the shape
(past the rule, stated now). Nested selection then over EYEC {0, 1, 2, 3, 5}. Prediction: EYEC 3 within ±0.05 of EYEC 2;
EYEC 5 lower; held-out over the set +0.05 to +0.12, below +0.3.

**§240 RESULT (14 Nov 12:14 UTC; code 6ff0a8e; `studies/14nov/eyeevents/score.sh`, `extview/compare.py ev8`; evidence
`evidence/eyeevents_14nov/{confusion,alone_ev8,compare_ev8}.txt`; `replay/check.py` OK; Mac CPU, ~1 min per order).**
V6Fr with `EXTRA_VIEW=runs/views_eye/ev8.npz` (8 columns, delivery-order rows; every log confirms the hook loaded) minus the
regular file, new people, mean of six bases, 80 % person intervals:

| within 0 | within 1 | full 0 | full 1 | runs 0 | runs 1 | mean 6 [80 %] | people up | orders up | V6r (fallback) |
|---|---|---|---|---|---|---|---|---|---|
| +0.17 | +0.08 | −0.29 | +0.00 | −0.24 | −0.01 | **−0.05 [−0.16, +0.06]** | 4/10 | 3/6 | −0.19 |

- **Against the rule:** not kept (below +0.3 and 7/10). Inside ±0.3: **"undecided"** as a direction, this form not kept.
  Prediction +0.1 (−0.2 to +0.4): inside the range, below the point. Gate 1 was passed over by 0.06 points (stated above);
  the outcome would not change any decision either way.
- **Reading:** the events carry real SUB/WORD information alone (64.7 BA, consistent signs incl. the 2 s flip) but their
  margins follow the members' (+0.35 within class × run, twice the eye net's), and SessionEM gains from a view only through
  independence (§188, §233, §234). Same lesson as §233: an accurate pair-specific view that shares the members' failure
  windows adds columns, not information. Small plus under within delivery, minus under full and runs.
- **Forms not tried (if the direction is reopened):** residualise the event columns on the mix's SUB/WORD margin before
  they enter the view; fewer columns (the two q 2 balances and v_le only); the panel's proposal 1 (eye regression out of the
  EEG members) first, which is meant to lower exactly this correlation.

**§236 RESULT (12:16 UTC; evidence/sesstype_14nov/, studies/14nov/sesstype/; replay/check.py OK).** v6f (V6Fr), new people, six
orders, 9 settings (eye off, eye ×2 / ×3 as copies, EM_LAM 0.25 / 1.0, TS_LAM 0.075 / 0.3, BTS 2 / 8); shipped-value no-op
file equal to shipped to 0.0.
- **Every setting as a global change:** eye ×2 +0.09, ×3 +0.09 (inside noise), eye off −0.36, BTS 2 / 8 −0.05 / −0.03,
  TS_LAM 0.075 / 0.3 −0.20 / −0.32, EM_LAM 0.25 / 1.0 −0.70 / −1.46. The shipped constants sit at the optimum of each axis.
- **Oracle ceiling (per session, labels):** all settings within +1.07 [+0.68, +1.50], full +0.70 [+0.58, +0.82], runs +1.18
  [+0.84, +1.55], mean +0.99; eye axis only +0.39 (§231's {off, on} alone: within +0.12 on two orders); stage constants
  only +0.89. With the causal switch at K 40 the ceiling is +0.98. The oracle's pick for a session changes from order to
  order (e.g. P01: bts8, eye×3, ts0.075, ts0.3, eye×3, em1.0), so most of the ceiling is per-order noise, not a session type.
- **Nested held-out (252 splits, K 40, fully nested): −0.28** (within −0.30, full −0.25, runs −0.28; person band [−0.40,
  −0.16]; 9 of 10 people ≤ 0). Global-setting control −0.01. Per descriptor (secondary): e1 −0.07, e2 −0.16, alpha −0.25,
  slow −0.32. K 20: −0.40; K 80: −0.21 (best single descriptor e2 −0.04). The most chosen rule (alpha low → EM_LAM 1.0)
  loses on held-out people.
- **Against the rule: killed** (below +0.3 nested; every form ≤ 0, intervals below 0). Prediction: ceiling +1.5 to +3 (missed
  low, +0.99), nested −0.2 to +0.4 (−0.28, at the low end). The alpha descriptor matches the panel (P01, P02, P08, P09
  peaks; others none), so the descriptor is right and the setting it selects is not.
- **Side finding:** §231's EYEW 0.5 / 2 equal w 1 to 0.0 in all six orders (z-scored SessionEM columns); §231's oracle is an
  on/off oracle. A real eye weight (extra copies) gives +0.09 as a global change: undecided, small.

**§235 RESULT (14 Nov 12:17 UTC; Leonardo CPU, shufcombo d80f97d unedited; evidence/sweeps_14nov/nested_s235.txt; code
studies/14nov/sweeps/).** No-op check exact (EM_LAM 0.5 TS_LAM 0.15 BTS 4 = no-knob file, 0.0 on six orders). Nested
held-out gain over 126 person splits, V6Fr minus shipped, mean [10 %, 90 % across splits], all | within | full | runs:
- EYEW > 0: exact no-op (wave 2 note). EYEW 0 (eye view removed): full data −0.32 (within −0.12, full −0.33, runs −0.51);
  never chosen, held-out 0. The eye view is worth keeping.
- Eye copies (EYEC = extra copies; 1, 2, 3, 5): full data +0.10 / +0.11 / +0.12 / +0.06 (within −0.01 to −0.12, full
  +0.12 to +0.22, runs +0.11 to +0.18; without P10 +0.09 to +0.15; 6–7 of 10 people up). Held-out over {1, 2}: **+0.09
  [+0.07, +0.11]** | −0.02 | +0.16 | +0.12; over {1, 2, 3, 5}: **+0.07 [−0.00, +0.11]** | −0.04 | +0.15 | +0.10. With REST_W
  {1, 2}: +0.05.
- REST_W 0.5 / 2 / 3: full −0.08 / −0.07 / −0.42; held-out **−0.06 [−0.16, 0.00]** | −0.00 | −0.08 | −0.10.
- EM_LAM 0.3 / 0.7 / 1.0: −0.30 / −0.36 / −1.52; held-out **−0.21 [−0.79, 0.00]** | −0.25 | −0.17 | −0.22.
- TS_LAM 0.1 / 0.2 / 0.3: −0.14 / −0.03 / −0.31; held-out **−0.04 [−0.07, 0.00]** | +0.02 | −0.07 | −0.07.
- BTS 2 / 8: −0.06 / −0.03; held-out **−0.03 [−0.05, 0.00]** | −0.04 | −0.02 | −0.02.
- Joint TS_LAM {0.1, 0.15, 0.2} × BTS {2, 4, 8}: best non-shipped −0.02; held-out **−0.07 [−0.15, −0.03]**.
- All 22 arms pooled: held-out −0.06 [−0.41, +0.09] (within −0.12, full +0.01, runs −0.06): selecting among many arms
  costs more than the best arm gains.
- **Against the rule:** no knob reaches +0.3 held-out, none is positive under every scheme (eye copies lose on within).
  **Undecided, the shipped constants stay.** EM_LAM 0.5, TS_LAM 0.15, BTS 4 and REST_W 1 sit at the optimum of this grid
  for v6.2hL-safe (every non-shipped value ≤ 0 on full data); EM_LAM's −0.3 neighbours are the only ones near the
  confirmation threshold. Eye weight 3–4 (copies) is the one positive direction, +0.07 to +0.09 held-out, inside the noise.
- Prediction: stage constants within ±0.15 (EM_LAM missed low at −0.21, the rest held); EYEW / REST_W 0 to +0.2 (REST_W
  −0.06, below); joint 0 to +0.25 (−0.07, below); no knob ≥ +0.3 (held).

### 241 result. OFF-TASK WEIGHTING, TERCILE GATE (14 Nov 12:17 UTC)

**Data.** `studies/14nov/offtask/prest.py` on Leonardo (job 64463250, 3 seeds, smoke 64463136 first): the 4 logits
reproduce the member's 3-way posterior to 1e-5 in every fold. The retrained twins are not bitwise the chain's rest files
(GPU training noise): argmax agreement 0.96 with runs/restshuf/within0 and runs/rest4/probs. p_rest (seed mean) is broad:
mean 0.25, q10/50/90 0.003 / 0.12 / 0.74. Files: runs/diag/restprest/<order>/ (local and Leonardo, gitignored).
**Tercile gate** (`tercile.py`, v6f V6Fr from s215_<order>_off, six bases, new people), accuracy by tercile low -> high
p_rest, bottom - top: within0 +1.62, within1 +1.77, full0 +6.13, full1 +4.76, runs0 +1.09, runs1 +2.37; **mean +2.96
[80 % person +1.49, +6.12]**, per person P01..P10 +9.6 +4.1 +3.0 -1.2 +11.2 -3.0 +14.2 -1.9 -1.7 +3.2 (6/10 > 0).
The rest net's own accuracy falls 67 -> 60 -> 58 over the terciles. With recording-order p_rest (not deployable in a
shuffled stream; correlation 0.90-0.97 with the order's own) the mean gap is +4.01.
**Verdict: gate missed by 0.04 points (below the 3.0 bar, inside its noise): undecided, not closed.** Per the brief, step 2
was not scored. Ready to run: `hooks.py` (weighted SessionEM / TSEM; w = 1 reproduces the stock stages exactly, random w
changes them), `chain.py` (shufcombo via runpy, unedited; W0 within0 reproduces s215_within0_off bit for bit, 46 s per
order on the Mac), `compare.py` (kill rule of s241). Step 2a costs about 10 CPU minutes for W1 and W3 on six orders.
The gap is concentrated in full orders (+5.4) and in three people (P01, P05, P07); 2b (FT label weights) not sized beyond
"one ftrel-style wave per order" since 2a did not run.
**241, exception (14 Nov 12:18 UTC, before any 2a result; coordinator):** the gate miss of 0.04 is inside its noise and 2a costs
about 10 CPU minutes, so 2a runs anyway as a stated exception: W1 (w = 1 - p_rest, primary) and W3 on all six orders,
judged by the pre-registered kill rule (mean below +0.3 or people up < 7/10 -> not adopted; inside ±0.3 undecided).
§242 amendment (14 Nov 12:18 UTC, before any s242 result): §235 kept the shipped stage constants (no knob reached +0.3; every
non-shipped EM_LAM / TS_LAM / BTS / REST_W value at or below zero), so stk2 changes no constant. Instead stk2 = stk1 plus two
extra copies of the eye view (eye weight 3, §235's only positive, +0.09: full +0.16, runs +0.12, within -0.02), via one
EXTRA_VIEW per order built inside the job (studies/14nov/stack/stk2view.py). Prediction: stk2 - stk1 +0.05 (inside noise;
reported, not used to choose unless positive in every scheme). Also: EYEW other than 0/1 now fails loudly (671465b).
**241 step 2a result (14 Nov 12:23 UTC; `chain.py` + `compare.py`, v6f chain under the s215 env, paired against s215_<order>_off,
W0 bit-identical on within0).** W1 (w = 1 - p_rest, mean w 0.75): within0 -0.29, within1 -0.23, full0 -0.18, full1 -0.26,
runs0 -0.40, runs1 -0.20; **mean -0.26 [80 % person -0.43, -0.09], 4/10 up**, without P10 -0.22; the same hook on v5.1b
(P0r) -0.24 to -0.70. W3: **mean -0.59 [-0.83, -0.35], 2/10 up**, negative on all six orders. **Verdict: refuted for this
form (kill rule fails; W1 negative on every order with the person interval below 0, and stronger weighting loses more).**
Down-weighting rest-like windows in the template estimates removes useful template mass; the tercile gap reflects
harder windows, not wrong teachers. 2b (FT label weights) not run.
- **§238 RESULT (14 Nov 12:32 UTC; runs/s228c_v62hL.txt, runs/rundetect/{package_returning,robust_v6f,robust_v62hL}.txt).**
  - New people, v6.2hL (V6Fr of v62hL_<order>.npz), gated filter: within (12) +0.31 [+0.18, +0.46], runs (4) +0.22
    [+0.08, +0.36], full (8) +0.00 [−0.04, +0.04]; true runs +0.32 / +0.23 / +1.11. False latches under full 17 of 112
    sessions (15 %), net cost 0. Windows 0–40 again cost (−0.05).
  - Returning people (seen_session protocol, retv6 outputs; mean over orders, P04 / P10): ER within (8) +0.08 (+0.08 /
    +0.08), runs (4) +0.11 (−0.19 / +0.40), full (4) −0.02 (−0.17 / +0.12); ERF +0.15 (+0.17 / +0.12), +0.11 (+0.09 / +0.14),
    +0.01 (−0.09 / +0.11). True runs give only +0.03 / +0.08 on within here: little room on these outputs. Sealed weighting
    (0.2 P04 + 0.8 P10) on ER: within +0.08, runs +0.28, full +0.06.
  - Robustness (synthetic sessions, chain outputs reused from within 0, balanced accuracy per session; v6f / v6.2hL; mean
    [80 % band], worst person): normal +0.58 / +0.75; drop_online +0.33 / +0.39 (−0.31); dup_online +0.47 / +0.54;
    online36 +0.20 / +0.32 (−0.37); no_acq +0.33 / +0.41; acq_last +0.45 / +0.50 (−0.66 / −0.60); double +0.40 / +0.42;
    drop10 +0.22 / +0.32 (−0.47). P10's third session (real online-36 deviation) +1.47 / +1.86 under normal. No violation
    mean is below −0.2 (all positive), so by the rule no self-check was added; single people lose up to 0.66 (v6f / v6.2hL).
  - Method statement: runs/rundetect/method_statement.md (corrected: the filter's blocks come from delivery position and
    the design sizes, weighted by the posteriors' fit to the urns, not from EEG grouping; the detector alone uses EEG).

**PV2 / PV3 RESULT (14 Nov; runs/pview/chain_all.log, chain_summary.txt; 16 orders × 6 bases; ER reproduced s214 exactly on
every order and base).** Change against ER (P04 71.05, P10 90.50 absolute), six-base mean, 80 % bootstrap over orders:

| arm | all 16: mean [80 %] | P04 | P10 | P10 > 0 | within / full / runs (mean) | verdict by the rule |
|---|---|---|---|---|---|---|
| O personal ocular columns | −0.16 [−0.22, −0.10] | −0.08 | −0.24 | 2/16 | −0.19 / −0.24 / −0.03 | undecided (inside ±0.3), negative |
| M personal ocular member | −0.44 [−0.52, −0.37] | −0.31 | −0.57 | 2/16 | −0.38 / −0.54 / −0.47 | this form closed |
| S SessionEM prior κ 20 | −1.92 [−2.16, −1.67] | −1.76 | −2.08 | 0/16 | −1.70 / −1.89 / −2.39 | this form closed |
| **T TSEM prior κ 20** | **+0.43 [+0.26, +0.60]** | **+0.81** | +0.04 | 9/16 | +0.21 / **+0.84** / +0.43 | misses P10 > 0 ≥ 12/16; undecided-positive |
| OST | −2.07 | −1.95 | −2.18 | 0/16 | | closed (S carries it) |
| W (LOO strength) | −2.01 | −0.53 | −3.49 | 0/16 | | closed (and §227) |
| H25 head share 0.25 | +0.22 [+0.10, +0.36] | +0.43 | +0.02 | 8/16 | +0.41 / −0.05 / +0.13 | undecided |
| H33 | −0.02 | +0.28 | −0.32 | 5/16 | +0.23 / −0.10 / −0.44 | undecided |
| H40 | −0.62 | −0.47 | −0.77 | 1/16 | | closed |
| B20 own beta 0.20 | +0.24 [+0.12, +0.36] | **+1.42** | **−0.94** | 0/16 | +0.31 / +0.03 / +0.32 | fails (P10 0/16) |
| B30 | −0.29 | +1.28 | −1.87 | 0/16 | | fails |

- **Against the predictions:** O +0.4 → −0.16; M +0.2 → −0.44; S +0.2 → −1.92; T 0 → +0.43; OST +0.5 → −2.07; H33 +0.3 → −0.02;
  H25 +0.2 → +0.22; H40 +0.2 → −0.62; B20 +0.1 → +0.24; B30 0 → −0.29. The ocular predictions are refuted.
- **Reading:** the personal ocular model alone is 73 % for P10 (41 % for P04) in every order, but P10's chain is already at
  90.5 % and the 73 % view adds disagreement, not information; as SessionEM columns it costs −0.24 for P10. The
  SessionEM prior at κ 20 is strongly harmful for both (possible cause, unverified: the prior's eye-net columns come
  from the LOSO eye net in recording order while the session's come from the re-streamed nets, and GE's event rates
  from recording order; a mismatch in coordinates would pull the templates the wrong way). The TSEM prior is the
  one personal-template form with a positive sign (P04 +0.81, positive on all 4 full orders for both people), but
  P10 is flat. Stronger strengths (§227) help P04 and hurt P10: own-space beta 0.20 is +1.42 / −0.94, the head share
  shows the same pattern more weakly. With two dev people, "common strength" is a P04-vs-P10 trade, and the sealed
  set weights P10-like people about 4:1 (STATE §4).
- **(iii) per-person strength:** W is the worst arm for P10 (−3.49) because LOO gives P10 strength 1.0 and P04 0.26 —
  the opposite of what the chain needs. Consistent with §227 (LOO ranks people wrongly).

### PV4. TSEM PERSONAL PRIOR, MORE FORMS (pview track; written 14 Nov, before any result)
T is the only personal-template arm with a positive mean, carried by P04; AGENTS asks for several forms of an idea with
a measured ceiling before setting it aside. Arms in chain.py (`TAG=pv4_ ARMS=ER,T10,T40,TH25`), same 16 orders:
T10 / T40 (κ 10 / 40), TH25 (T κ 20 + head share 0.25).
- **Prediction:** T10 +0.3 (P10 +0.05), T40 +0.4 (P04 +1.0, P10 −0.1), TH25 +0.55 (P04 +1.1, P10 0).
- **Rule:** as PV2. A form passes if mean ≥ +0.3, P10 > 0 on ≥ 12/16, no scheme mean below −0.3 for either person.

### 243. FINE-TUNING DRAWS FOR THE STACKED ARM (written 14 Nov 12:35 UTC, before any result; coordinator 12:32 item 2)
Mechanism: the stacked arm's fine-tuning (r15, and its control s15, on LP labels) re-run with fresh fine-tuning randomness:
ftrel.py DRAW=1,2 seeds torch and the augmentation generator with seed + 1000 DRAW exactly as ttt7 does (DRAW 0 unchanged,
bit-identical path; tag ttt18<order>d<DRAW>LP). Base nets are the same, so this measures fine-tuning noise only. Then the
s242 arms (ctl, stk1, stk2) on each draw, Leonardo CPU, SAVE_OUT s242<arm>d<DRAW>_<order>. Orders: the 20 with LP labels
(full4-7 have none; returning people have no r15 path, so their spread is not measured here).
Prediction: across the three draws, the spread (max - min) of the mean stk1 + filter - ctl over 20 orders is below 0.15,
and stk1 + filter > ctl in every draw. Refutation: spread above 0.3 or a draw with stk1 + filter <= ctl -> the stack's
number is draw-dependent and is reported with that range, not as a point.

### PV5. OCULAR VIEWS AND THE SESSIONEM PRIOR WITH MATCHED COORDINATES (pview track; coordinator; written 14 Nov, before any result)
**What actually ran in PV2 (checked in code and stores):**
- **Test side** (seen_session, sessions 2/3): GE48 = gaze columns (stateless, `session_em.gaze_features`) + event rates
  re-streamed in the delivery order (`runs/geshuf.npz` "<order>|seen_session", `session_em.EventRates`, global MAD
  prior), put in by retv6's apply_members. Eye columns = `runs/ocushuf_ss/<order>/oc_nomix-seen_session-s*.npz`
  (9 seeds; nets of the seen_session protocol, which trained on P04/P10 session 1; standardised in delivery order).
- **Personal ocular model (O, M):** trained on session-1 GE48 from `inputs.npz` all_loso. Checked: identical to the
  seen_session store on shared windows, and its event rates equal geshuf "rec" exactly. So the column definitions
  matched (same code path) and only the stream order of the training session differed (recording vs shuffled).
  Standardisation: training whole-session z; test causal prefix plus the training stats as 10 pseudo-windows.
  **O and M were not scored with mismatched coordinates.**
- **SessionEM prior (S, OST, W): mismatched.** Its eye columns came from `runs/ocunet/probs/oc_nomix-all_loso-s0..3`
  (LOSO nets that never saw P04 or P10, 4 seeds, recording order). The session's eye columns come from the
  seen_session nets above (which saw session 1). So two different nets fed the prior's means and the session's
  features. GE48 columns matched as for O; the O columns of OST/W were cross-fitted LORO outputs (matched model,
  different fit).
**Re-run** (chain.py, same 16 orders, against ER):
- **Columns, one code path:** gaze + event rates (geshuf) for training and test. The public session is streamed in
  the same scheme and seed as the test (geshuf "<order>|all_loso"); the test's are asserted to be geshuf
  "<order>|seen_session" and the gaze columns bit-equal.
- **OG48 / OG32:** personal ocular LR on gaze + event rates, or on gaze only, as 3 SessionEM columns.
- **SG48 / SG32:** SessionEM prior κ 20 on those columns only. The eye-net columns get no pull (per-column mask);
  that is the mismatched part removed.
- **Prediction:** OG48 ≈ O (−0.15 ± 0.15); OG32 −0.1. SG48 −1.0 (−1.8 to −0.3); SG32 −0.6. The mechanism I expect:
  cross-session personal GE templates are near chance for P04 (PV1 42–45) and at κ 20 against SessionEM's
  ≈ 7 windows per class at its start (window 20, early boost ×3) the prior sets the early templates.
- **Rule:** as PV2. If SG is still ≤ −0.3, close the SessionEM prior with that mechanism. If it is positive, report it
  per scheme with intervals.

### PV6. THE TSEM PRIOR ON A WITHIN-SESSION PROXY FOR ALL TEN (pview track; written 14 Nov, before any result)
`studies/14nov/pview/tproxy.py`. Each public session: the acquisition runs (one Graz, one BrainHero) play the public
session (class means of their core1 PCs, offline labels). The online runs get the prior (κ 20, 40) in TSEM.
- **Chain and scoring:** v5.1b's new-people tail (gm.pipe_g, members as recorded), orders rec, within 0–3, full 0/1,
  runs 0/1, six bases. Scored on online runs only; acquisition rows asserted unchanged.
- **Optimism:** the proxy has no between-session shift, so it is an upper bound for the real case.
- **Prediction:** T20 +0.8 mean over ten (people > 0: 7/10), T40 +1.0.
- **Reading:** if the proxy is ≤ +0.3 even in this optimistic setting, T has no case for the eight single-session
  people.

**PV4 RESULT (14 Nov; runs/pview/chain_pv4.log, pv4_summary.txt; 16 orders × 6 bases; ER = s214 on every order and base).**
The first launch died before any result: a KeyError when the arm list lacked `base`. chain.py now checks only the arms
it runs against s214 and requires ER. Change against ER, 80 % over orders:

| arm | all 16 | P04 | P10 | P10 > 0 | within | full | runs |
|---|---|---|---|---|---|---|---|
| T10 (κ 10) | +0.36 [+0.22, +0.49] | +0.65 | +0.06 | 8/16 | +0.13 (P04 +0.33, P10 −0.07) | +0.61 (+0.90, +0.32) | +0.56 (+1.04, +0.08) |
| T (κ 20, PV2) | +0.43 [+0.26, +0.60] | +0.81 | +0.04 | 9/16 | +0.21 (+0.54, −0.11) | +0.84 (+1.19, +0.50) | +0.43 (+0.98, −0.11) |
| T40 (κ 40) | +0.44 [+0.24, +0.65] | +0.90 | −0.01 | 7/16 | +0.25 (+0.59, −0.09) | +0.76 (+1.39, +0.14) | +0.51 (+1.02, +0.01) |
| TH25 (κ 20 + head 0.25) | +0.41 [+0.23, +0.59] | +0.87 | −0.05 | 6/16 | +0.31 (+0.67, −0.06) | +0.51 (+0.97, +0.05) | +0.52 (+1.17, −0.13) |

- **Against the predictions:** T10 +0.3 → +0.36 (P10 +0.05 → +0.06); T40 +0.4 → +0.44 (P04 +1.0 → +0.90, P10 −0.1 → −0.01);
  TH25 +0.55 → +0.41. T10 and T40 were close; TH25 does not stack.
- **Verdict:** no form passes (P10 > 0 on 6–9 of 16 against the 12 needed). Every form helps P04 (+0.65 to +0.90,
  largest under full) and leaves P10 at zero (its chain is at 90.5 %). The rule calls this "undecided". By the
  sealed weighting (about 0.2 P04 + 0.8 P10) it is worth about +0.15 on returning people. Not closed: it never hurts
  P10 on a scheme mean by more than 0.13.
§242 result (14 Nov 12:52 UTC; evidence/stack_14nov/s242_score.txt; replay/check.py not affected, Leonardo shufcombo = local).
New people, change against ctl (s15 on LP labels, no view, no filter; = v6.2hL-safe: LF s15 ref is +0.01 against it),
mean of schemes (equal weight), [80 % interval over people and orders], people up:
  filter alone   +0.15 [+0.05, +0.26] 7/10   (within +0.32, full -0.03, runs +0.18)
  stk1           +0.15 [+0.07, +0.23] 9/10   (within +0.09, full +0.26, runs +0.10)
  stk1 + filter  +0.28 [+0.16, +0.40] 10/10  (within +0.37, full +0.27, runs +0.21)
  stk2           +0.22 [+0.09, +0.34] 9/10   (within -0.00, full +0.48, runs +0.17)
  stk2 + filter  +0.34 [+0.19, +0.49] 9/10   (within +0.26, full +0.50, runs +0.27)
Below the +0.45 prediction, above the best single part (filter +0.15): the parts stack, not refuted. stk2's extra eye
weight moves gain from within (-0.11 against stk1 + filter) to full (+0.23); not positive in every scheme, so by the
amendment it is reported, not chosen. Returning people: only the filter is measured (§238: ER sealed-weighted within
+0.08, runs +0.28, full +0.06); r15 and the views have no returning-people runs (absent, assumed 0). Sealed-weighted
J (0.566 new + 0.434 returning, scheme means): stk1 + filter about +0.22, stk2 + filter about +0.25. Single fine-tuning
draw; §243 adds draws 1 and 2.

### 244. DEPLOYABLE EYE-EMBEDDING VIEW (written 14 Nov 12:57 UTC, before any result; package question from v52-build)
Finding (silent mismatch): s234's E view (and so §242's stk1/stk2) concatenates each fold net's HELD-OUT embeddings
(ocunet.run keeps only te rows), so every person's rows come from a different net and the PCA on "the other nine people"
mixes nine nets' coordinate systems. No package can compute that. Deployable analogue (studies/14nov/stack/ocufold.py):
per held-out person, the fold net (as ocushuf trains it, 3 seeds = the package's eye_s0-2) embeds all rows; PCA 4 on the
in-sample training rows (384 columns standardised there), applied to the held-out rows. GPU: 20 orders x 3 seeds, Leonardo.
Then stk1dep = r15 + Edep view, and the deployable-only arms already submitted: r15 alone, r15 + eye x3 (no E).
Prediction: Edep adds +0.00 to +0.10 over r15 alone (less than the dev E: in-sample embeddings are sharper than held-out).
Refutation: Edep minus r15 alone <= 0 over 20 orders -> the view is not packaged; the upload stack is r15 (+ eye x3 if it
holds) + filter.

**PV6 RESULT (14 Nov; runs/pview/tproxy.log, tproxy.npz; six bases; acquisition rows unchanged, asserted).** TSEM prior
from the acquisition runs, applied on the online runs of the same session. Change on online runs against v5.1b's
unprimed new-people tail (online base 77–81):
- **T20:** mean +1.93, people > 0 9/10 (P01 +4.65, P02 −0.60, P03 +2.38, P04 +1.53, P05 +1.85, P06 +1.10, P07 +1.99,
  P08 +0.32, P09 +3.38, P10 +2.67). Per scheme: rec +1.46, within +1.64 (4/4 orders > 0), full +2.46, runs +2.20.
- **T40:** mean +2.44, 9/10 (P02 −0.93). Per scheme: rec +1.81, within +2.20, full +3.02, runs +2.67.
- **Against the prediction:** T20 +0.8 → +1.93, T40 +1.0 → +2.44; people > 0 7/10 → 9/10. Higher than predicted.
- **Reading, with two limits:**
  1. Same-session templates, no between-session shift.
  2. The proxy's baseline has no personal head and no own-space stage (new-people tail), so the prior is the only
     personal information there. In the routed chain those two already carry much of it.
  The gap between this proxy (P10 +2.67) and the real across-session case (PV2/PV4: P10 0.00 ± 0.1) is the size of
  those two effects together, and the dev data cannot separate them. For the eight single-session sealed people
  the real case is across sessions, like P04 and P10. So the proxy shows the mechanism works inside a session and
  does not show it carries to a later one. P04, the one dev person where it does carry (+0.8), is also the weakest
  person under the chain.

**PV5 RESULT (14 Nov; runs/pview/chain_pv5.log, pv5_summary.txt; 16 orders × 6 bases; ER = s214 everywhere).**
- **Matching asserted on every order:** gaze columns bit-equal; test event rates = geshuf "<order>|seen_session"
  within 1e-9 (apply_members adds new − rec to the stored rec stream, so equality holds only to rounding; the
  first launch's exact-equality assert failed on runs0 for that reason and was relaxed to 1e-9). Training columns
  are geshuf "<order>|all_loso" for session 1.
- **Standalone personal ocular (matched):** G48 P04 41–43, P10 72–74; G32 P04 41–42, P10 56–58.

| arm | all 16 | P04 | P10 | P10 > 0 | within | full | runs |
|---|---|---|---|---|---|---|---|
| OG48 (matched O) | −0.17 [−0.22, −0.11] | −0.10 | −0.23 | 2/16 | −0.17 | −0.26 | −0.06 |
| OG32 (gaze only) | +0.04 [−0.02, +0.10] | +0.10 | −0.03 | 6/16 | −0.07 | −0.01 | +0.30 |
| SG48 (matched S, eye columns unpulled) | −1.65 [−1.91, −1.42] | −1.52 | −1.78 | 0/16 | −1.35 | −1.71 | −2.19 |
| SG32 (gaze only) | −0.36 [−0.53, −0.19] | −0.09 | −0.64 | 3/16 | −0.14 | −0.58 | −0.58 |

- **Against the predictions:** OG48 −0.15 → −0.17; OG32 −0.1 → +0.04; SG48 −1.0 → −1.65; SG32 −0.6 → −0.36.
- **Is PV2's −1.92 an artefact?** Only in small part. The mismatched eye-net columns account for about 0.27
  (S −1.92 → SG48 −1.65). The coordinates of O and M were already matched (same code path, only the training
  session's stream order differed), and OG48 reproduces O (−0.17 vs −0.16).
- **SessionEM personal prior: closed in this form (κ 20, unshrunk class means).** Most of the loss comes from the
  event-rate columns (SG48 vs SG32: −1.3).
  - **Likely mechanism (not isolated by a run):** SessionEM shrinks its own class means by m0 = 20, to
    Σ / (n + 20), while the prior adds full-size session-1 means. At the stage's start (window 20, ≈ 7 windows per
    class, early boost up to ×3), the templates are dominated by those means, so they are overconfident and placed
    where session 1's classes sat.
  - **Why that is the wrong place:** across sessions the person's GE templates are near chance for P04 (PV1 42–45)
    and only 72 % for P10, against a chain at 90.5 %. The prior therefore pulls early SessionEM decisions toward a
    weaker classifier.
  - Untested variant: a prior shrunk to the stage's own scale, i.e. pm × n_pub / (n_pub + m0), or a small κ.
- **Personal ocular as SessionEM columns (O / OG48 / OG32):** undecided-to-negative; inside ±0.3 for every form,
  and P10 never gains.

**§242 package (14 Nov 13:15 UTC, v52-build; written before the full checks, after one-session smokes): v6.4, r15 and
the count filter behind their own keys.**
- **Form:** `lean/src64` → `submissions/lean_v64` (`submission.py` `6ab43e4b…`, 674e0d6).
  - `v6_ft_relabel` (r15, ni26-push's R1/R2): at each refit at t, before recognition only, the label chain's SessionEM
    and TSEM state for window t is re-applied to windows 0..t-1 (`Tail.reapply`). TSEM selects templates from its
    as-emitted input; labels are argmax, weights max; the fine-tuning loss is unchanged. Constants checked equal to
    relabel.py's (em_nmin 20, boosts 2.0 / 4.0, …).
  - `v6_cf_on`: the count filter of v6.2c, now with its per-session log line from `__del__`.
  - The builder requires both keys. E_pca4 is **not** packaged: dev's view comes from a different fold net per person, so
    it has no package equivalent (ni26-push, 13:00 UTC).
  - Packages `cand_v64_r1iLu_{off,cf,r15,all}` on v6.2iLu ER's keys.
- **Smokes (P01, recording, Mac CPU):** r15 reference against relabel.py's formulas recomputed from the package's stored
  stage inputs: 17/17 refits, max difference 0.0, 0.13 ms per relabelling. All-off package bitwise equal to v6.2iLu ER.
- **Predictions:**
  1. all-off bitwise equal to v6.2iLu ER (14 sessions, both gate modes);
  2. r15 reference max difference ≤ 1e-10 in recording, within 0–2 and full 0–2 (14 sessions each);
  3. the filter's reference holds with r15 on;
  4. forced exception in the relabelling: errors 1, v5.1b from the refit on;
  5. stress cases complete;
  6. replays files unchanged; GPU worker image passes;
  7. r15 adds under 1 % of a refit's time.
- **Refutation rule:** any miss fails that key's package.

### 245. RETURNING-PEOPLE STACKED ARM: ER + TSEM PERSONAL PRIOR + COUNT FILTER (written 14 Nov 13:19 UTC, before any result; coordinator 13:20)
Mechanism: the returning half of the upload candidate as one arm. ER (v6.2hL-safe returning rows), plus the pview helper's
arm T (personal class means of the session-standardised core1 PCs from the person's public session as a TSEM template prior,
kappa 20, routed tail only; PV2), plus the detector-gated count filter (§238, package.py PROTO=seen_session). The pview chain
gets SAVE_OUT (per-window outputs, recording layout; ER asserted equal to runs/chainout/ret_<order>.npz).
Orders: within0-7, full0-3, runs0-3 (16). Report P04 and P10 separately and sealed-weighted (0.2 P04 + 0.8 P10), per scheme.
Prediction: ER + T + filter against ER raw, sealed-weighted: +0.25 (T's +0.43 is mostly P04 at weight 0.2; the filter adds
about +0.14 sealed-weighted). Refutation: T + filter below the filter alone, or P10 below -0.2 on a scheme mean -> T is not
carried (the sealed set weights P10 at 0.8).
§245 result (14 Nov 13:27 UTC; evidence/stack_14nov/s245_score.txt; ER per order and base equal to s214, asserted; saved ER equal
to runs/chainout/ret_<order>.npz). Change against ER raw, P04 / P10 / sealed (0.2 P04 + 0.8 P10), mean over orders per scheme:
  filter alone  within +0.08/+0.08/+0.08 | full -0.17/+0.12/+0.06 | runs -0.19/+0.40/+0.28 | mean sealed +0.14
  T             within +0.54/-0.11/+0.02 | full +1.19/+0.50/+0.64 | runs +0.98/-0.11/+0.10 | mean sealed +0.25
  T + filter    within +0.60/-0.01/+0.11 | full +1.12/+0.51/+0.63 | runs +0.72/-0.10/+0.06 | mean sealed +0.27
Not refuted (T + filter above the filter alone; P10 never below -0.2 on a scheme mean, lowest -0.10). Matches the +0.25
prediction. The filter adds little on top of T (+0.02 sealed; it helps P10 on runs without T, +0.40, and T takes that away).
Per order the sealed value ranges -0.66 to +1.55 (two people): wide, report with that range.
§242/§244 deployable arms (14 Nov 14:15 UTC; evidence/stack_14nov/s242_deployable.txt). New people, 20 orders, against ctl:
  r15            +0.10 [+0.03, +0.18] 9/10   (within +0.06, full +0.17, runs +0.07)
  r15 + filter   +0.25 [+0.13, +0.38] 9/10   (within +0.36, full +0.17, runs +0.22)   <- fallback stack, deployable as measured
  r15e3          +0.19 [+0.08, +0.31] 9/10   (within -0.02, full +0.36, runs +0.24)
  r15e3 + filter +0.30 [+0.17, +0.43] 9/10   (within +0.28, full +0.36, runs +0.25)
The dev E view adds +0.03 over r15 (stk1 + filter +0.28 against r15 + filter +0.25): small, so the stack does not hinge on
it. Eye x3 against r15 + filter: within -0.08, full +0.19, runs +0.03; not positive in every scheme, so by the §242
amendment it is reported, not chosen. Sealed-weighted J with §245's returning arm (+0.27): r15 + filter about +0.26,
r15e3 + filter about +0.29. Single fine-tuning draw.

**§239 RESULT, PART (1) TEMPLATES (14 Nov 14:19 UTC; runs/designdec/score_final.txt, proxy.txt; code 0e8d214 + scorer fix).**
Package form: shufcombo run unmodified through runpy with the stage installed (studies/14nov/designdec/wrap.py). No-op:
feedback off reproduces runs/chainout/within0.npz exactly (max |diff| 0, Mac and Leonardo); Leonardo and Mac outputs agree
to 0.0 on every shared key; replay/check.py OK. Change of variant + gated J filter against unmodified chain + the same filter
(the s239 decision number), new people, mean of 2 orders [80 % person band]; 6 bases where marked, else B2_pool:

| form (V6Fr) | within | runs | P0r within / runs |
|---|---|---|---|
| ext/J (6 bases) | +0.19 [−0.05, +0.42] | +0.10 [−0.09, +0.30] | +0.03 / +0.43 |
| rep/J/ts, TSEM only (6 bases; form added after the first results) | +0.27 [+0.05, +0.48] | −0.20 [−0.33, −0.08] | +0.11 / +0.27 |
| ext/J/ts (6 bases) | +0.16 | – (runs 0 only) | −0.09 / – |
| ext/true, ceiling (6 bases) | +0.30 [+0.07, +0.54] | +0.20 [+0.04, +0.36] | +0.03 / +0.68 |
| ext/true/ts (6 bases) | +0.20 | – | −0.02 / – |
| rep/J (B2_pool) | −0.41 | −0.93 | −0.05 / −0.59 |
| fwd/J (B2_pool) | −0.53 | −1.40 | −0.04 / −0.86 |
| rep/true (B2_pool) | −0.63 | −0.93 | +0.37 / −0.40 |

- **Kill met:** no J form reaches +0.3 over the gated filter on within (best registered form ext/J +0.19, best post-hoc
  form rep/J/ts +0.27 but −0.20 on runs). Inside the noise: direction undecided, not a package change.
- **Collapse (the panel's warning):** every form that replaces SessionEM's soft labels with the run posterior (rep, fwd,
  also with true runs) collapses under runs: order means −0.86 to −1.62 on V6Fr, single people −5 to −6.4. It is not
  boundary error (rep/true collapses too); SessionEM's EM seeds confirm the chain's own counts. Only the extrinsic
  message (ext) or TSEM-only feedback stays safe (person minima ≥ −1.0).
- **Ceiling:** with true runs, ext adds +0.30 / +0.20 over the true-run-gated filter, so the template channel holds about a
  third of the gap between the causal filter (+0.5) and the offline bound (+1.1); J-wprior keeps about two thirds of it on
  within, less on runs.
- **Self-contained replay** (members as recorded, B2_pool, within 0): ext/J chain alone +0.61, with the filter −0.17:
  the feedback and the output filter use the same information.
- **(2) proxy** (runs/designdec/proxy.txt): prefix-smoothed labels at s15 refits, J: +1.05 BA over the emitted labels on
  within (+1.00 runs), +0.30 over r15's relabels; true runs +1.57 / +0.82. Below the +2 rule. r15's +0.75 label gain gave
  +0.12 in §221, so the smoothed labels predict about +0.05 over r15.

**§245 package (14 Nov 14:25 UTC, v52-build; written before the full checks, after a one-session smoke): v6.5, T behind
its key, and the upload candidate.**
- **Form:** `lean/src65` → `submissions/lean_v65` (`dba089f1…`, 01e5288); v6.4's bundle untouched.
  - `v6_tpri_on`: route4's personal TSEM template prior, in RouteB's routed tail only. On a routed window with ramp r and
    person j, TSEM's class means are (Σ + 20 r · pri[j]) / (n + m0 + 20 r), with pview's denominator fix. v5.1b's Route,
    the gates and the shadow are unchanged.
  - pri (`lean/tools/fit_tpri.py`, `runs/tpri/tpri.npz`): for each of the ten people, the mean over all their public
    sessions of the class means of rlib3.stdz of the package's own core1 TSEM coordinates (128). Dev used session 1 of
    P04 / P10 only; that difference cannot be measured on dev.
  - Packages `cand_v65_r1iLu_{off, T, cfr15, cand}`. **cand** = filter + r15 + T on v6.2iLu ER's keys: the upload
    candidate next to the checked fallback `cand_v62_r1iLu`.
- **Smoke (P01, recording, gates as shipped):** the prior was applied on 180 windows from the first routed one (10); the
  TSEM steps equal pview's formula (difference 0.0); the stored prior equals route4's expression recomputed (0.0);
  windows before routing are unchanged against T off; T moves the posteriors by up to 0.09 and changes no label there.
- **Predictions:**
  1. T off is bitwise equal to v6.2iLu ER, and cfr15 to v6.4 "all" (14 sessions, both gate modes);
  2. the T reference (TSEM ≤ 1e-10, prior ≤ 1e-10, equal before routing) holds in recording, within 0 / 1, full 0 / 1,
     runs 0 / 1;
  3. on cand, the filter, r15 and T references all hold;
  4. forced exception at the TSEM and FT sites: errors 1, then v5.1b;
  5. stress cases complete; replays files unchanged; GPU worker image passes.
- **Not checkable here:** a window-exact comparison with pview's dev outputs. Dev's chain uses seen_session members and
  core1 trained on session 1 of P04 / P10; the package's are trained on all ten people.
§244 result (14 Nov 14:37 UTC; evidence/stack_14nov/s244_score.txt). Deployable eye-embedding view (fold net embeds all rows,
3 seeds, PCA 4 on the in-sample training rows), new people, 20 orders, against ctl:
  stk1dep          +0.14 [+0.04, +0.25] 10/10  (within +0.12, full +0.20, runs +0.10)
  stk1dep + filter +0.27 [+0.14, +0.40] 10/10  (within +0.40, full +0.19, runs +0.22)
Against r15 + filter (+0.25): +0.02 (within +0.04, full +0.02, runs 0.00). Inside the prediction (+0.00 to +0.10), not
refuted, but inside the noise: undecided. Not worth a package key (PCA weights + an embedding pass for +0.02); the dev
construction's +0.03 is the same size, so the mixed-net defect did not inflate it.

**§239 PART (2) LAUNCHED (14 Nov 14:50 UTC; no result read).** Gate: the best registered J form (ext/J) is +0.19 ≥ +0.15 on within,
so (2) runs (the proxy alone would not have opened it). Labels (studies/14nov/designdec/labels.py): r15's per-refit labels
smoothed by the J-wprior urn over each refit's prefix, from the latch on -> runs/labels/relab_dd239_<order>.npz. Label BA
over r15: within 0 +0.60, within 1 +0.54, runs 0 +0.81, runs 1 +1.05 (r15 over emitted about +1.2). FT: ftrel_dd.py =
ftrel.py unchanged with those labels and tag ttt18<order>LPD (s229's LP files untouched); smoke on Leonardo: control s15 vs
ttt7.stream 5e-7, SMOKE OK. Leonardo arrays 64471408–11 (within 0/1, runs 0/1, 24 tasks each); scoring jobs 64471438–41
(afterok): shufcombo through wrap.py (off/J = stock stages), FT_V r15 with FT_TAG LP and LPD, 6 bases ->
runs/designdec/sc_<order>_off_J_LP{,D}r15.npz on Leonardo; then score2.py. **Prediction:** smoothed-label r15 − r15 on V6Fr
+0.05 (−0.15 to +0.25) within, with the gated filter on top about the same. **Rule:** kept only if ≥ +0.3 over r15 + gated
filter on within with runs ≥ −0.1; inside ±0.3 undecided.

### 249. WHERE THE SEALED POINTS ARE: DEV LEVELS (14 Nov 15:08 UTC; coordinator 15:10 and its correction)
Candidate's dev levels, per-person BA, schemes weighted equally. New people (20 orders, r15 raw): 79.07, of which P01 76.7,
P02 89.1, P03 73.1, P04 52.6, P05 75.6, P06 83.4, P07 88.8, P08 86.2, P09 82.6, P10 82.5. Returning (16 orders, seen_session):
ER P04 70.6 / P10 90.2; T P04 71.5 / P10 90.3. Calibration, stated correctly: 0.820 is v5.1b's sealed score; v5.1b's own
dev levels are lower by the candidate's gains (about 1.2 new, 0.6 returning), giving about 0.566 x 77.8 + 0.434 x
(0.2 x 70.0 + 0.8 x 89.6) = 81.2 against sealed 82.0: the sealed set runs about +0.8 above dev for v5.1b. The
"0.8 P10-like" weighting of the eight single-session returning people is an assumption inside that sum.
Test of the assumption, closest number on disk: the within-session proxy (runs/pview/tproxy.npz, PV6: acquisition runs as
the public session, online runs scored, 8 shuffled orders), population new-people tail without / with the personal TSEM
prior: base mean 78.3 (P01 70.9, P02 91.8, P03 74.0, P04 48.0, P05 76.1, P06 83.7, P07 90.2, P08 86.2, P09 80.4, P10 81.7);
T20 80.4 (P01 75.9, P02 91.2, P03 76.4, P04 49.9, P05 78.0, P06 84.7, P07 92.7, P08 86.4, P09 84.1, P10 84.6). This is not
the routed path (no personal members or own-space): P10 is 84.6 here against 90.3 routed across sessions, so routing adds
about +6 for P10 beyond the prior. The single-session people sit from P01 76 to P07 93 on this proxy, spread like the new
people, not like P10 at 90. The routed level per person inside a session is what §246 (b) measures.

### 247. STRONG PERSONAL PATH FOR RETURNING PEOPLE (written 14 Nov 15:09 UTC, before any result)
Code `studies/14nov/strongpersonal/`, outputs `runs/strongpersonal/`. A whole personal system for a recognised person,
scored against the routed path (ER, and ER + T of §245) on the same test sessions of P04 and P10.
- **System (ceiling form first):** for test session S of person q, personal members trained with true labels on q's
  public sessions before S: (a) a tangent-space core (CAR, overlapping bands, full-window covariances, each session
  recentred on its own whole-session mean = Riemannian EA; the test session too, non-causal), logistic regression, in
  three training sets: q only; the nine others + q upweighted; q + the test session with ER's labels (whole session,
  the non-causal bound of pseudo-label adaptation); (b) the stored PFT nets of §29 (5 families x 3 seeds, fine-tuned on
  session 1; protocol B only); (c) their geometric mean. Fusion with the routed output: ER x personal^lambda, lambda
  picked per person on the test session itself (oracle), next to fixed lambda 0.5 / 1.
- **Protocols:** B = train session 1, test 2 and 3 (seen_session, compares with ER / T directly); A = train 1 + 2,
  test 3 (the package's case). The routed chain exists only with session 1 public, so on A the reference is ER on
  session 3 trained on session 1 (a lower bound of the routed path with two public sessions); the strong path is also
  scored s1 -> s3 so that one comparison is like for like.
- **Orders:** the 16 of §245 (within 0-7, full 0-3, runs 0-3) x 6 bases for anything fused with ER; the ceiling's personal
  part is order-free (non-causal).
- **Prediction:** personal alone below the routed path: B P04 64 (58-70), P10 86 (82-90) vs ER 70.6 / 90.2; A +2 to +4
  over B for the personal part. Fused at the oracle lambda: B P04 +2 (0 to +4), P10 +0.3 (-0.5 to +1); A P04 +2.5,
  P10 +0.5 over ER s1 -> s3.
- **Kill rule (stated before the run):** if the best ceiling form (oracle lambda, non-causal EA, true training labels)
  does not beat the routed path on the same sessions by >= +3 on P04 and >= +1 on P10 (mean over the protocol's test
  sessions and the 16 orders), stop: record the ceiling, build no causal version. If it passes, build the causal
  version (EA in delivery order, pseudo-label refits) and score it under within and full at least.

### 246. PERSONAL WITHIN-SESSION CEILING (written 14 Nov 15:09 UTC, before any result)
Code `studies/14nov/personal_ceiling/`, outputs `runs/personal_ceiling/`. Question: what does a fully personal system
reach inside one session, per person, and how far does the same system drop across sessions (P04, P10)?
- **Splits (within, W):** every session (14); train = the two acquisition runs + two online runs with true labels, test =
  the other two online runs; all six pairs of online runs (named primary pairs: last two online runs; online runs 1+2).
  Score = balanced accuracy per interface cell on the test windows, mean over the cells present, then over splits and the
  person's sessions. Cross-session (X): P04 / P10, train on all six runs of session 1, test sessions 2 and 3 on the same
  test-pair windows as W (and whole session).
- **Systems:** (a) classic personal pipelines on the 41-ch windows, each session recentred on its own whole-session mean
  covariance (non-causal, unlabelled; same rule in W and X): a1 filter-bank tangent space + LR; a2 FBCSP (OvR) + shrinkage
  LDA; a3 MDM 8-30 Hz; a4 slow waveform (< 4 Hz, decimated) + shrinkage LDA; aE = mean of the four probabilities (primary
  for a). (b) the four population member families (ttt5.ORDER: fbraw, fbl, sinc, fb), each trained on the fold's other
  nine people as ttt7 does, then fully fine-tuned (all parameters, eval-mode SessionBN, AdamW 5e-4, ttt5 augmentation)
  on the person's four training runs with TRUE labels, 60 and 200 steps; bE = mean of the four families (primary for b,
  200 steps if it is not below 60). (c) the population chain on the same test windows: s242ctl V6Fr (new-person path,
  all_loso), 20 orders x 6 bases, mean of the scores; for P04 / P10 sessions 2-3 also ret ER (routed returning path,
  16 orders). Fusion: product of experts, P_chain x P_personal (equal weight, renormalised), per order and base.
- **Predictions (W, ten-person mean):** chain (c) 78 (74-82) on the test runs; aE 72 (65-80); bE 83 (77-88); best of
  a/b fused with the chain 85 (80-90), i.e. +6 over the chain. P04: chain 52, aE 62, bE 70, fused 72. P10 (all three
  sessions): chain 82, bE 92. Cross-session drop of the same personal system (bE and aE, W minus X on the same windows):
  P04 -10 (-4 to -18), P10 -6 (-2 to -12).
- **Reading rule (stated now):** "the P04 / P10 gap is cross-session shift for known people" if the within-session personal
  ceiling (best of aE, bE, fused) on their sessions 2-3 is >= returning-chain ER + 5 on the same windows AND the same
  personal system loses >= 5 points from W to X. Its size = W personal (fused) minus ER. For the eight single-session people,
  W fused minus chain (c) is the room a session-internal personal system could at most claim; < +3 means the chain already
  sits near the personal ceiling. Bars are reading aids; within ±0.5 of a bar is "undecided".

### 248. ELIMINATION OF +1.5-POINT LEVERS (written 14 Nov 15:10 UTC, before any measurement)
- **Why:** on the live board two teams gained about +1.5 sealed in one upload and a new team entered at 0.851 (us 0.820).
  Everything tested today is +0.1 to +0.3. Question: which lever of 1.5 to 3 points do the rules allow that we lack?
  Method: elimination against the rules (FAQ and terms at the pinned rev and at upstream bddaff3, organisers' 13 Nov answer)
  plus minute-scale measurements from data on disk. Code `studies/14nov/levers/`, outputs `evidence/levers_14nov/`. CPU only,
  nothing trains, no download.
- **M1 design ceiling (perfect design knowledge, not only boundaries).** On `runs/chainout/s215_<order>_off.npz` (V6Fr and
  P0r, six bases, within/full/runs 0-1, new people), change against the raw output of: (b) the causal true-run count filter
  (runcount.filt, reference); (e) the same filter with each run's TRUE class counts as capacities; (c) offline exact-count
  assignment per run (Hungarian on log posteriors, true counts; non-causal, an upper bound); (d) offline exact counts per
  cell; (f) offline per session. Prediction (V6Fr, mean over orders): (b) +0.5; (e) − (b) <= +0.15; (c) +1.3 (80 % +0.8 to
  +2.2); (d) +0.8 (+0.3 to +1.4); (f) +0.6. Reading: if (c) < +1.5 the design lever cannot be the +1.5 even offline; the
  causal share is what matters.
- **M2 raw-data inventory.** From the BIDS events and EDF headers: seconds of eyes-open/closed rest per session and the
  4-s windows they give; task duration per trial (cue to end_trial) and how many 4-s crops at 0.5 s steps fit; inter-trial
  seconds. Prediction: rest about 1-3 min per run (20-90 windows per session); task 8 s (9 crops per trial at 0.5 s).
  Reading: crops were tested five times (negative, REVIEW_5NOV, §100); rest runs give unlabelled rows only.
- **M3 duplicates / overlap.** (i) Hash of the EDF signal payload of all 112 runs; (ii) max |correlation| of flattened
  120 Hz windows across different runs in `data/proteus_windows.npz`. Prediction: 0 duplicate runs; 0 cross-run window pairs
  above 0.9. A duplicate inside the public release would raise the prior that the sealed set copies public runs.
- **M4 run fingerprint under full shuffle.** Leave-one-out 1-NN run identity within each session (the oracle separability of
  runs, chance about 1/6) on (a) per-channel log-variance in 6 bands (§217's nuisance features, unprojected) and (b) a
  hardware view (per-channel log power 45-58 Hz and at the 50 Hz notch residual, channel-pair correlation at 30-58 Hz).
  Prediction: 0.45 (0.3-0.6) for both. Rule: >= 0.85 would reopen full-shuffle run recovery (§215 full arm +1.2 with true
  runs); below 0.6 that route stays closed in these forms.
- **M5 session dates.** EDF start dates per session (are sealed sessions same-day as any public one; gaps between
  sessions). Prediction: one session per day, gaps of days to weeks, or dates anonymised; FAQ says sealed sessions are
  "later sessions", so no same-day public twin exists.
- **Everything else** (foundation model, more people, eye-view oracle, metric quirks) is assessed from the record, cited.
**§248 amendment (written 14 Nov 15:12 UTC, after M4's first line, before these runs).** M4's hardware view identifies runs at 1-NN
0.79 (chance 0.19; the s217 log-variance view 0.49). M3 found 9 cross-run window pairs with |r| > 0.9 on flattened windows.
Added before running: (M3b) the same with each channel demeaned per window, and where the high pairs sit (same session?
same person?). (M4b) which block carries the fingerprint (high-band power, 50 Hz residual, 30-58 Hz channel correlations);
unsupervised per-session clustering (PCA 20 of the hardware view, Ward and GMM, k = true run count) with purity / ARI;
(M4c) those offline clusters as groups for runcount.filt on full 0/1 (V6Fr, six bases, new people) against true runs and
none. Prediction: purity 0.6 (0.45-0.75); M4c keeps 40 % (20-60 %) of the true-run filter gain under full (+0.78/+0.92).
Rule: >= 60 % of the true-run gain under full with a non-causal clustering makes a causal online form worth a day;
below 30 % the hardware route is recorded undecided-weak. Offline clustering is non-causal: an upper bound of the route.
**§248 amendment 2 (written 14 Nov 15:16 UTC, before this run).** M4c result: offline Ward clusters keep 48 % / −7 % (V6Fr full 0 / 1)
of the true-run filter gain: below the 60 % rule; route recorded undecided-weak in this form. 1-NN 0.79 against purity 0.50
says the information is local (neighbours), not cluster-shaped. (M4d) Does the hardware view recover TEMPORAL neighbours
(the s213 finding: about 80 % of the full-shuffle stage loss, −3.1 of −3.75, is non-stationarity)? Per window, the 1-NN
in the hardware view within the session: share in the same run and within ±1 / ±3 trials in recording time; Spearman of
view distance with time lag. Prediction: 1-NN within ±3 trials 0.45 (0.3-0.6) vs chance about 0.04. Rule: >= 0.5 makes a
content-neighbourhood stage (stage statistics weighted by fingerprint similarity instead of delivery recency) the top
one-day test for the full scheme.
**§248 RESULT (14 Nov 15:17 UTC; evidence/levers_14nov/, code studies/14nov/levers/; CPU, nothing trained, nothing downloaded).**
- **M1 design ceiling** (V6Fr, new, 6 orders x 6 bases, against raw): causal true-run filter +0.48; with true per-run class
  counts +0.49 (design knowledge beyond boundaries adds +0.01); offline exact counts per run (non-causal, no revision allowed)
  +1.16; per cell +0.15; per session +0.14. P0r: +0.64 / +0.66 / +1.29 / +0.42 / +0.42. Predictions: b hit, e hit, c hit
  (+1.3, 0.8-2.2), d miss (predicted +0.8), f miss (+0.6). Reading: even perfect, non-causal design knowledge on our
  posteriors is about +1.2; the causal part is about +0.5 and the deployable part +0.15 to +0.3. Not the +1.5 lever.
- **M2 raw data** (local; no download needed): 28 rest runs x 187 s = 87 min, 1,288 non-overlapping 4-s windows (92 per
  session); task period 8.0 s (cue to end_trial), 9 crops per trial at 0.5 s (23,895 rows) or 7 without the cue second;
  4.1 s between end_trial and the next cue; official windows use 32 % of the cued recording. EDF resolution 0.40-1.27 uV/bit,
  per file. Prediction hit. Crops: negative five times in the record; rest runs never used (unlabelled only).
- **M3 duplicates**: 112 distinct signal payloads; max cross-run |r| 0.934 (0.945 channel-demeaned), all large pairs are
  common-mode slow artifacts (per-channel r about 0.95 on every channel); the one cross-person pair (P08/P09 run 3) has
  different onsets and labels. No overlap inside runs (cue gap 12.15 s). Prediction "0 pairs > 0.9" missed in letter
  (9 / 22 pairs), not in substance: no duplicated trial.
- **M4 run fingerprint**: 1-NN run identity within session, chance 0.19: s217 log-variance 0.49; hardware view 0.79 (30-58 Hz
  channel correlations carry it: 0.79 alone; high band 0.62; line band 0.50). Prediction 0.45 missed high for the hardware
  view. Unsupervised clusters: purity 0.50-0.55, ARI 0.23-0.30. M4c, offline Ward clusters as filter groups under full:
  V6Fr +0.37 (48 % of true) / -0.06 (-7 %); P0r +0.46 / -0.36. Below the 60 % rule: undecided-weak in this form.
- **M4d temporal neighbours**: hardware 1-NN within +-3 trials 0.385 (chance 0.033), +-1 0.18 (0.012), same run 0.79,
  Spearman(distance, lag) +0.21; log-variance 0.16 / 0.07 / 0.49 / +0.14. Inside the predicted range, below the 0.5 rule.
- **M5 dates**: every EDF start date is 01.01.85 00:00 (anonymised); sealed sessions are "later sessions" (FAQ), so no
  same-day public twin exists; session-twin matching is closed by construction.
- **Ranking (expected sealed gain x probability)**, for the coordinator: (1) personal path for the eight single-session
  returning people (about 40 of 46 returning sessions; §246/§247/§249 running); (2) hardware-fingerprint run recovery /
  content neighbourhoods, worth something only if the sealed scheme is full; (3) eye-type view (oracle +3.8, §155; real
  views +0.1 to +0.3); (4) design filter beyond today's (<= +0.3); (5) rest runs / crops (<= +0.2); (6) foundation models (~0);
  (7) evaluation-structure quirks (0).
### 250. FINGERPRINT RUN RECOVERY UNDER FULL SHUFFLE (written 14 Nov 15:19 UTC, before any result)
- **Why (§248):** the hardware view (channel correlations 30-58 Hz, log power 45-58 / 48-52 Hz) identifies a window's run at
  1-NN 0.79 within a session, but unsupervised offline clusters reach purity 0.50-0.55 and keep 48 % / −7 % of the true-run
  filter gain under full 0 / 1. True runs under full: filter about +1.1 (§238), non-stationarity −3.1 of −3.75 (§213). The
  current chain's detector-gated filter (§228c) gives 0 under full. This is §248's one-day test.
- **Metric (offline, public people only):** per held-out person, a run-discriminant projection fitted on the nine others'
  TRUE runs: hardware features standardised on the nine, PCA, then LDA of between-run (run mean minus session mean) against
  within-run scatter with shrinkage, scaled so the within-run SD is 1 per direction. Settings grid: PCA 60 / 120, LDA dims
  8 / 16 / 32, shrinkage 0.1 / 1. The held-out person's runs are read only to score.
- **Causal online grouping (per session, delivery order, windows <= t, label fixed at arrival, nothing across sessions, no
  labels):** at most 6 groups per session (the public design: 2 acquisition + 4 online runs). Forms: (G1) sequential MAP in a
  conjugate Gaussian mixture (within-run covariance I in metric units, run-mean prior around the session's running mean with
  the training people's between-run variance, CRP weight alpha); (G2) online k-means with a new-group distance threshold
  (DP-means); (G3) nearest-earlier-window propagation (join the group of the nearest earlier window if its distance is below
  a threshold, else open a group; at 6 groups, nearest group). Hyperparameters (metric setting, alpha / thresholds) chosen
  nested: by mean online purity on the nine training people, each scored with a metric fitted on the other eight.
- **Measured first (full 0-3, new people, mean over sessions):** held-out 1-NN run identity in the metric; offline Ward
  purity / ARI (k = true run count); online purity, ARI, groups per session and false-merge rate (share of same-group window
  pairs from different runs), per form, nested choice reported.
- **Prediction:** metric 1-NN 0.85 (0.75-0.92); offline Ward purity 0.70 (0.55-0.85); online purity of the nested choice 0.60
  (0.45-0.75), false-merge rate 0.45 (0.3-0.6).
- **Stop rule (coordinator):** if the nested online purity is below 0.70 on full 0-3, report and stop: no chain scoring.
- **Chain scoring (only if purity >= 0.70):** groups into the count filter (jointcount.filt_groups, cap 12, τ 2, floor 0.5) on
  runs/chainout/s242r15_<order>.npz, V6Fr (P0r for reference), six bases, new people, within / runs / full 0-3. Arms: FP
  (fingerprint groups, always on) against raw; GFP (emit FP until the §228c detector latches, J-wprior/urn from the latch on)
  against the current gated filter (latchJ). Run offsets with the groups (shufcombo RUNOFF + RUN_IDS) only if the filter passes.
- **Prediction (if scored):** full FP − raw +0.25 (−0.1 to +0.6); within / runs GFP − latchJ −0.05 (−0.15 to +0.05).
- **Pass rule (coordinator):** GFP − latchJ >= +0.5 under full 0-3 AND >= 0 under within 0-3 and runs 0-3 (V6Fr, means over
  orders), with 80 % person intervals reported. Code studies/14nov/fingerprint/, outputs evidence/fingerprint_14nov/.
**§247 RESULT (14 Nov 15:21 UTC; runs/strongpersonal/ceil.txt, ceil2.txt, scheme.txt; code studies/14nov/strongpersonal/).**
Routed reference (ER of §245, 16 orders x 6 bases, scheme means): P04 s2 67.85 / s3 73.37, P10 s2 89.48 / s3 90.93
(T: 69.40 / 73.63, 89.15 / 91.44). Protocol A nets: pft12.py (s29's PFT code imported, base nets trained on all public rows
except q's session 3, fine-tuned on sessions 1 + 2; Leonardo 64473497, 15 tasks, smoke 64473434 first). Personal members alone
on the test session (non-causal EA for TS; PFT nets causal SessionBN, recording order):

| member alone | P04 s1>s2 | P04 s1>s3 | P04 s12>s3 | P10 s1>s2 | P10 s1>s3 | P10 s12>s3 |
|---|---|---|---|---|---|---|
| TSp (person only, best C) | 67.8 | 67.1 | 73.9 | 73.8 | 62.7 | 81.5 |
| TSw5 (nine others + person x5) | 54.6 | 64.4 | 69.2 | 85.6 | 72.9 | 84.5 |
| PFT nets, 15, pf60 / pf200 | 69.9 / 73.0 | 62.4 / 63.4 | 66.1 / 67.7 | 91.1 / 88.4 | 85.3 / 83.7 | 90.3 / 89.3 |

Ceiling = ER x TSp^a x TSw5^b x PFT^c, one setting per person and protocol picked on the test sessions (oracle), identity
known from window 0 (no gate), change against ER:
- **B (s1 -> s2, s3; like for like): P04 +3.96, P10 +0.88.** One setting for both (PFT200^0.25): P04 +2.70 (16/16 orders > 0),
  P10 +0.46 (11/16; within +0.16, full +0.72, runs +0.50).
- **A (s1 + 2 -> s3) against ER trained on session 1 only: P04 +7.78, P10 +2.33**; one setting for both (TSp10^0.25 x
  PFT12_200^0.5): P04 +7.78 (16/16), P10 +1.03 (9/16; within -0.24, full +2.42, runs +0.91).
- A-like (s1 -> s3): P04 +3.12, P10 +0.58.
- **Against the prediction:** personal alone below the routed path (hit: B P04 ~68-73 single members vs 70.6, P10 <= 91 vs
  90.2); fused B P04 +2 predicted, +3.96 oracle / +2.70 shared (above); P10 +0.3 predicted, +0.88 / +0.46 (inside).
- **Kill rule: met on the like-for-like protocol (B: P10 +0.88 < +1 even at the oracle).** A passes the bars, but only
  against a reference that has one public session less; most of A's gain over A-like is the second session itself (TSp
  alone gains +6.8 P04 / +18.8 P10 from it, the nets +4 / +5), which the routed path's personal head, own-space stage and
  T would also receive, and A's P10 gain sits in full orders (where ER routes late or never; the ceiling has identity from
  window 0) and is negative under within. Not packaged; no causal version built. Sealed-weighted (0.2 P04 + 0.8 P10) the
  like-for-like oracle is +1.50 on returning sessions, the shared setting +0.91 (both upper bounds: oracle setting,
  oracle identity, non-causal EA). What would reopen it: the routed chain rerun with sessions 1 + 2 public (core1, head,
  own-space, T) so that A has a like-for-like reference; or late fusion of the PFT nets (ER x PFT200^0.25, the shared B
  form, +2.70 / +0.46) scored behind the gate under shuffle.
- **Side finding (not a lever as measured):** ceil.py's TSwpl member reproduced the argmax of the mean ER posterior over
  the 96 runs (16 orders x 6 bases), which beats a single run by +4.1 / +2.7 (P04 s2 / s3) and +1.0 / +0.9 (P10): the
  routed path's spread across delivery orders and bases is that large on returning people.
**§250 amendment (written 14 Nov 15:31 UTC, before the nested run; after a smoke on three held-out people).** The smoke showed
the held-out person's scale in the metric differs up to 30x (median within-run d^2/d 0.07 to 2.05), so absolute thresholds put
every window in one group. Changed before the nested run: all three forms are scale-free in the session's own causal spread
(windows <= t: per-direction variance v_t, mean m_t; G1 within-run variance f * v_t and run-mean prior (1 - f) * v_t, f in
0.2 / 0.35 / 0.5, alpha 0.1 / 1 / 10; G2 / G3 thresholds on d^2 / (d x mean pairwise d^2 / d)); no new group before window 5;
metric grid + PCA 200 / 64 dims (shrinkage 0.1 / 1); inner selection on full 0-1, outer scoring on full 0-3. The smoke's
non-nested purities (P01-P03, full 0, one setting) were 0.22-0.53; the prediction and the 0.70 stop rule stay as written.
Option, not a plan (14 Nov 15:22 UTC; from §247's side finding): averaging the candidate's posterior over the six bases instead of
shipping one gives new people within +0.14, full +0.06, runs +0.25 (r15 raw, 20 orders), at six times the run time. §247's
larger gain from averaging over orders and bases (P04 up to +4) is mostly the orders, which a package cannot average.
- **Stamp correction (§250 amendment):** its heading says 15:31 UTC; `date -u` at writing was 15:21 UTC (the nested run started then).

### 247a/b. AMENDMENTS: TWO-SESSION ROUTED REFERENCE AND PACKAGE-FORM LATE FUSION (written 14 Nov 15:24 UTC, before any result; coordinator after §247)
**(a) The routed chain with sessions 1 + 2 public, tested on session 3** (`studies/14nov/strongpersonal/chain12.py`; pview
chain.py's machinery imported, not edited). The replay cannot rebuild every member with session 2 in training without new
Leonardo exports of all members (35 nets, 30 SessionBN nets, teacher, core1, Hilbert, eye and rest nets, their delivery-order
re-streams). Closest honest version: the population members and core1's basis stay as in seen_session (session 1 in
training); everything the routed path builds from a person's public data gets session 2 added: the v2 gate's gallery and
basis (rlib3.gate_model's per-person sessions), the bio gate's enrolment rows, the personal head (ps1_heads), T's prior
(pri), and the own-space stage's three models (ownspace.Personal on sessions 1 and 2). Session 2's core1 PCs are the
fold's stored causal test features (training sessions use whole-session ones; per-session z-scoring in stdz / contrast
removes most of the difference). Only session 3 is scored; session 2's outputs are leaky and dropped. Zero-change check
first: with the additions off, ER and T on session 3 equal s245_ret exactly (every order and base).
- **Prediction:** ER(1+2) − ER(1) on session 3: P04 +2.5 (0 to +5), P10 +0.5 (−0.5 to +1.5); the strong path's ceiling on
  top of ER(1+2) (§247 grid, oracle) keeps about half of its A gain: P04 +4, P10 +1.
- **Reading rule:** the strong path is worth a causal build only if, on top of ER(1+2) [and of ER(1+2) + T], its
  one-setting-for-both form gives >= +3 P04 and >= +1 P10 on session 3 (mean of the 16 orders).
**(b) Package-form late fusion** (`fusion.py`, nets `pftshuf.py`): output = ER x PFT^(w r) on routed rows, r the gate's ramp
(0 before the first routed window, so those rows emit ER exactly), PFT = the geometric mean of the 15 PFT nets (5
families x 3 seeds, s29's code) of the routed person, now re-streamed in each delivery order (causal SessionBN /
alignment in delivery order; s29's stored outputs were recording order). Primary: pf200, w = 0.25, fixed for everybody.
Honest cross-person setting: (variant, w) in {pf60, pf200} x {0.125, 0.25, 0.5, 1} chosen on P04 and scored on P10 and vice
versa. Also on T, and with the count filter (score_ret.py on SAVE_OUT keys <order>|<base>|<arm>). 16 orders x 6 bases.
- **What differs from v5.2's personal nets** (about 0 to −0.4 in package form): a geometric factor on the final emitted
  output after all stages, at a small exponent, instead of nets swapped into the member mix (which also feeds SessionEM,
  TSEM and the shadow); the §29 PFT nets (fine-tuned 200 steps on the person's public session, all five families) rather
  than v5.2's pnets; on the routed tail only, behind the gate's ramp.
- **Prediction:** F(pf200, 0.25) against ER: P04 +2.0 (+0.5 to +3), P10 +0.3 (−0.3 to +0.8), sealed-weighted +0.6; the
  cross-person setting within 0.3 of it; on T about the same increments; the filter adds ≤ +0.1.
- **Rule:** a package candidate if sealed-weighted ≥ +0.3 on the 16-order mean with no scheme below −0.3 for P10 and the
  cross-person setting also ≥ +0.3; inside ±0.3 undecided; ≤ −0.3 closed for this form.
**§250 RESULT (14 Nov 15:33 UTC; evidence/fingerprint_14nov/group.txt, code studies/14nov/fingerprint/{metric,group}.py; Mac CPU,
5 min, nothing trained beyond LDA, no download; group labels runs/fingerprint/groups_G{1,2,3}.npz, not scored).**
- **Metric (nested setting, held-out people, per session):** 1-NN run identity 0.803 (unsupervised z-scored view 0.794, §248);
  offline Ward purity 0.564, ARI 0.33 (unsupervised 0.50 / 0.23). The supervised metric adds almost nothing: the nine people's
  run directions do not transfer to the tenth beyond what the raw view already has.
- **Online, causal, ≤ 6 groups, full 0-3, nested choice, mean over people [80 % person interval]:** G1 purity 0.527 [0.491,
  0.564], ARI 0.28, false-merge 0.63; G2 0.484 [0.457, 0.511], 0.68; G3 0.489 [0.454, 0.525], 0.67. Best person P07 0.71 (G1),
  worst P05 0.36. Inner (selection) purities 0.46-0.55, so the choice did not overfit.
- **Predictions:** metric 1-NN 0.85 missed low (0.80); offline Ward 0.70 missed low (0.56, inside the 0.55-0.85 range at its
  edge); online purity 0.60 missed low (0.53, inside 0.45-0.75); false-merge 0.45 missed high (0.63, outside 0.3-0.6).
- **Stop rule met (purity < 0.70 on full 0-3, upper interval 0.56): no chain scoring, no run offsets.** With about two of three
  same-group pairs from different runs, the groups are mostly mixtures; §248 M4c (offline Ward at 0.50 purity) already kept
  48 % / −7 % of the true-run gain. Reading: the fingerprint is local (nearest neighbours lie in the same run 80 % of the time)
  but runs do not form separable groups in it, supervised or not; route recorded closed in the grouping form. Not tested: soft
  neighbourhood forms (stage statistics weighted by fingerprint similarity, §248 M4d), which need no groups.
- **Stamp correction (§250 RESULT):** its heading says 15:33 UTC; `date -u` at writing was 15:27 UTC.

### 251. SOFT FINGERPRINT NEIGHBOURHOODS (written 14 Nov 15:32 UTC, before any result)
- **Why (§248 M4d, §250):** the hardware fingerprint is local (1-NN same run 0.80, within ±3 trials 0.39) but hard online
  groups reach purity 0.53, so grouping forms are closed. Remaining form: nothing grouped; each window's stage statistics
  are weighted by fingerprint similarity to the session's earlier windows (delivery order, windows ≤ t, nothing across
  sessions, no labels at use). Code `studies/14nov/softfp/`, outputs `runs/softfp/`, evidence `evidence/softfp_14nov/`.
- **Weights (fpw.py, fixed before any score):** s248 hardware view (runs/fingerprint/hw.npz) z-scored with the other nine
  people's mean / SD; r_tj = log(d²_tj / median_{j<t} d²_tj); w_tj = P(same run | r_tj), 40 quantile bins, isotonic,
  calibrated per held-out person on the nine others' TRUE runs under full 4-7 (disjoint from the scored orders). Checked
  before writing this (weights only, no chain or score): under full 0 the weight-averaged same-run share is 0.27 (base
  0.16); the top-weight earlier window is in the same run 0.61 (1-NN among earlier windows 0.67).
- **(0) Hard diagnostic:** §250's G1 groups (runs/fingerprint/groups_G1.npz) through runcount.py (RUN_IDS, s215 full 0/1
  files) and through jointcount.filt_groups (cap 12, τ 2, floor 0.5) on runs/chainout/s242r15_full{0-3}.npz V6Fr, against
  raw. Prediction: full +0.15 (−0.2 to +0.4) (M4c offline Ward +0.37 / −0.06; G1 purity is lower).
- **(1) Soft count filter (softcount.py):** prior_c ∝ max(CAP − k_c, 0) + FLOOR with k_c = Σ_{j<t} w'_tj q_j(c), q_j the
  filter's own emitted output, likelihood P_t^τ (τ 2, FLOOR 0.5, CAP 11, fixed); output ∝ prior × likelihood. Six
  settings: kernel w' = γ w, γ ∈ {0.5, 1, 2}; top-k w' = w on the k smallest r_tj, k ∈ {5, 10, 20}. Arms on s242r15 V6Fr
  (six bases, new people): SCF (always on) − raw; **GSCF** (SCF before the §228c latch, latchJ from the latch on) −
  latchJ (the current system's emission). Setting chosen nested: per held-out person by the nine others' mean GSCF −
  latchJ over the 12 orders (schemes equal). Orders full 0-3 (primary), within 0-3, runs 0-3.
- **(2) Soft offsets (softoff.py hook + wrap, shufcombo via runpy unedited):** each pc128 and GE row (delivery order) minus
  n/(n + κ) (m_w − m_s), m_w = Σ_j w'_tj v_j / n, n = Σ_j w'_tj, m_s the session prefix mean (current row excluded), κ 5
  (as §210's true-run offsets). Two forms fixed in advance: kernel (w' = w) and top-10 (w' = w on the 10 smallest r).
  Installed by wrapping gm.em_g / gm.tsem_g for all_loso; env as the s242 r15 arm (STRICT, GATE_P0, ttt18<order>LP,
  FT_V r15). Zero-change check first: hook with n ≡ 0 reproduces s242r15_<order>.npz V6Fr exactly. Scored V6Fr against
  s242r15, raw and both through latchJ (primary: latchJ(SOFF) − latchJ(r15)).
- **Predictions:** (1) SCF full +0.10 (−0.15 to +0.35); GSCF − latchJ within / runs −0.05 (−0.2 to +0.05). (2) full +0.15
  (−0.1 to +0.4), within / runs 0.00 (−0.15 to +0.10). The weights carry about a quarter of their mass in the true run,
  so I expect a fraction of the true-run gains (§215 filter full about +0.8, §210 offsets full +0.9 to +1.25).
- **Pass rule (coordinator):** ≥ +0.5 mean over full 0-3 and ≥ −0.1 over within 0-3 and over runs 0-3 (V6Fr, new people,
  primary arm), 80 % intervals resampling people and orders. Between 0 and +0.5 under full with within / runs ≥ −0.1:
  undecided (not closed). (3) Local templates in TSEM run only if (1) or (2) reaches ≥ +0.25 under full; otherwise
  recorded as not run.
§239 part 2 result (14 Nov 15:34 UTC; scored by the parent session with studies/14nov/designdec/score2.py). Fine-tuning on r15's
labels smoothed by the run urn (tag LPD) against r15 (LP), V6Fr, new people, 2 orders per scheme, [80 % person interval]:
  within: no filter +0.07 [-0.03, +0.17] | gated J filter -0.10 [-0.18, -0.01] | gT -0.04
  runs:   no filter +0.09 [-0.01, +0.21] | gated J filter +0.06 [-0.12, +0.25] | gT +0.14
Killed (below +0.3; with the filter, which the candidate carries, it is about 0). Fix in score2.py: its exact-equality check on
P0r failed because each fine-tuning run re-writes its base outputs (GPU float noise up to 0.05 on single-seed bases, <= 1
argmax flip in 2655); the check is now argmax agreement > 0.998, still loud on a real difference.
§243 draw 1 (14 Nov 15:34 UTC; evidence/stack_14nov/s243_d1.txt). Same arms on fine-tuning draw 1 (its own s15 control), new
people, 20 orders: r15 +0.08 [-0.02, +0.20] 6/10; r15 + filter +0.23 [+0.09, +0.37] 7/10 (within +0.41, full +0.05, runs
+0.22); stk1 + filter +0.34; stk2 + filter +0.38. Draw 0 had r15 + filter +0.25, stk1 + filter +0.28, stk2 + filter +0.34.
Spread of the candidate's new-people half over two draws: 0.02. Draw 2: 13 of 20 orders in.
**§251 amendment (written 14 Nov 15:37 UTC, after (0) and (1) were read, before (1b) ran; (2) running, not read).** (1) nested
GSCF − latchJ: full +0.30 [+0.06, +0.53], within +0.03, runs −0.02 (undecided by the rule); the kernel strength curve is
sharp (γ 0.5 / 1 / 2: full +0.09 / +0.30 / −0.21). Mechanism read: with k_c = γ Σ w q, the soft counts grow with t
(Σ_j w_tj ≈ 0.17 t under full) and reach CAP near the session's end at γ 1 and halfway at γ 2, where every class sits on the
floor and the prior turns flat; the form is badly scaled, so γ 1 may be a lucky spike. **(1b) normalised soft count:** the
same filter with k̃_c = k_c · min(1, M / n_t), n_t = Σ_j w'_tj (the local class share among similar earlier windows, applied
as M seen windows of the run; never saturates for M ≤ CAP · 3 / 2), kernel weights, M ∈ {5, 10, 15, 20, 30}; the old form's
γ ∈ {0.75, 1.25, 1.5} added to show the curve's shape (reported, not in the nested set). Nested choice over the five M
values as before, and over all 11 settings (reported). Prediction: (1b) nested full +0.25 (0 to +0.5), within / runs
−0.03; the γ curve between 0.75 and 1.5 is not smooth (a spike at 1). Same pass rule.

**§246 RESULT (14 Nov 15:42 UTC; studies/14nov/personal_ceiling/, evidence/personal_ceiling_14nov/score.txt; nets: Leonardo array
64473623, 4 families x 2 seeds; classic on the Mac via run_capped).** Within session (train 4 runs, test 2 online runs, six
pairs), cell-mean BA, ten people: chain (s242ctl V6Fr, 20 orders, schemes equal; whole-session levels reproduce the dev
table exactly, P01 76.6 ... P10 82.2) 79.6 on the test runs; classic aE 79.8 (a1 FB-TS-LR 76.9 best single); frozen
population nets alone 65.9; nets fine-tuned on true labels bE (200 steps) 84.4 (+4.8 [+2.9, +6.8], 8/10; 400 steps 84.8,
added after the smoke, secondary); chain x aE 85.3 (+5.8, 9/10); chain x bE 85.2 (+5.6, 10/10); chain x aE x bE 87.1
(+7.5 [+5.7, +9.4], 10/10). Both primary pairs agree (last two online runs 87.3, online 1+2 87.3).
- P04 (three sessions): chain 48.9, aE 65.0, bE 64.1, all fused 66.8. On P04 s2/s3 the routed path ER is 61.3 / 69.5 on the
  same windows; personal alone 64-65 (no better than ER); ER x aE 69.9 / 74.5 (+6.8). P10 s2/s3: ER 92.6 / 91.0; bE 93.6 /
  94.2; ER x bE 94.6 / 94.9 (+3.0).
- Cross-session drop of the same personal system (s1 -> s2/s3, same windows): classic aE P04 -11.7 / -16.1, P10 -22.2 /
  -16.7 (band-power features do not transfer, even with per-session EA); fine-tuned nets bE P04 0.0 / -1.4, P10 -6.4 / -6.2.
  Cross-session personal fused with ER: P04 65.4 / 68.4 (bE), P10 93.6 / 92.8: about ER's level, no gain.
- **Reading rule:** P04 meets the size bar only fused with ER (+6.8 >= 5) and the drop bar only for aE (bE does not drop);
  P10 misses the size bar (+2.1 to +3.0). So the P04 / P10 shortfall is not mainly cross-session shift: P04 stays at 61-75
  even with labels from her own session (a hard person, not a shifted one), and ER already sits within 2-3 points of P10's
  within-session personal ceiling. The large room is for new people: the chain at 79.6 against 87.1 for chain x personal
  with in-session labels (+7.5, every person up). Sealed-weighted labelled ceiling about +5.9 (0.566 x 7.5 + 0.434 x (0.2 x
  6.8 + 0.8 x 3.0)). It needs labels; the question becomes how much of it label-free in-session adaptation can reach.
- **Against the predictions:** chain 79.6 (78), bE 84.4 (83), fused +7.5 (+6): hits; aE 79.8 (72) above; P04 bE 64 (70) and
  fused 67 (72) below; drops P04 aE -14 (-10) hit, P04 bE -0.7 (-10) miss, P10 bE -6.3 hit, P10 aE -19 (beyond -12).
- **Limits:** EA and SessionBN statistics are non-causal / whole-session; the two test runs are adjacent to labelled runs of
  the same session; one fusion weight (equal); orders resampled only inside the chain average, intervals over people.

### 252. LABEL-FREE PERSONAL CEILING (written 14 Nov 15:43 UTC, before any result)
Follow-up of §246 (coordinator): how much of chain x aE x bE's +7.5 survives when the four training runs carry the chain's
own outputs instead of true labels? Code `studies/14nov/personal_ceiling/` (same splits, scoring, pipelines, nets).
- **Chain:** the candidate's r15 chain, runs/chainout/s242r15_<order>.npz V6Fr. Per order the pseudo-label source is the
  mean over its six bases; fusion and the reference score use each base as in §246. The labelled ceiling is re-scored on
  the r15 chain and on the same orders, so gains compare like with like.
- **Label forms (training runs only; test runs never labelled):** (i) hard = argmax of V6Fr; (ii) soft = argmax weighted by
  its max posterior (ttt's wpl; weighted LR / MDM, weighted shrinkage LDA); (iii) filt = (ii) on the detector-gated count
  filter output (rundetect/package.py: latch + J-wprior urn, as stack/score.py builds it), deployable. No true-run filter.
- **Systems:** aE (refit per order: all 20 orders, Galvani CPU); bE (4 families x 2 seeds, 200 steps, Leonardo; orders
  within0, within1, full0, full1, runs0, runs1); fused c x aE, c x bE, c x aE x bE per order and base. Primary = c x aE x bE
  with filt labels, on the six bE orders; per person, 80 % interval resampling people and orders.
- **Predictions (gain over the r15 chain on the test runs, ten people):** pseudo-label accuracy on the training runs about
  80 %; aE alone about 72, bE alone about 79 (below the chain); c x aE +1.0 (-0.5 to +2.5); c x bE +0.8 (-0.5 to +2); primary
  c x aE x bE +1.5 (0 to +3), about 20 % of the labelled gain; filt about +0.3 over hard; P10 and P04 both small.
- **Rule:** primary >= +2 with the people interval above 0 -> sketch the causal package form (refit on windows < t every k,
  chain's emitted labels, run time against the worker budget). +0.5 to +2: undecided, a causal form only if cheap (aE).
  < +0.5: the labelled ceiling is not reachable by self-training on the chain's labels in this form; recorded, not closed.
**§251 amendment 2 (written 14 Nov 15:46 UTC; (1b) read, (2) read on full 0/1 only, (3) running, not read).** (1b): normalised soft
count m15 / m20 GSCF − latchJ full +0.16 / +0.16 [+0.04, +0.28] 8/10, within +0.03, runs −0.02; nested over M +0.02 (the
choice flips between m15 and m30); the γ curve 0.75 / 1 / 1.25 / 1.5: +0.16 / +0.30 / +0.05 / +0.12, so γ 1's +0.30 is a
spike (prediction hit). **(4) Stack of the soft parts in one chain:** the soft-offset chain (form kern, fixed now; full 0/1
read: kern +0.12, top10 +0.09) with the normalised soft count m15 on its output (GSCF: before the latch; latchJ after),
against latchJ(r15). m15 is chosen after reading (1b) on all ten people (stated; nested M reported next to it). If (3)
tloc is positive under full it is not added (separate chain variants; no third chain run in this entry). Prediction: full
+0.25 (0 to +0.5), within / runs −0.05 to +0.05. Same pass rule.
**§247a/b RESULT (14 Nov 15:58 UTC; runs/strongpersonal/{compare12,fusion_B,fusion_A}.txt, filter_{B,A}.log; chain12 S2=0
reproduced s245's ER and T on all 16 orders x 6 bases; pftshuf's recording-order stream agrees with s29's stored nets on
98-100 % of windows).**
- **(a) Routed path with session 2 public, session 3 (scheme means):** ER 73.37 -> 72.84 (P04, −0.53), 90.93 -> 91.67 (P10,
  +0.74); T 73.63 -> 73.14 (−0.49), 91.44 -> 92.50 (+1.06). The routed path draws almost nothing from a second session for P04
  (predicted +2.5) and about +1 for P10. Strong path on top (§247 grid, members from window 0): on ER(1+2) oracle +7.61 / +2.09,
  one setting for both (TSp10^0.25 x PFT200^0.5) +7.29 / +0.67 (sealed +1.99); on T(1+2) oracle +7.05 / +1.52, one setting
  +6.86 / −0.10 (sealed +1.30). **Reading rule (>= +3 / >= +1 one-setting on T(1+2)): missed on P10.** The second session's
  value sits in the personal members (and nets), not in the routed path's personal parts.
- **(b) Package-form late fusion ER x PFT^(w r), nets in delivery order**, change against ER (P04 / P10 / sealed):
  - B (one public session): primary pf200 w 0.25 **+1.05 / −0.11 / +0.12** (P10 > 0 on 7/16); cross-person −0.02 on ER,
    +0.09 on T; on T: TF vs T +0.76 / −0.32 / −0.10; with the filter TF+filter +0.21 against T+filter +0.27. **Undecided
    (inside ±0.3), not a candidate.** Best post-hoc (pf60 0.25) +0.37 / +0.28 / +0.30.
  - A (two public sessions, against ER(1+2)): primary **+4.31 / +0.33 / +1.13** (P10 > 0 11/16, P10 scheme min −0.08);
    cross-person +1.01. On T(1+2): +3.99 / −0.22 / +0.62, but P10 −0.90 under full (T's own full gain taken back);
    cross-person on T +0.48. Filter (score_ret, session-2 rows held at s245 ER, numbers doubled): F+filter +1.34, TF+filter
    +1.62 against T+filter +0.92. By the rule: passes on ER, fails on T (P10 full −0.90).
- **Against the predictions:** B primary +2.0 / +0.3 / +0.6 predicted, +1.05 / −0.11 / +0.12 observed (miss, low); A not predicted.
- **What differs from v5.2's nets, and what did not help:** late geometric factor behind the gate instead of members in the
  mix; with one public session it still does not reach P10 (−0.11), as v5.2 did not. The positive case needs two public
  sessions, and in the sealed set only P04 and P10 have more than one (8 of 10 returning people have one: the B case).
- **Verdict:** not packaged as a general returning-person stage. A person-specific form (active only for a recognised person
  with >= 2 public sessions, i.e. P04 and P10, with 3 public sessions in the package) is the only case with a measured gain:
  about +4 P04 / +0.3 P10 per session on ER, smaller on T. On the board that is about 6 of ~100 sessions (order +0.1 before
  transfer), and it rests on one dev protocol (P04 s3, P10 s3).

### 253. CAUSAL IN-SESSION PERSONALISATION (written 14 Nov 16:00 UTC, before any result)
The package form of §246 / §252 for new people. Code `studies/14nov/causalpers/` (imports personal_ceiling/classic.py,
splits.py; edits neither), outputs `runs/causalpers/`. §252 not read when this was written.
- **System.** Per session, in delivery order (`runs/shuffle_perms.npz` `<scheme>|<seed>|all_loso`), per base of the r15
  chain (`runs/chainout/s242r15_<order>.npz` V6Fr; each base is its own chain, so labels, personal model and fusion are per
  base, never the base mean). From window T0 = 20, every k windows (k in 5 / 10 / 20) the classic personal pipeline is refit
  on the session's windows < t with the chain's emitted outputs as labels and predicts windows t .. t+k-1. Label forms:
  hard = argmax of the raw chain; soft = the same weighted by its max posterior (s252's weighted LR / MDM / CSP / LDA);
  filt = soft on the detector-gated count-filter output (rundetect/package.py gate + J.filt_joint 'urn'). Labels are the
  chain's outputs before the fusion (no feedback of the personal model into its own labels). Features per window only
  (zero-phase band filtering inside the 4 s window, OAS band covariances, < 4 Hz waveform), **no session alignment** (the
  s246 EA used the whole-session mean; dropped, the TS reference and LDA are fit on windows < t). Pipelines: a1 FB tangent
  space + LR, a2 FBCSP + LDA, aE mean of a1-a4 (classic.fit_predict).
- **Fusion.** out = chain x personal^w(t), renormalised; w = 0 before t0, then cap x min(1, (t - t0 + 1) / R) (R = 0: step).
  Two tracks: R (fused into the raw chain; reference r15 raw) and F (fused into the count-filtered chain; reference r15 +
  count filter, and also reported against r15 raw).
- **Design choice.** Grid: form x pipeline {a1, a2, aE} x k {5, 10, 20} x t0 {20, 40, 60, 90} x R {0, 40, 80} x cap
  {0.25, 0.5, 0.75, 1}. **Primary = track F, chosen nested**: for each person, the configuration with the best
  scheme-equal mean change on the other nine people (all 20 orders, mean over bases), scored on that person. Fixed in
  advance as a second line: filt, aE, k 10, t0 40, R 40, cap 0.5. Best-in-hindsight is printed and labelled as such.
- **Scoring.** New people, 20 orders (within0-11, full0-3, runs0-3); per-person change, mean over the six bases; means per
  scheme and scheme-equal; 80 % intervals resampling people and orders (as stack/score.py); people up. My scorer must equal
  replay score.person on r15 raw and filt for every order and base (zero-change check, fails loudly).
- **Run time.** Mac CPU, one process: ms per refit and per window (features + predict + refit / k) per pipeline.
- **Predictions.** Track F nested vs r15 + filter +0.4 (-0.3 to +1.2), people up 6/10, every scheme >= -0.2; track R nested
  vs raw +0.5 (-0.2 to +1.3); filt and soft labels >= hard by about +0.2; a1 alone within 0.2 of aE; the nested cap 0.25-0.5
  and t0 40-60; k makes < 0.1 difference between 5 and 20. a1 refit about 100 ms, aE about 300 ms; per window (k 10) a1
  about 25 ms, aE about 60 ms. The gain is smaller than §252's batch number (half the training data on average, causal).
- **Pass (stated by the coordinator):** track F nested >= +1.0 scheme-equal mean, >= 0 in every scheme, people up >= 8/10
  -> ready as a package stage if §252 is positive. +0.3 to +1.0 or a scheme < 0: undecided (report form, cost, and which
  part to push). < +0.3: this causal form does not carry the in-session personal gain; recorded, not closed (other forms:
  nets, feedback labels, fusion before the filter).
**§251 RESULT (14 Nov 16:07 UTC; evidence/softfp_14nov/{softcount,softcount_v2,runcount_G1,score_off,score_tsem,stack}.txt; code
studies/14nov/softfp/; Mac CPU, shufcombo unedited via runpy; zero-form checks exact on all 12 orders (offsets and TSEM
hooks); SCF γ 0 equals raw).** New people, V6Fr, six bases, change against the current emission latchJ(s242r15) unless
noted, mean over orders [80 % interval, people and orders resampled] people up:

| arm | full 0-3 | within 0-3 | runs 0-3 |
|---|---|---|---|
| (0) hard G1 groups, filter − raw | +0.22 [+0.04, +0.40] 7/10 | – | – |
| (1) GSCF nested (γ / top-k) | +0.30 [+0.06, +0.53] 6/10 | +0.03 | −0.02 |
| (1b) GSCF m15 / m20 (smooth form) | +0.16 / +0.16 [+0.04, +0.28] 8/10 | +0.03 | −0.02 |
| (1b) GSCF nested over M | +0.02 [−0.18, +0.19] | +0.03 | −0.02 |
| (2) soft offsets kern | +0.13 [−0.02, +0.28] 6/10 | **−0.17** [−0.25, −0.08] 1/10 | +0.01 |
| (2) soft offsets top10 | +0.17 [+0.00, +0.35] 6/10 | **−0.32** 1/10 | −0.16 |
| (3) TSEM local templates | −0.12 [−0.39, +0.13] 4/10 | −0.06 | +0.01 |
| (4) stack kern + m15 / m20 | +0.22 / +0.25 [+0.08, +0.41] 8/10 | **−0.15** 1/10 | +0.01 |

- runcount.py with G1 groups on s215 (V6Fr): full 0 +0.26, full 1 +0.34 (P0r +0.38 / +0.02).
- **Against the pass rule (≥ +0.5 full, ≥ −0.1 within / runs): no arm passes.** The count-filter forms keep within / runs
  (the latch hands over to J) and give +0.16 under full in the smooth form; nested (1)'s +0.30 is γ 1's spike (γ 0.75 /
  1.25 / 1.5: +0.16 / +0.05 / +0.12). Offsets and the stack fail within (1/10 people up). TSEM local templates are ≤ 0.
- **Predictions:** (0) +0.15 hit (+0.22); (1) SCF full +0.10 hit (+0.30 nested, smooth form +0.16), within / runs hit;
  (1b) +0.25 missed low (nested +0.02, fixed m15 +0.16), the spike prediction hit; (2) full +0.15 hit, within 0.00 missed
  (−0.17 / −0.32); (4) full +0.25 hit, within missed.
- **Reading:** soft fingerprint neighbourhoods recover about 0.15-0.25 of the full-shuffle loss (true runs: filter about
  +0.8, offsets about +1), consistent with the weights carrying only about a quarter of their mass in the true run (0.27
  against a 0.16 base). Under within, the offsets remove real class structure (neighbours in recording time share task
  blocks and slow drift that the stages already use). Status: **undecided-weak** for the count-filter form (m15 / m20 as
  GSCF: +0.16 full, within / runs inside ±0.05, 8/10 people, deployable as an emission-only addition before the latch);
  offsets and TSEM local templates in these forms fail the within bar. Larger gains would need a sharper same-run weight
  than this fingerprint gives (top-weight neighbour same run 0.61).
**§253 amendment (written 14 Nov 16:24 UTC, before any result; no personal output scored yet).** Run time and access: Leonardo
became unreachable (ssh time-outs on every login node) and a classic.fit_predict refit takes 0.4-1 s, ~90 % of it in the
Riemannian means of TangentSpace / MDM. Pipelines replaced by `causalpers/fast.py`: l1 = log-Euclidean FB tangent space
(per-window matrix log, the reference absorbed by the scaler) + LR; a2 / a4 as classic (FBCSP, slow waveform, weighted LDA);
lE = mean of l1, a2, a4 (MDM dropped). True-label s246 W splits, no EA: l1 76.0 (s246 EA a1 75.1), a2 66.9 (71.8), a4 59.2
(60.0), lE 77.3 (aE 78.2). Refit 44 / 65 / 101 ms at 40 / 100 / 180 windows. k 5 dropped (k in 10 / 20). Grid pipelines
{l1, a2, lE}; the fixed line becomes filt, lE, k 10, t0 40, R 40, cap 0.5. Predictions and pass rule unchanged (read a1 as
l1, aE as lE).
**§253 amendment 2 (written 14 Nov 16:30 UTC).** Stated plainly: written after the scorer's smoke on the first 2-6 order x base files
(within0, full2), which showed small changes (+0.2 to +0.5), picks at the grid edge (hard, lE, cap 0.25, R 80) and the
personal model alone at 63 % against the chain's 77 % on windows >= 40 (argmax agreement 66 %). The primary (pre-registered
grid, nested, track F) and the pass rule are unchanged. Secondary lines, added now: (a) EXT grid: caps 0.1 / 0.15 and R 120
added (nested again; reported separately); (b) label form 'top': hard labels, refit only on the more confident half of the
prefix (median of the chain's max posterior over windows < t), unweighted - a purity-for-quantity form (pp_*_top.npz, TOP=1;
nested over the grid with it); (c) fusion before the count filter: filter(raw x personal^w) with the urn on the fused
posterior, small grid (hard, lE, k 10, R 80, t0 40 / 60, cap 0.15 / 0.25 / 0.5), nested, against r15 + filter
(prefilt.py). Predictions: (a) +0.1 over the primary; (b) within ±0.2 of hard; (c) within ±0.2 of track F.

**§252 RESULT, part (a) (14 Nov 16:43 UTC; evidence/personal_ceiling_14nov/score_pl_classic.txt, randnoise.txt; Galvani CPU 3102464-84,
check job: true labels through the new weighted code reproduce classic.npz to < 1e-8).** Pseudo-label accuracy on all rows
0.74-0.78 (filter +0.0 to +0.6). r15 chain on the test runs 79.7 (20 orders); labelled ceiling re-scored on r15: c x aE x bE
+7.40 [+5.67, +9.22], c x aE +5.67. Classic pipelines on the chain's labels of the four training runs, 20 orders:
| form | aE alone | c x aE gain [80 % people x orders] | people up | share of labelled +7.4 |
|---|---|---|---|---|
| hard | 67.8 | -0.98 [-1.98, +0.07] | 3/10 | -13 % |
| soft (wpl) | 67.3 | -1.04 [-2.14, -0.02] | 3/10 | -14 % |
| filt (gated count filter) | 67.7 | -0.88 [-1.91, +0.13] | 4/10 | -12 % |
- aE falls 12 points from true to chain labels (79.8 -> 67.8); the same error rate drawn at random gives 63.6 (within0), so it is
  the label noise itself at about 136 training windows, not the chain's systematic errors, that sinks it. Fused, it costs
  about 1 point; only P09 (+3.6) and P08 (+1.3) gain. Prediction (+1.0, -0.5 to +2.5) refuted; c x aE by the rule: < +0.5,
  the classic personal factor on self-labels is not a route in this form (not closed for denoised or confidence-selected labels).
- Part (b) (nets on pseudo-labels, Leonardo 64475142, 16 tasks) ran, but every Leonardo login node times out since about
  16:20 UTC, so its outputs (runs/personal_ceiling/ftpl_*.npz there) are not fetched yet; score with score_pl.py once fetched.

**§252 amendment 1 (written 14 Nov 17:00 UTC, before any part-(b) result is read).** Leonardo is down on CINECA's side; part (b)
(ftnet_pl.py, 16 tasks) reruns unchanged on Galvani A100s (galv_array.sh, smoke first). The predictions and rule of §252
stand; Leonardo outputs, if they arrive, are a duplicate and are not scored.

**§252 amendment 2: SIZING THE LABEL-FREE PERSONALISATION CLASS (written 14 Nov 17:00 UTC, before any result).** One map, CPU.
- **(a) Dose-response on TRUE labels:** c x aE gain over the r15 chain (same 84 splits, six orders x six bases for the fusion)
  when the personal model sees n of the four training runs' windows (n = 20, 40, 70, 100, all ~136; random subset, each class
  >= 2) at label purity p = 100, 95, 90, 85, 80, 75 %. Noise two ways: random (a wrong label drawn uniformly from the other
  two classes) and chain-like (per person the chain's own confusion on its sessions: class-wise error rates scaled to 1 - p,
  wrong label drawn from the chain's off-diagonal row). One draw per cell (seed fixed). aE x bE is not cheap (GPU per cell):
  left out, stated. Prediction: +1 contour at about p >= 92 % for n = 136 and p >= 97 % for n = 40; at p 100 / n 136 +5.7
  (the labelled ceiling); random-noise and chain-like within 1 point of each other (randnoise.txt: 63.6 vs 65.3 alone).
- **(b) The chain's operating curve:** per order (20) and session, at delivery position t = 40, 80, 120 and end, the windows
  delivered before t ranked three ways: max posterior of the r15 chain (base mean); agreement count of the chain's argmax with
  the eye net (runs/ocushuf/<order>, mean of 9 seeds) and the member mix (A.mix B2_pool, shipped families), ties by max
  posterior; the gated count filter's posterior (its own argmax and max). Purity = accuracy of the handed-over labels for the
  top k (k = 10, 20, ...). Prediction: at the end of session the chain hands over about 60 windows at 95 % and about 100 at
  90 %; agreement ranking about +2 points of purity at equal k; the filter about +0.5.
- **Verdict rule:** a (b) point (k windows at purity q, scheme mean) "falls in" (a)'s region if (a)'s chain-like gain,
  interpolated bilinearly at (n = k, p = q), is >= +1. If no point does, the label-free personalisation class (personal
  model on the chain's own labels, fused late) is sized and closed for this feature set; if one does, that point (t, ranking,
  k, q) names the form to build. Caveat stated now: (a)'s n windows are the four training runs, while (b)'s k come from the
  delivered prefix; the map treats a window as a window.

### 254. POPULATION-ANCHORED PERSONAL CLASSIC MODELS (written 14 Nov 17:01 UTC, before any result)
Coordinator's form after §252 (refit from scratch on chain labels: c x aE −0.9; labelled +5.7): pull the personal classic
model toward the population solution instead of refitting it. Code `studies/14nov/anchored/` (imports personal_ceiling
classic.py / classic_pl.py / splits.py, edits nothing), outputs `runs/anchored/`, Mac CPU (≤ 2 processes).
- **Space:** s246's recentred features (classic_feat.npz: every session EA-aligned on its whole-session mean covariance,
  non-causal, unlabelled; the ceiling form; the causal form would recentre on the delivered prefix). Tangent space at the
  identity (log-Euclidean, logm upper triangle x sqrt 2), which is a common space across people after EA. One scaler per
  held-out person, fit on the population rows (other nine people, all their sessions, TRUE labels: public data).
- **Members:** (L) multinomial LR on the six-band tangent features (5166 dims), loss weighted by the label weights, penalty
  ||w − w_pop||² / 2C (intercept free); w_pop = the same LR (C 0.1) on the other nine people. Grid from population to
  personal: pop (w_pop, b_pop) | C 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3 | pers (anchor 0, C 0.1 = s246's a1 form in
  this space). (D1) LDA on the 8-30 Hz tangent features (861) and (D2) LDA on s246's slow waveform (1312): class means
  (n_c mu_c + k mu_pop,c) / (n_c + k), covariance (N Sigma_LW + k Sigma_pop) / (N + k), priors likewise, pseudo-count k in
  inf (pop) | 1000, 300, 100, 30, 10, 3 | 0 (pers, = weighted shrinkage LDA). E = mean of L, D1, D2 (one k for D1, D2).
- **Labels (training runs only):** true; chain (classic_pl.labels on plabels_r15: hard, soft = wpl weights, filt = soft on
  the gated count-filter posterior).
- **Fusion and scoring:** as §252's scorer: c x personal (product, equal weight, fixed now), per r15 base, 20 orders, the 84
  W splits; gain over the chain alone on the same order; 80 % intervals resample people and orders (schemes equal).
  Strength chosen nested across people (for person p the grid point, jointly (C, k) for E, with the best scheme-equal mean
  gain on the other nine); the whole curve is printed with the nested pick marked. Zero-change: the scorer's chain-alone
  level must equal score_pl's (79.7) and its vectorised BA S.cell_score; the pers LR endpoint must equal sklearn's LR (same
  scaler, C 0.1) to 1e-4.
- **Primary:** c x E, filt labels, nested (C, k). Secondary: hard / soft, each member, the true-label curve.
- **Predictions:** pop alone (E) about 62; c x pop E +0.3 (0 to +0.8). True labels: c x pers E +5 (s246 c x aE +5.7 with
  FBCSP / MDM), best anchored point +5.5, so anchoring costs the labelled ceiling < 0.5 at the nested pick, and below 0.1 the
  small-C end falls toward the pop line. Chain labels: pers end about −1 (as §252); nested c x E filt +0.6 (−0.2 to +1.4),
  people up 6/10; the nested pick sits in the middle of the grid (C 1e-3 to 1e-2, k 30 to 300); filt >= hard by 0.1-0.3;
  L carries most of it (D2 slow alone <= +0.2).
- **Pass (coordinator):** with chain labels, c x anchored E (nested) >= +1.0 over the chain, >= 0 in every scheme, >= 8/10
  people up. Then sketch the causal form (refit every k windows on the delivered prefix with the chain's labels, prefix EA)
  for §253's helper. +0.3 to +1.0 or a scheme < 0: undecided (report which member and end to push). < +0.3: anchoring does
  not rescue the self-labelled classic factor in this form; recorded, not closed.

### 255. TEST-TIME TRAINING WITH A SELF-SUPERVISED LOSS (written 14 Nov 17:04 UTC, before any result)
Explore, ground-up form (coordinator). §246: +7.5 of in-session headroom with true labels; §252: the chain's labels at ~75 %
destroy it (label noise). s15 / r15 fine-tuning on pseudo-labels gives about +0.1 to +1. Untried: TTT (Sun et al. 2020), an
encoder adapted to the session by a label-free auxiliary loss, so label noise never enters. Code `studies/14nov/tttssl/`,
outputs `runs/tttssl/` (Galvani A100, then scored on the Mac).
- **Net.** The shipped FBLight SessionBN recipe (`fblight_ea_rot05_sbn`, ttt5 "fbl", 40 epochs, rot 0.05) with its output
  layer widened: rows 0-2 classify, the rest is the auxiliary head (the encoder = filter bank, spatial conv + SessionBN,
  attention; 32 features). Trained JOINTLY (CE + 0.5 x aux) on the nine other people, all_loso folds, seeds 0-5. Forms:
  **C** SimCLR (NT-Xent, t 0.1, 64-d head; views: shift <= 15, noise 0.2 sd, 10 % channels dropped, amplitude
  U(0.8, 1.25), as §147); **P** a spatial "rotation" pretext: left-right hemisphere mirror of the (EA-whitened) window,
  2-way (mirroring the aligned input equals aligning the mirrored raw window), applied to a random half. Twin: the same
  recipe, 3 outputs, same seed (= the shipped member).
- **Ceiling arms (non-causal: the whole held-out session, recording order, eval-mode SessionBN, then classify the same
  windows).** Joint nets: na (no adaptation); sA10 / sA40 / sA100 (aux loss, all trainable parameters, AdamW 1e-3, 10 / 40 /
  100 full-batch steps); sB40 (aux loss, SessionBN affine only, 5e-3, 40); pl40 (s15's loss: confidence-weighted CE on an
  augmented view + KL to the clean view, chain labels, all, 1e-3, 40); spl40 (pl40 + aux). Twin: na, pl40, tent40 (entropy
  minimisation, SessionBN affine, 5e-3, 40). Pseudo-labels: r15 chain V6Fr, mean of six bases, order within0 (one order for
  every ceiling arm; stated limitation). PL arms are in-sample distillation of the chain on the windows scored, so their
  net-alone number is biased upward against the SSL arms; read them in fusion.
- **Scoring.** New people (10), replay score.person on all_loso. Net alone: per seed, then mean over seeds; per-person change
  vs the same net's na; 80 % intervals resampling people and seeds. Fusion: chain (s242r15 V6Fr, per base) x net^w, net =
  6-seed mean, w 0.5 primary (0.25 / 1 reported), 12 orders (within0-3, full0-3, runs0-3) x 6 bases; change of c x arm^w
  against c x na^w and against the chain alone. Diagnostics: aux loss before/after per session, mirror accuracy on the
  held-out session before adaptation (P).
- **Predictions.** Joint na vs twin na: C -0.3 (-1.5 to +1), P -0.3 (-1.5 to +1). Net alone, primary sA40 vs na: C +0.5
  (-1 to +2), P +0.3 (-1 to +1.5); best SSL arm over forms below +2 (70 %). sB40 within 0.5 of sA40. Twin pl40 alone +4
  (+1 to +8, distillation); twin tent40 +0.5 (-1 to +2). Fusion w 0.5: SSL primary vs na +0.1 (-0.3 to +0.5); pl40 +0.2.
  Held-out mirror accuracy 60-75 % (above chance, so the pretext transfers).
- **Kill rule (coordinator).** Primary sA40 (either form) net alone >= +2 over its na with the 80 % people interval above 0
  -> causal form (update every k windows on windows < t, as s15) under shuffled orders, within / full / runs, people x order
  intervals. Below +2: no causal form in this round; +1.5 to +2.5 is "undecided" (bars are reading aids); < +1: "label-free
  TTT of this encoder does not reach the in-session headroom in this form", recorded, not closed (other encoders, aux tasks,
  members). Best-in-hindsight arms are printed and labelled as such; they do not trigger the causal form alone.
**§253 amendment 3 (written 14 Nov 17:07 UTC; coordinator after §252's result: non-causal chain-label aE fused -0.9 to -1.0, label
noise the cause).** The plain forms (hard / soft / filt) are scored as the confirmation on what the Mac finishes (no new
compute for them). Compute goes to selection forms, all hard raw-chain labels, unweighted refit on a selected prefix:
top (confidence >= prefix median), cq50 / cq25 (per chain class the top 50 / 25 % most confident, at least 3, as TSEM picks
templates), agree (raw and count-filter argmax agree), agcq (cq50 inside agree); Galvani CPU, 20 orders x 6 bases
(pp_*_b<b>_top_cq50_cq25_agree_agcq.npz). Small caps (0.1, 0.15) in the grid (EXT). Nested over the selection forms with the
same rule. **Stop rule (coordinator):** if no nested line reaches +0.5 (track F vs r15 + filter, or R vs raw), the verdict
is "not a route". TSEM overlap: reported from the redundancy diagnostic (personal-alone accuracy and agreement with the
chain); a before-TSEM fusion needs a chain run with the personal posterior inside the stages and is only done if a form passes.
Predictions: selection forms +0.2 to +0.5 over hard in the same nested design; the best nested line +0.4 (0 to +0.8).
**§254 amendment (written 14 Nov 17:07 UTC; stated plainly: after the scorer's smoke on the TRUE-label arm, three orders, no chain-label
output read).** The smoke showed the population classic solution is near chance on held-out people in this space: pop LR
alone 44.6 (sklearn LOSO check: C 1e-4 / 1e-3 / 0.01 / 0.1 give 48.5 / 47.2 / 44.9 / 44.8), pop LDA 8-30 Hz 39.0, slow 45.9;
and that the pre-registered C grid does not interpolate to the personal end (large C = weakly penalised refit around a large
w_pop, not the s246 C 0.1 fit): L pers 76.0 alone, L C 0.3 66.4. Primary, grid and pass rule unchanged. Added, secondary: (La)
the anchor scaled at the personal strength, penalty ||w − a w_pop||² / 2(0.1), a in 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1
(a 0 = pers), and Ea = mean(La, D1, D2) on the joint (a, k) grid, nested the same way, all four label forms. Prediction: the La
curve is flat to falling from a 0 (the anchor carries little: <= +0.2 for the nested pick over a 0, true and chain labels),
so anchoring to a classic population solution cannot rescue the self-labelled factor here; Ea nested within ±0.3 of E nested.
**§255 RESULT (14 Nov 17:24 UTC; evidence/tttssl_14nov/{score.txt,diag.log}; Galvani array 3102597, 18 tasks, all EXIT 0;
base-vs-harness max diff <= 2.2e-16 on every fold; replay/check.py OK).** New people, FBLight, 6 seeds. Net alone, change vs
the same net's na [80 % people x seeds] people up:

| arm | C (SimCLR) | P (mirror) |
|---|---|---|
| na vs twin na (57.88) | **-3.07** [-5.18, -0.90] 2/10 | -0.54 [-1.10, +0.03] 4/10 |
| sA10 | +0.55 [+0.09, +1.05] 8/10 | -0.79 2/10 |
| **sA40 (primary)** | **-0.18** [-1.05, +0.62] 3/10 | **-2.17** [-3.08, -1.28] 2/10 |
| sA100 | -1.39 3/10 | -3.17 2/10 |
| sB40 (SessionBN affine only) | +1.19 [+0.66, +1.77] 9/10 (hindsight best; still 1.9 below twin na) | -0.12 2/10 |
| pl40 / spl40 (in-sample distillation) | +23.8 / +14.4 | +22.2 / +21.4 |

Twin: pl40 +22.1, tent40 -21.1 (collapse to one class at 5e-3).
- **Fusion with the r15 chain, w 0.5 (12 orders x 6 bases), SSL arm vs its na:** C sA40 +0.16 [+0.03, +0.28], sA10 +0.09,
  sB40 0.00; P sA40 +0.25 [+0.01, +0.51], sB40 0.00. Every un-adapted net fused is below the chain alone (C na -0.56, P na
  -1.00, twin na -1.36 at w 0.5; -0.10 to -0.22 at w 0.25). The pl arms fuse to +2.3 to +2.7 over the chain, but that is not a
  pseudo-label TTT ceiling: the labels are the six-base mean of the chain in order within0 over the whole session, so fusing
  them adds a base and order ensemble of the chain itself (smallest on within, +1.1 to +1.3, the order the labels came from).
- **Diagnostics.** Held-out mirror accuracy before adaptation 0.90 (55 of 84 sessions > 0.95): the pretext is trivially solved
  on new people (left-right montage asymmetry, not task structure), so it carries no session signal. The contrastive loss
  falls 1.47 -> 0.44 in 40 steps on one session: the encoder fits the views and the fixed classifier drifts.
- **Against the predictions:** joint na vs twin: C -3.1 missed low (-0.3 predicted), P -0.5 hit. Primary sA40: C -0.18 inside
  the range (-1 to +2), P -2.17 missed low. Best SSL arm below +2: hit. sB40 within 0.5 of sA40: missed (C +1.19 vs -0.18).
  Twin pl40 alone +4 missed high (+22, in-sample distillation of a 78 % chain); tent40 +0.5 missed (collapse). Fusion SSL
  +0.1: hit (+0.16 / +0.25). Mirror accuracy 60-75 %: missed high (0.90).
- **Verdict by the kill rule:** primary sA40 net alone -0.18 (C) / -2.17 (P), < +1 for both -> **no causal form.** Label-free
  TTT of this encoder (joint SimCLR or mirror pretext, FBLight) does not reach the in-session headroom in this form; recorded,
  not closed. The only positive is C sB40 (+1.19 alone, 9/10, hindsight), which recovers a third of the joint head's -3.1 cost
  and is flat in the chain (0.00). Forms not run: a reconstruction aux task (masked channel / masked time) that cannot be
  solved trivially, larger trunks (sinc, fbraw), an aux weight below 0.5 so that joint training costs nothing.
**§255 amendment (second wave, written 14 Nov 17:26 UTC, before any result of it; coordinator, several-forms rule).** Code
`studies/14nov/tttssl/tttssl2.py` (imports tttssl.py; wave 1 unchanged), same folds, seeds 0-5, FBLight, Galvani.
- **Forms.** **R10 / R25**: masked reconstruction at aux weight 0.1 / 0.25. Input masked per window: one contiguous time block
  of U(25 %, 40 %) of the samples and a random U(25 %, 40 %) of the channels set to 0; the encoder's 32 features (input of the
  output layer) go through a light decoder (32 -> 128 ELU -> 369) to the clean window's log band power in the net's own nine
  filter-bank bands x 41 channels, z-scored per dimension over the batch (detached), MSE. The loss is therefore 1 - R2 of the
  reconstruction: 1 = saturated, 0 = trivial. (Raw-sample reconstruction is impossible from FBLight's band-power features, so
  the target is set at the level the encoder can represent.) **C10**: wave 1's SimCLR at aux weight 0.1. Twin: wave 1's twin
  na files (same code path, unchanged).
- **Arms, fixed now.** Primary **sB40** (aux loss, SessionBN affine only, AdamW 5e-3, 40 full-batch steps on the whole session);
  secondary **sA10** (aux loss, all trainable parameters including the decoder, 1e-3, 10 steps); na. No pseudo-label arms.
- **Report first:** joint cost (form na vs twin na, net alone); held-out reconstruction loss before adaptation (per session).
- **Predictions.** Joint cost: R10 -0.3 (-1 to +0.5), R25 -0.6 (-1.5 to +0.3), C10 -0.8 (-2 to +0.3). Held-out reconstruction
  loss before adaptation 0.4-0.8. Net alone sB40 vs na: R10 +0.3 (-0.5 to +1.2), R25 +0.4 (-0.5 to +1.5), C10 +0.5 (-0.3 to
  +1.3); sA10 within ±0.7 of 0. Fusion (w 0.5, 12 orders x 6 bases) sB40 vs na +0.05 (-0.2 to +0.3).
- **Kill rule (unchanged in kind).** sB40 of any form net alone >= +2 over its na with the people x seeds interval above 0, or
  fusion >= +0.5 over c x na^0.5 with the people x orders interval above 0 -> causal form under shuffled orders. Otherwise
  record "label-free TTT not reached in two forms (contrastive / mirror, masked reconstruction)" and stop this line.
**§255 amendment RESULT (14 Nov 17:46 UTC; evidence/tttssl_14nov/{score2.txt,diag2.log}; Galvani array 3102704, 18 tasks, all EXIT
0, base-vs-harness <= 2.2e-16; LAM checked in the log, 0.1 / 0.25; replay/check.py OK).** New people, 6 seeds.
- **Joint cost first (form na vs twin na 57.88, net alone):** R10 -0.24 [-0.57, +0.07], R25 -0.24 [-0.57, +0.12], C10 -0.03
  [-0.80, +0.77]: all within the 0.5 budget (wave 1's C at 0.5 cost -3.07).
- **Held-out reconstruction loss before adaptation** (1 - R2, session mean): R10 0.90, R25 0.89 (neither trivial nor
  saturated, but weak: R2 about 0.1); sB40 moves it only to 0.89 / 0.88. C10 NT-Xent 1.77 -> 1.37 under sB40.
- **Net alone, change vs na [80 % people x seeds] people up:** primary sB40: R10 +0.10 [-0.53, +0.75] 4/10, R25 +0.09 5/10,
  C10 +0.80 [+0.11, +1.56] 7/10 (P10-type spread: one person +5.1); secondary sA10: R10 +0.33, R25 +0.10, C10 +0.12.
- **Fusion with the r15 chain, w 0.5 (primary), sB40 vs na [people x orders]:** R10 -0.06, R25 +0.04, C10 +0.27 [+0.08, +0.47]
  9/10; sA10 +0.11 / +0.04 / +0.19. w 0.25: C10 +0.09; w 1: C10 +0.63, but every net fused at w 1 sits 3+ below the chain.
  All fused nets stay below the chain alone at w 0.5 (-0.7 to -1.4).
- **Against the predictions:** joint cost R10 -0.3 hit (-0.24), R25 -0.6 hit (-0.24, within the range), C10 -0.8 hit (inside
  the range, high). Held-out loss 0.4-0.8 missed (0.89-0.90: the reconstruction is weaker than expected). sB40 alone R10 +0.3 /
  R25 +0.4 hit (inside the ranges, low), C10 +0.5 hit (+0.80). sA10 within ±0.7: hit. Fusion +0.05: R10 / R25 hit, C10 +0.27
  hit (inside -0.2 to +0.3).
- **Verdict by the kill rule:** no sB40 reaches +2 net alone (best C10 +0.80), and none reaches +0.5 in fusion (best C10
  +0.27, interval above 0). **Label-free TTT not reached in two forms (contrastive / mirror; masked band-power
  reconstruction; aux weights 0.1-0.5; SessionBN-affine and full-encoder adaptation), FBLight; this line stops here.** The
  only consistent signal is contrastive SimCLR with SessionBN-affine adaptation (wave 1 C sB40 +1.19 alone / 0.00 fused;
  wave 2 C10 +0.80 / +0.27 fused, 9/10), small and below the noise of a stacked member. Not tested: larger trunks (sinc,
  fbraw), a causal form (not triggered).

### 256. C10 CONTRASTIVE TTT AS A STACKED MEMBER (written 14 Nov 17:46 UTC, before any result; coordinator after §255 wave 2)
Mechanism: §255's only consistent signal, the FBLight net trained jointly with a contrastive head at weight 0.1 (C10) and
adapted with the contrastive loss on SessionBN affine parameters only (non-causal: fused +0.27 [+0.08, +0.47], 9/10;
wave 1 at weight 0.5: 0.00), made causal and stacked. Per held-out person (nine-people nets, 6 seeds, studies/14nov/tttssl),
per session in delivery order: from window 20, every 10 windows, 5 adaptation steps on the windows delivered so far
(< t; the s15 schedule); window t is classified with the current parameters; seed-mean posterior. Fused with the
candidate's output as chain x net^0.5 (w fixed in advance; rows before window 20 unchanged), then the gated count filter.
Arms: (A) C10 adapted causally, (B) the same nets unadapted (control for the fusion itself). New people, the 20 orders,
against the candidate (r15 + filter), scheme-equal mean, people-and-order intervals, sealed weighting with returning
people at 0 (no returning-people form; reported as absent). Prediction: A minus candidate +0.15; A minus B +0.20.
Pass: A minus candidate >= +0.2 with the interval above 0 and >= 7/10 people up; else small option, not packaged.
Run time: time per window of the causal adaptation on an A100 and on a 2080, against the worker budget; reported with
the result, required before any package.
Third form (pre-registered here, run in parallel): the same C10 joint training and SessionBN-affine adaptation on a larger
trunk (fbraw or sinc, whichever ttt7 / ftrel already fine-tune), non-causal ceiling as in §255. Continue to the causal form
only if the net alone gains >= +1.5 or the chain fusion >= +0.5 (wave 2's C10: +0.80 / +0.27).

**§252 amendment 3 (written 14 Nov 17:48 UTC, after the amendment-2 map was read, before this test ran).** The map's rule fires,
marginally: the scheme-mean operating point at the END of the session, max-posterior ranking, k = 130 of ~190 windows at
83.4 % purity, interpolates to +1.2 in the chain-like grid (within +1.7, runs +1.25, full +0.6); no point at t <= 120 reaches
0 (best -0.9). The map treats a selected window like a random one; confident windows are the easy ones, so it is optimistic.
Direct test of the named form, before building anything: §252's splits, classic aE trained only on the top f of the four
training runs' windows ranked by the r15 chain's max posterior, with the chain's argmax labels (f = 85, 70, 55 %), fused with
the chain; six orders (within0/1, full0/1, runs0/1). Prediction: top 70 % gives -0.5 (-1.5 to +0.5), below the map's value
(the selection bias). Rule: >= +1 with the people interval above 0 -> the form goes to §253 as the end-of-session refit;
otherwise the class is closed for this feature set (the map's +1.2 does not survive selection).

**§256 implementation record (14 Nov 18:03 UTC, before any result; design unchanged).** Code `studies/14nov/tttssl/{causal,stack,third}.py`.
Choices the pre-registration left open, fixed now: refits warm-start the SessionBN affine weights and use a fresh AdamW (5e-3, wd 0,
5 steps, as ttt7's gC5), contrastive views as §255 (`view`), loss on the eval-mode forward of windows < t; windows are classified
with SB.to_stream's StreamBN (state carried over the session, affine copied in at each refit; zero-change: refits off equals
the eval forward, max diff 6e-8 on a smoke net). Windows 0-19 of each session position unadapted (A = B there, asserted). Nets are
retrained per task (same recipe and seeds as §255 wave 2; not saved), 30 tasks = 6 seeds x 5 groups of 4 orders; A and B share the net.
Fusion w 0.5 on session delivery positions >= 20, gated filter via the stack/score.py call, 6 bases, 20 orders; scorer `stack.py`
(W = 0 returns the chain, asserted). Third form: FBCNet raw (ttt5 "fbraw": fbcnet_raw_sbn, k 20, no pretrained trunk), C10 weight 0.1,
arms na and sB40, twin na for the joint cost; tttssl.run unchanged, scored by score.py FORMS=twin,C10. Run time: causal.py time on
an A100 and on a 2080, first fold of P01, one-epoch nets (run time does not depend on training), per session. Smoke: both scripts ran on the Mac
(one fold, one epoch).

### 257. COST OF CAPPING FINE-TUNING REFITS (written 14 Nov 18:08 UTC, before any result; coordinator, insurance for A100 time slack)
Mechanism: the package can stop fine-tuning refits after window N and keep the last net (build6 --cap N; A100 worst-case
slack 1.37x now, about 1.50x at N 150, 1.74x at N 100). ftrel.py CAPS=100,150,250 derives r15c<N> from the SAME r15 run:
identical to r15 before the first refit at or after N (asserted in the smoke), then the frozen net predicts block by block
from the prefix (SessionBN causal, as r15). Tag ttt18<order>LPC (the candidate's LP files are not touched); the uncapped r15
of the same run is the reference. Orders within0-3, full0-1, runs0-1 (8), new people, the full candidate system (shufcombo
FT_V=r15c<N> and r15, SAVE_OUT, then the gated count filter), people-and-order intervals.
Prediction: cost (capped + filter minus uncapped + filter) N 250 -0.02, N 150 -0.08, N 100 -0.15 (refits after window 100
touch about a third of each session's windows; r15's whole gain is +0.10).
Report: the cost per N; no pass rule (a reading for the time-slack decision).

**§252 amendments 2-3 RESULT (14 Nov 18:16 UTC; evidence/personal_ceiling_14nov/dose_map.txt, opcurve.txt, score_pl_sel.txt; Galvani CPU).**
- **(a) Dose-response, true labels, c x aE over the r15 chain (six orders, chain 79.95):** chain-like noise, purity 100 / 95 /
  90 / 85 / 80 / 75 %: n 20 -13.0 ... -18.8; n 40 -2.8 ... -8.2; n 70 +2.4 / +0.7 / -1.1 / -2.3 / -3.7 / -5.3; n 100 +4.4 / +3.1
  / +1.8 / +0.1 / -1.2 / -2.7; n all (129) +5.4 / +4.4 / +2.9 / +1.8 / -0.1 / -1.1. Random noise within about 1 point of it.
  **+1 contour:** n 70 needs 100 %, n 100 needs about 88 %, n 129 needs about 82 %; at n <= 40 nothing reaches +1 even clean.
  Predictions: contour at n 136 p >= 92 % (found ~82 %: noise hurts less than predicted, but small n hurts much more); n 40
  p >= 97 % (found: never). Consistent with part (a): n 129 at about 77 % purity gives -0.6 on the grid, -0.9 measured.
- **(b) Operating curve (20 orders, scheme mean, max-posterior ranking):** t 40: 10 at 89 %, 40 at 73 %; t 80: 40 at 86 %, 70
  at 78 %; t 120: 40 at 89 %, 70 at 85 %, 100 at 80 %; end: 40 at 91 %, 100 at 86 %, 130 at 83 %. The chain's purity tops out at
  about 93 % even for its 10 most confident windows. Agreement ranking (eye net + mix) is worse than max posterior (by 1 to 4
  points at equal k); the count filter is within ±0.5. Predictions (end: 60 at 95 %, 100 at 90 %; agreement +2) refuted:
  the chain does not hand over high-purity windows.
- **Map verdict (rule of amendment 2):** one region fires: end of session, k 130 at 83.4 % -> +1.2 (within +1.7, runs +1.25,
  full +0.6); nothing at t <= 120 reaches 0. **Direct test (amendment 3)** of the named form, top-f of the training runs by
  max posterior with chain labels (purity of the kept windows 80 / 83 / 86 %): c x aE top85 -0.94 [-1.87, -0.02], top70 -1.59
  [-2.59, -0.60], top55 -3.41 [-4.53, -2.14]; all below plain -0.96 except top85 (equal). The map's +1.2 does not survive
  selection: confident windows are the easy ones and teach the personal model less than random windows at the same purity.
  **By the stated rule the label-free personalisation class (classic personal model on the chain's own labels, fused late)
  is sized and closed for this feature set.** Not covered: the fine-tuned nets on pseudo-labels (part b, Galvani array
  3102582 queued behind smoke 3102581, A100 queue full) and label forms that are not the chain's own (e.g. a teacher).

### 258. CO-TRAINING LABELS FROM THE EYE NET (written 14 Nov 18:18 UTC, before any result; coordinator 19:17 CET item 4)
Mechanism: §252 showed that the chain's confident windows are the easy ones and teach a personal EEG model less than random
windows at the same purity. Co-training needs a labeller whose errors are independent of the learner's features. The eye net
reads slow frontal potentials, and its errors are nearly independent of the EEG members (§188, §234: error correlation about
+0.15). So: label the training runs by the eye net (9-seed mean posterior from runs/ocushuf/<order>), keep its most
confident windows per class (top q, q in {30, 50, 70} %), optionally only where the eye net and chain agree (the A arm), and
train the personal classic EEG pipelines (aE, studies/14nov/personal_ceiling/classic.py) on them; fuse chain x aE as §246.
Same 4-train / 2-test splits and scoring as §246 / §252, the r15 chain, 20 orders, people-and-order intervals.
Prediction: eye-labelled windows at q 50 are about 80 % pure (eye net alone about 65 % BA, top half more), and because their
errors are independent of the EEG features they behave like random noise of that purity at n about 65: about -1 to +0.5
(the §252 map's n 70 row at 80 % gives -3.7, so a positive result needs the independence to help beyond the map).
Pass (non-causal ceiling): c x aE >= +1.0 over the chain with the interval above 0 and >= 8/10 people up; then a causal
form. Else closed.

### 259. BALANCED TRANSDUCTIVE CLUSTERING IN A PERSONAL FEATURE SPACE (written 14 Nov 18:18 UTC, before any result; ground-up)
Mechanism: inside a session, personal models trained with true labels reach 80-90 % (§246), so the classes are separable
in personal feature spaces (FB tangent space, FBCSP-like projections). Instead of labels from the chain (75 % pure), take
labels from the session's own cluster structure: cluster the session's windows into 3 clusters with each run's class
balance as a constraint (balanced k-means / constrained GMM, or optimal transport with uniform class marginals per run), in a
space fitted label-free on that session (tangent space at the session mean, then a projection learned on the nine other
people's labelled data, e.g. LDA directions from the population in the recentred space). Map clusters to classes by the
chain's posterior (Hungarian assignment on summed posteriors). Use the cluster labels (a) directly as a third factor
(chain x cluster posterior), (b) as training labels for aE.
Ceiling first, non-causal (whole session): cluster-label purity per person against the chain's 75-78 %, then the fused
gain. Prediction: purity 70-80 % (the population projection carries the between-person shift), fused gain -0.5 to +1.
Gate: purity >= 85 % on average, or fused gain >= +1.0 with >= 8/10 people up, before any causal form. Else closed.
Both CPU-only (Mac at most 2 processes; cpu-galvani).
§259 amendment (14 Nov 18:19 UTC, before any §259 result; coordinator's guards). (b) The balance constraint must be a design
constant known in advance: per-run balance needs run ids, which sealed delivery does not have. The primary form uses
PER-SESSION balance (uniform class marginals over the session's windows); the per-run form is reported only as an oracle arm,
labelled as using true run ids. (a) Next to the transductive ceiling, report the causal prefix form: cluster only the
windows delivered before t (t = 80, 120, end of session) in the delivery orders, and score the windows from t on. The gate
is read on the per-session-balance arm; the package could use only its causal form.
§258 amendment (14 Nov 18:20 UTC, before any §258 result). Degenerate splits: when the kept eye-net labels give a class fewer
than 2 training windows (E30 7, E50 1, E70 1 of 1680 splits; A30 357, A50 192, A70 111, concentrated in P01, P04, P10, where
eye net and chain rarely agree on one class), that split falls back to the chain alone (gain 0); each arm reports its
fallback share next to the gain. Arm A = top q % per eye-argmax class first, then kept only where eye argmax = chain argmax.
Noted before the run: the eye net's argmax accuracy is 51 % on within 0 (chain 77.7 %), so the predicted 80 % purity at
q 50 is likely high; the prediction is left as written.
**§253 result (written 14 Nov 18:22 UTC). Verdict: not a route, sized by §252's map** (label-free classic personalisation on this
feature set needs n >= 100 at ~88 % purity for +1; the causal prefix never gets there). Scorer equals replay score.person
on every order x base (zero-change OK). r15 + filter vs raw +0.14. Evidence `evidence/causalpers_14nov/`.
- **Plain forms** (hard / soft / filt; Mac, 82 order x base files, 14 orders: within 0-11, full 0 (4 bases) + 2, runs 2):
  primary nested track F vs r15 + filter **-0.01** [-0.32, +0.26], 6/10 (within +0.05, full -0.19, runs +0.10); fixed line
  -0.36; nested R vs raw +0.16; best in hindsight +0.23. EXT grid: nested F -0.13. Prediction (+0.4) missed.
- **Selection forms** (amendment 3: top, cq50, cq25, agree, agcq; Galvani + Mac, 94 of 120 files, all 20 orders; jobs
  cancelled at the coordinator's stop): nested F vs r15 + filter **+0.04** [-0.13, +0.20], 7/10 (within +0.15, full -0.11,
  runs +0.07); EXT grid (caps 0.1 / 0.15) +0.15 [+0.03, +0.28], 7/10, every scheme >= 0; vs raw +0.29, 9/10. Best in
  hindsight +0.16 / +0.22. No line reaches +0.5; pass (+1.0) far off.
- Small caps help only by switching the personal model nearly off: cap 1.0 is the worst column of every grid.
- **Why:** on windows >= 40 the personal model alone is 56-63 % against the chain's 77.5 %, argmax agreement 60-65 %:
  it relearns the chain's label noise with fewer windows. TSEM (confidence-seeded label-free LDA on core1 PCs) and r15's own
  prefix relabelling already sit in the chain; a before-TSEM fusion was not run (no form passed, as amendment 3 stated).
- **Run time** (Mac CPU, one thread, `timing.txt`): refit l1 41 ms / lE 88 ms at 120 windows; per window lE 25 ms at k 10,
  21 ms at k 20 (features 16 ms of it). Cheap enough; the gain is not there. Amendment-2 prefilt was stopped unscored.
Plan note (14 Nov 18:22 UTC; coordinator): §253's capped form (selected labels, fusion cap 0.10 / 0.15: +0.15 [+0.03, +0.28],
7/10) is kept as a stack candidate, recorded undecided. When §256 reads out, a joint stack test gets its own pre-registration:
candidate + C10 (only if §256 passes) + capped cpers, all three schemes, nested choices. Time cost noted: cpers 25 ms per
window x about 20k windows is about 8 min of worker CPU against a worst-case slack of 1.37x, so a cheaper form (k 20, or
refits only up to t = 150) is required before any package.

**§259 RESULT (14 Nov 18:38 UTC; studies/14nov/transclust/space.py, score.py; evidence/transclust_14nov/score.txt, space.log; Mac CPU via run_capped; the 18:19 amendment applied: per-session balance primary, per-run as oracle).** Space: FB tangent space at each session's Riemannian mean, standardise, PCA 200, shrinkage LDA (2 directions) fitted on the nine other people; balanced k-means (exact Hungarian balance), 20 restarts picked by loss; mapping by summed chain posterior; Sinkhorn soft posterior (eps 0.1 x median cost, numpy: POT not installed); weight 1; 20 orders.
| arm | mapped-cluster purity | chain purity (same windows) | (a) c x soft cluster | (b) c x aE(cluster labels) |
|---|---|---|---|---|
| per-session balance (PRIMARY) | 48.2 % | 79.2 % | -26.97 [-30.01, -23.86], 0/10 | -12.71 [-15.20, -10.15], 0/10 (aE alone 46.2) |
| per-run, true run ids (oracle) | 50.7 % | 79.2 % | -23.74, 0/10 | -9.84, 0/10 (aE alone 51.5) |
With the best cluster-to-class permutation chosen from the labels (diagnostic only) purity is 49.9 % (sess) / 52.5 % (run): the clusters do not follow the classes in this space, so the chain mapping is not the cause. Causal prefix (a), windows from t on, nearest centroid: t=80 purity 46.6 % vs chain 80.6 %, gain -29.6 [-33.8, -25.4] 0/10 (within -31.2, full -27.6, runs -30.2); t=120 purity 45.8 % vs 80.7 %, gain -30.4 [-35.2, -25.8] 0/10 (one empty person x order cell filled); end of session = the per-session arm. Prediction (purity 70-80 %, gain -0.5 to +1) refuted: purity far below.
**Verdict by the gate (purity >= 85 % or gain >= +1.0 with >= 8/10 up): closed in this form, on the primary arm and the oracle.** Not a closure of clustering in other spaces: one fixed space (population LDA on FB tangent features, 2 directions). Limits: the soft posterior is sharp, so a confident wrong cluster multiplies the chain at weight 1 (a tempered form not run; the gate is the purity); (b) uses hard labels; no causal form of (b); 4-train / 2-test splits as §252.

**§254 RESULT (14 Nov 19:09 UTC; evidence/anchored_14nov/{score,score_alpha}.txt; code studies/14nov/anchored/; Mac CPU; checks: pers LR =
sklearn to 7e-7, LDA k 0 = classic.wlda to 3e-13, scorer chain alone 79.70 = score_pl, a 0 = L pers on every split).** Change
of c x system over the r15 chain, 20 orders x 6 bases, nested pick [80 % people x orders] people up:
| labels | E nested (primary family) | E pers end | L nested | La nested (amend.) | Ea nested (amend.) | best of 106 in hindsight |
|---|---|---|---|---|---|---|
| true | +3.55 [+1.39, +5.68] 8/10 | +2.89 | +3.63 (picks pers) | +3.51 | +3.48 | +3.63 (L pers) |
| hard | −2.44 | −2.18 | −2.90 | −2.83 | −1.94 | −1.91 |
| soft | −2.49 | −2.10 | −2.66 | −2.43 | −1.84 | −1.80 |
| filt | **−2.25 [−3.84, −0.68] 2/10** | −1.92 | −2.44 | −2.25 | −1.77 [−3.39, −0.31] 2/10 | −1.66 |
- **Primary fails the pass rule** (needs >= +1.0, >= 0 per scheme, 8/10): −2.25, every scheme < 0 (w −2.97 f −1.78 r −2.00), 2/10.
- **Why:** the population classic solution carries almost nothing for a held-out person in this space: pop LR alone 44.6
  (fused −25.1), pop LDA 8-30 Hz 39.0, slow 45.9 (E pop end −13.6 fused, smoke). Every step toward it costs: with true labels
  the La curve falls monotonically from a 0 (+3.63) to a 0.3 (+1.56) and a 1 (−6.61); the nested picks are the personal end.
  With chain labels anchoring buys at most +0.2 over the pers end (La a 0.1 −2.25 vs −2.44; Ea a 0.3 −1.77 vs E pers −1.92),
  far inside noise and far below 0. D2 (slow) alone is best at k 100-300 (shrinkage helps it), but stays −8 to −11 fused.
- **Caveat on the instrument:** this E (LR + two high-dim LDAs, product fusion) is weaker than s246's aE: labelled c x E pers
  +2.9 vs aE +5.7, chain-label pers end −1.9 vs §252's −0.9; the LDA members are overconfident (fused alone −11 to −23). The
  conclusion rests on L / La, whose pers end equals s246's a1 form (alone 76.0) and which shows the same picture.
- **Predictions:** pop E 62 (missed: ~46); c x pop +0.3 (missed: strongly negative); true pers +5 (missed: +2.9), anchoring
  costs < 0.5 at the nested pick (hit: −0.08 La, nested = pers end); chain-label nested +0.6 (refuted: −2.25); La flat to
  falling, <= +0.2 over a 0 (hit).
- **Verdict:** anchoring a personal classic model to a population classic solution does not rescue the self-labelled
  classic factor; < +0.3, recorded. Not closed for anchors that transfer: the population nets (the chain's own members
  reach ~80 on new people) are the anchor that carries information, which is what s15 / r15's warm start already uses. A
  head anchored on the nets' embeddings (classic-cost refit of the last layer toward the population head) is the open form.
  No causal form is sketched (rule not met).
§243 draw 2 (14 Nov 19:17 UTC; evidence/stack_14nov/s243_d2.txt). r15 + filter +0.21 [+0.09, +0.34] 9/10 (within +0.37, full
+0.14, runs +0.13); r15 +0.07. Over three draws the candidate's new-people half is +0.25 / +0.23 / +0.21 (spread 0.04,
mean +0.23), every draw positive in every scheme. §243's prediction (spread < 0.15, positive in every draw) holds.

**§258 RESULT (14 Nov 19:21 UTC; evidence/cotrain_14nov/score.txt; code studies/14nov/cotrain/{eyelab,score}.py; Mac CPU, 20 orders, r15 chain 79.70).**
Rule of the amendment applied: a split whose kept labels leave a class < 2 windows falls back to the chain alone (gain 0). Arm A = top q % per eye-argmax class first, then agreement with the chain. c x aE gain over the chain, scheme-equal mean [80 % people x orders], people up; n and purity of the kept labels over non-fallback splits; dose-map = §252 chain-like grid at that (n, purity), purity clipped at 75 (an upper bound below it):

| arm | n | purity | fallback | dose map | gain | people up |
|---|---|---|---|---|---|---|
| E30 | 39 | 60.3 % | 0.4 % | -8.9 | -12.63 [-15.64, -10.02] | 0/10 |
| E50 | 65 | 57.0 % | 0.1 % | -5.8 | -11.22 [-14.14, -8.34] | 0/10 |
| E70 | 90 | 54.4 % | 0.1 % | -3.5 | -10.25 [-13.34, -7.30] | 0/10 |
| A30 | 25 | 83.0 % | 21.2 % | -14.0 | -10.71 [-13.69, -7.98] | 0/10 |
| A50 | 39 | 81.7 % | 11.4 % | -7.5 | -8.64 [-11.35, -6.12] | 1/10 |
| A70 | 52 | 80.7 % | 6.6 % | -5.9 | -7.02 [-9.91, -4.29] | 1/10 |

Schemes (within / full / runs) differ by < 1 point in every arm. **Prediction refuted:** purity of the eye-labelled windows is 54-60 %, not about 80 % (the eye net is 51 % accurate on within0 and its confidence barely selects: top 30 % is 60 %); the prediction was -1 to +0.5 and the arms are -7 to -13. **Verdict by the pass rule (c x aE >= +1.0, interval above 0, >= 8/10 people up): fails in all six arms; no causal form.** The agreement arms reach 81-83 % purity but at n 25-52 (and 7-21 % fallback), where the §252 map already gives -6 to -14; measured results are at or below the map (E arms 2-7 points below at purity < 75, where the map is an upper bound; A arms 0.5 to 3 above at n 25-52). No sign that the eye net's independence helps beyond the map; the A70 arm (about +1 over the map at equal n and purity) is the only one near it. Closed in this form (classic aE on eye-net co-training labels); not covered: the eye net with the member mix as second labeller, or labels from a much stronger independent source.

### 260. CAPPED CAUSAL PERSONALISATION, FIXED FORM, ON FRESH DRAWS (written 14 Nov 19:24 UTC, before any result; research thread)
Mechanism: §253's nested picks were nearly one form for all ten held-out people: labels cq50 (or agcq), FBCSP pipeline a2,
refits every k 10 from t0 90, weight ramp R 120 to cap 0.10, fused into the count-filtered r15 output (track F); nested +0.15
[+0.03, +0.28] 7/10, that form in hindsight +0.22. The form was chosen on draw 0 of the r15 chain, so its value is biased.
Replication on chains it never saw: the same code (studies/14nov/causalpers, chainf / run / score) on the r15 draw-1 and
draw-2 chains (runs/chainout/s242r15d1_*, s242r15d2_*; 20 orders x 6 bases each), ONE fixed form, no grid: cq50, a2,
t0 90, R 120, cap 0.10, k 20 (primary: 11.3 ms per window against 12.9 at k 10, timing.txt) and k 10 (reported). Reference:
r15 + filter of the same draw (the candidate's new-people half). New people only; returning people are absent (cpers was
never built for ER; stated). Zero-change: cap 0 equals r15 + filter on every order x base (the scorer's existing check).
Prediction: +0.12 per draw (80 % range -0.02 to +0.25), two draws within 0.1 of each other, >= 6/10 people up.
Rule: both draws > 0 with the pooled people-and-order interval above 0 -> the form joins the candidate list with its run
time (it then needs a package-form check); one draw <= 0 or pooled interval across 0 -> "undecided", kept out of the
package. The §253 plan note's joint stack test reduces to this: C10 drops out (§256), and the candidate's new-people half is
the reference here.

### 261. V6.5b: empty_cache AFTER EACH REFIT (written 14 Nov 19:30 UTC, build thread, before the GPU checks)
Mechanism: A100 timing (3102619) showed the CUDA caching allocator reserving 41.5 of 42.4 GB (peak allocated 6.7-7.2 GB)
with retry warnings on 378- and 800-window unrecognised sessions; each refit's batch grows with the prefix and
empty_cache ran only in reset_state. Change: FineTune.refit calls torch.cuda.empty_cache() in a finally after the refit
(guarded by torch.cuda.is_initialized(); code lean/src65b, bundle submissions/lean_v65b, keys as v6.5 cand). No other change.
Predictions: (1) CPU: bitwise identical outputs to v6.5 cand (no CUDA); Mac replay 0.9945788530. (2) GPU: outputs vs v6.5
GPU within non-determinism (<= 0.05), err 0, labels 100 % vs CPU. (3) A100 peak_reserved on the 378 / 800 sessions
<= 12 GB, zero allocator retry warnings, ref wall time within +5 % of v6.5 (empty_cache cost small against 0.35 s refits).
Refutation: any CPU output difference, any err, reserved > 20 GB or a retry warning, or run time > +10 % -> v6.5b is not
the candidate; v6.5 stays.

**§256 RESULT (14 Nov 19:38 UTC; evidence/tttssl_14nov/stack.txt (primary), score3.txt (third form); stack.py W 0.5, 30 causal
files = 6 seeds x 5 groups, Galvani GPU 3102959; checks: W 0 returns the chain raw and filtered ("w0 ok", tolerance 1e-6 for
float32-derived rows), arm A = arm B before window 20).** New people, 20 orders x 6 bases, change [80 % people x orders], people up:
| contrast | change | per scheme (within / full / runs) |
|---|---|---|
| A (C10 adapted causally) + filter minus candidate (r15 + filter), PRIMARY | **-0.84 [-1.41, -0.27] 3/10** | -0.87 / -0.49 / -1.16 |
| B (same nets unadapted) + filter minus candidate | -0.92 [-1.52, -0.33] 3/10 | -0.97 / -0.58 / -1.21 |
| A minus B (filtered) | +0.08 [-0.02, +0.18] 7/10 | +0.10 / +0.08 / +0.05 |
| candidate minus r15 raw (filter) | +0.15 [+0.05, +0.25] 9/10 | +0.31 / -0.00 / +0.15 |
Per person A+f minus candidate: -3.0 -2.6 -2.2 -0.9 -0.2 -1.5 +1.3 -0.3 +0.2 +0.8 (P01..P10).
- **Verdict: fails the pass rule** (needs >= +0.2, interval above 0, >= 7/10). Predictions: A minus candidate +0.15 (refuted:
  -0.84); A minus B +0.20 (missed: +0.08, inside noise).
- **Why (a misreading in the pre-registration, recorded):** §255's "C10 fused +0.27" was C10 adapted minus C10 unadapted inside
  the same fusion, never fused minus chain. The net alone is about 62.5 % on new people against the chain's 79 %, so any
  product fusion at w 0.25-1.0 costs -1.2 to -6 (score3.txt shows the same for the fbraw trunk). The adaptation itself is a
  small, consistent +0.08 to +0.10 on top of a fusion that loses about 0.9.
- **Third form (fbraw trunk, non-causal, score3.txt):** net alone sB40 minus na +0.09 [-0.64, +0.95] 4/10 (rule >= +1.5: no);
  fusion sB40 minus na +0.26 at w 0.25, +0.54 at w 0.5, +0.80 at w 1.0, but every fusion minus chain is -1.2 to -5.5. Read
  literally the rule's "fusion >= +0.5" fires at w 0.5; it measures the same within-fusion contrast as above, which does not
  reach the chain, so no causal form is run. Stated as going past the letter of the rule.
- **Run time** (reported by the ni26-push collector from Galvani timeA100-3102961 / time2080-3102962 logs): causal C10
  adaptation about 20 ms per window on an A100, 15 ms on a 2080. Not needed: no package.
- **Closed in this form** (a weak TTT-SSL net fused as a product factor). Not covered: contrastive TTT applied inside the chain's
  own nets (r15's fine-tuning step) instead of as a separate member. §253's plan note's joint stack test reduces to §260.

**§252 amendment 4 (written 14 Nov 19:37 UTC, before any part-(b) score is read; coordinator's decision).** Amendment 1 set the Leonardo
ftpl files aside only because Leonardo was down and a Galvani rerun was planned; Leonardo is reachable again and holds all 16
(ftpl_{fb,fbl,fbraw,sinc}_s{0,1}_g{0,1}.npz, written from 17:32 UTC, job 64475142). Checked: Leonardo's ftnet_pl.py, classic_pl.py, ftnet.py,
splits.py, classic.py match git e649e98 byte for byte (md5 of ftnet_pl 90e3bbec...); HEAD differs from that only by 9e47ed5's
skip-if-output-exists line (no training change), which the Galvani run uses. plabels_r15.npz md5 1c009212... is identical on
Leonardo and local (Galvani's copy not re-hashed: unreachable at writing; to be checked). Seeds and tasks are fixed in code (task t:
family ORDER[t % 4], seed (t // 4) % 2, group t // 8). Not available: the Leonardo job log (not looked for beyond file names).
The 16 Leonardo files are the primary part-(b) score (fetched to runs/personal_ceiling_leo/, never mixed with Galvani files;
score_pl.py takes the dir from env PL_FTPL_DIR and fails if it is missing or holds fewer than 16 files). The Galvani copy
(3102582), once complete, is scored from runs/personal_ceiling/ as a reproducibility check, per-arm difference reported.
Predictions and rule of §252 unchanged.

**§252 part (b) RESULT (14 Nov 19:5x UTC; evidence/personal_ceiling_14nov/score_pl_b.txt; the 16 Leonardo files of job 64475142 as primary per amendment 4, scored with PL_FTPL_DIR=runs/personal_ceiling_leo; r15 chain 79.95 on the six bE orders).** Scorer fixes: GROUPS defined locally (the ftnet_pl import needs torch), bE only for hard / soft / filt (the loop over the top* forms would have raised KeyError); the primary reads the part-(b) files, no silent skip.
| form (six orders, [80 % people x orders]) | bE alone | c x bE gain | people up | c x aE x bE gain (primary) | people up |
|---|---|---|---|---|---|
| hard | 77.1 (-2.8) | +0.83 [+0.00, +1.67] | 6/10 | +0.18 [-0.72, +1.10] | 5/10 |
| soft | 77.2 (-2.7) | +0.86 [+0.01, +1.68] | 6/10 | +0.34 [-0.51, +1.25] | 5/10 |
| filt | 77.3 (-2.6) | +0.83 [-0.01, +1.68] | 6/10 | +0.42 [-0.55, +1.38] | 5/10 |
- Labelled ceiling on the same orders: c x aE x bE +7.18 [+5.40, +9.11]; so filt c x aE x bE is 6 % of it. Fine-tuned on the chain's labels the net alone is 2.6 below the chain (bE 77.3 vs 79.95).
- Prediction: c x bE +0.8 holds (filt +0.83); primary c x aE x bE +1.5 refuted (filt +0.42, aE pulls it down by about 0.4: c x aE -0.91).
- P04 (the 49-point person): c x bE -1.5, primary -2.8; P10: c x bE +3.9, primary +3.0. Both reported; neither decides. Gains concentrate in P09 (+5.1) and P10 (+3.0); P04, P06, P07 lose.
- Rule (>= +2 with people interval > 0 -> causal sketch; +0.5 to +2 undecided; < +0.5 recorded, not closed): primary +0.42 [-0.55, +1.38] is below +0.5, and the interval spans zero: recorded, not closed and no causal sketch. c x bE alone (+0.83, lower edge 0.00) would be undecided, but it is not the pre-registered primary.
- Reproducibility check (Galvani 3102582 copy, runs/personal_ceiling/) pending: scored when 16 files are there; per-arm difference to be appended.

**Wave plan, 14 Nov 19:45 UTC (research thread, from the independent solution-space map of 14 Nov evening and the coordinator).**
Pre-registered sealed estimate for v6.5 (if uploaded): 0.823-0.828. The record's dev-to-sealed transfer is 0.41-0.7x
(STATE), and §249's +0.94 sealed-weighted dev gain over v5.1b at that transfer gives +0.4 to +0.65; 0.829-0.831 would need
about 1.0x. Main front: base accuracy, because every label-based lever is gated by label purity (§252 dose map: purity
75 / 80 / 85 / 90 % gives -1.1 / -0.1 / +1.8 / +2.9 at n 129). Then returning people on all ten (43 % weight, n = 2 on dev).
Deferred to the next wave, with reasons: the pseudo-session selector (it needs stored chain outputs under several stage
settings; feasibility is checked first) and in-domain SSL pretraining (2080 queue contested; after the core test reads out).

### 262. RETURNING-PEOPLE PATH ON ALL TEN PEOPLE, WITHIN-SESSION PROXY (written 14 Nov 19:45 UTC, before any result; map lever 1)
Mechanism: dev measures the routed path for returning people on P04 and P10 only; the eight single-session returning
people (about 40 of 46 sealed returning sessions) are assumed P10-like. The within-session proxy (§249, PV6,
runs/pview/tproxy.npz: each public session's acquisition runs, one Graz and one BrainHero, play the public session; the
online runs are scored; 8 shuffled orders x 6 bases) exists for all ten people. Score the routed path with ORACLE
recognition (person known; the gate's recall is a separate question) on it: personal heads and the own-space stage fitted on
that person's acquisition runs exactly as the package fits them on a public session, ER and ER + T (§245), against the
population new-people tail on the same windows (base 78.3, T20 80.4 on PV6). Reuse the package's personal-stage code path,
not a re-implementation; zero-change: with the personal parts switched off the score equals the PV6 tail.
Prediction: routed minus tail per person, median over the eight non-P04/P10 people +2.5 (80 % range +0.5 to +5); P10 about
+6 (§249), P04 large. Rule (map's kill line): routed gain < +2 for 6 of the 8 -> the 0.8-P10-like assumption is wrong and the
returning weighting of every dev number is revised downward; >= +2 for 5 of 8 -> tune the returning path's weights on all ten
people (nested by person), not on P04 / P10. Either way, report the sealed-weighted dev level under both assumptions.

### 263. CORE FEATURE SCALING, CORE ALONE FIRST (written 14 Nov 19:45 UTC, before any result; map lever 2)
Mechanism: core1 (CAR, 17 overlapping bands x full and two half windows, tangent space, 44k-dim LR) loses 4 points at 512
PCs (§49), so the fine feature space carries signal, and the feature set was never scaled. Variants, core1's own LR recipe and
regularisation search unchanged, LOSO over the ten people (new-people protocol, recording order is fine for a core-alone
screen; it is a feature test): (A) 34 half-octave-spaced bands, same windows; (B) 17 bands x 5 overlapping time segments
(0-2, 1-3, 2-4, 0-4 s, 0.5-3.5 s); (C) A + B; (D) core1 + regional CAR blocks (frontal, central, parietal-occipital
re-referenced separately, extra covariance blocks). CPU, Galvani; feature caches keyed by every variant setting (fail loudly
on a key without them). Reference: core1 alone, new people, about 69.3.
Prediction: best variant 70.5 (80 % range 69.0 to 72.0); (C) best; (D) +0.3 at most. Rule (map): best >= 71.0 with >= 7/10
people up -> rebuild TSEM's view and the routing features on it (coreshuf tooling, §175) and score the chain under 8
shuffled orders as its own pre-registration; 70.0-71.0 -> undecided, one more form (more bands x segments) before setting
aside; < 70.0 -> recorded, this scaling family set aside (not the core redesign).

### 264. EXTERNAL PEOPLE: WITHIN-CORPUS VERSUS CROSS-CORPUS PROBE (written 14 Nov 19:45 UTC, before any result; map lever 3)
Mechanism: every external-people form so far pooled raw rows or froze a teacher; cross-corpus transfer to PROTEUS is about
45 % (§216ff). If external people decode well WITHIN their own corpus, the loss is a domain gap that a corpus-aware pool
(per-corpus recentring, corpus intercept, shared slope) can bridge. Probe: a core1-like member (tangent space at each
session's mean, LR), LOSO inside each corpus: Scherer2015 (9 people, the PROTEUS tasks, 3 classes), Shin2017B (29,
mental arithmetic vs rest, 2 classes), Shin2017A (29, MI left/right, 2 classes); data/ on the Mac and Galvani. Report BA per
person, mean, and the same member trained on all of PROTEUS tested on Scherer (cross-corpus, reference).
Prediction: Scherer within-corpus 60 (80 % range 50 to 70); Shin2017B 70 (60-80); Shin2017A 62 (55-72).
Rule: Scherer >= 60, or both Shin corpora >= 65 -> pre-register the hierarchical pool (core1 on k PROTEUS people plus the
corpus, per-corpus recentring, corpus intercept; pass +1.5 core-level at k 5, >= 6/10 up); else the domain-gap hypothesis
is not supported for these corpora (recorded).

### 265. NETS FINE-TUNED ON CHAIN LABELS (c x bE), THREE VARIANTS (written 14 Nov 19:45 UTC, before any result; §252 part b open form)
Mechanism: §252 part (b) c x bE (filt labels) +0.83 [-0.01, +1.68] 6/10 over the r15 chain: 4 families x 2 seeds fine-tuned
200 steps on the four training runs' chain labels, tested on the two held-out runs. It rises with base purity (dose map),
so it stays open. Variants (same splits, scorer score_pl.py, the 16 Leonardo files as the base):
(V1, CPU, existing files) fusion weight c x bE^w, w 0.5 / 1 / 1.5, and each family alone and leave-one-family-out (which
families carry it). (V2, GPU) delivery-prefix split, the deployable direction: per order, bE fine-tuned on the session's
windows delivered before t = 120 (any run; chain filt labels of those windows), tested on windows from t on, against the
r15 chain on the same windows; 6 orders (within0/1, full0/1, runs0/1), 4 families x 2 seeds. (V3, GPU) seeds 4 per family
(ensemble size), split as §252.
Predictions: V1 best w 0.5-1.0, within 0.2 of +0.83; no single family above +0.5 alone. V2 +0.4 (80 % range -0.3 to +1.0;
fewer, earlier windows than the run split). V3 +0.2 over the 2-seed form.
Rule: V2 >= +0.5 with the people interval above 0 -> sketch the causal package form (r15's refit extended to these families,
run time per window on an A100 against the 1.37x slack) as its own pre-registration; V2 0 to +0.5 -> undecided, one more form
(t = 80 and t = end-40); V2 < 0 -> the run-split gain does not carry to delivery order (recorded). P04 and P10 reported, neither decides.

**§262 AMENDMENT (14 Nov, before any result; implementation choices the design left open).** Code `studies/14nov/pview/proxy_routed.py`,
derived from chain.py's `system()` (ER inputs, rest member at weight 1, eye log columns on SessionEM's GE view, a = 1/16, own-space beta 0.10)
and tproxy.py (all_loso store, shuffled orders, six bases, online rows scored). Package functions reused, not re-implemented: `R3.ps1_heads`
(C 0.1), `R3.slot_mix` (g 3, a 1/16), `RR.causal_std`, `ownspace.Personal` (T, C, S members), `gm.pipe_g`, route4's PM/KP prior path (kappa 20).
- **Oracle recognition:** the routed person is the session's own person, ramp r = 1 on every online row (from the first routable window of the
  delivery order). Acquisition rows play the public session and are never routed (r = 0, they emit the tail's output); their labels reach only the
  personal heads, the prior and the own-space members, so no label reaches a scored row's pseudo-labels through unrouted rows.
- **Personal parts fitted on a session's acquisition rows** (one Graz + one BrainHero run), per person x session: head on the fold's core1 PC256
  (feats_exact/all_loso-<P>), prior = class means of PC128 standardised on those rows, own-space members on those rows.
- **ER inputs as found for all_loso:** eye net runs/ocushuf/<order>/oc_nomix-all_loso-s0..2.npz (log of the 3-seed mean, clip 1e-3), rest member
  runs/restshuf/<order>/rest4_sh_r1-all_loso-s0..2.npz; both in recording layout, permuted with the order. Members otherwise as recorded
  (tproxy's limit); the heads' PC256 are the recorded running-recentred ones.
- **Arms:** tail0 (no ER, no personal; asserted equal to tproxy.npz base per order and base), ERoff (ER inputs, r = 0), ER, ERT (kappa 20).
  Orders: within0-3, full0-1, runs0-1 (the 8 shuffled orders of tproxy.npz). Reported: routed minus tail0, and routed minus ERoff (what the
  personal path alone adds). Rule and prediction as above, applied to routed minus tail0 (the pre-registered comparison).

**§263 amendment 1 (14 Nov 19:47 UTC, before any §263 result; research thread).** Choices the pre-registration left open, fixed
now (code studies/14nov/s263/, Galvani work dir ni26_s263, jobs C 3103503/04, A 3103505/06, B 3103507/08, D 3103509/10,
Z prep 3103511; the Z fold array was refused by the per-user submit limit and is submitted when the queue drains):
primary head lr10 for every variant (core1's reference 69.29; lr1 and lr3 reported, not used for the rule). A's 34 bands =
the ov17 bands, the 16 geometric midpoints between consecutive bands, and (1, 2) Hz. B's segments = 0-2, 2-4, 0-4, 1-3,
0.5-3.5 s. D's regions = frontal (16 ch, AF/F/FC), central (16, C/CP/TP), parieto-occipital (9, P/PO), each with its own CAR
and 17 bands x 3 windows of extra blocks. Zero-change: Z (core1 settings through the new code) within 0.1 per person of
runs/diag/ts_core/res/car_ov17_fs2_t0.1_le_k2.json; no variant is read before Z passes.

**§264 RESULT (14 Nov 20:50 CET; code studies/14nov/extharness/probe264.py, cached core features of runs/extharness, Mac via run_capped,
no Galvani; evidence/extprobe_14nov/s264.txt, per person BA and windows).** Member: core feature set (17 bands + slow block),
every session centred at its own full mean, standardised, LR C 0.05, LOSO inside the corpus; BA = mean of per-session BA. 80 %
intervals are bootstrap over people (10-90 % of the mean). Chance: Scherer 33.3, Shin 50.
| corpus | predicted | measured | 80 % interval of mean | causal-centring variant |
| Scherer2015 (9 people, 3 classes) | 60 (50-70) | 46.6 | [43.3, 49.9]; people 35.4-58.8 | 46.2 |
| Shin2017B (29, arithmetic vs rest) | 70 (60-80) | 77.5 | [75.6, 79.4]; people 55.0-95.8 | 65.1 |
| Shin2017A (29, MI L/R) | 62 (55-72) | 63.8 | [61.0, 66.6]; people 45.0-87.5 | 61.7 |
| PROTEUS (10 people, all windows) -> Scherer | reference | 45.2 | [41.6, 48.8] | 43.7 |
Rule: Scherer >= 60 no (46.6); both Shin >= 65 no (B 77.5 yes, A 63.8 just below, interval [61.0, 66.6] includes 65). The rule does
not fire: the domain-gap hypothesis is not supported for these corpora (recorded). Scherer within-corpus (46.6) equals the
cross-corpus reference (45.2, difference inside both intervals), so on Scherer there is no gap for a corpus-aware pool to bridge:
the tasks are hard for this member whichever way it is trained (8 training people per fold). Shin2017B is far above chance
within its corpus, and Shin2017A is at the prediction; neither is the PROTEUS task set, so they do not speak to the PROTEUS gap.
Silent notes: (i) PROTEUS class ids 0-2 are assumed to match the Scherer export's c0-c2 (the export is documented as the PROTEUS tasks;
not re-verified here); (ii) Shin2017B's 77.5 depends on offline full-session centring, with the causal running mean it is 65.1
(a deployable-centring cost of 12 points there, 0.4 to 2 on the other corpora), so the primary number is the optimistic one;
(iii) the 1.7-point margin of Shin2017A under 65 is inside the dev noise, so "both Shin >= 65" is undecided rather than clearly
failed; no pre-registration written, no run of the hierarchical pool. Suggestion only if the coordinator reads A as >= 65: the pool
would pass on the Shin corpora alone, which are not the PROTEUS tasks, so it should still be scored on Scherer or PROTEUS people.
Note (i) checked, 14 Nov 19:55 UTC (research thread): scripts/export_external.py maps Scherer's description.pdf codes
{4 right-hand MI: 0, 2 mental subtraction: 1, 1 word association: 2} (navigation and feet dropped), which matches PROTEUS
(0 motor imagery, 1 mental subtraction, 2 word generation). The cross-corpus 45.2 is not a label-permutation artefact.
Context for both Scherer numbers: 22 of the 41 PROTEUS-montage channels in the export are spherical-spline interpolated.

**§261 RESULT, partial (14 Nov 21:30 CET; evidence/v65b_14nov/, jobs 3103438, 3103439; bundle 5323fd8b…, sha printed in both logs).**
Prediction (2) holds: 2080 Ti check err 0, relabel diff 0.0, label agreement 100 %, output diff 0.039 / 0.044. Prediction (3) holds:
A100 peak_reserved 4.34 / 8.81 GB (v6.5 in the same job 30.38 / 41.55), no allocator warning in the v6.5b arm (6 in the v6.5 arm),
run time v6.5b / v6.5 1.019 and 1.000. The unpaired ratios against v5.1b (1.97 / 2.28 / 1.84) come from a node whose v5.1b baseline
was 2x faster; the v6 arm's times did not move. Prediction (1), CPU bitwise identity (3103433), is still running; v6.5b is not the
candidate until it passes. Mac replay equal to v6.5 (0.9945788530).

**§261 RESULT, complete (14 Nov 22:05 CET).** Prediction (1) holds: CPU bitwise identity v6.5 vs v6.5b on P01 and P10|1, max_diff 0.0 (job 3103433, evidence/v65b_14nov/cid65b-3103433.out). No refutation clause fired: v6.5b replaces v6.5 as the candidate (ZIP v65b_r1iLu_cand a7b13e57…850d6, bundle 5323fd8b…).

### 262. V6.6 TRACK B: CHEAPER REFITS (written 14 Nov 22:20 CET, before any speed number; build thread, coordinator's order)
Motivation: the 60-min margin is node-sensitive (sealed 33-37 min at 57 % unrecognised, 42-49 at 100 % on the 3103439 node),
the refit path costs 14-20 sealed minutes (0.35 s + 2.1 ms x prefix per refit, 17-39 refits per unrecognised session), and
research's §265 (nets fine-tuned on chain labels) would add refits. New tree lean/src66 (copy of src65b; v6.5b's bundle and ZIP
untouched). Stage 0 = profile of one refit (torch profiler, 2080 Ti, prefix 190 and 378, v6.5b code): where the 0.35 s fixed and
2.1 ms/window variable parts go. Candidate forms, each scored whole in the session path: (a) per-session GPU buffer for the
prefix X (no np.stack / H2D per refit); (b) dtheta from one on-device reduction, one sync (today a float() per parameter);
(c) clean-path FIR cache: the FIR banks are frozen and per-window, so spectral_filtering(X_clean) for windows already seen is
reused by the clean forwards (5 per refit) and the final state pass; the augmented path cannot be cached; (d) one shared
augmented batch where members hold equal X (checked, else dropped); (e) fewer launches (no per-step zero(m) double calls
that are no-ops, fused state reset) only if profiled hot.
Target: refit path 14-20 -> 2-4 sealed minutes (estimated from 2080-measured per-refit time ratios applied to the A100 timing,
then checked on A100 with v6_time.py), run time at 378 windows v6.6/v6.5b <= 0.6.
Identity: rule stated before the run. (a), (b) change no arithmetic: CPU outputs bitwise equal to v6.5b (cpu_identity.py on
P01, P10|1, 17/18 refits). (c), (d) may change float summation through different batch shapes: allowed tolerance max output
difference <= 1e-5 on CPU with 100 % label agreement and equal refit/relabel counts, GPU <= 0.05 with labels 100 % (the
non-determinism band of v6.5). Any label change or any err refutes that form.
Prediction: profile shows the clean forwards + final pass + dtheta syncs + staging >= 50 % of refit time; (a)+(b)+(c) together
give >= 1.8x on 2080 at prefix 190 (80 % range 1.4-2.6x). Refutation: < 1.4x for the stack -> the remaining cost is the
augmented forward/backward (irreducible without changing arithmetic) and track B stops at the gain reached (reported, not hidden).

**§262 RESULT (14 Nov, 21:5x CET; evidence/returning10_14nov/s262.txt, runs/pview/routed262_*.npz, routed262.log; code studies/14nov/pview/proxy_routed.py + score262.py, run on the Mac via run_capped, 48 order x base cells, 49 s each).**
Zero-change asserted on every order and base: tail0 (no ER, no personal parts) equals tproxy.npz base (atol 1e-9). Routed minus tail0, mean over 8 orders x 6 bases, [80 % over orders]:
- ER: mean over ten +6.26; **median of the eight (non-P04/P10) +5.08** [+4.64, +5.65 over orders; +3.28, +7.69 resampling people]; >= +2 on **7/8**. P01 +11.2, P02 +0.5, P03 +6.6, P04 +9.3, P05 +7.7, P06 +2.2, P07 +3.5, P08 +3.3, P09 +8.8, P10 +9.4.
- ERT (kappa 20): median +5.74 [+4.95, +6.50], >= +2 on 6/8 (P06 +1.5, P02 0.0); T adds +0.5 to the median, as in PV6 it helps unevenly (P01, P03 up, P05, P06 down).
- Personal parts alone on the tail mix (P / PT, no ER inputs): median +4.94 / +5.49; ER inputs alone (ERoff) +0.78 median, so about +4.3 of the gain is heads + prior-free own-space stage.
- **Against the prediction:** median +2.5 (80 % +0.5 to +5) -> +5.1, above the range; P10 about +6 -> +9.4; >= +2 for 5 of 8 required -> 7/8. Rule: >= +2 for 5/8, so the "0.8-P10-like" assumption is NOT refuted and the downward revision is not triggered; the second branch fires: tune the returning path's weights on all ten people (nested by person), not on P04/P10.
- **Limits, stated plainly:** (1) the proxy is within-session: heads, own-space members and prior are fitted on the same session's acquisition runs (no between-session shift), so these are upper bounds for the sealed case; P10's own across-session routed gain over the tail is +6, the proxy gives +9.4 for P10, which calibrates the optimism at about 1.5x on P10. (2) The tail0 baseline has no personal parts, so the gain includes the whole personal path, not T alone. (3) Oracle recognition and r = 1 from the first online window; the gate's recall and ramp are a separate loss. (4) Members and PC256 as recorded (tproxy's limit). (5) Acquisition rows are about a third of a session, so the personal models see fewer rows than a full public session.
- **Sealed-weighted dev level (§249 calibration J = 0.566 new 77.8 + 0.434 returning, returning = 0.2 P04 70.0 + 0.8 other):** (a) other returning P10-like 89.6: J 81.22. (b) measured ten-person spread (proxy absolute routed level of the eight: ER 87.5, ERT 87.8): J 80.48 / 80.58. (c) §249 new-people level of the eight 81.9 + proxy gain (+5.5 / +5.8): J 80.46 / 80.56. So the P10-like assumption overstates dev J by about 0.7 if the eight sit where the proxy puts them, and the proxy itself is optimistic, so (b)/(c) are an upper bound for that alternative.

### 266. RETURNING PATH: RECOGNITION ON TEN PEOPLE AND NESTED TUNING OF THE SHIPPABLE WEIGHTS (written 14 Nov 20:32 UTC, before any result; §262's rule branch, coordinator)
Mechanism: §262 shows the routed personal path is worth a median +5 on the within-session proxy for the eight people dev
never measured, but with oracle routing and with weights chosen on P04 / P10. Two things decide what ships: whether the gate
recognises those people causally and label-free, and whether the weights are right for ten people rather than two.
Part (a), recognition (no tuning): run the package's label-free gate (routing v4.1 code path, the package's thresholds) on
the PV6 proxy for all ten people, gallery = that person's acquisition runs plus every other person's public data as the
package holds them, 8 orders. Report per person: recognised yes / no, time to recognition (delivery position of the first
routed window), share of online windows routed, false routes of other people's sessions to that person (on the new-people
LOSO sessions). Then the §262 path with the REAL gate (ramp as shipped) instead of the oracle: routed minus tail0.
Part (b), tuning: only parameters that ship as constants: the own-space stage weight (0.10 now; grid 0.05-0.30), the
personal-head slot weight (grid 2-6), the TSEM personal-prior kappa (10-40), the gate ramp speed and threshold (3 values each
around the shipped ones). Nested by person: for each held-out person pick on the other nine, score on the held-out one,
real gate. Primary: nested tuned minus shipped settings, ten people, 8 orders x 6 bases. Check on the real cross-session
protocol: the nested picks applied to P04 / P10 seen_session (16 orders) must not lose more than 0.3 there.
Transfer to the sealed estimate (stated before the result): sealed gain = proxy gain x 0.64 (P10: about +6 across sessions /
+9.4 on the proxy) x recall from part (a) x 0.434 (returning weight) x 0.41-0.7 (dev-to-sealed transfer).
Run time: weights and thresholds only, so no new per-window cost (reported as measured, expected within noise of v6.5).
Predictions: (a) recall for the eight >= 6 of 8 recognised, median time to recognition <= 40 windows; real-gate gain median
+3.5 (80 % range +1.5 to +5). (b) nested tuned minus shipped +0.4 on the ten (80 % range -0.2 to +1.0), P04 / P10 check
within -0.3. Rule: (b) >= +0.3 with the people interval above 0, >= 6/10 up and the P04 / P10 check passing -> the tuned
constants go to the build thread as a v6.5 variant, with the sealed estimate from the transfer line; else the shipped
weights stay (recorded). If (a) recognises < 5 of 8, the gate's recall becomes the lever and gets its own pre-registration.

**§266 amendment 1 (14 Nov 21:50 CET, before any part (b) result; part (a) seen on one smoke cell only, within0 x B2_pool; build thread).** Choices the pre-registration left open:
- *Gate under test.* The bio half of routing v4.1: `experiments/bio_gate.Bio` + `fold_params` unchanged (model bio_model_psd_d64, tau 0.26, ramp 0.05, t0 10, as route4's `bio_stream` builds them); only the window feature is served from a cache (`studies/14nov/pview/gate266.py`). **Not applied: the v3 half of v4.1's `max`** (its gallery contrasts need the labelled acquisition rows and the shadow tail); the gate's r here is therefore a lower bound on the merged r. Stated as a limit, not closed.
- *Gallery and stream.* Own session: the nine other people's rows of that person's LOSO fold + the session's acquisition rows as a one-session entry (m = 1); the gate sees the online rows in delivery order (acquisition rows play the offline public session), t0 counts online windows. A top person other than the session's own gets r = 0 (counted, reported). Because the gallery entry comes from the same session this is optimistic on recognition; added, for P04 / P10 only, a cross-session stream (gallery = nine others + all rows of the person's other sessions, as the package holds them). False routes: each session of another person p (all rows, delivered in order) against fold p's gallery with q's full rows replaced by q's acquisition rows of one session (all session pairs, 8 orders).
- *Zero-change.* The stub "correct person, r = 1 from the first online window" through the new code path must equal §262's ER and ERT arms (atol 1e-9, asserted per order x base); a second stub, r = 1 from the gate's first routable window (online window 10), separates the t0 loss from recognition and ramp loss. tail0 is asserted equal to tproxy.npz base as in §262.
- *Primary path.* ERT (kappa 20; the shipped v6.5 has the TSEM prior). ER is reported next to it.
- *Grid (part b), full factorial.* own-space beta {0.05, 0.10*, 0.20, 0.30}; head slot gain G3 {2, 3*, 4, 6} (the head's share at r = 1 is G3 a / (1 - a + G3 a), a = 1/16); TSEM kappa {10, 20*, 40}; gate threshold tau {0.22, 0.26*, 0.30}; ramp {0.03, 0.05*, 0.08} (* shipped). 432 configs on 8 orders x 6 bases, real gate r (raw scores, `r = clip((s - tau) / ramp, 0, 1)`).
- *Nested pick.* For held-out person q: per config the six-base, eight-order mean online BA averaged over the other nine people; the pick is the argmax, **but the shipped config unless the argmax beats it by >= 0.05 on that nine-person mean** (noise floor, stops tie-driven picks); q's score is that config's. Primary = nested tuned minus shipped, mean over the ten, with 80 % intervals over orders (picks fixed) and resampling people (picks fixed), count of people up.
- *Identifiability, stated before the result.* Own-session top scores sit far above tau + ramp = 0.31 (smoke: min 0.31, median 0.57), so on this proxy tau and ramp move r only at the corners of the grid, and the proxy contains no false-route cost. A non-shipped tau / ramp pick is therefore not evidence for the gate constants; if the nested picks move them, I additionally report the primary with tau / ramp held at shipped, and the false-route windows of gate266 at the picked tau / ramp against the shipped ones.
- *P04 / P10 check.* seen_session, 16 orders x 6 bases, `chain.py`'s machinery (new generic arm X<G3>_<beta>_<kappa> next to the T arm, which is shipped v6.5); P04 and P10 each get their own nested pick (made on the other nine of the ten proxy people); pass = mean of (pick minus T) over P04 and P10 >= -0.3, each person also reported. If a pick moves tau / ramp, the gate is run with that tau / ramp (env hook in route4, default unchanged).
- *Transfer line (unchanged):* sealed gain = proxy gain x 0.64 x recall (a) x 0.434 x 0.41-0.7; recall (a) = the share of the eight recognised in all streams.

**§262 STAGE-0 RESULT (14 Nov 22:50 CET; evidence/v66_14nov/rprof-3103933.out; studies/14nov/refit_profile.py, scripts/cloud/rprof.sbatch; RTX 2080 Ti, v6.5b code, prefix = dev stream order concatenated, labels/weights are model outputs on it).**
One refit (4 members x 5 steps): t=190 1285 ms, t=378 2535 ms (spread < 0.2 %); per window 6.6 ms, fixed part ~0 here (the ledger's 0.35 s
did not show up; it is a first-refit/process warm-up). Shares at t=190: clean forward 25 %, augmented forward 25 %, backward + Adam 40 %,
final state pass 5 %, staging 2 %, dtheta 0.1 %. FIR (spectral_filtering) is ~50 % of a forward (9.2 of 17.7 ms per member at 190).
Members' X differ and strong() draws differ across members: form (d) dropped on its stated condition.
**Prediction refuted:** clean forwards + final pass + staging + dtheta are 32 % of refit time, not >= 50 %. Ceiling of the identity-preserving
forms: caching the clean-path FIR saves ~half of (25 + 5) % = ~15 %, staging 2 %, dtheta 0.1 %: about 1.2x (below the 1.4x
bar of the refutation rule) -> (a)+(b)+(c) alone cannot reach 2-4 sealed minutes. 65 % of the time is the augmented forward + backward
(conv backward, depthwise conv, 1570 small elementwise calls: launch-heavy), which only arithmetic-changing forms touch.

### 267. c x bE: SINC-WEIGHTED FAMILY SETS AT A FIXED WEIGHT (written 14 Nov 21:12 UTC, before any result; §265 V1 follow-up)
Mechanism: §265 V1 (16 Leonardo files, §252 split, filt labels, six orders) found c x bE^0.5 +1.20 [+0.52, +1.90] 8/10, with
leave-one-family-out: without sinc -0.37, without fbl +0.25, without fb +0.86, without fbraw +1.28. The weight 0.5 and the
family pattern were read after the fact, so they need fresh nets. New nets, seeds 4-7 (never used), same recipe as ftnet_pl.py
(200 steps, chain filt labels, §252 split), fusion weight FIXED at 0.5:
(S1) sinc only, 4 seeds; (S2) sinc + fb + fbl (no fbraw), 2 seeds each; (S3) all four families, 2 seeds each (the §252
set on fresh seeds: the control for seed luck). Score = c x bE^0.5 over the r15 chain, six orders, people-and-order intervals.
Predictions: S3 +0.9 (80 % range +0.2 to +1.6; regression toward §252's w 1 value); S2 - S3 +0.2 (-0.3 to +0.6); S1 +0.3
(-0.6 to +1.0; one family alone was negative at w 1 in V1).
Rule: S2 or S3 >= +0.8 with the people interval above 0 and >= 7/10 up -> the family set and w 0.5 are fixed for the §265 V2
causal form (delivery prefix), which is then rerun with that set; S2 - S3 >= +0.3 with interval above 0 -> drop fbraw from
the set; else the four-family set stays. P04 and P10 reported, neither decides. GPU, Leonardo boost_usr_prod.

**§265 V1 + V2 RESULT (14 Nov 21:48 UTC; research thread; score265.py v1 / v2, PL_FTPL_DIR=runs/personal_ceiling_leo,
S265_V2_DIR=runs/personal_ceiling_v2_leo, Leonardo jobs 64493438/39, smoke 64493417 passed the t < 120 assert; V3 pending).**
- **V1** (existing 16 files, §252 split, six orders, chain 79.95): c x bE^0.5 +1.20 [+0.52, +1.90] 8/10 (P04 -0.40, P10 +3.57);
  w 1 +0.83 [0.00, +1.67] 6/10; w 1.5 +0.42. Single family alone -1.1 to -2.9 at w 1; leave one out: without sinc -0.37, fbl
  +0.25, fb +0.86, fbraw +1.28. Predictions: best w 0.5-1.0 within 0.2 of +0.83 (missed: +1.20 at 0.5, read after the fact,
  hence §267); no single family > +0.5 (hit).
- **V2, delivery-prefix split (the deployable direction):** nets fine-tuned on the session's windows delivered before t 120
  with the chain's filt labels, tested on windows from 120 on (chain 81.35 there), 6 orders x 8 nets: bE alone 76.03 (-5.33);
  **c x bE -1.23 [-2.41, -0.14] 4/10** (P04 -0.77, P10 +0.42); per scheme within -2.65, full -1.30, runs +0.27.
  Prediction +0.4 (80 % range -0.3 to +1.0): refuted, below the range.
- **Verdict by the rule (V2 < 0): the run-split gain does not carry to delivery order** (recorded). The scheme pattern says
  why: only under the runs scheme, where the prefix holds whole runs as in §252's split, is it near 0; under within (all
  runs interleaved) it costs 2.65. The §252 split's +0.83 / +1.20 lean on training on whole other runs of the session, which
  sealed delivery does not hand over in that form. r15 (the chain's own in-session fine-tuning) already takes the
  deployable share. §267 (fresh seeds, fixed w) still reads the run-split ceiling; its rule's "feed the V2 rerun" branch is
  void unless a V2 form at w 0.5 or with a later t is positive (not pre-registered here; scored only as a labelled
  exploratory column if run).

**§266 RESULT (14 Nov 23:2x CET; evidence/returning10_14nov/s266.txt; code studies/14nov/pview/{gate266,proxy266,score266a,score266b}.py, chain.py arm X; Mac via run_capped, grid 4 jobs x about 20 min, check 16 min; commit see git log).**
Zero-change: stub gate (correct person, r = 1 from window 0) through the new path equals s262 ER / ERT at atol 1e-9 on all 48 cells; tail0 == tproxy base; grid's shipped config == ERT_gate exactly; chain arm X3_0.1_20 == T.
- **(a) recognition: prediction holds.** Own sessions (same-session gallery, so optimistic): 10/10 recognised in every order, the eight 8/8, first routed online window 10 (= t0, the floor) for all, routed share 0.92, no wrong-person window; own top score median 0.57 (r = 1 needs 0.31). Cross-session P04 / P10: routed from window 10 in 24/24 streams each, P10 71 windows with a wrong top person. False routes (q's acquisition gallery, all session pairs): 10/1360 streams, 737/257,832 windows (0.29 %), all P10-session to P03 (8 orders, mean ramp 0.27) and P03 to P10 (14 windows); at tau 0.30 they are 0, at 0.22 2019. Their cost to the new-people tail is not scored here.
- **(a) real-gate path minus tail0** (ERT / ER, mean orders x 6 bases): median of the eight **+4.80** [+2.76, +8.32 people; +4.03, +5.65 orders] / +4.82 (prediction +3.5, range +1.5 to +5: inside the range, top end); >= +2 on 6/8 / 7/8 (ERT: P02 +0.3, P06 +1.4). Gate minus oracle: -0.32 ERT / -0.18 ER, all of it the t0 = 10 wait; recognition and ramp cost 0.00. Mean over ten ERT +6.09.
- **(b) refuted.** Grid 432 configs x 8 orders x 6 bases. Nested tuned minus shipped: **-0.17** mean over ten (80 % over orders [-0.32, -0.03], over people [-0.61, +0.26]), 5/10 up; tau / ramp held at shipped -0.07; beta and G3 only -0.00. Prediction +0.4 (range -0.2 to +1.0): below. In-sample best (g3 k40 tau .30 ramp .08 beta .20) +0.33 over shipped, which nested validation does not reproduce. One-at-a-time: beta 0.20 +0.32 mean but -1.6 for some people; tau and ramp +0.00 (identifiability as stated: not tunable on this proxy). seen_session check, gate constants shipped (deviation from the amendment: picks moved tau / ramp, which would need a route4 hook; moot since the rule already fails): P04 +0.23, P10 -1.04, mean -0.40 < -0.3 -> fails.
- **Rule:** (b) needs >= +0.3 with the people interval above 0, >= 6/10 up and the check passing: none holds -> **shipped weights stay**. (a) recognises 8/8 (>= 5 required), so the gate's recall does not become the lever.
- **Transfer line** for the shipped routed path (not for tuning): +4.80 x 0.64 x recall 1.0 x 0.434 x 0.41-0.7 = +0.55 to +0.93 sealed J.
- **Limits, plainly:** same-session gallery (recognition optimistic; cross-session only P04 / P10); v3 half of v4.1 not applied; wrong-person routes zeroed (none in own sessions); false-route harm unscored; the gate's t0 = 10 counts online windows only; proxy optimism (s262 limit 1) carries over; 0.64 transfer for the eight assumed.

**§265 V3 RESULT (15 Nov 00:0x CET; score265.py v3, V3 dir runs/personal_ceiling_v3_leo = Leonardo jobs 64493421/22, seeds 2-3 with the 16 base files = 4 seeds per family; Galvani copies 3103487 / 3103514 not used, never mixed).** Evidence evidence/personal_ceiling_14nov/s265.txt. Six orders, chain 79.95, filt form:
c x bE^1 +0.95 [+0.12, +1.82] 7/10 (P04 -1.58, P10 +4.09) against +0.83 with 2 seeds, i.e. +0.12; prediction +0.2 over the 2-seed form: hit within its noise (undecided, not a seed-count effect worth a rule). c x bE^0.5 +1.29 [+0.62, +1.98] 7/10 (P04 -0.26, P10 +3.54); w 1.5 +0.54. Leave one family out (w 1): without fbraw +1.44 8/10, without fb +1.16 8/10, without fbl +0.43, without sinc -0.28: same pattern as V1. Single families -1.0 to -2.8. Run-split ceiling only; V2 above is the deployable reading.

**§267 RESULT (15 Nov 00:0x CET; fresh seeds 4-7, Leonardo 64493618/19, runs/personal_ceiling_s267, w fixed 0.5, six orders; evidence/personal_ceiling_14nov/s267.txt).** The smoke 64493611 was still pending when the arrays went out; the s267 path had no separate smoke, its first tasks ran without error and wrote all 20 files; seeds 4-7 were asserted not to collide.
| set | gain over the chain [80 % people x orders] | people up | P04 | P10 |
|---|---|---|---|---|
| S1 sinc x 4 seeds | -0.13 [-0.95, +0.65] | 5/10 | -2.03 | +0.98 |
| S2 sinc+fb+fbl x 2 | +1.53 [+0.88, +2.18] | 8/10 | -0.49 | +3.40 |
| S3 four families x 2 | +1.19 [+0.60, +1.81] | 8/10 | -0.32 | +3.16 |
| S2 - S3 | +0.34 [+0.19, +0.50] | 9/10 | -0.17 | +0.24 |
Predictions: S3 +0.9 (+0.2 to +1.6): hit (+1.19, above the centre). S2 - S3 +0.2 (-0.3 to +0.6): hit (+0.34). S1 +0.3 (-0.6 to +1.0): hit (-0.13; sinc alone does not carry it, the family mix does).
**Rule fires on both branches:** S2 >= +0.8, interval above 0, 8/10 -> the family set and w 0.5 are fixed for the V2 rerun; S2 - S3 >= +0.3 with the interval above 0 -> fbraw dropped. Run-split ceiling on fresh seeds: +1.2 to +1.5 at w 0.5, P04 negative in all sets (-0.3 to -2.0), P10 carries a large share (+3.2 to +3.4); neither decides.

**§267 branch: V2 with the S2 set at w 0.5 (15 Nov; score265.py v2s2, the existing V2 Leonardo files seeds 0/1, no new training; evidence/personal_ceiling_14nov/s267_v2s2.txt).** Note: §265's V1+V2 RESULT called this branch void before §267 read out; §267's rule pre-registered it, so it runs. Same windows (position >= 120), same r15 chain (81.35), families sinc + fb + fbl x 2 seeds = 6 nets; the refactored v2 reproduces the 4-family c x bE -1.23 exactly.
c x bE^0.5 **+0.03 [-0.65, +0.67] 7/10** (P04 -0.11, P10 +0.70); per scheme within -0.94, full -0.11, runs +1.14. c x bE^1 (reference) -0.67 [-1.60, +0.24] 5/10 (P04 -1.59, P10 +0.51). bE alone 76.87 (-4.49).
**Verdict by §265's V2 rule (>= +0.5 with interval above 0 -> causal sketch; 0 to +0.5 undecided; < 0 recorded): +0.03 is in 0 to +0.5, undecided (interval spans 0, inside the dev noise).** The S2 set at w 0.5 removes about 1.3 of V2's loss (-1.23 -> +0.03) but does not give a gain; no causal sketch is triggered. Gain only under the runs scheme (+1.14), as before. Not run: t = 80 / t = end-40 forms (§265's one-more-form clause for the undecided band), which stay open.

**§260 RESULT (15 Nov 07:14 UTC; research thread; evidence/causalpers_14nov/s260.txt; Galvani cpu 3103445 / 3103446, 240 pp files;
scorer zero-change OK on 120 order x base per draw: cap 0 equals replay r15 + filter).** Fixed form cq50 / a2 / t0 90 / R 120 /
cap 0.10, track F, change vs r15 + filter of the same draw, new people:
| draw | k 20 (primary) | k 10 |
|---|---|---|
| d1 | +0.26 [+0.11, +0.41] 7/10 (w +0.20 f +0.26 r +0.31) | +0.29 [+0.14, +0.44] 8/10 |
| d2 | +0.19 [+0.03, +0.36] 8/10 (w +0.18 f +0.08 r +0.32) | +0.23 [+0.07, +0.38] 8/10 |
| pooled | **+0.22 [+0.07, +0.38] 8/10** (w +0.19 f +0.17 r +0.31) | +0.26 [+0.11, +0.41] 8/10 |
Per person (pooled k 20): -0.2 +0.2 +0.0 +0.3 +0.3 +0.1 -0.1 +0.4 +1.1 +0.2 (P01..P10); draw difference +0.06.
- Predictions: +0.12 per draw, range -0.02 to +0.25 (d1 +0.26 just above, d2 inside); draws within 0.1 (hit, 0.06); >= 6/10 (hit).
- **Verdict by the rule (both draws > 0, pooled interval above 0): the form joins the candidate list** with its run time (a2
  at k 20: 11.3 ms per window on the Mac CPU, timing.txt), and needs a package-form check (build thread). New people only;
  returning people absent. It replicates §253's draw-0 nested +0.15 on chains it was not chosen on, and sits on top of the
  candidate's new-people half (+0.23 over three draws), so the stack is about +0.45 for new people on dev.
- Fixes found while scoring (silent-no-op hunt): score.py had a stray leading space at line 26 since db13394 (import error,
  so the §260 waiter never scored); pp files carry no "draw" key despite db13394's description; run.py names them by the
  draw suffix and reads chain<draw> under an assert (code and chain md5 identical on Galvani and the Mac), so score260.py
  now checks the name and asserts the key only where present.

### 268. V6.6 = V6.5b + CAPPED CAUSAL PERSONALISATION (§260 form) (written 15 Nov 08:20 CET, before any build check; build thread, coordinator's order)
Form (study studies/14nov/causalpers, fixed in §260): chain labels = argmax of the package's own emitted posterior p of windows < t
(before the count filter); refit points t = 20, 40, ... (k 20); training set = per chain class the top 50 % most confident of
windows 0..t-1 (at least 3); 8-band FBCSP (4-8 ... 32-40 Hz, nf 2, one-vs-rest, classic.csp_feats) on per-window zero-phase band
covariances (OAS), sklearn LDA (lsqr, auto shrinkage), unweighted; window i >= 20 takes the latest refit's posterior q; output =
argmax[log clip(o, 1e-6) + w(i) log clip(q, 1e-6)] with o the count-filter output (p before the latch), w(i) = 0.10 clip((i - 89)/120, 0, 1)
(t0 90, R 120, cap 0.10). A refit with < 2 chain classes gives q = uniform. New people only: the stage stops for the rest of the
session at the first routed (recognised) window and is never applied after it. Package form: module cpers.py in lean/src66,
key v6_cp_on (0: v6.5b byte-for-byte path, no CausalPers object); all state per session, reset_state; any exception: counted,
one stderr line, count-filter output for the rest of the session (as the count filter).
Predictions: (1) cp_on 0: v6.6 bundle outputs bitwise equal to v6.5b on CPU (record/cpu_identity, P01, P10|1, plus a
recognised session); recognised sessions (stage stops at recognition before window 90) equal to v6.5b with cp_on 1 too. (2) Package
form against the study form: on a few cells (order x base x person, unrecognised) the package's q (hard q per window) equals the
study's pp file within 1e-6 and the fused labels agree on 100 % of windows; the person score of the package form equals the
study's within 0.05. (3) Cost on the worker CPU (Galvani CPU / Leonardo): <= 25 ms per window of an unrecognised session
(Mac 11.3 ms x the 1.9 worker/Mac ratio of the count filter = 21), so <= 4.5 sealed min expected (57 % unrecognised), <= 8 worst.
(4) A100 timing: sealed worst case 42-49 min (slow node) + cost; the stage needs total <= 50 min (>= 1.2x slack) in the worst case,
else it is reported as not fitting and a cheaper form (k 40, fewer bands) is pre-registered separately.
Refutation: any bitwise difference with cp_on 0, any err on the unforced run, package-vs-study label disagreement > 0.5 % of windows
(a porting bug until explained), or the timing rule failing. v6.5b stays the upload candidate until v6.6 passes all.
