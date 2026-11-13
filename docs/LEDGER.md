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

### 101. FIX2: refinements of the augmented-input fine-tuning (written 12 Nov 19:09 UTC, before any job ran; code `studies/12nov/ttt5.py`, Leonardo)

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

### 105. Sealed expectation for v5.2 over v5.1b (written 12 Nov 21:43 UTC, before any upload of v5.2 is discussed)

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


### 106. V53: in-session fine-tuning for unrouted sessions, runtime-feasible forms (written 12 Nov by 21:49 UTC [stamp corrected from the commit time; it read 22:20 UTC], before any result; code `studies/13nov/ttt_reduce_score.py`, v52-build session)

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


### 108. TTT6 (exploit track): arms that may raise the gain of the §101 fine-tuning (written 12 Nov by 21:59 UTC [corrected from the commit time; it read 23:20 UTC], before any job ran; code `studies/13nov/ttt6.py`, Leonardo, 24 tasks)

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

### 119. RETSTACK (explore → design input): the returning-people stack at stage strength 0.15 (written 12 Nov 23:26 UTC [date corrected 13 Nov: it read 13 Nov], before any result)

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

### 130. STACK (explore track, gap 1 of `docs/SOLUTION_SPACE_MAP.md`): the replicated small positives for new people as one system (written 13 Nov 00:05 UTC, before any result; code `studies/13nov/stack.py`, Mac, stored outputs only)

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

**§138 addendum (13 Nov 00:35 UTC, before any six-base result):** the same replay on fresh seeds (FT from ttt8 seeds
6–11, eye-net seeds 3–8) is queued on Leonardo (job 64325439, after §135 and §134; `runs/night13/pkg_fresh.txt`). The
§138 bar is read on both seed sets.
