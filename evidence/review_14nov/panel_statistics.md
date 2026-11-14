# Panel answer: statistics, validation and competition strategy (14 Nov 2026)

Read-only checks behind the numbers below ran on the public windows (`data/proteus_windows.npz`) and the stored dev
posteriors (`runs/replay_data/inputs.npz`, the v3 chain, new people, six bases). Limits of those checks: the chain's
posteriors were computed in recording order and permuted afterwards; run membership was given (oracle) where marked
with *; the six bases share the same ten people. The scripts (`quota.py`, `quota2.py`, `runid.py`, `runid2.py`,
`bagchain.py`, `bagchain2.py`) are in the session scratchpad under `stat/`; no tracked file was changed.

## 0. Facts I checked that the brief does not state

- **Every run is exactly balanced.** Acquisition-Graz 12/12/12 in 14 of 14 runs; Online-Graz 10/10/10 in 28 of 28;
  Online-BrainHero 10/10/10 in 26 of 28 (the other two 12/12/12); Acquisition-BrainHero 33 to 36 windows (a 12/12/12
  design with 0 to 3 windows lost). A session is six runs, three per interface, runs 1 and 2 are the acquisition runs.
- **Every Graz cell is exactly 32/32/32** (14 of 14). BrainHero cells have 90 to 103 windows, 29 to 35 per class.
- So within a cell **balanced accuracy is plain accuracy** (exactly for Graz, to within a window for BrainHero), and
  the two cells of a session are almost the same size. The ranking metric is, to a very good approximation, the mean
  over sessions of plain accuracy. It rewards nothing that accuracy does not. The optimal decision is the maximum of
  the true posterior, where "true" includes the design: labels are drawn without replacement from a known urn per run.
- Self-transitions in the cue sequence are 0.27 to 0.29 against 0.33: that is the urn, not a cue-order trick.
- **Small cells and the mean.** A cell of 96 windows at 80 % has a sampling standard error of 4 points; 200 cells
  bring that to 0.3 on the mean if they were independent. They are not: cells cluster in about 20 people with a
  person standard deviation near 10 points, so the sealed level has an error of about 2 points as an estimate of
  "the method", while differences between two systems on the same sealed set are much tighter (section 3). Cells
  count equally, so a window in a 380-window session counts half as much as one in a 190-window session.

## 1. Diagnosis

1. **The error is between people, not between windows.** One person at 54 % and nine at 75 to 90 % means a person
   standard deviation of about 10 points. The sealed set has about ten new people; one or two of them will be like
   P04 and will carry a third of the new-person error. Nothing in the chain is built for such a person: the stages
   sharpen whatever the population model believes, they cannot replace a wrong class geometry.
2. **Decisions cannot be revised.** The offline equal-count assignment per run (run membership given) scores +1.1
   over the causal chain on the same posteriors. The first 30 to 60 windows are decided before the session's
   templates exist. Under a full shuffle that share of weak decisions is the same but no longer confined to the
   first run.
3. **The system does not use the experimental design.** It uses "classes are about balanced in a session" through a
   30-window share. It does not use "exactly 10 or 12 per class per run", the block structure of the stream, or the
   fact that the delivery scheme can be read from the stream itself (section 2, proposal 1).
4. **The ensemble has no variance left to remove.** About 70 members trained on the same nine or ten people share
   their person-level bias. Averaging removed seed and architecture noise long ago; what is left is common to all
   members, and the stages then correct part of it per session. That is why a better member moves the end by its
   weight share and no more. I also checked the obvious next step, averaging whole chains instead of members, and it
   gives nothing (end of section 2). Headroom is in information the chain does not have yet (the design, the
   delivery scheme, other training people), not in more or better copies of what it has.

## 2. Five proposals, ranked

### 1. Run-level count filter as the last stage, switched on by a delivery-scheme detector

**Mechanism.** Treat the labels of a run as an urn with 10 (or 12) balls per class. Run an exact forward filter over
the class counts (state = counts of two classes, the third follows; 13 x 13 states per run, 65 x 65 per session),
with the chain's final posterior raised to a fitted temperature as the likelihood. The decision is the filter's
posterior maximum. It needs to know which run a window belongs to. Under the "runs shuffled" and "within-run
shuffle" hypotheses runs are contiguous blocks of known sizes (36, 33 to 36, 30, 30, 30, 30), so membership follows
from position plus a change-point step. Under a full shuffle it is switched off (or falls back to the session urn).
Which case holds is detected inside the session from a contiguity statistic.

**Why it should work.** It is a view whose errors are independent of the EEG by construction, the kind the brief says
the stages exploit. Measured on the stored final posteriors (new people, six bases, base 77.29):

| Constraint (causal filter) | recording | full shuffle | within-run shuffle |
|---|---|---|---|
| session urn (64 per class) | +0.06 | +0.29 | +0.11 |
| cell urn* (32 per class, interface given) | +0.22 | +0.29 | +0.21 |
| **run urn*** (10 or 12 per class, run given) | **+0.30** (5/10 people up, person SE 0.33) | **+0.68** (9/10, SE 0.15) | **+0.68** (8/10, SE 0.19) |
| offline equal-count assignment per run* (bound, not causal) | +1.10 | +1.10 | +1.10 |

At the mix level the team's recent-share division followed by the run urn* gives +0.59 / +0.98 / +0.81 over the
division alone (10 of 10 people up under the full shuffle).

**Detector (measured).** Ratio of the mean feature distance between consecutive delivered windows to that between
random pairs, in a 12-dimensional projection learned on other people's runs: full shuffle 1.000 (sd 0.016, range
0.962 to 1.038); within-run shuffle 0.869 (sd 0.051, range 0.764 to 0.971); recording order and shuffled runs 0.773
(sd 0.054). "Full or not" separates almost perfectly on a whole session (threshold near 0.95); "shuffled inside the
run or in order" is only two standard deviations apart. Measured on whole sessions with crude band-power features;
how early in a session it is reliable is not measured.

**Expected gain.** New people +0.3 to +0.7 if the stream is block-contiguous, 0 to +0.3 if it is fully shuffled;
returning people the same (it acts on labels, not on the person). Confidence: medium for the contiguous case (the
measurement used oracle run membership and recording-order posteriors), low for the full shuffle.

**Cheapest deciding experiment (CPU, 2 to 3 hours).** On v5.1b's final posteriors with members re-streamed under the
three schemes: (a) boundaries from position plus change points instead of the oracle; (b) the detector deciding per
session; (c) the same on returning people. **Kill:** below +0.3 on both contiguous schemes with inferred boundaries,
or below −0.1 under the full shuffle with the detector in the loop.

**Notes.** Causal; microseconds per window; soft floor on the urn (I used 0.02) so that lost windows and the two
12/12/12 online runs do not zero a class; fit the temperature on other people (it came out near 2 on the final
posteriors). Sessions of about 380 windows are presumably twelve runs: the block logic covers them, the session urn
does not unless the length is known. Under a full shuffle the run can only come from the signal: a window's run is
readable with labels (70 % six-way within a session, chance 17 %) but unsupervised clustering reaches a purity of
only 0.36 with raw band powers and 0.52 with the learned projection. That is not enough yet; a better learned
same-run metric is the explore version of this proposal.

**The cell itself is not worth inferring.** The interface of a single window is read at 59 % across people (chance
50 %; online against acquisition runs 52 %), and even with the interface given the cell urn adds only +0.2 to +0.3.
The cell matters for the metric only through the run: three runs make a cell, and the run is where the design binds.

### 2. One set of stage constants per delivery scheme, chosen in the session by the same detector

**Mechanism.** The stages lose 1 to 4 points under shuffled delivery, and the loss differs by scheme (the ledger has
−3.8 full, −2.1 runs, −1.1 within for the shipped stages). A single re-tuned set is a compromise between schemes
that want opposite things: a full shuffle wants long memory and strong whole-session templates, block schemes want
templates that follow the run. The detector of proposal 1 separates a full shuffle from the block schemes cleanly
(and the two block kinds less sharply), so the package can carry one constant set for "full" and one for "blocks"
(each tuned under that scheme, nested over people) and switch. Until the detector is sure it runs the set with the
best worst case.

**Why.** It turns the unknown scheme from a risk to be averaged over into something observed. It is also the hedge
the brief asks for: no bet on a hypothesis, and no loss in recording order if the organisers' scheme is mild.

**Expected gain.** The regret of the compromise set against the scheme's own set: new +0.2 to +0.8, returning
similar; confidence medium that it is positive, low on the size. **Experiment (CPU, the existing re-tune grid):**
best set per scheme against the best single set, nested over the two people halves, with the detector's actual
decisions and its early-session errors in the loop. **Kill:** regret of the single set below 0.3 under every scheme.
**Notes.** Causal; the detector costs one projection and one distance per window; fix the threshold on public
sessions under simulated schemes and report its error rate at windows 60, 100 and 150.

### 3. A person-specific trust weight for the personal route, fitted before the session

**Mechanism.** The own-space stage enters at one strength (0.10) for everybody, and its measured benefit is +3.5 for
one person and +0.1 for the other. Estimate for each of the ten public people, on that person's public data by
leave-one-run-out, how much the personal models beat the population chain on held-out runs, and turn that into a
person-specific strength by empirical-Bayes shrinkage towards the common value. Nothing is fitted during the session.

**Why.** 43 % of sealed cells are returning people and the only sealed evidence the team has is on this route. A
single strength chosen on two people, one of them an outlier, is the weakest estimate in the system.

**Expected gain.** Returning +0.3 to +1.0 (sealed +0.1 to +0.4); new 0. Confidence low to medium. **Experiment:**
CPU: leave-one-run-out advantage per person, then check on P04 and P10 sessions 2 and 3 and on the ten-person
within-session proxy whether the advantage predicts the best strength. **Kill:** rank correlation below 0.3 between
the leave-run-out advantage and the best strength.

### 4. Replace the dev measure by a predictor of the sealed score, and decide with shrinkage

**Mechanism.** H = 0.57 x new + 0.43 x returning. "new" is leave-one-person-out under the three delivery schemes
(members re-streamed, several permutations). "returning" is not the mean of P04 and P10: use a ten-person estimate
with P04 down-weighted to its sealed share (3 of about 46 returning sessions). Report every change as a paired
per-person delta with a person jackknife, and multiply by the measured transfer before deciding.

**Why.** See section 3. It gains no dev points; it stops the team from shipping noise and from holding back real
gains. I put its value at +0.1 to +0.3 sealed in expectation over two uploads. **Experiment:** recompute the last
two uploads' predictions with H: if H predicts +0.27 and +0.36 within 0.15, adopt it. **Kill:** it predicts them no
better than J x 0.45.

### 5. Members bagged over training people, stacked on out-of-person predictions

**Mechanism.** Train each net on a random seven or eight of the ten people instead of on all with a different seed.
Diversity then sits where the error is (people), and every member has honest out-of-person outputs for all ten
people, so mixing weights and one temperature can be fitted without leakage. Match the stacked mix's sharpness to
the current mix before the stages (the earlier fitted pool lost in the chain for exactly that reason, I suspect).

**Expected gain.** New −0.3 to +0.8: each member loses about 2 points from fewer people (4.7 per doubling), the
ensemble has to win that back through diversity. Confidence low. **Experiment:** leave-two-people-out folds for two
net families, about 4 GPU hours, scored in the chain. **Kill:** the bagged mix is not above the seed mix at equal
member count before the stages.

### Checked here and dead: averaging whole chains ("bag the pipeline")

My first instinct for a system whose stages carry the score was to average after the stages instead of before. On
the replay (v3 chain, new people, recording order) it does nothing:

- Three single-seed chains averaged after the stages against one chain on the pooled seeds: −0.10 (person SE 0.11);
  two half-pools against the pool: −0.14 (SE 0.07). Single-seed chains score 80.21, the pooled chain 80.31.
- Six chains with jittered family weights, averaged: −0.04 (jitter sd 0.15), −0.31 (sd 0.3), −0.58 (sd 0.7, eight
  chains).

Two side findings. (1) A single seed set of the SessionBN slots is as good as three in the chain, so there is run
time to take back. (2) The family weights sit on a sharp optimum: a single chain loses 0.16, 0.55 and 2.2 points at
jitter sd 0.15, 0.3 and 0.7. Weights that sharp, tuned on the ten people they are scored on, make the absolute dev
level optimistic; it is one more reason not to read dev levels as sealed levels.

## 3. One contrarian claim

**"Development gains reach the sealed score at 0.41 to 0.46" is not a property of the dev measure, and the "±0.3
noise" is the wrong yardstick.**

- Both ratios come from returning-person levers. Dev "seen" is two people; one is an outlier that holds 3 of about
  46 returning sealed sessions. With the benefit at +3.5 and +0.1, a person-equal mean of +1.8 was never a credible
  forecast; +0.6 to +0.8 per returning session, which is what the sealed set showed, sits between the two.
- **No new-person lever has been measured on the sealed set yet.** Its transfer is unknown. It could be higher (new
  people are person-equal in both) or lower (the dev numbers were mostly taken in recording order).
- The ±0.3 band comes from permuted-view null arms on the same ten people. That is seed noise. The uncertainty that
  decides transfer is across people: two systems that differ on 6 % of 2,655 windows have a paired standard error of
  about 0.5 points before clustering, 0.5 to 1.1 with it. **A +1.0 gain is 1 to 2 standard errors; +0.3 is 0.3 to
  0.6.** With dozens of arms the best one is inflated by selection; a posterior-mean reading (delta x tau² / (tau² +
  se²), tau about 0.5 from the upload history) turns +1.0 ± 0.5 into about +0.5 and +0.3 ± 0.5 into about +0.1.
- A sealed difference is itself noisy: about 20,000 windows, person-clustered, give roughly ±0.2 to ±0.3 on a paired
  difference. Two ratios of two noisy numbers do not make a constant.

**Test:** recompute both uploads' forecasts with the weights of proposal 4; and treat the next upload as the first
measurement of new-person transfer, with the prediction written down first.

Second, smaller: **"rebalancing is worth almost 6 points, so its form matters"**. At session level the form is
already at its optimum: on the mix the recent-share division scores 74.1 and the offline equal-count assignment over
the session 74.25. It is a per-session bias correction, not a quota. An exact session quota adds +0.1 to +0.3 on the
final posteriors, and a pure quota without the division loses 0.6 to 2.2. The unused part is the run level.

## 4. One question for the organisers, one check

- **Question:** "Are the windows of one run still delivered contiguously, or is the permutation over the whole
  session?" If they decline, the detector of proposal 1 answers it per session without them.
- **Check on the data:** what the sessions of about 380 windows are (twelve runs, or two visits joined), since it
  sets the urn sizes and the run-time worst case.

## 5. The leaderboard and the two uploads

**Reading the board.** Two accounts of one team at 0.847 and 0.846 mean a stable method, not a lucky draw. The pack
(0.819 to 0.833, nine teams within 1.4 points) is one recipe with small differences; much of the order inside it is
within two to three standard errors of a paired sealed difference. The leaders' margin over the pack (1.4) is the
size of one extra mechanism, not of a different regime. Candidates, in my order of likelihood: (1) a much stronger
use of the 43 % returning sessions (within-person, cross-session accuracy is usually well above cross-person; +4 on
those sessions is +1.7 on the board); (2) exact use of the label design at run level; (3) more target-task training
people from a source the pack has not found. They lead on the board scored without chronology, so the lead does not
rest on cue order. I could not check any of the three; they are guesses from the rules and the shape of the board.

**Decision frame.** The repository notes say a team keeps its best score (`docs/REVIEW_5NOV.md`); confirm on the
live board that it shows the best and not the latest upload. If so, each upload is an option with a floor of 0.8198.
That favours variance: do not spend an upload on a +0.2 candidate. A failed run costs no quota, so keep the last day
as a spare.

- **Upload 1 (16 or 17 Nov): measurement and option in one.** v5.1b untouched on recognised sessions; on
  unrecognised sessions the new-person stack (fine-tuning + eye view + rest member) with stage constants chosen by
  the worst case over the three schemes, not their mean (or per scheme by the detector, if proposal 2 passes); plus
  the detector-gated run filter if it passes its experiment. Under a full shuffle the filter is inert, so the sealed
  delta then measures the new-person stack alone; under a block scheme it measures both, and both would be kept.
  Written forecast before upload: 0.57 x (shuffled new-person gain) x 0.65, about +0.3 to +0.5, so 0.823 to 0.825.
  Reading rule (a paired sealed difference carries about ±0.2 to ±0.3): above +0.4 the new route transfers; below
  +0.1 it does not; between the two it is undecided and upload 2 keeps the route only if dev under shuffle still
  supports it; below −0.3 something in the package path is broken.
- **Upload 2 (20 Nov, with 21 Nov as the spare day):** if upload 1 confirmed the new route, everything that passed
  since, including the returning-route additions and the person-specific trust weight. If it did not, v5.1b plus
  returning-route work only. In both cases the highest-ceiling build that has a worker-image replay, not the safest
  one.
- **Honest odds:** third place (0.831 today, moving) needs about +1.2 sealed, which is +2.5 to +3 on dev at the
  transfer seen so far. The levers measured to date do not add up to that. Only the returning route or a run-level
  mechanism under a block scheme can close it.
