# Panel answer: cognitive neurophysiology (14 Nov 2026)

Source: the brief, `docs/STATE.md`, ledger §103 and §196, `experiments/session_em.py`, `experiments/ts_core.py`, and
about a dozen throwaway read-only snippets on `data/proteus_windows.npz` (2,655 windows, 10 people, 14 sessions).

**How to read my numbers.** They come from crude linear models (log band power per channel, 32 ocular features,
logistic regression, per-session standardisation over the whole session, CPU seconds). "LOSO" is leave-one-person-out,
"within" is 5-fold cross-validation inside a session with that session's labels. They show direction and structure.
They are not chain numbers and nothing here was scored in `replay/`. No script was saved; the feature definitions are
given so each number can be rebuilt in a few lines.

## 0. What I measured (facts the rest rests on)

| Check | Result |
|---|---|
| Montage | F, FC, C, CP, P rows plus AF3/AF4 and PO7/PO8. No Fp, no F7/F8, FT, T7/T8, no O1/Oz/O2, no mastoids |
| Reference | Not average-referenced: the channel mean has sd 23.5 µV of 29.7 µV total. After CAR, 5 to 12 µV per channel (2.3× between people) |
| Spectrum | No roll-off before 60 Hz; 50 Hz line about +4 dB; flat from 13 to 60 Hz in the raw reference |
| MI against the two cognitive tasks | Carried mostly by 1–7 Hz power, strongest frontal (AF4 4–7 Hz d = −1.05, 9 of 10 people): eye activity. Mu ERD is weak and sits at C4 and Cz (Laplacian 8–13 Hz d = −0.15 to −0.36), not C3 |
| SUB against WORD | Band power barely separates them (mean abs d per band 0.06 to 0.19). Best single features are ocular: horizontal gaze level (left fronto-lateral level, CAR, d = +0.4 to +0.63, t up to 4.4) and the late rise of vertical activity |
| Time course | SUB minus WORD in vertical ocular 1–3 Hz power flips sign 2.0 s into the window: d = −0.45 in the first half second, +0.45 to +0.48 at 2.0–3.0 s. MI minus SUB grows from about 0 to −0.96 at the same point |
| Gaze direction | SUB −5 to −6 µV, WORD +2 to +8 µV on F6−F5, same sign in all four interface × run-type cells; single-trial sd 18 to 30 µV |
| Alpha / mu peak | P03, P04, P05, P06, P10: no peak above 1.5 dB in the task window. P01, P08, P09: 6 to 9.5 dB. Where a peak exists it lies at 10.5 to 12.5 Hz |
| Ocular view, 32 features | LOSO 0.47, within 0.66. Per person LOSO → within: P10 0.35 → 0.81–0.84, P02 0.63 → 0.81, P03 0.59 → 0.77, P07 0.53 → 0.76, P04 0.31 → 0.38–0.50, P01 0.52 → 0.47 |
| Band power, 287 features | LOSO 0.50, within 0.70. Confusions LOSO: WORD is the weak class (0.44 recall), split between MI and SUB |
| De-ocularised band power (H and V regressed out of every channel, per session) | LOSO 0.506, within 0.653. Error correlation with the ocular view: 0.14 → 0.01 (LOSO), 0.17 → 0.08 (within). Product of the two views unchanged: 0.533 / 0.529 and 0.747 / 0.749 |
| Run clustering of errors (sd of run accuracy over binomial sd) | Ocular 1.6 (LOSO), 2.1 (within); band power 1.4. Lag-1 autocorrelation of correctness inside a run: −0.01 |
| Oracle standardisation per run, per interface, per interface × run type | Ocular within 0.658 → 0.672 at best, band power 0.701 → 0.710; LOSO unchanged; run clustering unchanged |
| Transfer inside a session | A model fitted on one interface loses 3 to 6 points on the other (ocular 0.65 → 0.59–0.61, band power 0.65 → 0.61–0.63). Models on one interface (96 windows) are as good as pooled models (190 windows) |
| Interface identity | With the session's labels 84 %; across people 59 %; 2-means inside a session 58–60 %. Run identity with labels 73–76 %, k-means ARI 0.1 |
| Variance explained by run / by class | 20–60 Hz: 16–23 % / 2–4 %. 1–13 Hz: 7–9 % / 8–16 % |
| EMG view (40–58 Hz at 12 edge channels against central, 4 time bins) | LOSO 0.37, within 0.44; adds nothing to the product |
| Cardiac artefact | Not visible (no peak at heartbeat lags in the common-mode or left-right signal, except faintly in P01) |
| Drift removal with label-free 50 Hz and 52–60 Hz components | Loses 0.6 to 1.7 points on band power; their first component is not time (rho 0.2) |

## 1. Diagnosis

**Where the error is.** Three sources, in this order.

1. **Word generation has no private signature in this recording.** Its cortical sites (left inferior frontal,
   anterior temporal) lie under F7, FT7 and T7, which the montage lacks. Its peripheral signature is covert
   articulation: muscle activity peaks at 50 to 150 Hz, and 120 Hz sampling leaves only the tail below 58 Hz. My EMG
   view found nothing usable. WORD is therefore decided by exclusion (not MI, not SUB), which is why rebalancing is
   worth almost 6 points and why WORD stays the weakest class.
2. **The class code is a person's strategy, not the task.** Half of the public people show no alpha or mu rhythm in
   the task window, so for them "motor imagery" has no rhythm to desynchronise. The eye code is strong in four people
   (0.76 to 0.84 with own labels), absent in two, and P10's is unrelated to the population's. A population model
   cannot pass about 50 % on either view; a session's own labels reach 66 to 70 % with the same crude features.
3. **The code changes inside a session, and no recentring repairs it.** Errors cluster by run at 1.4 to 2.1 times
   the binomial spread, also for models that had the session's labels. Oracle standardisation per run or per
   interface changes nothing. So it is not an offset. It is behaviour: runs or trials where the person performs the
   task differently or not at all, and feedback runs where the feedback takes the eyes.

**What the system cannot capture as built.**

- **The views are not separate.** Core1's low bands (1–3 to 4–8 Hz) under CAR carry the eye signal on every channel.
  Five points of a session's own band-power accuracy are ocular (0.701 → 0.653 after regression). SessionEM's 32 gaze
  features, the eye net and the top PCs of core1 therefore partly see the same quantity in the same window. A product
  of experts counts it several times.
- **Sign and latency of eye events.** `EventRates` is sign-free and uses four bins. The SUB against WORD difference
  is a direction (left against right) and a latency (blink release after about 3 s post-cue in SUB, 1 to 2 s in WORD
  and MI). The 16-bin levels keep direction but depend on the DC level.
- **A reference state.** ERD is a change against a baseline, and the window has none. The session mean over three
  active tasks stands in for it. The rest member is the right repair; I would use it further (proposal 3).

**What is adequately covered, so leave it.** Hemispheric asymmetry and connectivity (tangent space and the Hilbert
covariances contain them); slow levels (present in SessionEM and the own-space stage); time resolution of core1 (the
half split at 2.0 s happens to sit on the sign flip; thirds lost in §103a). Individual alpha frequency is not worth
aligning: the measurable peaks span 10.5 to 12.5 Hz against 4 Hz overlapping bands, and half the people have none.
Amplitude scale and 1/f slope are handled by recentring. Cross-frequency coupling needs gamma above 60 Hz. Cardiac
and respiratory views do not exist in these data.

**On shuffled order.** Fatigue and drift live above 20 Hz (run explains 16 to 23 % of the variance there, class 2 to
4 %). Removing them did not help in any form I tried, with or without oracle run labels. The loss of 1 to 4 points
under shuffling is therefore unlikely to be drift that recording order let the stages track. Look for it in the
statistics of the label sequence (the "recent share" in rebalancing is tightly balanced only in recording order).

## 2. Five proposals, ranked

### 1. Let the eye signal enter once: remove it from the EEG members

**Mechanism.** Regress the two ocular derivations of `session_em.py` (horizontal: F5, AF3, F3 minus F6, AF4, F4;
vertical: AF3, AF4 minus P1, P2, Pz), low-passed at 8 Hz, out of all 41 channels before CAR and band-pass. Core1,
its PCs for TSEM and the Hilbert core then describe the brain; SessionEM and the eye net describe the eyes.

**Why here.** The team's own estimate is that an eye view with independent errors is worth about +1.0 more. §196
tried to decorrelate from the eye side (centring per run or session) and found nothing; the listed next step was
residualising on the members. This is the same step in physiological form, at the source. My check: error
correlation between the two views falls from 0.14–0.17 to 0.01–0.08 while the plain product stays equal. The gain
can only appear in the EM stages, which I did not run.

**Expected.** New people +0.3 to +0.8, confidence 35 %. Returning +0 to +0.3 (personal heads must be refitted on
the new features). A real chance of −0.3 if the stages currently profit from the double counting.

**Cheapest test.** CPU. Rebuild core1 features with the regression, refit, swap in with stages and routing using
the new features (not posterior-only), all shuffle schemes. Report alongside: correlation within class × run
between the eye-net log posterior and the member mix's log posterior, before and after.

**Kills it.** Chain change below +0.2 with the conditional correlation unchanged; or core1 alone below 66 %.

**Notes.** Causal form: running sums of R'R (2 × 2) and R'Y (2 × 41) over delivered windows, or fixed population
coefficients for the first 10 windows. Order-free. Microseconds per window. Low-pass the regressors so frontal
theta, alpha and beta are not removed with the eye signal. The nets are not retrained in the first test.

### 2. Weight the peripheral view by how much this person has of it

**Mechanism.** The ocular code is bimodal across people (own-label accuracy 0.38–0.50 or 0.76–0.84, few between).
Give SessionEM's exponent a per-session factor from a label-free index of the view's class structure: template
separation (mean pairwise Mahalanobis distance of the fitted class templates) against the same quantity under
permuted responsibilities, or the agreement between the ocular template posterior and the EEG posterior on the
windows so far. Range 0 to about 3 times the current value, ramping in with the window count.

**Why here.** One exponent for everyone under-uses P10, P02, P03 and P07 and adds noise for P04 and P01. The team
found +1 to +2 for an oracle choice per session and no label-free rule; this is a rule with a physiological reason
for expecting it to be readable from the session itself.

**Expected.** New +0.3 to +1.0, confidence 30 %. Returning: P10-like people +0.5 to +1.5, P04-like 0; a known
person's factor can be stored from the public sessions (P10's ocular code holds at 0.81 to 0.84 in all three).

**Cheapest test.** CPU replay, oracle first: set the factor per session from the supervised accuracy of the same
32 features. Then the label-free index. Also report, per session, the accuracy of SessionEM's own templates at
session end against that supervised ceiling; a gap above 10 points for three or more people means the seeding fails
for unusual codes, and restarts from permuted seeds chosen by cross-view agreement are the follow-up.

**Kills it.** Oracle factor below +0.3 on new people.

**Notes.** Causal, order-free, no run-time cost. The eye net's posterior is a population code: for P10 it is close
to uninformative (LOSO 0.35), so the factor should apply to the template view, not to the net's weight.

### 3. Use the rest member as a trial-compliance weight

**Mechanism.** Run-level error clustering that survives oracle recentring is behaviour: lapses, late starts, runs
done half-heartedly. All three tasks suppress parietal alpha against rest, and a lapse looks like rest. Multiply
each window's responsibility in the template EM, and its weight as a pseudo-label in the fine-tuning, by
(1 − p_rest) from the member that already knows rest; optionally add parietal alpha relative to the session median.
Templates and fine-tuned nets are then learnt from trials where the task was performed.

**Why here.** The stages and the fine-tuning learn from every delivered window with equal trust. With true labels
the fine-tuning gives +1.9 more, so label quality is the limit, and non-compliant trials are wrong labels by
construction. The team already observed that the eye view makes the stage use the rest-trained member.

**Expected.** New +0.2 to +0.6, confidence 25 %. Returning +0 to +0.3.

**Cheapest test.** CPU, existing outputs: chain accuracy by tercile of p_rest. If the top tercile is less than 3
points worse than the bottom one, stop. Otherwise replay with weighted responsibilities.

**Kills it.** No relation between p_rest and correctness, or a weighted replay below +0.2.

**Notes.** Causal, order-free, free at run time. Do not change the prediction for a rest-like window, only its
influence on learning.

### 4. A view built for SUB against WORD: signed, latency-coded eye events

**Mechanism.** Add to SessionEM's view: counts of leftward and rightward velocity events per half window (signed,
not absolute), the latency of the first blink (censored at 4 s), and late-minus-early contrasts of vertical
activity. Events do not depend on the DC level. In my check events were slightly better than levels (within 0.594
against 0.563) and a little less clustered by run (1.73 against 1.89).

**Why here.** This is the pair the EEG separates worst, and the eye code for it has the same sign in both
interfaces and both run types. The sign flip at 2.0 s means anything averaged over the window cancels.

**Expected.** New +0.1 to +0.4, confidence 25 %. Returning similar. §58's extended eye view lost 0.8 to 2.8, so the
prior is not good; that test added dimensions to a view also present in core1, which proposal 1 removes.

**Cheapest test.** First the chain's confusion matrix by class pair and run type. If SUB ↔ WORD is under 40 % of
the errors, drop this. Then a CPU replay with 6 to 8 added columns, with and without proposal 1.

**Kills it.** Below +0.2 in both settings.

**Notes.** Thresholds from the running session median as in `EventRates`. Causal, order-free, free at run time.

### 5. Class templates per interface (the form §103b did not test)

**Mechanism.** Give each class template an interface-specific offset shrunk to the pooled template. §103b closed
the interface axis on rebalancing and the teacher weight per interface. My measurement puts the difference in the
class code itself: a model from one interface loses 3 to 6 points on the other, and half the data per interface
equals all data pooled.

**Expected.** Oracle +0.3 to +1.0. Label-free +0 to +0.4, confidence 15 %: the interface cannot be read from a
window across people (59 %) or by clustering (58–60 %). It becomes usable only if runs arrive contiguously, where
forgetting or a reset at detected run boundaries gives it for free.

**Cheapest test.** CPU replay with the true interface label in SessionEM and TSEM. **Kills it:** oracle below
+0.5. **Notes.** Worth nothing under a full shuffle unless a context detector appears.

## 3. Contrarian claim

**"The eye net's errors come in streaks within a run" is not what §196 showed.** The diagnostic that helped
(shufrun) permutes the eye outputs within person × class × run. That breaks two things at once: the order in time,
and the pairing of each eye output with the other views of the same window. It was scored under within-run
shuffled delivery, where the order inside a run is already random for the stages. What remains is the pairing. In
my data correctness has no serial dependence inside a run (lag-1 −0.01), while the eye view and the EEG view share
errors (0.14 to 0.17) because they share the signal. I read the +0.22 / +0.67 as the price of conditional
dependence between views, not of streaks.

**Test.** Run shufrun under the full-shuffle scheme: if the gain is the same, time is not the cause. Then compute
the within class × run correlation of the eye-net log posterior with SessionEM's gaze columns and with the member
mix. Proposal 1 is the remedy if I am right; centring and smoothing in time are not.

A smaller one: "more people is the strongest lever" is true of the population part, and the reason is that the code
is a strategy. Ten people sample perhaps three strategies. External corpora without the eye and slow-potential
behaviour of this paradigm cannot add strategies, which fits their never helping.

## 4. One question, one check

**Question for the organisers.** What is on the screen from 1 to 5 s after the cue in each interface: where the cue
symbol sits, and whether feedback driven by an online decoder is visible in the online runs? If SUB-left and
WORD-right is screen layout, it holds for every sealed person and deserves a strong population prior. If it is a
personal gaze habit, only the per-session fit can use it and proposal 2 gains weight.

**Check on the data that would change my ranking.** The conditional correlation in §3. If it is below 0.05,
proposal 1 drops to last and proposal 2 leads.

## 5. The two uploads

- **Upload A (15 or 16 Nov).** The team's measured candidate as it stands (fine-tuning, eye view and rest member on
  unrecognised sessions; recognised sessions as v5.1b), provided it holds under all three shuffle schemes. At the
  observed transfer of 0.41 to 0.46 that is about +0.4 to +0.5 sealed: useful, not third place. Nothing of mine
  goes in unmeasured.
- **Upload B (20 or 21 Nov).** A plus whichever of proposals 2 and 3 passed its bar in the chain; both are a few
  lines in the stages with no run-time cost. Proposal 1 only if the whole chain was rebuilt on the new features and
  scored with stages and routing, because it changes core1, the TSEM basis and the personal heads at once. If
  nothing passes, upload A's design with the best-measured fine-tuning form and keep the slot's risk low.
