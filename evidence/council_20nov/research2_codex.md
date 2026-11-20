# Research 2 · Six tests for the shuffled EEG stream

20 November 2026. First run the class-coupled SessionEM test and the returning-person test in parallel. Forecasts are planning estimates in balanced-accuracy
percentage points (pp).

## Rules and shared test protocol

- Upstream main `8447e181` supplies one `(1, 41, 480)` tensor and requires one immediate, final label. Use the current window and strictly earlier deliveries
  of the same session. Reset mutable state after each session. Frozen public weights and enrolment are allowed. Query labels, interface, person ID and
  recording order stay with the scorer. The budget is 3,600 s on one RTX PRO 4500. Declare any licensed external training data.
  [FAQ](https://github.com/neural-interfaces26/2026-competition/blob/8447e181/codabench/pages/scientific_faq_bci.md#streaming-and-adaptation) ·
  [Terms](https://github.com/neural-interfaces26/2026-competition/blob/8447e181/codabench/pages/terms.md)
- Run the complete stack on each arm. Use hardcb on the held-new chain, cap250 for the Scherer calibration tests and hardcb-B for returning stress. Lane S2
  gives a fast corrected `full0` and `within0` comparison. Its frozen shadow, fine-tuning banks and unrouted held-new arm limit attribution. Confirm gains on
  the live, full-shuffle Solver. [Lane S2](../wave_20nov/laneS2_session_stages.md) · [NI-V](../wave_19nov/NIV.md)
- Register each run in the ledger before scoring. Require zero-change, pinned-order and prefix/reset parity, no worker errors and active changes. The v3
  replay guard printed OK. Seal blind predictions, then report official cell-mean balanced accuracy, corrections and regressions, and paired 80% intervals
  resampling people then orders. Show P04/P10 separately and J beside the **unverified** 57/43 mix and 20/80 returning sensitivity. A nonpositive upper bound
  refutes a contrast on its instrument. Results inside person-level noise leave the direction undecided. [Ledger](../../docs/LEDGER.md) ·
  [Composition audit](../audit_20nov/sealed_composition.md)
- Smoke before the parallel wave. Use Galvani for CPU jobs and Leonardo for live workers. The target is Saturday 21 Nov, 10:00 CET, given queue slots. Hardcb
  projects to 3,211 s on the RTX proxy. As a planning example, 1 ms/window adds 20 s across 20,000 windows. Measure paired warm runs including reset and label
  return on both recognition paths. [Hardcb timing](../wave_20nov/final_stack.md)

## 1. Couple the class displacements inside SessionEM

**Bottlenecks 1, 3, 5.** GRAFT shares prototype displacement across related classes through a small graph and fits it with a closed-form update. The ICLR 2027
NeuroContext-TTA submission translates every source prototype together when short EEG histories give unreliable class coverage. These supply two ends of a
testable bias–variance trade-off. [GRAFT §4, 17 Nov](https://arxiv.org/html/2611.23591v1) ·
[NeuroContext-TTA, 18 Sep submission](https://openreview.net/forum?id=KJ0mwI2BjI)

In the outer SessionEM only, estimate source class anchors `A` from GE (ocular feature) rows in prefix age bands 20–39, 40–79 and 80+, with equal-person/class
weighting and held-person exclusion. In its existing two updates, let `N` be the diagonal of soft class masses and `S = Qᵀ Z` the weighted sums of
standardized prefix features. With graph Laplacian `L`, solve `(N + 20 I + 12 L) D = S − N A` and use means `A + D`. Start with one graph edge between
subtraction and word generation. The test is whether their **displacements** are related. Keep current likelihood strength and downstream TSEM, CountFilter
and CSP. Use only past GE and soft responsibilities.

Lane S2’s single-constant moves in hardcb SessionEM all scored at or below zero. Lane D’s full-session Gaussian gained +0.292 pp on full0 with an interval
spanning zero and lost on P04. Sharing only reliable changes can rescue a weak class whose own pseudo-labels are sparse, without re-learning the whole map
from those labels. Lane Z’s −2.40 pp without rebalance and −2.26 pp without EM+TSEM justify testing this within the stack.
[Lane D](../wave_19nov/laneD_prefix.md) · [Lane Z](../wave_20nov/laneZ_chain_profile.md)

**Saturday test.** In `laneS2/chain/final_s2.py`, compare exact base, anchors only (`L=0`), one shared displacement, the cognitive edge and a word–motor edge
at the same dose. Source-only check whether within-person, across-interface labelled displacements are closer on the cognitive edge than on both wrong edges.
Predict +0.25 pp [−0.40, +0.90] vs base and +0.15 [−0.30, +0.60] vs anchors alone. Require a gain over anchors alone and the wrong edge through the terminal
stage. Check minority-class recovery by age 20–60. Galvani CPU, roughly 25–45 min including anchor extraction. Budget <0.5 ms/window for the three-class
solve. If confirmed, repeat the hook on the live outer RouteB tail with the same reset.

## 2. Calibrate what existing expert votes add to the terminal decision

**Bottlenecks 2, 6.** The NeurIPS LLM ensemble paper measures conditional information under correlated errors. A second accepted paper frames post-hoc
deferral as density-ratio estimation. ICLR submission *A Query Is Not a Commitment* shows why biased expert answers can still carry information.
[Ensemble selection, NeurIPS 2026](https://neurips.cc/virtual/2026/poster/150011) · [Density ratios, 10 Nov revision](https://arxiv.org/html/2605.19557v2) ·
[Expert answer correction, 16 Sep submission](https://openreview.net/forum?id=FDEKHiOd3X)

Take the complete terminal top two classes `a,b`. Bin their log-odds gap at fixed 0.25 and 0.65. In each ordered-pair/gap bin, estimate the odds of `a`
versus `b` conditional on the **joint** votes from C1 and the effective raw-FBC family. Shrink the conditional table by 40 weighted pseudorows to the
votes-blind table and clip the difference in log odds to ±log(2). Add half the difference to `a` and subtract half from `b` at emission. The joint table
accounts for correlated votes. Subtracting votes-blind odds removes ordinary pair calibration. The edit uses the current call’s existing votes and frozen
tables.

R1’s nested mass changes were near zero or harmful. §363 used four scores from two whole chains and selected which existing endpoint to trust. F5 lost
3.28–3.98 pp externally with a learned waveform residual. Here the new variable is the current *pattern* of class confusions inside a single stack, and the
signed edit compares the same ordered class pair with and without that pattern. [R1](../wave_20nov/laneR1_bigstack.md) ·
[§363](../../docs/LEDGER.md#363-source-nested-terminal-branch-selector-prospective) · [F5](../../friday/errorfit/external-result.md)

**Saturday test.** Capture these contributors with Lane Z’s mixer hook. Fit the table on public fit-excluded `full0` source tapes, then lock it before scoring
exact cap250 on NI-V B-new orders 71000/71001. Stock on both sides avoids the Scherer exposure of the hardcb-C source banks. Verify that the live capture
reads the same effective family and causal adaptation path. Compare zero, votes-blind calibration, the joint table and a table fit with source vote tuples
permuted within each pair/gap bin. Audit oracle top-two coverage on stock misses after sealing. Forecast +0.15 pp [−0.45, +0.75] vs cap250. A gain over both
controls supports the conditional-vote effect. ~20 min CPU setup and <1 h live workers with slots available. Target <0.05 ms/window. A new public held-person
result needs nested source tapes that also exclude its outer held person.

## 3. Estimate cross-day drift from one labelled enrolment

**Bottleneck 4.** Speaker verification carries utterance uncertainty into scoring and calibration. Few-shot call-sign spotting bounds acoustic context so
strong lexical decisions survive. Labelled stimulus anchors with source shrinkage also improve bias estimation in EEG-image retrieval.
[Li and Lee, 13 Sep revision](https://arxiv.org/html/2609.01221v2) · [Call-sign spotting, 18 Sep ICLR submission](https://openreview.net/forum?id=AXTuklqgdP)
· [SA-LBC, 12 Nov](https://arxiv.org/html/2611.16257v1)

At public enrolment, store class centroids and within-class variance of the first 12 existing tangent PCs, with run-split uncertainty. After recognition, use
strictly earlier windows and the stock population shadow to estimate a **common** offset from those class anchors. In each of two interleaved prefix halves
average `u_i − Σ_c p_i(c) μ_enrol,c`. Require each half to have at least 20 windows and their offsets to have positive dot product in enrolment variance
units. Shrink the pooled standardized offset Δ using its estimated noise. Use `r = clip(1 − trace(V_noise)/||Δ||², 0, 0.5)`, where `V_noise` is the variance
of the estimated offset. Shift centroids by `rΔ` and apply the diagonal Gaussian log-likelihood difference for shifted vs original centroids, clipped to ±0.3
logit per class after RouteB. Class contrasts, personal head, identity and fine-tuning work are fixed. Use one PC coordinate system and causal extraction
recipe for enrolment and query.

The shipped personal TSEM prior pools class means with current pseudo-labelled means at a fixed prior dose. This arm estimates a day offset in common
coordinates and uses a bounded correction when two prefix halves corroborate it. R4 gains +3.68 pp BA2 on NETBCI but costs runtime and loses near the P04/P10
ceiling. The split-prefix estimate can suppress a near-zero or noisy offset. [Personal TSEM](../../lean/src66/stages.py) ·
[R4](../wave_20nov/laneR4_keepadapt.md) · [NETBCI](../wave_20nov/laneB10_returning_stress.md)

**Saturday test.** Extend `niv/enrol.py` to write the frozen moments for each actual enrolled row. Stream day 2 and day 4 for all 19 NETBCI people with
`laneB10/run.py`, orders `full:72000,full:72001`. Compare exact hardcb-B, fixed half-shift, uncertainty-gated half-shift and R4’s already recorded matched
streams. Check zero before recognition, real post-latch activation and correct rows after external enrolment. Run Lane A returning P04/P10 on full0/full1 as a
ceiling stress and NI-V wrong-enrolment as a harm check. Forecast +1.2 BA2 pp [−0.5, +2.9] on NETBCI and +0.00 pp [−0.45, +0.45] on P04/P10. Require a
positive gated-vs-fixed contrast to credit the uncertainty step. NETBCI uses two placeholder enrolment classes for rest and scores binary BA2. ~1–2 h of
parallel Leonardo workers when scheduled. Target <0.5 ms/window with two moment accumulators.

## 4. Verify the top two classes with within-window cross-view prediction

**Bottlenecks 1, 2, 5.** TTC tests a candidate by how its subspace responds to the example. The ICLR 2027 RISD submission selects neural features against
residuals from grouped out-of-fold readouts. Their common lesson is to ask what an extra feature explains conditional on existing evidence.
[TTC, NeurIPS 2026, 5 Oct](https://arxiv.org/html/2610.06147v1) · [RISD, 17 Sep submission](https://openreview.net/forum?id=WBLUYlTW3x)

Extract separate vectors of theta/alpha/beta log power in central, frontal and lateral ROIs from the first and last two seconds of the current tensor. Fit
source-only ridge maps from the first vector to the second, one per class with shrinkage to a shared map. Compare regularized residual likelihoods for the two
candidates. Add 0.15 times the centered log likelihoods, clipped to ±0.5 logit after TSEM. Run CountFilter/CSP normally. P04’s enrolled head shows a class map
exists. A class-conditioned relationship between two views can carry information that marginal powers discard. F-C31 measured output sensitivity to a
reconstructed sensor view and lost to its lag control. Lane P2 uses segment dispersion and shared pseudo-labels. Here a class hypothesis explains an observed
second view. §290 fit independent early/late scores. Labels enter only the frozen source fit. Within-window time is part of the supplied tensor.
[F-C31](../../friday/cap250c31/README.md) · [Lane P2](../wave_20nov/laneP2_package.md)

**Saturday test.** Extract once on CPU, fit ten leave-one-person-out folds and inject into the Lane S2 pipeline. Compare full stack zero, class-specific
marginal likelihood of the second view, and the same conditional model fit with second views permuted within source session and class. Fix ridge penalty at 10
in standardized coordinates. Predict +0.15 pp [−0.45, +0.80] vs base and +0.10 [−0.45, +0.65] vs matched permuted pairs. Refute the specific cross-view claim
if the upper bound vs either evidence control is ≤0. ~30–60 min Galvani CPU. Target <1 ms/window with shared filters and small matrices.

## 5. Estimate a feasible class prior for template fitting

**Bottleneck 3.** Label-shift constraints bound feasible class proportions without inverting a noisy confusion matrix. Uniform Sinkhorn assumes a known target
marginal. SCALAR’s small-batch ablation uses a 1,024-item queue. Lane D’s strict-prefix balancing reversed across delivery types and lost 0.183 pp on Scherer.
[Label-shift constraints, 8 Nov revision](https://arxiv.org/html/2609.14802v2) · [SCALAR, 9 Nov](https://arxiv.org/html/2611.12242v1) ·
[Lane D](../wave_19nov/laneD_prefix.md)

From public fit-excluded predictions, bound class-conditional frequencies for hard votes and high-margin votes. After 40 delivered windows, intersect the two
feasible regions of the class-proportion simplex using strictly past frequency intervals. Require a nonempty region that narrows at least one class’s prior
range below 0.20. Use the feasible prior closest to uniform in the outer SessionEM responsibilities. Multiply it by the source-calibrated pre-rebalance
posterior divided by its source prior, then normalize. Keep the stock rebalance. Update emission through the fitted class likelihood. Under pure label shift,
equal recall weighting cancels the target prior in a single cell. Cross-person covariate shift can invalidate the moment bounds.

**Saturday test.** In Lane S2 compare stock, uniform and feasible prior with identical template updates. For honest calibration use the public fit-excluded
tapes. Score the frozen test with exact cap250 on Scherer orders 71000/71001. Test class-prior error vs uniform by prefix age after sealing predictions and
count infeasible/active prefixes. A nonpositive upper bound on template gain vs uniform defeats this form. Unchanged responsibilities fail the activity check.
Forecast +0.08 pp vs uniform [−0.45, +0.55]. CPU prototype ~30–60 min followed by <1 h of paired live workers. Target <0.1 ms/window by solving the
two-dimensional intersection every 20 calls. Treat the bounds as diagnostics under cross-person shift.

## 6. Test a covariance feature invariant to shared sensor mixing

**Bottlenecks 1, 2, 5.** SPDAlign models shifts in the EEG forward map. Notaro et al. show that congruence alignment preserves affine-invariant within-person
task geometry, so it cannot fix every cross-person mismatch. The proposed feature eliminates a shared mixing term within each window.
[SPDAlign, 5 Oct](https://arxiv.org/html/2610.06315v1) · [Notaro et al., NeurReps workshop, 11 Nov](https://arxiv.org/html/2611.15204v1)

Compute sorted log generalized eigenvalues of (mu covariance, broad covariance) and (beta covariance, broad covariance) from the same EEG window. Keep four
low and four high eigenvalues per pair. Under full-rank shared sensor mixing both matrices transform as `C ↦ M C Mᵀ` and the generalized eigenvalues are
unchanged. Add 0.001 times the denominator to the numerator to preserve invariance. Fit a source-only ridge logistic readout, then use its centered log
evidence at 0.10 after TSEM with the live tail. This gives a signal that F-C29’s transport of pseudo-class covariance and global EA never measured. It
discards spatial orientation. It may fail if the usable task signal is localized or sensor mixing is band-dependent.
[F-C29](../../friday/cap250c29/result-19nov.md)

**Saturday test.** Reuse the bands and raw tensor from the corrected chain. Run zero, same-size absolute log-eigenspectra and paired spectra through both
orders on Galvani CPU. Require invariance on real covariance pairs transformed with fixed invertible matrices before scoring blinded predictions. Fit the same
way on the other nine people. Forecast +0.10 pp [−0.50, +0.70] vs base. A nonpositive paired upper bound vs the absolute-spectrum control refutes the proposed
incremental benefit. ~25–50 min for extraction and replay. Budget ≤1 ms/window sharing covariance extraction.

## Search decisions

- Primary-source search ran 20:27–20:57 CET. Conference schedules and OpenReview notes distinguish NeurIPS acceptances from ICLR 2027 submissions. Recent
  arXiv versions included the 8 Nov label-shift, 10 Nov deferral and 18 Nov MAGE revisions. I checked the dated arXiv lists, Hugging Face model cards and
  author code. [NeurIPS schedule](https://neurips.cc/Conferences/2026/AcceptedPapersInitial) ·
  [ICLR venue](https://openreview.net/group?id=ICLR.cc/2027/Conference) · [arXiv recent](https://arxiv.org/list/cs.LG/recent) ·
  [MAGE v2](https://arxiv.org/html/2611.09928v2) · [CBraMod](https://huggingface.co/weighting666/CBraMod) ·
  [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- SafeCut's published neighbours use the entire target domain. GALA predicts before each update but averages repeated presentations of the same image. JPTA’s
  reference online batch is 128. Realistic few-shot evaluation also finds that fixed balance can conceal failures.
  [SafeCut, 2 Oct](https://arxiv.org/html/2610.02981v1) · [Author code](https://github.com/SeongJunLee-net/SafeCut_NIPS26) ·
  [GALA, 18 Nov](https://arxiv.org/html/2611.24206v1) · [JPTA code](https://github.com/MR-Sherif/JPTA) ·
  [Few-shot evaluation, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/4d7a968bb636e25818ff2a3941db08c1-Abstract.html)
- Recent cross-day intracortical calibration uses three labelled sentences from the query session. PRISM uses separate target support and chronological query
  bins. The fMRI-to-ECoG paper shares stimulus features in an encoding task. Brain-OF uses modality-specific experts. The protocols require task-specific
  support or matched stimuli. [Cross-day, NeurIPS 2026](https://neurips.cc/virtual/2026/poster/149445) ·
  [PRISM, 17 Sep submission](https://openreview.net/forum?id=MyPsZDr55E) · [fMRI–ECoG](https://neurips.cc/virtual/2026/poster/153546) ·
  [Brain-OF, fMRI/EEG/MEG](https://neurips.cc/virtual/2026/poster/155505)
- Low-data speaker codes use multiple ASR hypotheses. ZP-KWS separates speaker and word evidence. Cold-start LDCO transfers observed user evidence and GAIN
  bounds continual corrections with posterior-predictive uncertainty. These support low-dimensional, uncertainty-aware updates. Their effect sizes apply to
  their source tasks. [Speaker codes, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/vandalen25_interspeech.html) ·
  [ZP-KWS, Interspeech 2026](https://arxiv.org/abs/2606.20106v2) · [LDCO, 8 Sep submission](https://openreview.net/forum?id=xCYiobaQXx) ·
  [GAIN, 16 Sep submission](https://openreview.net/forum?id=JzjNi4rkcX)
