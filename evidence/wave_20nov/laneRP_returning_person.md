# RP · one enrolled EEG session, later-day decoding

## What the comparison measures

The test subject's identity is available to the evaluator for splitting and scoring. During inference the gate sees the current EEG prefix and a gallery of named training sessions. A wrong match selects the wrong person's actual trained decoder. Every probe day starts fresh. A recording from the eight single-visit public people can test false recognition when that person is held out. It cannot test true recognition on a later day.

On the production folds, native recognition increased P04 by 9.14 pp [7.97, 10.32] versus the same weights with recognition disabled. For P10 it changed accuracy by −0.50 pp [−0.83, −0.17]. Removing the whole person from source training further reduced both baselines. This distinguishes the value of identity routing from the value of including the person in source training.

The primary whole decoder uses cap250's new-person output until a match, then emits the personal expert's label. The protected alternative keeps enrolled stock running and switches a decision only when the two experts agree and their mean posterior is at least 0.70. Stock never consumes the alternative's outputs. This permits exact decision-level composition of its tapes without changing its future state. Reference and alternative use the same row order.

| System | EEG identity and fitted decoder |
|---|---|
| Feature experts | From window 8, both spectral shape and spatial correlation must identify the same gallery row. The mean nearest/runner-up distance ratio must be below 0.80. Equal posterior mixture of multiview kernel ridge and a nonlinear neural readout. Each has a population fit on public first visits and a supervised fit on the enrollee's first visit. Both see band covariance, band envelopes and the slow waveform, standardized to the delivered prefix with eight enrollment pseudowindows. |
| Waveform experts | From window 11, use cap250's externally trained spectral likelihood with a one-day gallery. Require score above 0.26 and a lead above 0.04 over the next person. Fine-tune the actual stock raw FBCNet and aligned FBLight traces on the person's first visit, then average their posteriors. Causal batch normalization and covariance alignment continue on the probe. Fifty-fifty posterior weights and all thresholds were fixed before scoring. |
| Controls | Gate disabled, permissive feature gate, individual ridge/neural decoders, actual stock returning, cap250 treating the same person as new. A waveform diagnostic supplies the correct identity from window 11; this requires information unavailable on the worker. |

**PROTEUS.** Real stock packages from production-20261120-32a94ab7. P04/P10 visit 1 supplied enrollment and visits 2–3 supplied the test data. Ten separate folds held out an entire person. Four whole-session shuffles, seeds 71000–71003. For the returning counterfactual, run precisely the same fold weights with both native recognition gates suppressed, allowing cap250 to continue its unrecognized-person adaptation. The person-held-out fold separately measures the effect of omitting all that person's training data. Earlier Lane A used conditional member tapes and a different delivery bank.

**Scherer.** Nine people. Enroll visit 1 and test visit 2 against all ten public and all nine external people. Hold the target out of the gallery to test both visits as new. Repeat with the ten public people alone. The matched protected test gives both the outer gate and stock the same enrollment roster, ten public people plus one external person. Its wrong-enrollee control runs each of eight other enrollees against each probe, both visits, full 71000. Full and within-run seeds 71000–71003 elsewhere. Twenty-two of 41 channels are interpolated.

**NETBCI.** Nineteen people. Enroll day 1 and probe days 2–4, with 29 competing identities, and with the target absent. The matched protected test uses ten public people plus the external enrollee. Full and within-run shuffles, seeds 72000–72003. Same B10 label construction for both expert systems and stock. Day-1 right-hand imagery maps to task 0. Alternating rest rows supply the two placeholder labels. The score averages imagery recall and rest specificity. It measures a two-class foreign-lab task.

PROTEUS uses official subject/session/interface balanced accuracy with equal cell weights. External cohorts use per-visit balanced accuracy. Deltas and 80% intervals pair predictions on the same rows and resample people, then orders within people and delivery scheme (20,000 draws, seed 20261120). Per-person intervals resample orders only. On false-enrollee pairs the probe people are resampled conditional on the donor panel. Reused people and repeated shuffles limit all intervals.

[Registration](../../docs/LEDGER.md#rp--one-enrolled-day-independent-expert-and-open-set-recognition-20-nov-before-scoring) · [Composition audit](../audit_20nov/sealed_composition.md) · [Reference production pipeline](cap250_pipeline.md) · [Experiment source](../../laneRP)

## Feature decoder results

All changes below are paired percentage points. The two columns headed “new” distinguish a fold that excluded the person completely from a fold that included visit 1 in source training and then disabled recognition.

| Returning cohort | Whole person held out / stock new | Same-source stock new | Stock enrolled | Feature decoder | Change vs enrolled [80%] | Protected change [80%] |
|---|---:|---:|---:|---:|---:|---:|
| P04 | 46.96 | 61.06 | 70.20 | 60.39 | −9.81 [−11.40, −8.22] | +0.22 [−0.32, +0.77] |
| P10 | 84.59 | 89.72 | 89.22 | 78.08 | −11.14 [−11.68, −10.60] | −4.21 [−4.63, −3.79] |
| P04/P10, equal cells | 65.77 | 75.39 | 79.71 | 69.24 | **−10.47 [−11.39, −9.38]** | −1.99 [−4.26, +0.28] |
| Scherer, competing gallery | 48.74 | 48.74 | 57.35 | 52.71 | −4.63 [−7.69, −1.28] | See matched gallery |
| Scherer, one external enrollee | 48.74 | 48.74 | 57.35 | 52.53 | −4.82 [−7.91, −1.38] | −0.23 [−1.79, +1.46] |
| NETBCI, competing gallery, BA2 | 52.18 | 52.18 | 68.65 | 68.24 | −0.40 [−3.54, +2.82] | See matched gallery |
| NETBCI, one external enrollee, BA2 | 52.18 | 52.18 | 68.65 | 72.47 | +3.82 [+1.09, +6.64] | +4.47 [+2.87, +6.14] |

The strict feature gate recognized all 16 public returns correctly. Twelve matched at window 8 and the last matched at window 15. Despite that, the decoder changed 849 of 3,048 labels and lost 10.47 pp. Ridge alone, the neural readout alone and the permissive gate lost 8.78, 10.84 and 10.57 pp. On NETBCI, correct identification fell from 89.5% with one external enrollee to 55.3% with all competitors. Wrong identity accounted for 7.0% of the latter streams.

| New-person test | False matches | Feature change vs genuinely new [80%] | Protected change vs genuinely new [80%] |
|---|---:|---:|---:|
| PROTEUS, target excluded | 3/56 | −1.16 [−2.49, 0.00] | −0.77 [−1.66, 0.00] |
| Scherer, all other people enrolled | 0/144 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Scherer, ten public people only | 0/144 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Scherer, single wrong external enrollee | 36/144 | −2.25 [−3.49, −1.10] | −2.24 [−3.56, −0.93] |

The three public false matches came from P03 visit 1 twice and P10 visit 1 once. With less gallery competition, the nearest/runner-up ratio let the wrong external enrollee through more often. On that same Scherer control, ordinary wrongly enrolled cap250 already lost 1.64 pp [−2.71, −0.57] relative to genuinely new. The protected alternative lost another 0.60 pp [−0.94, −0.27] against wrongly enrolled cap250 itself.

**Sealed sensitivity, unverified.** Let U be the alternative's unknown change on later visits of the eight single-visit public people. With the withdrawn 60/6/40 visit counts, the full feature decoder's change in sealed score is −1.25 + 0.377 U pp. Substituting the nine-person Scherer result gives −3.00 pp [−4.34, −1.60]. The protected alternative gives −0.55 + 0.377 U, or −0.63 [−1.42, +0.15] using matched Scherer. The legacy 57/43 and returning P04/P10 20/80 weighting gives −5.34 and −1.87 pp respectively. These are dev sensitivities. The eight people's U remains unmeasured. The sensitivity intervals resample the public single- and multi-visit groups separately, sharing returning people's draws between roles, and resample the external people independently.
