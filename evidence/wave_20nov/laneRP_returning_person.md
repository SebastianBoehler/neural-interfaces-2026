# RP · one enrolled EEG session, later-day decoding

**Decision, Fri 20 Nov 2026. Keep cap250’s returning-person path.** Both gates identified P04 and P10 in every later-session shuffle. The complete replacements lost 10.47 pp (feature experts) and 8.97 pp (waveform experts) against enrolled cap250. The strongest protected rule reached −0.13 pp [80% −0.49, +0.20] on the two public returners and +0.52 [+0.07, +0.97] on matched Scherer. It missed the +2.5 pp returning target. The eight single-visit people's later days remain unobserved.

## What the comparison measures

The test subject's identity is available to the evaluator for splitting and scoring. During inference the gate sees the current EEG prefix and a gallery of named training sessions. A wrong match selects the wrong person's actual trained decoder. Every probe day starts fresh. A recording from the eight single-visit public people can test false recognition when that person is held out. It cannot test true recognition on a later day.

On the production folds, native recognition increased P04 by 9.14 pp [7.97, 10.32] versus the same weights with recognition disabled. For P10 it changed accuracy by −0.50 pp [−0.83, −0.17]. Removing the whole person from source training further reduced both baselines. This distinguishes the value of identity routing from the value of including the person in source training.

The primary whole decoder uses cap250's new-person output until a match, then emits the personal expert's label. The protected alternative keeps enrolled stock running and switches a decision only when the two experts agree and their mean posterior is at least 0.70. Stock never consumes the alternative's outputs. This permits exact decision-level composition of its tapes without changing its future state. Reference and alternative use the same row order.

| System | EEG identity and fitted decoder |
|---|---|
| Feature experts | From window 8, both spectral shape and spatial correlation must identify the same gallery row. The mean nearest/runner-up distance ratio must be below 0.80. Equal posterior mixture of multiview kernel ridge and a nonlinear neural readout. Each has a population fit on public first visits and a supervised fit on the enrollee's first visit. Both see band covariance, band envelopes and the slow waveform, standardized to the delivered prefix with eight enrollment pseudowindows. |
| Waveform experts | From window 11, use cap250's externally trained spectral likelihood with a one-day gallery. Require score above 0.26 and a lead above 0.04 over the next person. Fine-tune the actual stock raw FBCNet and aligned FBLight traces on the person's first visit, then average their posteriors. Causal batch normalization and covariance alignment continue on the probe. Fifty-fifty posterior weights and all thresholds were fixed before scoring. |
| Controls | Gate disabled, permissive feature gate, individual ridge/neural decoders, actual stock returning, cap250 treating the same person as new. A waveform diagnostic supplies the correct identity from window 11 using information unavailable on the worker. |

**PROTEUS.** Real stock packages from production-20261120-32a94ab7. P04/P10 visit 1 supplied enrollment and visits 2–3 supplied the test data. Ten separate folds held out an entire person. Four whole-session shuffles, seeds 71000–71003. For the returning counterfactual, run precisely the same fold weights with both native recognition gates suppressed, allowing cap250 to continue its unrecognized-person adaptation. The person-held-out fold separately measures the effect of omitting all that person's training data. Earlier Lane A used conditional member tapes and a different delivery bank.

**Scherer.** Nine people. Enroll visit 1 and test visit 2 against all ten public and all nine external people. Hold the target out of the gallery to test both visits as new. Repeat with the ten public people alone. The matched protected test gives both the outer gate and stock the same enrollment roster, ten public people plus one external person. Its wrong-enrollee control runs each of eight other enrollees against each probe, both visits, full 71000. Full and within-run seeds 71000–71003 elsewhere. Twenty-two of 41 channels are interpolated.

**NETBCI.** Nineteen people. Enroll day 1 and probe days 2–4, with 29 competing identities, and with the target absent. The matched protected test uses ten public people plus the external enrollee. Full and within-run shuffles, seeds 72000–72003. Same B10 label construction for both expert systems and stock. Day-1 right-hand imagery maps to task 0. Alternating rest rows supply the two placeholder labels. The score averages imagery recall and rest specificity. It measures a two-class foreign-lab task.

PROTEUS uses official subject/session/interface balanced accuracy with equal cell weights. External cohorts use per-visit balanced accuracy. Deltas and 80% intervals pair predictions on the same rows and resample people, then orders within people and delivery scheme (20,000 draws, seed 20261120). Per-person intervals resample orders only. On false-enrollee pairs the probe people are resampled conditional on the donor panel. Reused people and repeated shuffles limit all intervals.

[Registration](../../docs/LEDGER.md#rp--one-enrolled-day-independent-expert-and-open-set-recognition-20-nov-before-scoring) · [Composition audit](../audit_20nov/sealed_composition.md) · [Reference production pipeline](cap250_pipeline.md) · [Experiment source](../../experiments/lanes/returning-participants/open-set-personal-experts)

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

## Waveform decoder results

| Returning cohort | Enrolled cap250 BA | Waveform BA | Waveform change [80%] | Protected change [80%] | Correct identity from window 11, change [80%] |
|---|---:|---:|---:|---:|---:|
| P04 | 70.20 | 56.35 | −13.85 [−15.86, −11.85] | −0.46 [−0.59, −0.33] | −13.46 [−15.21, −11.72] |
| P10 | 89.22 | 85.14 | −4.08 [−4.50, −3.66] | +0.20 [+0.07, +0.33] | −4.08 [−4.50, −3.66] |
| P04/P10, equal cells | 79.71 | 70.74 | **−8.97 [−14.16, −4.03]** | **−0.13 [−0.49, +0.20]** | −8.77 [−13.73, −4.03] |
| Scherer, competing gallery | 57.35 | 56.34 | −1.00 [−5.06, +3.16] | See matched gallery | +0.28 [−3.12, +3.91] |
| Scherer, one external enrollee | 57.35 | 57.63 | +0.28 [−3.12, +3.91] | +0.52 [+0.07, +0.97] | +0.28 [−3.12, +3.91] |
| NETBCI, competing gallery, BA2 | 68.65 | 68.24 | −0.41 [−2.24, +1.52] | See matched gallery | −1.97 [−4.04, +0.15] |
| NETBCI, one external enrollee, BA2 | 68.65 | 67.58 | −1.07 [−2.74, +0.73] | +0.43 [+0.12, +0.76] | −1.97 [−4.04, +0.15] |

The protected waveform rule changed 11 of 3,048 public returning decisions and zero of 10,620 held-new decisions. The standalone waveform decoder changed 684 returning decisions. Giving it the correct identity early recovered 0.20 pp. Most of its public deficit persisted after recognition.

| Test | Correct / wrong / missed, % of stream-orders | Consequence |
|---|---:|---|
| P04/P10 returning (16) | 100 / 0 / 0 | P10 and P04 visit 3 matched at window 11 on every order. P04 visit 2 matched at 11, 11, 25 and 39. |
| Held-new PROTEUS (56) | — / 0 / 100 | Both waveform arms reproduced every new-person label. The empirical zero interval covers ten people. |
| Scherer returning, competing (72) | 93.1 / 6.9 / 0 | Correct recognition 80% interval [83.3, 100.0]%, median correct age 11. |
| NETBCI returning, competing (456) | 75.4 / 9.2 / 15.4 | Correct recognition [63.2, 86.8]%, median correct age 11. |
| Scherer new vs ten public identities (144) | — / 0 / 100 | Neither waveform arm changed a label. |
| Scherer new vs gallery including other Scherer people (144) | — / 84.7 / 15.3 | Whole decoder −3.66 [−5.73, −1.56] pp versus genuinely new. |
| Scherer single wrong enrollee (144) | — / 100 / 0 | Whole decoder −3.20 [−5.38, −0.99] and protected −1.36 [−2.47, −0.24] pp versus genuinely new. The protected arm recovered +0.29 [+0.14, +0.43] versus stock with the same wrong enrollee. |
| NETBCI new vs competing gallery (456) | — / 16.2 / 83.8 | Whole decoder +2.68 [−0.08, +5.73] pp versus genuinely new. A wrong identity can still predict the proxy's two task groups. |

P01, P02, P03, P05, P06, P07, P08 and P09 each lack a labeled later day. Their true match rates, latencies and decoder gains are unknown. Counts by window 8, 20, 50 and session end are available for each observable public visit and each external participant/day. [Recognition by person and day](laneRP_recognition.tsv)

**Sealed sensitivity, unverified.** For the whole waveform decoder, ΔJ = −0.51 + 0.377 U pp, or −0.89 [−2.43, +0.72] with the Scherer proxy. For the protected waveform rule, ΔJ = −0.007 + 0.377 U pp, or **+0.19 [+0.02, +0.36]** with matched Scherer. The corresponding returning-only proxy gain is **+0.44 [+0.04, +0.83] pp**. The eight actual people's mean gain would need to be +2.89 pp to bring all returning cells to +2.50. Legacy 57/43, P04/P10 20/80 gives −2.59 pp for the whole decoder and +0.03 pp protected. Intervals condition on the evaluated arm and the assumed proxy. Choosing the strongest measured arm adds selection uncertainty. Neither cohort reconstruction nor proxy transfer is validated.

**Runtime.** On a Leonardo A100, three warm 100-window passes took 0.538–0.539 s for both fine-tuned waveform members. Spectral feature extraction, projection and gating took 0.039–0.077 s. The sequential path reproduced every diagnostic label and passed the posterior tolerance. These timings exclude cap250 itself and come from one public session. The sealed worker allows 3,600 s on an RTX PRO 4500 and stock used 2,555 s. Neither candidate earned package qualification.

## Reproduction and checks

- Leonardo release `/leonardo_work/AIFAC_P02_1675/ni26_laneRP_20261120`. `out/` contains the frozen feature decisions and same-source population-only tapes. `wave/` holds the waveform results. The code reads B10's per-person archives and the Galvani NI-V cap250 tapes. Enrollment fit seeds and bootstrap seed are 20261120.
- Galvani 3144944 printed all six v3 reference scores and `OK`. Leonardo 64985932 reproduced all 189 stock labels on P04 visit 2, full 71000. Disabling recognition preserved its pre-recognition labels and produced zero routed windows. Job 64986260 ran that exact suppression on all 16 returning stream-orders.
- Feature extraction 64986058 passed per-window real EEG checks on all three corpora before extracting rows. Jobs 64986261, 64986289 and 64986291 each ran a corpus-specific smoke before the blind replay. All official public stock scores reproduced to 1e−10. Independent logs confirm zero native recognitions on all 56 held-new public stream-orders. Unenrolled external tapes had zero native errors and zero routes.
- Registration `bce0192e`, feature implementation `b7b833e2`, waveform and scorer `cf3f99de`. Waveform smoke 64986661 stopped before scoring when one FBLight posterior differed by 0.000091 across inference batch sizes. Repair `cad72876` uses batch size one for every test convolution and disables cuDNN TF32. Authorized recovery 64987360 ran the real prefix, reset and live-stream checks before its full replay. One original smoke failed. No R4b job was repeated.
- Local PROTEUS and Scherer window SHA-256 prefixes are `2be4187fe33a` and `a61887997532`. External identity calibrator prefix `204f2040c217`. The official metric comes from the pinned code. The unverified composition and session constraints were checked against upstream rules at `8447e181`.
- Recompute using `python laneRP/score.py DECISIONS B10_OUT RESULT.json --stock PRODUCTION_RECEIPTS`. Add `--wave` for waveform tapes. The production receipts directory contains `stock/scores.json`, `jobs.jsonl` and original `logs/`. Coverage, independently regenerated row order, zero-switch prefixes, native stock scores, error counts and completeness all have separate checks.
- All 249 decision archives per system passed scoring coverage. Aggregate SHA-256 for feature tapes is `749fcd7d7a6f6beb269c7812745915fece2aee0ecc17b41f7565d7a299ee25bc` and for waveform tapes `26eb7ae269c4cff65bba21a4b4cc746d48bf5759e054ea2f8c7b9a04a64d4349`. Nine Slurm jobs including the approved recovery. Recovery completed in 21m27s. No upload.
