# §413 Does regional high-frequency activity explain the burst-duration reversal?

Removing the source-estimated contribution of local 34–45 Hz activity reduced the whole-classifier cohort gap by **0.485 percentage points** [80% +0.068, +0.938]. The original gap was +3.372 points. After correction it was still +2.887 [+1.364, +4.511]. This rejects the registered one-point reduction for this correction. First-session-only reduction was +0.555 [−0.040, +1.185]. The extra reduction over another person's amplitude-matched spectral ratio was +0.783 [−0.106, +1.714], leaving the registered +0.5-point contrast undecided.

| Measure | Six held source people | Four held development people | All 28 official cells |
|---|---:|---:|---:|
| Whole classifier, intact duration | 36.233% | 34.921% | 36.075% |
| Whole, shuffled duration | 38.247% | 33.563% | 36.447% |
| Whole, true spectral correction | 36.609% | 34.812% | 36.304% |
| Whole, wrong-person correction | 36.020% | 35.006% | 36.056% |
| Full frozen B2_pool | 78.592% | 81.216% | **76.279%** |
| Full, intact | 78.390% | 81.395% | 76.144% |
| Full, true spectral correction | 78.425% | 81.404% | 76.178% |
| Full, wrong-person correction | 78.470% | 81.380% | 76.186% |

**FULL-SYSTEM DEV.** True minus wrong was −0.008 pp [−0.058,+0.043], below the registered +0.05. True minus frozen was −0.101 [−0.276,+0.083]. P04 and P10 changed +0.023 and −0.605 versus frozen. These new-person replays inserted motor probability before the actual count filter and causal personalization. Official cell means weight P04/P10 three times as heavily as other people; the first two columns weight people equally. The selected public people and repeated common panels do not establish a sealed or returning-person result.

**Independent physical test.** The endpoint here is held-epoch duration prediction, in squared error normalized by source outcome variance. Positive numbers mean lower error with the true spectral ratio. All twelve bipolar pairs used measured native electrodes, with 800 motor-imagery epochs from 20 PhysioNet people. These are study-specific associations of EEG statistics. There is no peripheral muscle recording in this test.

| Training → testing | Reduction vs wrong-person ratio [80%] | Reduction vs independently fitted low-band power alone [80%] |
|---|---:|---:|
| PROTEUS → PhysioNet | +.04702 [+.03002,+.06498] | +.00234 [−.00489,+.00957] |
| PhysioNet → PROTEUS | +.00192 [−.00771,+.01051] | −.00679 [−.01352,−.00062] |
| PhysioNet → held PhysioNet | +.01682 [+.01181,+.02218] | +.00887 [+.00277,+.01497] |
| PROTEUS → held PROTEUS | +.01209 [+.00343,+.02051] | +.00262 [−.00084,+.00600] |

Within-study slopes explained a small amount of duration variation on held PhysioNet people. For both target populations, swapping the within-study model for the cross-study model reduced its incremental performance above low-band power. The same-target-scale differences were −.00970 [−.01645,−.00253] on PhysioNet and −.00942 [−.01635,−.00298] on PROTEUS. The two studies differ in task and acquisition. The donor comparison alone overstates evidence that the ratio adds information above low-band power.

**Protocol.** Six frozen §410 five-person panels, source-only weighted least squares, unchanged amplitude features and kernel readout, no parameter tuning. Each row's donor was the nearest training row in standardized twelve-pair log 8–33 Hz power after excluding that person. Median twelve-dimensional distances varied from 1.658 to 2.950 across panels. Amplitude matching leaves residual differences; the ratio is an EEG proxy, with no verified physiological source. The full and whole comparisons resampled source people with their panel, development people across those same panels and orders within each family, 5,000 times with seed 413. Physical intervals resampled target people conditional on the fitted source panels.

**Validation.** v3 replay OK, real feature and fit smokes passed. Six actor smokes checked prefix, zero and stop. All 48 pinned order workers passed and their frozen predictions equaled §410 exactly. A scorer smoke recovered the official 76.2793% anchor and exact zero/constant paired effects. The first collection failed before reading scores because it used the hash of the predecessor's batch script for its scorer. Verified the published score digest against the repository and remote source; collection 3124563 passed on unchanged outputs. Fit revision 6f45123; scorer 0487c90. Jobs 3124446, 3124449–54, 3124461, 3124473 and 3124560/61/63.

[All official levels and per-person effects](nuisance413_result.json) · [Independent transfer losses](nuisance413_transfer.json) · [Within-study controls](nuisance413_within.json) · [Registered protocol](../../docs/LEDGER.md)
