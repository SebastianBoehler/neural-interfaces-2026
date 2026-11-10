# PROTEUS source experiments, 10 November 2026

Forty-eight Galvani runs, eight methods by six auxiliary bootstrap seeds
2400–2900. Each uses the three archived overlap seed offsets and the
same three participant-disjoint source folds. All seven source people
contribute only session 1. Keep P04, P09 and P10 outside fitting,
tuning, selection and scoring. Reconstruct fixed 10% from source
traces and require exact comparator parity.

| Kind | Method | Matched comparisons |
|---|---|---|
| Refinement | Compact Gaussian-rank covariance added to Hann and bounded lag energy | Same electrode pairs with Pearson correlations, and Hann plus lag energy alone |
| Refinement | Relative lag energy after removing broadband scale | Bounded raw energy, log energy, lag-slot permutation, and Hann alone |
| Refinement | Within-hemisphere Hann coherency | Same-width mirror and rewired feature banks, plus power |
| Refinement | Mid-band dense rank covariance | Ordinary correlations and equal-width outer-band rank and ordinary correlations |
| New | Three-point ordinal motif statistics | Matched shared Fourier phase randomization, and Hann plus lag energy alone |
| New | Three-frequency phase coupling | Matched phase-randomized signals and Hann alone |
| New | Spatial localization of dominant covariance modes | Eigenvalues only and channel-rewired localization |
| New | Nonlinear random Fourier readout | Same projection dimension, Gaussian draws and head regularization without cosine, and the original linear head |

The shared phase-randomization control preserves whole-window auto and
cross spectra. The bispectral estimator uses shorter tapered segments,
so their individual spectra need not be exactly preserved. The ordinal
test counts the six orderings of three samples rather than fitting a
sequence model. The spatial mode test is an electrode-level statistic,
not source localization.

Every arm replaces the same 10% probability vote. The frozen source
screen requires +0.8 percentage points overall, +0.5 BrainHero, no Graz
loss, +0.3 vs every designated matched control, and a positive overall
gain in five of six auxiliary seed averages. No held-out score or
Codabench upload is scheduled.

Slurm accepted array 3083051 (tasks 0–47, up to eight at once) and
summary job 3083052, which depends on successful completion of the array.
The parent rank/temporal fusion comparison is job 3083044.

All 48 array tasks and the summary job completed. The remote summary
matched a local aggregation of all checksum-verified results.

| Method | Overall | BrainHero | Graz | Smallest lead over a control |
|---|---:|---:|---:|---:|
| Compact rank | +0.645 | +0.778 | +0.513 | −0.040 |
| Relative lag energy | +0.616 | +0.381 | +0.852 | −0.069 |
| Within-hemisphere coherency | +0.127 | +0.030 | +0.223 | −0.484 |
| Mid-band rank covariance | +0.706 | +1.041 | +0.372 | +0.130 |
| Ordinal motifs | +0.074 | −0.207 | +0.355 | −0.611 |
| Bispectrum | +0.332 | +0.177 | +0.488 | −0.278 |
| Spatial covariance modes | +0.310 | +0.256 | +0.364 | −0.301 |
| Nonlinear interactions | +0.342 | +0.312 | +0.372 | −0.347 |

All figures are source percentage points versus fixed 10% except the
last column, which is relative to the strongest matched control. None
passed. Mid-band rank was positive across six of six auxiliary seed
averages. Follow-up 3083135 tests equal mid-band rank and Hann/lag
probabilities versus the two branches, same-band linear fusion, and
outer-band rank fusion on fresh seeds 3000–3500. The overall, BrainHero,
Graz and matched-control bars are unchanged. No protected scoring.

## Parent fusion result

3083044 completed successfully. The primary comparison is replacement
of the 10% vote. Values are source percentage point deltas from fixed 10%.

| Arm | Overall | BrainHero | Graz |
|---|---:|---:|---:|
| Rank fusion | +0.518 | +0.502 | +0.533 |
| Rank covariance alone | +0.403 | +0.546 | +0.260 |
| Hann with bounded lag energy alone | +0.551 | +0.259 | +0.843 |
| Ordinary covariance fusion | +0.506 | +0.528 | +0.484 |

Against rank covariance, Hann/lag and ordinary covariance fusion,
respectively, rank fusion won 2/4, 3/4 and 3/4 auxiliary seed averages on
BrainHero, and 3/4, 1/4 and 2/4 on Graz (one tie in the last comparison).
The secondary added-vote comparison lost 0.067 points on BrainHero
and gained 0.831 on Graz. No fusion advanced.

## Mid-band fusion and branch selector

3083135 completed with six auxiliary draws and exact fixed-10% parity.
The mid-band rank fusion averaged +0.509 points overall, +0.497 on
BrainHero and +0.521 on Graz. Mid-band rank alone gained +0.780 on
BrainHero and beat the fusion in all six averages. Hann plus lag energy
gained +0.893 on Graz and also beat it in all six. The fusion was close
to mid-band linear fusion and stronger than outer-band rank fusion.

3083136 trained a branch selector in reciprocal inner source folds.
The training target was which branch was correct on decisive
disagreements. The selector used branch probabilities and margins and
used the equal blend at intermediate confidence. It averaged +0.624
overall, versus +0.627 for equal fusion and +0.622 for a shuffled-target
selector. An oracle using the recorded interface gave +0.757 overall.
The oracle was a diagnostic. It was never available to the selector.
Neither run passed the source screen.

## Overnight source batch

Galvani accepted another 48 runs across the same three person-disjoint
source folds. Every method gets the same six fresh auxiliary draws
4200–4700 and three archived overlap seeds. The preregistered source
thresholds and 10% replacement match the earlier breadth batch.

| Kind | Method | Matched controls |
|---|---|---|
| Refinement | Smoothly bounded mid-band covariance | Ordinary and rank covariance, plus power |
| Refinement | Geometric mean of half-window lag energies | Arithmetic mean, full-window lag energy, and Hann |
| Refinement | Dense real Hann coherency at 8–20 Hz | Time-domain ordinary and rank correlations in the same bands, plus power |
| Refinement | Three class-pair readouts with reconciled log odds | Multinomial heads on the same views |
| New | Class templates in log covariance space | Ordinary covariance templates and rank logistic readout |
| New | Class-conditional Markov transitions on quantized EEG tiles | State histogram, shuffled transitions, and pooled logistic readout |
| New | Learned temporal patch network | Same network on shuffled patches and a pooled logistic head |
| New | Learned product interactions between rank and temporal views | Same-capacity additive head, within-cell shuffled partners, and a temporal-only head |

The first 16 tasks are array 3083524. Its two class-pair tasks used
the three-class fit guard and were rerun as 3083546 after switching
to a binary fit with the same regularizer. Array 3083525 becomes eligible
at 21:00 CET on 10 November, and array 3083526 at 04:30 CET on
11 November. Each array runs up to four tasks concurrently. Job
3083527 waits for the successful retries and later arrays, then verifies
all 48 result files. Slurm determines actual starts.

The experiment source and its frozen dependency files live under
`runs/rank1_8nov/runtime_review/`. The Galvani scripts require the
PROTEUS cache, source-only archived overlap predictions, and prior
fit receipt referenced in their input manifests. Those generated
inputs, Slurm logs, model weights, and outputs remain outside Git.
P04/P09/P10 were excluded from all fits and tuning. No upload was made.

## Sealed board

Verified on Codabench on 10 November. Submission 1040847 was sixth,
with recorded score 0.8240266154 and displayed score 0.824.
The top three displayed 0.838, 0.834 and 0.833. Approximately +0.9 sealed
points is needed to match the current third-place score and +1.4 to
match the leader. The source runs do not give a sealed score estimate.

[Codabench sealed results](https://www.codabench.org/competitions/17982/#/results-tab)

- [Bandt and Pompe, ordinal patterns](https://doi.org/10.1103/PhysRevLett.88.174102)
- [Nikias and Raghuveer, bispectrum estimation](https://doi.org/10.1109/PROC.1987.13824)
