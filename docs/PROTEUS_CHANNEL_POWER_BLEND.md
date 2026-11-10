# PROTEUS source-fitted channel-power blend · 10 November 2026

Galvani job 3081056 finished in 5 seconds with exit 0. The source folds
selected **0%** auxiliary weight. The frozen replay reproduced the overlap
predictions exactly. The fixed 10% blend improved the combined held-out
score at every seed and reproduced job 3081051's scores and switch counts
exactly.

| J change versus overlap, balanced-accuracy points | Seed 0 | Seed 100 | Seed 200 | Mean |
|---|---:|---:|---:|---:|
| Source-fitted weight, 0% | 0.000 | 0.000 | 0.000 | 0.000 |
| Fixed 10% | +0.473 | +0.793 | +0.538 | +0.601 |

J is the equal mean of the overall changes in the two slices. The later
slice overlaps the all-unseen slice. The three seeds reuse the same
participants and windows.

| Slice | Scope | Source-fitted balanced accuracy | Fixed 10% balanced accuracy | Fixed minus source-fitted |
|---|---|---:|---:|---:|
| All unseen | Overall | 68.05% | 68.73% | +0.68 pp |
| All unseen | P10 | 78.43% | 78.94% | +0.51 pp |
| All unseen | BrainHero | 71.09% | 71.68% | +0.59 pp |
| All unseen | Graz | 65.01% | 65.78% | +0.77 pp |
| Later unseen | Overall | 65.55% | 66.07% | +0.52 pp |
| Later unseen | P10 | 81.37% | 81.69% | +0.33 pp |
| Later unseen | BrainHero | 68.16% | 68.86% | +0.70 pp |
| Later unseen | Graz | 62.93% | 63.28% | +0.35 pp |

The channel-power head uses five trace-normalized spectral bands and
the same shrinkage, standardization and multinomial logistic readout as
the diagonal control in job 3081051. The weight minimizes true-class log
loss over the linear probability blend, bounded to [0%, 20%]. The
objective gives equal mass to each seed, person, interface and class.
Every source prediction comes from a fold that excludes that participant
from the head's fit. Mean source log loss was 0.660919 at 0% and 0.678269
at 10%. The derivative at 0% was +0.136378. The convex objective therefore
selects the lower bound. The fixed 10% control's source balanced accuracy
also decreased by 0.245 points on average.

The fit used 1,323 session-1 windows from P01, P02, P03, P05, P06, P07
and P08. P04/P09/P10 entered inference and scoring only after the head and
weight were frozen. The all-unseen slice has 1,332 windows. The later-unseen
slice has 762 windows from P04 and P10. The same population supports both
the 10% comparison in job 3081051 and this replay. The source objective
failed to recover the gain seen on that reserved population. Both Codabench
uploads remain unused.

## Replay and provenance

[Weight-fitting driver](../scripts/fit_channel_power_blend.py) uses the
archived campaign readers, scorer, and three overlap traces in
`runs/rank1_8nov/runtime_review`. The campaign directory also retains the
Slurm script and input hashes. Run the driver from the project environment
with the campaign artifacts restored.

```bash
review=runs/rank1_8nov/runtime_review
wave="$review/proteus_channel_power_weight_wave"
control="$review/proteus_covariance_overlap_wave"
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/fit_channel_power_blend.py \
  "$review/actual_runner_wave/artifacts" "$review/actual_seed_wave/seed100" \
  "$review/actual_seed_wave/seed200" "$control/fit-3081051.json" \
  "$control/result-3081051.json" "$wave/fit-replay.json" "$wave/result-replay.json"
```

Job 3081054 stopped before reserved inference because float32 softmax
rounding tripped source probability validation. Job 3081056 restored the
float64 out-of-fold logits used by the previous replay and passed both
source and held-out parity checks.

Verified Galvani output SHA-256 hashes.

```text
812540e9dab25d3ee9758a2a92f8af3f4dc29b98c8f48eb39aed187b2b827ea5  fit-3081056.json
845c459d1bc3c5c188643b7219cb031f7af04470547477719419b77c468ee580  result-3081056.json
```
