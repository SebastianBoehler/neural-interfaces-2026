# PROTEUS paired covariance confirmation · 10 November 2026

Slurm array 3083806 was still pending when these comparisons were run locally
with the same frozen script, cached source EEG, archived overlap scores, and
participant-disjoint source folds. The local bridge reproduced every archived
overnight rank-logistic score and the bootstrap receipts for draws 4200/4300.
The eight fresh local draws are 6000–6700. These are not results from Slurm.

| Draws | Comparison | Overall | BrainHero | Graz |
|---|---|---:|---:|---:|
| 4200/4300 | 8–13 Hz Fisher minus rank logistic | +0.037 | −0.248 | +0.322 |
| 6000–6700 | 8–13 Hz Fisher minus rank logistic | +0.067 | −0.033 | +0.167 |

Values are cell-balanced accuracy percentage points. Fisher won on
BrainHero in 0/2 bridge draws and 4/8 fresh draws. It won on Graz in
2/2 and 5/8 respectively, with one fresh Graz tie. The unmodified Fisher
arm averaged +0.720 overall, +0.920 BrainHero and +0.521 Graz against
fixed 10% on fresh draws. It did not pass the frozen confirmation gate.
Capping its 8–13 Hz correlations at ±0.95 increased mean BrainHero to
+1.146 but reduced Graz to +0.223. This is evidence of an interface
trade-off, not evidence of a specific cortical or artifact mechanism.

The overnight arrays still had 16 of 48 results. 3083525 and 3083526
were waiting for their scheduled start times. 3083527 waits for the
remaining runs before aggregation.

Queued 3083835 to compare a single regularized classifier using Fisher
rank correlations at 8–13 Hz and matrix-log covariance edges at
13–25 Hz against Fisher plus raw rank, the reciprocal band swap, log
edges in both bands, and the archived rank-logistic control. Keep the
same power features and width across the four edge-based arms. Fit
on eight fresh auxiliary draws 6800–7500 with the original source
folds, three archived overlap seeds, and fixed 10% parity. The primary
hybrid must improve overall and on BrainHero, preserve Graz, and beat
all matched controls by the frozen margin. P04/P09/P10 stay out of
fitting, tuning, and scoring. No upload.
