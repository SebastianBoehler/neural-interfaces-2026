# Lane TB · Team B in hardcb_safe

Leonardo R1 seed-0 refits used production-20261120-32a94ab7. Four fresh orders 71120–71123 tested ten held-new folds and P04/P10 visits 2–3. The 190-window pinned smoke and all-fold 71120 repeat changed zero labels. [Results and receipts](laneTB_blend.json)

| Blend | Held-new Δ pp [80%] | Returning Δ pp [80%] | Returning P04 / P10 | J Δ pp [80%] | Assumed 57/43 Δ pp |
|---|---:|---:|---:|---:|---:|
| 9 × 15% | +0.73 [+0.20, +1.20] | +1.23 [+0.80, +1.90] | +1.47 / +0.99 | +0.98 [+0.64, +1.31] | +0.88 |
| 9 × 25% | +0.86 [+0.19, +1.42] | +1.82 [+1.33, +2.55] | +2.13 / +1.51 | +1.34 [+0.95, +1.71] | +1.19 |
| 3 × 15% | +0.32 [-0.06, +0.65] | +0.93 [+0.50, +1.46] | +1.00 / +0.86 | +0.63 [+0.35, +0.89] | +0.57 |
| 3 × 25% | +0.43 [-0.16, +0.97] | +1.21 [+0.61, +2.09] | +1.35 / +1.07 | +0.82 [+0.41, +1.23] | +0.73 |

Paired intervals resample people and orders. The 57/43 mix with returning P04/P10 at 20/80 is unverified. Warm A100 adds 33.93 ms/window for hardcb and 7.82/3.54 for nine/three, projecting 3,394/3,308 s before +0.25/+0.34 s load against 3,600 s. The 9 × 25% ZIP includes the dead-session guard and passed exact CPU blend-zero mixer, step and labels on 379 windows (64997383). RTX PRO 4500 timing is unmeasured.
