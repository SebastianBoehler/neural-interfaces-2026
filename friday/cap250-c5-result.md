# F-C5 results — Defer unused personalized covariance work

The candidate reproduced all 27,336 continuous historical cap250 score vectors bitwise. It reduced median paired personalizer time by 17.0% in held-new streams and 79.9% in historical returning streams. These are CPU times for the personalizer inside an authenticated full-stack replay.

| Historical role | Exact vectors | Stock → deferred covariance calls | Median stage time ratio, paired 80% |
|---|---:|---:|---:|
| Held new, ten people × eight orders | 21,240 | 21,240 → 17,214 (−18.95%) | 0.8303× [0.8243, 0.8384] |
| Returning, P04/P10 × eight orders | 6,096 | 787 → 226 (−71.28%) | 0.2012× [0.1536, 0.2520] |

The full-system accuracy difference is exactly 0.000 pp on both historical roles. All selection lists, refit counts, fusions and recognition stops matched. The smoke confirmed 10-to-0 feature calls on an early-recognized session. A held-new prefix matched independently through the first live personalized outputs. The replay preserved all original refits, unlike §369. The largest association with speed was whether recognition arrived before the first refit.

The held-new call reduction fell 1.05 percentage points below the registered 20% advancement bar. Returning call savings and paired stage speed passed their bars. The exact staged A100 worker comparison remains unopened under this rule. The paired CPU times exclude upstream stages. End-to-end worker headroom and sealed timing remain unmeasured for this change.

Galvani smoke 3127921 finished in 3m07s. All eight paired jobs 3127930 passed in 3m06s–4m13s. Collector 3127931 finished in 1 s. Replay printed OK. Source commit `bb952f4`, manifest `3facf7f63b60d87a932fc3bbb39369086fffcf0014369568226338d105cb090f`. [Exact timing and input receipts by order](cap250-c5-result.json) · [Registration](cap250-c5.md)
