# F-C9 · Previous-window agreement inside live adaptation

The adaptive scale scored **48.4689% FULL-SYSTEM EXTERNAL**, versus **48.7169%** for exact cap250. Both delivery-order point estimates were negative. The adaptive rule also scored 0.3208 percentage points below the matched fixed scale, with the entire paired interval below zero. The experiment missed its registered release condition.

| Comparison | Paired change, pp | Paired 80% interval, pp | Order 411, pp | Order 1411, pp |
|---|---:|---:|---:|---:|
| Adaptive vs stock | −0.2480 | [−0.6647, +0.1687] | −0.3241 | −0.1720 |
| Fixed 0.75 vs stock | +0.0728 | [−0.2579, +0.4034] | −0.1918 | +0.3373 |
| Adaptive vs fixed 0.75 | −0.3208 | [−0.5556, −0.1025] | −0.1323 | −0.5093 |

The fixed scale scored 48.7897%. Adaptive scaling changed 109 of 4,260 decisions versus stock, correcting 36 errors and introducing 47. Fixed scaling changed 89, with 36 corrections and 33 regressions. The median adaptive scale was 0.7332, with a 10th–90th percentile range of 0.5052–0.8478. The policy changed 14,085 of 14,160 live adapted-family corrections. This confirms activity at the intended in-chain mixing point. [Per-person estimates and checksums](cap250-c9-result.json)

All 4,260 stock labels and the original inner v5.1b labels and work counters for every arm matched the exact §411b archives. Both tested policies read only delivered windows. They propagated their changes through the actual downstream routing, count and personalization stages. The stock gain contrasts remain undecided. The fixed-control contrast refutes the registered positive adaptive direction on these nine previously examined Scherer people. [Exact stock preflight](cap250-readiness-18nov.md)

A100 smoke 3128678 and the nine-worker array 3128702 finished with exit 0. Eight workers took 5m40s–6m13s. One took 19m33s. CPU scorer 3128703 authenticated all decisions and completed at 18:13:35 CET on 18 November 2026. Result SHA-256 `152ab42dd2be99a07e6bba046660b7cb576cf57f1c81be563563b391ded73a5b`. Before-label decision receipt `5d284d6e083b502afb238f8a085afc424ff7aff0469c2ef6c9d35c11bebcfccb`. [Preregistration, source and smoke](cap250-c9-queue.md)
