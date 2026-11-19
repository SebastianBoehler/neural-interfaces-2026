# F35 · Source-person head contribution balance

Completed 19 November 2026 at 09:59:56 CET. The balance arm improved whole accuracy over exact hard in both seeds. Both preregistered bounded comparisons remain **undecided**. The mirror also improved over hard, with nearly the same gain at seed 1. This is standalone DEV on ten fit-excluded people, 28 official cells and eight deliveries. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact A100 hard | Balance | Mirror | Balance − hard, pp [80%] | Balance − mirror, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 58.6790% | 59.1088% | 58.8172% | +0.430 [−0.090, +0.849] | +0.292 [−0.014, +0.520] |
| Bounded | 1 | 59.1089% | 60.0328% | 59.9337% | +0.924 [+0.458, +1.406] | +0.099 [−0.241, +0.403] |
| Ordinary | 0 | 58.1123% | 58.9476% | 58.3430% | +0.835 [+0.329, +1.312] | +0.605 [+0.202, +0.964] |
| Ordinary | 1 | 58.7062% | 59.5056% | 59.6491% | +0.799 [+0.498, +1.143] | −0.144 [−0.507, +0.230] |

Bounded P04/P10 changes against hard were +1.614/+0.461 and −0.464/+1.656 pp. Bounded motor/subtraction/word changes were +0.852/+0.930/−0.493 and +0.225/+2.145/+0.402 pp. Both source tapes used the exact original order. They shifted loss mass on 13,090 and 13,074 original paired-person updates across folds. Mean absolute head-gradient imbalance on the anchor fell 58.35% and 57.90%. On the actual weighted trajectories, relative imbalance for balance was 59.31% and 58.93% below mirror, with the same fixed transfer amounts. The analytic versus autograd error was ≤3.576e−7. This diagnostic measures final-affine contributions. The shared waveform gradients still receive scalar loss reweighting.

Source `3135480/3135487`, independent CPU `3135486/3135488` and collector `3135489` passed. The native hard checkpoint bytes, zero-transfer smoke and 84,960 hard decisions reproduced F33a. There were 672 exact first windows, 1,344 exact resets and maximum posterior prefix drift 2.423e−6. The collector sealed all predictions before reading held labels.

| Artifact in `ni26_friday_s1c_balance_a100_20261119` | SHA-256 |
|---|---|
| Source | `b153df05e923c0b40795adaed413d8d024087e49e4b2be51445d81b6180db42e` |
| Scorer | `eaa05cb8ee29a01106c56cf5c17fd457ce607ab7426468f7e1e2046a7c7be6a4` |
| Blind bank | `1ebeb5737c35c1a7e83083bfd4979db87e5bd22c37fb96d87d8a3c90bf84c530` |
| Whole results | `d22fcd1b314ac37cc3e6ee42b53b580e124209c08d77550db5df6ea91790a5ee` |
