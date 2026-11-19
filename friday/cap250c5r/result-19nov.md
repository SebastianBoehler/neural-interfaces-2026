# F-C5r · Deferred covariance on the RTX 2080 Ti proxy

19 November 2026, 04:36:49 CET · Exact cap250 worker · No accuracy score

Deferred covariance preserved every same-input terminal score bit and every independent decision across both measured orders. It cut the unrecognized personalizer's measured stage time by **28.4% and 31.5%**. The warm entire worker was **1.80% and 0.64% faster**. The second order missed the registered 1% threshold, leaving the consistent worker benefit undecided. Per-window p95 increased in both orders. [Preregistration](../../docs/LEDGER.md#f-c5r--same-input-personalizer-parity-and-paired-rtx-worker-timing)

| Measured on Scherer G visit 1, 120 windows | Order 0 | Order 1 |
|---|---:|---:|
| Covariance calls, eager → deferred | 120 → 85 | 120 → 81 |
| Same-input personalizer wall time, seconds | 2.699 → 1.932 | 2.703 → 1.851 |
| Warm worker wall time, mean seconds | 33.396 → 32.793 | 31.070 → 30.872 |
| Deferred / eager mean worker wall time | 0.9820× | 0.9936× |
| Deferred / eager mean per-window p95 | 1.1248× | 1.1775× |
| Max independent stock-to-stock score delta | 1.1421e−5 | 5.7356e−7 |
| Max independent stock-to-deferred score delta | 9.2170e−6 | 5.1785e−7 |

Each timing block used the pinned Linux worker image and one RTX 2080 Ti allocation. Each compared four separate warmed `Solver` instances in alternating order. Both variants received the same canonical rows, preserved all archived decisions and performed the same real refits, count updates and CSP fusions. The independent score drift stayed inside the prespecified bound and below the same-block stock-to-stock drift. The two instances of the personalizer that consumed identical live inputs matched byte for byte. This is separate from F-C5a's original independent-process bitwise condition. [A100 attribution diagnostic](../cap250-c5a-result.md)

The public P10 visit 3 recognized 31-window prefix stopped at age 10. Deferred variants computed zero covariances instead of ten. Warm whole-worker ratios were 0.9542× and 0.9331× on this repeated prefix, and all independent scores were bitwise equal. The two timed orders reuse the same people. They provide neither a long-stream worker timing distribution nor an estimate of the official runtime limit.

Smoke `3133816` completed at 04:31:20 CET. Timing blocks `3133817_0/1` completed at 04:36:49/04:36:45 CET. The archives were mounted read-only and no task labels were opened. Remote release `ni26_friday_c5r_rtx_v1_20261119`.

| Receipt | SHA-256 |
|---|---|
| `source.sha256` | `cfdbd89da2f1a5a87abe11074f78ac66032d0bc659073f222a83e9bbe5aae2d7` |
| `smoke.json` | `4da9b1b595175a7134c28b23fc42d6642d5666461f740abb9d644b20e1643d32` |
| `profile_0.json` | `04e8a1162bfa8fd91cafbe597d9402ff3b7d34bb1f147e81c24f2451f0e571ef` |
| `profile_1.json` | `6ad9d58a7ca7b1daf2280599a0f40c091a69169e38891402b67cc7ca5eba6c42` |
