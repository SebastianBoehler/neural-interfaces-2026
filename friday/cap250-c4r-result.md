# F-C4r · Balanced neural refit weights on alternate delivery orders

Retain the staged v6.6cap250 ZIP. The balanced policy scored **48.4259% FULL-SYSTEM EXTERNAL** versus **48.7169%** for exact cap250 on all nine Scherer participants, two visits and §411b orders 411/1411.

| Delivery | Balanced minus stock, pp |
|---|---:|
| Both orders | −0.2910 [paired 80% −0.8102, +0.1984] |
| 411 | −0.5026 |
| 1411 | −0.0794 |

The original orders 0/1 had gained +0.053 and +0.185 pp respectively on the same people. The positive point estimates did not reproduce on the alternate orders. The pooled interval crosses zero, so the average positive direction remains undecided and the registered packaging condition was unmet. The balanced arm changed 116 decisions, with 38 corrections and 51 regressions. [Original F-C4 comparison](cap250-c4-result.md) · [Per-person replication and paired receipt](cap250-c4r-result.json)

The stock run reproduced all 4,260 archived labels. Both arms matched the original work counters on all 36 streams and the labels before the first neural refit. The original F-C4 hook reduced the mean dominant pseudo-class cross-entropy mass by 3.168 pp across all 356 refits. The difference is measured after the worker's float32 weight cast. These checks confirm the intervention was active at the real optimization step.

Smoke 3128879, all nine A100 tasks 3128892, and CPU scorer 3128893 completed with exit 0. Worker allocations took 4m58s–5m41s. The scorer verified both arms and the official metric, and finished at 18:42:37 CET on 18 November 2026. The before-label receipt is `d9878801f5d8fd7194c4f605c03d7feb7796eee1b364bbd9cc3ef1669f4b927b` and the result SHA-256 is `5a4e2ac3a7aaf7abdb6a7042d0663ff65d18891e8d9941e0e201d648eb0754dc`. [Registration, source and smoke](cap250-c4r-queue.md)
