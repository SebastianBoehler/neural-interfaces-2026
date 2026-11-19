# F-C11 · Hard-target correction in the exact cap250 refit

Keep the submitted cap250 weights. The geometry arm scored **48.4954% FULL-SYSTEM EXTERNAL** and exact stock scored **48.4557%**. Geometry beat the matched edit rotation. The point gain against stock was small and changed sign across the two orders, so the registered replication and release conditions were unmet.

| Comparison on nine Scherer people | Difference, pp [paired 80%] | Order 0, pp | Order 1, pp |
|---|---:|---:|---:|
| Geometry vs stock | +0.0397 [−0.3571, +0.4398] | −0.3770 | +0.4563 |
| Geometry vs rotation | +0.4233 [+0.0661, +0.7837] | +0.3307 | +0.5159 |
| Rotation vs stock | −0.3836 [−0.6978, −0.0728] | −0.7077 | −0.0595 |

The live hook changed 751 of 11,480 hard-target exposures (6.54%) across 356 real refits. There were 272 refits with a different edit placement under rotation. The geometry-edited rows had 319 correct original targets and 239 correct edited targets, a −10.65 pp change. The matched rotation had 340 and 194 correct targets, a −19.44 pp change. The two-view geometry placed the fixed types of edits better than the confidence-ranked rotation. Its selected replacements still reduced training-target correctness. These are refit exposures, so the same earlier trial can appear several times.

Geometry changed 97 whole-system labels relative to stock, correcting 36 errors and introducing 34. Geometry and rotation produced different labels on 115 deliveries. The median warmed session-time ratios were 1.008× and 1.013× respectively. Both measured timing ratios use the A100 experiment. [Paired results and target diagnostics](cap250-c11-result.json)

The worker reproduced all 4,260 archived exact stock predictions and original inner labels, and preserved all route, neural refit and personalization work counters. Original refit targets and float32 weight hashes matched exactly across arms. The two added views read only past EEG covariance. The scorer opened labels after all decisions and refit traces were authenticated. The cohort was used in earlier experiments. Its estimates give no calibrated private score forecast.

Both real smoke tasks 3129894, all nine paired workers 3130025 and scorer 3130026 completed with exit 0. The score completed at 20:36:12 CET. Allocation times were 7m46s–8m28s on seven tasks and 21m22s/22m11s on the other two. Result SHA-256 `77b70067ac5963ba9e5698f79f1a10518796a010f2c8eaa0e2db3239a9d771bd`. Before-label receipt `8a00d0715bf58a8e93d4ad7852da7e817de05b0e871a30a2510f7cb52684dbc6`. [Source, preregistration and smoke](cap250-c11-queue.md)
