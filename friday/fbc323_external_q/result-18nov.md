# F15eq same-weight fixed-feature ablation on Scherer

Zeroing the standardized 287 mean/power inputs improved balanced accuracy on Scherer for both frozen checkpoint types, both seeds and both evaluation normalizers. The primary bounded contrasts met the preregistered support rule. The replay changed only the fixed-feature contribution at inference, with all four all-PROTEUS F15e checkpoints held fixed. Scherer was excluded from training and reused as the nine-person external development cohort, with its two visits and full delivery orders 411 and 1411. [Preregistration](../../docs/LEDGER.md#f15eq--same-weight-fixed-feature-ablation-on-scherer)

| WHOLE EXTERNAL DEV | Seed 0 full q | Seed 0 zero q | Seed 1 full q | Seed 1 zero q |
|---|---:|---:|---:|---:|
| Original, ordinary | 42.0271% | 44.0377% | 40.4729% | 43.4888% |
| Original, bounded | 42.0503% | 44.7520% | 40.7937% | 44.2758% |
| Whole-feature dropout, ordinary | 41.8221% | 44.3353% | 40.7341% | 43.4028% |
| Whole-feature dropout, bounded | 42.1925% | 44.9173% | 40.5754% | 44.0972% |

| Zero q minus full q in percentage points [paired 80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Original, bounded | +2.7017 [+0.9521, +4.4114] | +3.4821 [+2.1528, +4.8251] |
| Dropout, bounded | +2.7249 [+1.3261, +4.1204] | +3.5218 [+2.3843, +4.6925] |
| Original, ordinary | +2.0106 [+0.5258, +3.4358] | +3.0159 [+1.6204, +4.4444] |
| Dropout, ordinary | +2.5132 [+1.3128, +3.7073] | +2.6687 [+1.4815, +3.8889] |

Every order-level contrast was positive. The separate F16e raw hard-CE actor scored 45.9392% and 44.5437% under bounded inference. Its residual point advantage over the corresponding F15e feature-zero dropout checkpoint was 1.0218 and 0.4464 pp. Those context figures compare different training objectives. [F16e raw-only evidence](../fbc323_distill_external/result-18nov.md)

The independent CPU replay recovered all 34,080 full-feature F15e archive decisions exactly. Each feature mode matched its native Factor control. The zero mode also matched the raw path that omits feature computation on real prefixes. Both normalizers passed 288 exact first-window agreements and 576 exact interleaved restarts. Maximum truncated posterior error was 1.638e−6. All 2,130 external labels matched the pinned native codes and cues after the decision seal. Intervals resample people, visits and orders in paired draws.

Galvani G smoke 3131605 passed in 17 seconds. All nine replay jobs in 3131606 passed, and collector 3131607 completed 18 November 2026 at 21:47:42 CET. The same-source F15 PROTEUS held-new and this all-source Scherer instrument have different cohorts and fit rosters. [F15 PROTEUS results](../fbc323_skip/result-18nov.md)

| Galvani release `ni26_friday_s1c_external_q_20261118` | SHA-256 |
|---|---|
| `score.sha256` | `dfc5bc5757b02e919e60a2de16acbc7d1ed040f32137a4dbc4be43c0d97f6ab2` |
| `before_labels.json` | `2a2d0ae56110d86ab5c2ddec7a11793e0d94034aebc15697aee04ae2f653e140` |
| `result.json` | `8cb971bf19c57b5f3733349912f1e342f85ba991c1102957b6b4f0cf297773a8` |
