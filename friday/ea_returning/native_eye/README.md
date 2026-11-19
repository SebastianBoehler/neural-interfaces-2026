# D-EArE · Source-fold Eye on conditional returning full0

Both original §321 returning bundles load the same three Eye TorchScripts as the archived cap250 ZIP, byte for byte. These original traces were trained on all public windows. The new fit calls the exact owning Eye training function with one source-only data load, using the original architecture, no-mixup schedule and augmentation. Each fold includes its target's first visit and excludes that target's two later visits before features, running normalization or labels are made available. This follows §321's separate source folds. The other target's data remain present as source in each corresponding fold.

The native replay holds production Rest/Core2, classical/event priors and the old out-of-fold Eye and gallery calibration fixed. Both authentic baselines must reproduce exact decisions and posteriors. The only candidate change is the three source-trained Eye traces. The actual computed seven traces enter the same native Eye actor. Its three log probabilities pass to RouteB's actual EM evidence. The inner mixture, recognition, strict-past warm state and outer probability mixture must match exactly.

| Galvani job | Scope and immutable release |
| --- | --- |
| `3133765_0/1 → 3133766` | P04/P10 real source smoke, then all six 60-epoch source-only Eye fits. `ni26_friday_return_eye_fit_v1_20261119`, source `a2566e46…65dd`. |
| `3133784 → 3133785` | Both actual 31-window native prefixes, then all four held visits. `ni26_friday_return_eye_native_v1_20261119`, source `ae14ac08…b28f`. |
| `3133787 → 3133788` | Independent four-visit complete label-blind seal, then separate official metric. `ni26_friday_return_eye_score_v1_20261119`, source `143f56f4…4b41`. |

The real fit smokes passed at 03:41:21 CET for P04 and 03:43:10 CET for P10 on 19 November. Both made 36 actual AdamW updates, excluded their later target visits, and matched the source trace prefix and interleaved reset exactly. Receipts `b0cc3568…8b03` and `0f5bfd1e…9623`. The separate registered score applies only to 762 reused full0 target windows and eight official cells after completed, authenticated native visits.

```bash
ssh mlcloud-galvani 'sacct -X -j 3133765,3133766,3133784,3133785,3133787,3133788 -o JobID,State,Elapsed,Start,End,NodeList'
```

All six full source fits completed by 03:46:28 CET, with 2,160 real updates per fit. The native prefix jobs completed by 03:48:59 CET. They retained both exact original native baselines, the exact seven ocular traces, matching first-17 resets, fixed recognition ages, and original warm activity. Only the Eye evidence changed before the real consuming outer path.

| Native prefix | Eye log evidence max change | Terminal posterior max change | Decisions changed | Recognition age and warm steps |
| --- | ---: | ---: | ---: | --- |
| P04 visit 2, 31 rows | 6.907754 | 0.0741342 | 1 | 10 and 0 |
| P10 visit 2, 31 rows | 6.905282 | 0.0185562 | 0 | 28 and 20 |

Native smoke receipts `d2e8c1cc2dc20ca26ea0b7f71b135add256cc9274760b1d9b346f1b152ed2bd7` and `62e65e3a0f550eea12cd824833fc053bd64a26d7df957c54abc0f08b77b5012c`. All decision counts above precede scoring.
