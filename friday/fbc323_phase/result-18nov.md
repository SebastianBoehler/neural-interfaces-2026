# F17 matched waveform phase perturbation

F17 did not establish a benefit specific to dispersive phase augmentation. Bounded inference with phase augmentation reached 59.2684% and 59.9326% WHOLE DEV across seeds 0 and 1. A displacement-matched two-sample jitter reached 59.4005% and 59.9529%. Both augmentations improved the seed-1 estimate over the exact raw hard-CE control. The seed-0 intervals crossed zero. [Preregistration](../../docs/LEDGER.md#f17--dispersion-matched-common-phase-augmentation-for-raw-s1c)

| Whole-decoder balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Exact F16 hard CE | 58.5458% | 59.1613% | 58.3882% | 58.8654% |
| Rigid jitter | 58.7817% | 59.4005% | 59.2789% | 59.9529% |
| Dispersive phase | 58.8073% | 59.2684% | 59.3912% | 59.9326% |

| Paired contrast, percentage points [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded phase minus hard CE | +0.1070 [−0.4924, +0.5374] | +1.0672 [+0.4742, +1.5111] |
| Bounded phase minus jitter | −0.1322 [−0.5441, +0.2482] | −0.0203 [−0.4930, +0.3174] |
| Bounded jitter minus hard CE | +0.2392 [−0.3963, +0.7477] | +1.0875 [+0.5783, +1.5523] |
| Ordinary phase minus hard CE | +0.2614 [−0.3185, +0.6962] | +1.0030 [+0.3636, +1.4840] |
| Ordinary phase minus jitter | +0.0256 [−0.3861, +0.4055] | +0.1123 [−0.3290, +0.4462] |
| Ordinary jitter minus hard CE | +0.2358 [−0.3430, +0.6857] | +0.8907 [+0.4256, +1.2731] |

Both registered effects remain undecided. Bounded phase-minus-hard was +1.1639/+1.4488 pp for P04 and +0.6305/+2.4006 pp for P10. Bounded phase-minus-jitter was −0.3899/+0.4639 pp for P04 and +0.5551/+0.7274 pp for P10. These people were fit-excluded under the same held-new protocol. The paired intervals resample people and delivery orders.

Each active arm perturbed the same 489,600 of 1,958,400 source row appearances. The maximum per-row normalized squared displacement mismatch was 2.7073e−7. The maximum sampled relative error across all electrode-pair cross-spectra was 1.8207e−7, and maximum relative DC/Nyquist error was 5.0563e−8. Held scoring used unchanged input windows and no fixed-feature computation. The hard arm matched the F16 states and all 84,960 archived held decisions. The independent replay passed 672 exact first-window comparisons and 1,344 exact restarts. Maximum truncated-prefix probability error was 1.771e−6. [Exact F16 control](../fbc323_distill/result-18nov.md)

Ferranti 511174 passed the real source smoke in 12 seconds. Galvani 3132104 passed the independent held smoke in 37 seconds. The second smoke released the held Ferranti dependency and both full arrays completed. Source array 511175 produced 60 fitted actors. Galvani 3132105 produced 20 blind folds. Collector 3132106 completed at 22:54:51 CET on 18 November 2026.

| Artifact in `ni26_friday_s1c_phase_20261118` | SHA-256 |
|---|---|
| `actor.sha256` | `7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372` |
| `score.sha256` | `43424b1ad5311025d546b20c10b820d1ff65fadaf253fc51acbdae252506b498` |
| `before_labels.sha256` | `77a25afbba7061dde2c781f3e8fb810b49d653aa57d09dcbc4b5abc0bb2abb37` |
| `whole_results.json` | `9c2ef83eb850d671950be72024393c43c2b2e9952303d5f5bf0718b618e62916` |
