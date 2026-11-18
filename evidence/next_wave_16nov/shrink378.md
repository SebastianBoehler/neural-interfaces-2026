# Source-estimated normalization strength (§378)

The session has unknown scale. Each causal variance estimate interpolates between a fixed source variance and the variance of delivered EEG activations. The original §332 variance-only decoder uses 20 prior windows for every latent feature. This study estimates a separate shrinkage weight per feature from training people, before receiving any held windows. [Original same-weight variance intervention](moments332.md) · [Original two-seed full-system controls](calibration339_full_handoff.md)

For one latent feature let the energy of a window be its mean squared activation about the original training mean. With equal weight per source session, estimate W as the average within-session sample variance of that energy. Estimate T as the mean squared deviation of each session's mean energy from the original source variance, less that session's sample variance divided by its number of windows. Truncate T at zero. After n delivered windows, weight their empirical first and second moments by nT/(W+nT). The center used for normalization stays fixed. Thus only scale changes. The effective prior count is W/T for T>0. The finite-sample subtraction assumes independent windows within a session. Dependencies and the uneven source-person session counts can affect this estimate.

| Arm | Inference |
| --- | --- |
| RF | Original frozen population normalization |
| RV | Original causal variance-only normalization with 20 prior windows |
| G | Common prior count, the geometric median of finite source estimates |
| K | Per-feature source-estimated prior count |
| P | Same estimates rotated one feature position |

Each of twenty fixed checkpoints excludes the held person from training. Each estimate uses only the nine other people's unlabeled activations. Both seeds produce predictions for the full held person on the eight pinned orders. RF and RV in the complete evaluator must reproduce the original §339 streams and final labels. Report the ten-person new-person routing and actual full system on six bases and B2_pool. Genuine returning decisions require a separate held-session producer.

The implementation checks the original A100's RV endpoint for **bitwise** agreement and the k20 analytical endpoint for identical decisions and at most 5e−5 posterior difference. On real P01 source data, the repaired A100 smoke reproduced k20 with zero probability error. P01 seed 0 estimated a finite median of 1.994 and range of 0.362–160.495 across latent features. These values estimate shrinkage strength from nine source people. They provide no held accuracy result.

The first CPU comparison failed the cross-device archived probability guard on eight of ten people. Two CPU folds finished and were excluded from the homogeneous A100 matrix. The first A100 smoke caught a missing GPU-to-host transfer before producing any predictions. [Source implementation](../../studies/18nov/shrink378/produce.py) · [Estimator](../../studies/18nov/shrink378/estimate.py) · [Registration and job record](../../docs/LEDGER.md)

All ten A100 producer folds completed, with 20 source-excluded checkpoints across both seeds. Every archived variance-only comparison passed bitwise. Across the twenty models, the median finite prior count was 1.53–2.98 windows. Individual feature estimates ranged from 0.247 to 703.79; every feature had a positive between-session estimate. The common-strength control tests whether this large shift in typical prior count, by itself, explains the effect of the learned feature assignments. The disjoint-future variance diagnostic (§378b) estimates scale error separately from classification accuracy.

## Prediction of future variance (§378b)

After observing 1, 8, 32 or 90 windows, each estimator predicted the variance of the next 32 windows. The future rows were disjoint and used only as diagnostic targets. We averaged the squared log ratio across features, prefix lengths, two seeds, source shuffle schemes, sessions and ten held people.

| Arm | Mean squared log ratio (lower is better) |
| --- | ---: |
| Common fitted prior G | 0.16251 |
| Per-feature K | 0.17689 |
| Rotated P | 0.18082 |
| Original 20-window prior RV | 0.28018 |

K reduced error by 36.87% versus RV (80% paired people/order interval +28.97%, +43.42%) and 2.17% versus rotated P (+0.10%, +4.33%). Both predictions held. G outperformed K at every tested prefix length; K's mean squared error was 8.85% higher than G's (+5.16%, +12.86%). The large effect comes from using a shorter typical prior. Correct feature assignment gives a smaller benefit relative to rotation, and the simple shared prior performs better still on this dispersion diagnostic. These are scale-prediction errors, not classification scores. [Diagnostic](../../studies/18nov/shrink378/mechanism.py) · [Paired report](shrink378_variance_result.json)

## Held classification and the complete consumer

The per-feature estimate improved standalone mean accuracy by 0.423 pp versus the original 20-window prior. It lost 0.155 pp against a rotated assignment of the same weights. Both paired 80% intervals span zero. On B2_pool, the registered primary full system, both comparisons were near zero.

| Contrast (pp) | Standalone decoder | Full B2_pool | Full six-base mean |
| --- | ---: | ---: | ---: |
| K − RV | +0.423 [−0.121, +1.066] | −0.017 [−0.150, +0.145] | −0.052 [−0.125, +0.029] |
| K − rotated P | −0.155 [−0.440, +0.201] | +0.008 [−0.077, +0.096] | +0.044 [+0.003, +0.082] |
| K − shared G | −0.003 [−0.323, +0.289] | −0.023 [−0.130, +0.089] | +0.059 [−0.004, +0.115] |
| Shared G − RV | +0.427 [−0.004, +0.985] | +0.007 [−0.113, +0.143] | −0.111 [−0.184, −0.012] |
| RV − static RF | +4.089 [+2.340, +6.455] | −0.058 [−0.162, +0.073] | −0.069 [−0.133, −0.003] |

The full B2_pool control scored 76.2793%, static RF 76.2908%, fixed RV 76.2325%, shared G 76.2390%, per-feature K 76.2160% and rotated P 76.2079%. K changed the control by −0.0634 pp [−0.3134,+0.2405]. The K−RV effects were −0.2965/−0.0646 pp for P04/P10 under new-person routing. Their K−P effects were −0.1327/+0.0813 pp.

The six-base mean favors the correct feature assignment over a rotation by 0.044 pp. Its comparison with fixed RV remains negative and uncertain. The two registered primary comparisons missed their predicted gains and both intervals cross zero. The classification benefit of channel-specific weights remains undecided. The sizable gain in predicting future variance did not identify an improvement over the current full decoder.

All ten held people, two source-excluded seeds, and eight delivery orders completed. The 56 complete consumer jobs (3123106/3123107) consumed 21.545 CPU core-hours. The 3123108 scorer finished in 38 seconds, after replay OK, all 336 base/order/arm outputs and historical endpoint decisions passed. Inference used new-person routing throughout. [Complete person/order intervals and contrasts](shrink378_result.json)
