# F8 · Synthetic covariance style on held people

**The covariance augmentation did not resolve a gain for the whole raw decoder.** The full-style arm improved on the variance-matched diagonal arm at seed 460 and trailed it at seed 1460. Both intervals crossed zero. Both terminal insertions trailed the complete historical cap250 stock.

| Genuine history, balanced accuracy | Seed 460 | Seed 1460 |
| --- | ---: | ---: |
| Whole F8 covariance style | 47.282% | 46.272% |
| Whole F8 diagonal style | 46.667% | 46.753% |
| Whole F7 unaugmented supervised | 47.097% | 47.277% |
| Full cap250 with F8 covariance actor | 73.056% | 72.918% |
| Full cap250 stock | 76.104% | 76.104% |

| Paired public development contrast | Seed 460, pp [80%] | Seed 1460, pp [80%] |
| --- | ---: | ---: |
| Whole covariance − diagonal | +0.615 [−0.047, +1.297] | −0.482 [−1.258, +0.289] |
| Whole covariance − unaugmented | +0.186 [−0.798, +1.267] | −1.005 [−2.454, +0.509] |
| Whole covariance genuine − blank | +0.026 [−0.583, +0.450] | +0.173 [−0.468, +0.725] |
| Full covariance − diagonal | −0.092 [−0.431, +0.278] | +0.038 [−0.291, +0.387] |
| Full covariance − stock | −3.048 [−4.173, −1.563] | −3.186 [−4.317, −1.553] |

The preregistered paired direction rule leaves all three augmentation predictions undecided. The stock comparison rules out a gain from this terminal insertion on these folds. The whole covariance−diagonal contrast changed 3,758 and 4,135 of 21,240 paired decisions. For held-new P04, the whole differences were +0.758 and +1.133 pp. For P10, they were −0.067 and −1.616 pp. [F8 preregistration](../../docs/LEDGER.md#f8--train-the-raw-decoder-with-synthetic-session-styles)

## Whole decoder contribution by delivery age

These terms add to the official cell-weighted differences above. The late term accounts for most of the seed-dependent reversal.

| Comparison and seed | Window 1 | Windows 2–10 | Windows 11–40 | Windows 41+ |
| --- | ---: | ---: | ---: | ---: |
| Covariance − diagonal, 460 | +0.010 | +0.027 | +0.120 | +0.458 |
| Covariance − diagonal, 1460 | +0.003 | +0.052 | −0.072 | −0.463 |
| Covariance − unaugmented, 460 | −0.004 | +0.030 | +0.002 | +0.157 |
| Covariance − unaugmented, 1460 | −0.013 | −0.011 | −0.029 | −0.952 |

Ferranti smoke 510134 and both 20-fit arrays 510135/510136 completed 0:0 in 284 allocated H100-seconds combined. Training checked active cross-channel mixing, variance matching on the fitted covariance, finite waveform changes and active optimizer updates. The held scorer authenticated exactly 300 transformed episodes per 600-step fit and excluded the evaluated person from all source rows. The actual evaluation windows came from the original, untransformed recording export.

Galvani preflight 3127482 reproduced the pinned eight orders and exact zero change. The held smokes 3127483/3127486 verified first-window, reset and active history checks. All four scorer jobs 3127484/85/87/88 and paired comparisons 3127489–92 completed 0:0. Each full scorer sealed its decisions before reading held labels. The four comparison files contain all ten participant effects and the eight per-order results. The same public PROTEUS participants supported earlier model choices. This experiment has no independent external, returning-route or sealed estimate.

The owning Galvani release is `ni26_friday_wave_style_20261118`. The training source revision is `f001ac4`. The independent scorer revision is `db8a334`.

| Artifact | SHA-256 |
| --- | --- |
| `actor.sha256` | `93f2dcaf3f64badc6735a40c175a777b2c11f65c1df1ac5f9a5f74c01d4b21a5` |
| `score.sha256` | `2c49a909264dc36c8184e83a8a0d16cb4c2a1f4d6a24f796c9917184b4d9d371` |
| `comparison_raw_style_full_460_vs_raw_style_diagonal.json` | `7573dc0c0ccd573ab142f322133a70baef7386de7e1f15c372775f00a09a9ca0` |
| `comparison_raw_style_full_1460_vs_raw_style_diagonal.json` | `4a10dfbcbf4d26936eed6bdec31a1cbbc6ed206cdde1499c9cdabb8998302aaf` |
| `comparison_raw_style_full_460_vs_raw_supervised.json` | `873102adda7fead2bec3cdcc2e01b979c2c03a5c3895155b7ca3c2c1d5aa3b8a` |
| `comparison_raw_style_full_1460_vs_raw_supervised.json` | `88d0c9fe0acbaa9baa446edfd7f068f0245be289995c0f02ad6cc6f1dbca5a4a` |
