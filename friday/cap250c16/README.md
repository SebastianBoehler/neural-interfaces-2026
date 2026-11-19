# F-C16 · First-visit waveform prototypes in the native returning head

F-C16 replaces only the positive-ramp RouteB personal head. The package's native raw FBCNet seed 0 supplies the learned penultimate waveform vector. Both prototype heads use the same source standardizer, cosine concentration of eight, and class counts. The owner head uses that person's first public visit. The population head samples matched class counts across the nine other first visits. [Preregistration](../../docs/LEDGER.md#f-c16--frozen-waveform-prototypes-at-the-native-returning-head)

At each positive recognition ramp the adapter subtracts the original head's contribution and inserts the replacement at the actual 1/16 share and gain 3. The live RouteB tail consumes the resulting mixture. The original native personal head supplies the placebo. The adapter consumes the designated current waveform feature in one call and leaves all fitted geometry frozen.

## Authenticated source and smoke

Source fitter `3132877` completed at 01:12:23 CET on 19 November. P04/P10 used 1,152-coordinate penultimate features that reproduced their native member probabilities and 31-row first-visit prefixes bit for bit. Both new heads excluded all four measured visits. Their own and population class counts were 62/63/64 for P04 and 64/64/64 for P10.

Real smoke `3132907` completed at 01:19:09 CET. On P10 visit 2 it reproduced both the native §321 zero and the actual matched D-EArN causal-EA replay, on 39 delivered windows starting at row 1238. The placebo was bitwise native. All arms recognized at age 28 and kept the authentic zero-ramp gap at age 30. The actual age-20 refit ran 20 total optimizer steps and emitted through age 27 in every arm with identical strict-past targets and warm probabilities. The interleaved 29-row repeat passed exactly.

| Arm | Native positive-ramp rows | Max expected-to-consumed mixture error | Max final posterior movement | Final decisions changed |
|---|---:|---:|---:|---:|
| Placebo | 10 | 1.12e-16 | 0 | 0 |
| Population prototypes | 10 | 1.12e-16 | 0.07159 | 0 |
| Own first-visit prototypes | 10 | 1.12e-16 | 0.09310 | 2 |

No held labels were read by these jobs. The conditional package retains the original production rest, eye, core2, classical/event priors and historical gallery calibration. Array `3132916` runs all four visits at concurrency two, with an 18-minute limit per visit. Scorer `3132917` seals the four canonical outputs before reading measured labels.

| Artifact in `ni26_friday_c16_waveform_head_v1_20261119` | SHA-256 |
|---|---|
| `fit.sha256` | `37da6a192481d7636c754d4e7fb4d090929d5b570268ae971a0e0dfc98a68bec` |
| `replay.sha256` | `2850876c72bea3d0946bbd50eda2733e3f549044a42112077a09bcccb9e3f9d6` |
| `scoring.sha256` | `7e59b6dc1f2bf3e784a21ee41aca492f0b4f4cd174cf4e0b2f3acdd7c9896d49` |
| `fit/P04.npz` | `0cc03183f1d58a7dcf057684da98ee97c55ad418a607d148b2ddad3d58af22a7` |
| `fit/P10.npz` | `209696fbdd17327ad44970ce893d22cdc6db462293cd8cc60fc733e74da103be` |
| `smoke/result.npz` | `8c786729ca696fd47390c1766ee368d2f3810450d6455b624862053a72d86f51` |
| `smoke/result.json` | `e4f204f964f5a5fa98e05790c228a2c9fe531a92c647c1c995525d0838d0be6f` |
