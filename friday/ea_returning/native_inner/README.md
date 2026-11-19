# D-EArN · Native returning mix and warm targets

19 November 2026 · Full0 returning delivery · Conditional source-fit package replay

The worker uses the actual §321 CPU `base` package for the selected `P04_s23` or `P10_s23` fold. The native TangentCore, PCA256, personal head, TSEM prior and own-space arrays match the authenticated D-EArU source fit. The primary P10 case includes 2,270 fit rows and 192 own visit-1 rows and holds 385 later-visit rows. Its live raw and Sinc FineTune blobs match D-EArW.

The two matched arms use the same D-EAr source fits. The recording arm feeds each group's recording-order mean to the native aligned TF-FBCNet pair and the five native aligned Shallow members. The causal arm feeds their causal full0 means. The model divides stored member weights by 1.875, giving the two groups total masses 1/30 and 1/15. The five TF and six Shallow source seeds are averaged before entering their differently sized native groups. The other members and their weights stay fixed.

The matched package retains §321 production rest, eye, the second TangentCore and other classical weights, global and event priors, and historical out-of-fold gallery calibration exposure. These results establish conditional worker behavior. The independent `returning_pair_cpu1` native zero supplies the terminal comparison. The D-EAr0 legacy zero and D-EArW's older conditional targets establish separate checks.

## P10 visit 2 smoke

Job `3132709` stopped before prediction because the preflight compared stored member weights to the solver's normalized weights. Commit `79a53489` applies the same normalization before the preflight check. The new immutable job `3132715` passed the 31-window smoke on the clean CPU SIF. The raw window sequence starts at global row 1238.

| Observation | P10 visit 2 |
| --- | ---: |
| Native zero terminal posterior and decision mismatches | 0/31 |
| Maximum causal minus recording inner mix change | 0.0134670 |
| Maximum weighted-delta arithmetic error | 1.63e−16 |
| First native positive route age, each arm | 28, P10 |
| Live refit age and optimizer work, each arm | 20, 5 steps per family |
| Actual warm prediction ages, each arm | 20–27 |
| Maximum strict-past posterior change | 0.0330641 |
| Changed target confidence weights and hard classes | 20/20 weights, 0/20 classes |

All four families used the actual native optimizer. The maximum matched warm changes were 0.00282118 (FB EA), 0.00186465 (FBLight), 0.00861472 (FB raw), and 0.00104089 (Sinc). The independently truncated 29-row prefixes and interleaved resets compared exactly. Independent readback verified all three native relabel tensors contain exactly the 20 strictly earlier global windows.

The successful source manifest SHA is `3f513cf5e0fd9f2db3cfdb15c9f09ae1d09e8bbbf9092b0f2e3176444fed8626`. Galvani release `ni26_friday_returning_native_inner_v2_20261119` contains:

| Read-only artifact | SHA-256 |
| --- | --- |
| `smoke/native_inner.json` | `7574a7386c38a324b7bb0d65cf86df704a71f4a6944698cc2c16f45a7416f07c` |
| `smoke/native_inner.npz` | `7b754c57959ac6305a58d6a9ce63873951cbbdfb48e37c59f61b7d4ad98a9f3c` |

The authenticated full-visit successor is array `3132716`. Collector `3132740` verifies the four source-bound tapes and all 762 held rows from the full0 permutation. No held gold or accuracy is used.

## Complete four-visit replay

The four `3132716` tasks and independent collector `3132740` completed at 00:41:34 CET on 19 November. All 762 native zero posteriors and decisions matched the original §321 reference exactly. The same four live recognition ages held and only P10 visit 2 refitted. Its strict-past target change and eight warm outputs per family reproduced the smoke.

| Session | Windows | First recognition age | Max native mix change | Final decision changes | Max final posterior change |
| --- | ---: | ---: | ---: | ---: | ---: |
| P04 visit 2 | 189 | 10 | 0.0416955 | 1 | 0.0582207 |
| P04 visit 3 | 188 | 10 | 0.0328508 | 2 | 0.0526357 |
| P10 visit 2 | 186 | 28 | 0.0263445 | 0 | 0.0362305 |
| P10 visit 3 | 199 | 10 | 0.0318012 | 1 | 0.0308195 |

The changes compare the causal EA convention with the matched recording convention. The four native decisions that differ carry no accuracy interpretation. The collector checked every pinned global row, hash and emitted class against its own posterior. The read-only aggregate SHA-256 is `1e656138b910827defc26de4cc7854b5735908d31bfea4d47d111c73f9675c64` at `ni26_friday_returning_native_inner_collect_v1_20261119/returning_native_inner.json`. The retained production and calibration exposures remain.
