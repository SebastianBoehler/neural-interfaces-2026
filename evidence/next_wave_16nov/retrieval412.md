# §412 — Supervised retrieval from earlier windows in a shuffled session

The trained metric did not establish a full-system gain. On four source-excluded people and eight delivery orders, both seeds slightly improved the native waveform average. The matched learner trained with permuted labels improved it more.

| Official development score or paired contrast, pp | Seed 412 | Seed 1412 |
|---|---:|---:|
| Whole A6 plus learned retrieval | 60.876% | 60.971% |
| Whole − original A6 (60.785%) | +0.091 [−0.165,+0.327] | +0.186 [−0.000,+0.360] |
| Whole real − destroyed supervision | −0.106 [−0.277,+0.050] | −0.101 [−0.304,+0.120] |
| Full frozen real − stock (81.216%) | −0.018 [−0.325,+0.252] | −0.042 [−0.284,+0.164] |
| Full JS real − stock (81.236%) | −0.087 [−0.193,+0.028] | −0.132 [−0.275,+0.017] |
| Full JS real − destroyed supervision | +0.046 [−0.075,+0.174] | −0.087 [−0.219,+0.037] |
| Full JS real − cyclic reassignment | −0.053 [−0.166,+0.062] | −0.108 [−0.317,+0.106] |
| Full JS real − source donor history | −0.068 [−0.158,+0.021] | −0.033 [−0.174,+0.110] |

Each fit trained a 32-dimensional metric on pairs from one source session at a time. Held inference applied this metric to a query and its strictly earlier windows. Their static A6 posteriors supplied votes. Each correction subtracts the uniform prefix vote. Reversing the first 14 windows preserved the 15th query's correction to 1e−6, including recomputation of waveform features and votes. Both seeds changed complete system decisions. The original A6 and complete frozen and JS endpoints reproduced their pinned official scores. The full paths used terminal log-posterior correction; upstream personalized training remained fixed.

Source sample order and initialization matched within each real/destroyed pair. Across the final eight training episodes the supervised loss averaged 3.430/3.339, versus 3.903/3.905 against the destroyed targets. This records optimization on source participants. Both training seeds scored the same four exposed people. Intervals resample people and orders and do not measure sealed transfer. Per-person effects, changed-label counts and every control are in the [paired results](retrieval412_result.json). The [source fit receipts](retrieval412_fits.json) pin input, schedule and checkpoint hashes.

Ferranti smoke 509548 and all four 509549 H100 fits passed (52 allocated GPU-seconds together). Galvani 3124487 passed v3 replay and the first held-order smoke. All four actors 3124488 passed. Scorer 3124489 stopped before labels because two pinned historical modules use the same import name. Corrected scoring job 3124562 consumed the original sealed actors and completed successfully. Source commits 36a4cd6 and 252db6d. Raw result SHA-256 373a24cc8ea6191bfffbe662391a940e79466a3d4ad39584f60a7dc17fae9c0c.
