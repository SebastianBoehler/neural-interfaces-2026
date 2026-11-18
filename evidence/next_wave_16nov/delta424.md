# §424 Counterfactual history distillation on identical queries

Keeping session history improved mean accuracy across four fit-excluded PROTEUS people. Three of four improved at seed 382 and four of four at seed 1382. Matching the explicit teacher’s change in logits gave similar accuracy to matching teacher changes from different query windows. Both seeds scored below the matched label-only fit. On 26 Shin2018 people, the estimated benefit of intact history was positive but uncertain. Both preregistered superiority criteria remain unmet.

| Measurement, pp | Seed 382 | Seed 1382 |
|---|---:|---:|
| Held individual decoder, T / shifted / CE | 59.509 / 59.629 / 59.877 | 57.674 / 57.594 / 58.250 |
| Held T minus shifted, paired 80% | −0.120 [−0.317, +0.065] | +0.080 [−0.013, +0.180] |
| Held T minus CE | −0.368 [−1.064, +0.344] | −0.576 [−1.069, −0.084] |
| Held T minus own blank history | +5.704 [+3.443, +7.949] | +6.736 [+4.128, +9.047] |
| External 60-window word selectivity, T / shifted / CE | 12.692 / 12.628 / 11.667 | 12.564 / 12.372 / 12.949 |
| External T minus shifted, paired 80% | +0.064 [−0.192, +0.321] | +0.192 [−0.128, +0.513] |
| External T minus own blank history | +2.692 [−1.346, +6.731] | +1.667 [−2.372, +5.769] |

Original explicit A6 reached 60.785% on these four previously exposed people and eight orders. Frozen and JS full systems reached 81.216% and 81.236%. External selectivity measures the active-minus-fixation difference in WORD-minus-SUB responses. Original explicit A6 scored 10.577 pp on the same 26 people and two orders. Published v5.1b and staged cap250 scored 14.808 and 20.128 pp. Held intervals resample people and orders within shuffle schemes. External intervals resample people after averaging their two pinned orders, as in §418.

Each source example contained the same 24 query waveforms twice, once after a 24-window prefix from the same session and once after a different person's prefix. All optimizer arms saw identical supervised queries, weights and episodes. T matched the centered explicit k20 logit change per query. S shifted those targets by 12 positions. This preserves the episode’s mean class shift. C used weighted query cross-entropy only. The joint spatial, memory and head gradients and final weights differed between all three arms. On 24 independent draws from the source streams, teacher-change mean squared error for T/S/C was .004196/.004194/.010829 and .005829/.006242/.012493. Both auxiliary targets improved this source measure relative to CE. The data do not isolate a gain from matching each query to its own teacher change. Source probes reuse the fit people and rows.

Six source-only fits and smoke took 121 allocated H100 seconds on Ferranti (509596/509597). Code revision 2d51bb4. Galvani 3124746–48 locked all checkpoints, reproduced the original A6 and population controls across four people and eight orders, verified replay prefixes and independently reset each external 20-window block. The cn102 reservation would have delayed scoring approximately 12 hours, so the same partition supplied other nodes with the same numerical guards. Collector 3124749 passed replay and completed held scoring, then indexed a one-horizon package reference as a two-horizon record during external contrast aggregation. Corrected 3124849 kept all participant outputs and reproduced the historical external endpoints. These results concern whole individual decoders and the external task contrast. No model was inserted into the frozen package or submitted.

[Held scores](delta424_held.json) · [External scores](delta424_external.json) · [Six source fit records](delta424_fits.json) · [Held prediction hashes](delta424_held_commitments.json) · [External prediction hashes](delta424_external_commitments.json) · [Training implementation](../../studies/18nov/delta424/train.py)
