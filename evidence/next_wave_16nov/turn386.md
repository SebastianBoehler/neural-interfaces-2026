# §386 and §386b — Does changing the historical training rows improve cap250?

The positive four-person §384 shuffle result did not replicate when every arm exchanged the same number of rows. Both seeds scored below stock on the six additional people. The paired intervals still admit small gains or losses.

## §386 matched turnover

The frozen B2_pool cap250 stock scored 76.2793% on all ten new-person replays. The table shows percentage-point differences in the official cell-balanced metric. Intervals are paired 80% intervals over people and delivery orders.

| Intervention and reference | Ten people | Six outside §384 | P04 | P10 |
|---|---:|---:|---:|---:|
| Fresh seed 386 − stock | −0.064 [−0.198, +0.095] | −0.120 | −0.457 | −0.101 |
| Fresh seed 1386 − stock | +0.002 [−0.147, +0.150] | −0.040 | −0.046 | −0.162 |
| Fresh − persistent, seed 386 | +0.001 [−0.133, +0.156] | −0.016 | −0.054 | −0.118 |
| Fresh − persistent, seed 1386 | −0.020 [−0.191, +0.137] | −0.018 | +0.275 | 0.000 |
| Geometry − stock | +0.079 [−0.123, +0.278] | +0.022 | −0.209 | −0.066 |
| Geometry − permuted geometry | +0.142 [−0.072, +0.358] | −0.003 | −0.088 | +0.127 |

Every alternative exchanged exactly 6,593 stock selections over 1,008 refits. Among rows available at consecutive refits, fresh draws changed membership 20,109/19,997 times versus 4,996/5,109 under persistent priorities. Stock changed membership 3,286 times. Turnover therefore changed as intended, while the score difference stayed small. The confidence-boundary arm beat the fresh arms by 0.089/0.022 points, also with intervals spanning zero.

Galvani ran smoke 3123492, core 3123493, geometry 3123494, collector 3123495. Exact CPU stock scores matched all 112 archived session/order trajectories. Native stock labels and pinned row permutations matched. The smoke checked session reset/prefix and recognition-stop behavior for all eight arms. Public labels were opened after all 40 worker outputs had been sealed. The six additional people are a transfer check for this question, drawn from a previously used development set.

Source snapshot SHA-256 `5fe10977cd9e510da3106442a0ffa6ddd94a08f87c137a7cc5654b9197f471b7`. Raw result SHA-256 `aaf5795a46496e729efe3766182b57b52065fc936d1e1d01dc17f9671cc63cf8`. [Full metrics and traces](turn386_result.json)

## §386b independent seeds and paired heads

Neither preregistered dose improved the complete stock endpoint. Averaging the fully fused log scores of heads built with fresh priorities from seeds 2386 and 3386 also failed to show a gain over persistent priorities. All four separate seed/dose estimates against stock were negative. Their intervals include zero.

| Fresh pair contrast | Ten people, pp | Six outside §384, pp | P04, pp | P10, pp |
|---|---:|---:|---:|---:|
| 1/16 − stock | −0.077 [−0.236, +0.119] | −0.147 | −0.564 | +0.006 |
| 1/4 − stock | −0.028 [−0.177, +0.135] | −0.013 | −0.273 | +0.006 |
| 1/16 − persistent pair | −0.096 [−0.237, +0.066] | −0.173 | −0.286 | −0.168 |
| 1/4 − persistent pair | −0.008 [−0.139, +0.121] | +0.020 | −0.109 | +0.111 |
| 1/4 − 1/16 | +0.049 [−0.112, +0.206] | +0.135 | +0.291 | 0.000 |

Control, schedule and label timing checks passed. Galvani jobs 3123537, 3123538 and 3123539 completed. Source SHA-256 `fa32ff5aec071ad8673a0505f6f89211181955a888d50ab9b0373350b6f233e0`. Raw result SHA-256 `99f047c1b708ba2e7bd601c7dca39669e2a0f581704d809286d2c6f79a3f9f95`. [All seed, dose and subgroup results](dose386b_result.json)

The §384 shuffled waveform term favored rows near the confidence cutoff. Uniform random row exchange covers a broader range of confidence values. The deterministic boundary arm tested nearest-cutoff swaps. Random selection within a bounded confidence band remains a separate mechanism.
