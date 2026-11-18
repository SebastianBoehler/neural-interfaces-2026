# F15 whole-feature dropout in the original S1C decoder

Whole-vector dropout reached **59.9374% and 60.5966% held-new WHOLE DEV** with full fixed features and bounded evaluation. It beat deterministic feature scaling by **+0.9165 and +0.9208 pp**. Both paired intervals were positive. At zero-feature evaluation, dropout's advantage over a decoder trained without those features remains undecided across the two seeds.

| Source training | Ordinary, full q | Bounded, full q | Ordinary, zero q | Bounded, zero q |
| --- | ---: | ---: | ---: | ---: |
| Original | 58.9325 / 59.4776 | 59.2454 / 60.2583 | 57.4266 / 58.5591 | 57.6684 / 59.2137 |
| Zero q | — | — | 58.5458 / 58.3882 | 59.1613 / 58.8654 |
| Fixed 0.75 scale | 58.6651 / 59.2245 | 59.0209 / 59.6758 | 57.4041 / 58.5529 | 57.8056 / 59.0979 |
| Whole-vector dropout | 59.3990 / 60.1509 | **59.9374 / 60.5966** | 57.8174 / 58.9110 | 58.4405 / **59.6064** |

Absolute entries are official cell balanced accuracy percentages for seeds 0 / 1. Training with zero q leaves the fixed-feature columns without a loss gradient. Its evaluation uses zero q.

| Paired contrast, bounded evaluation | Seed 0, pp [80%] | Seed 1, pp [80%] |
| --- | ---: | ---: |
| Dropout − fixed scale at full q | +0.9165 [+0.5028, +1.2935] | +0.9208 [+0.5073, +1.3146] |
| Dropout − zero training at zero q | −0.7208 [−1.4146, −0.0988] | +0.7411 [+0.0473, +1.4569] |
| Dropout − original at full q | +0.6920 [+0.2909, +1.0392] | +0.3383 [−0.0627, +0.7329] |

At full q, dropout minus fixed scale was +1.7048/+1.6758 pp for P04 and +0.6091/+0.4955 pp for P10. At zero q, dropout minus zero training was +0.4879/−0.7827 pp for P04 and −0.9262/+1.7602 pp for P10. The intervals resample people and delivery orders. Fresh original q-on control states and both q-on prediction banks exactly reproduced the F14 same-platform control. [F14 comparison](../fbc323_train/result-18nov.md)

Four matched source arms covered ten held people and two seeds. The separate mask generator selected exactly 489,600 of 1,958,400 batch-row appearances in the dropout arm. It preserved the default CPU and CUDA RNG states. Zero q had an exact zero loss gradient in the skip columns. Original, fixed-scale and dropout training had maximum per-fit skip gradients of 0.562–1.056, 0.430–0.769 and 0.487–0.874, respectively. The independent CPU replay authenticated all 80 fitted states and 896 first-window comparisons across checkpoint/session/order groups. The maximum prefix posterior difference was 2.421e−6. Twenty held decision banks were sealed before opening labels.

Ferranti smoke 511043 passed in 11 s and array 511044 completed all 20 matched allocations in 36–48 s each. Galvani smoke 3130934 passed in 53 s. Independent held replay 3130935 and collector 3130936 completed on 18 November by 20:46:45 CET. Training release `~/ni26/friday_s1c_skip_20261118`, held release `ni26_friday_s1c_skip_20261118`, source `935cce6`.

| Artifact | SHA-256 |
| --- | --- |
| Ferranti actor manifest | `04ec8dd8e7593088a1b882e8fcf75cf3ac9bf67ff737c14c1903b70f97ea1be3` |
| Galvani scorer manifest | `9fbb2d8b80415c142219c776417a584d48c80451e94661b2d9201dd172394e2c` |
| Blind decisions | `92015e3dcc743708bb18f19ba090d5370645f3c184eeec5d635a5ccc83e54b7f` |
| Whole-model score | `eb06cd1fa8c52540706045850a4a76fa74113f7febe668e5229dc1a12984842e` |
