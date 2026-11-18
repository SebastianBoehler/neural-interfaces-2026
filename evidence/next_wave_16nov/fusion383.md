# Does the learned A6 session history improve the full decoder? · §383

A 5% terminal contribution from the jointly trained A6 waveform model changed the full JS decoder’s accuracy by +0.002 and +0.045 percentage points on the two fixed training seeds. In both cases, a matched source-donor history scored about 0.076 pp above the genuine history. All primary paired intervals included zero. The within-A6 session benefit in §382 did not establish incremental value after the complete decoder. [Earlier standalone experiment](joint382.md)

| Full JS B2_pool contrast, 5% weight | Galvani seed 382, pp [80% interval] | Ferranti seed 1382, pp [80% interval] |
| --- | ---: | ---: |
| J real − unchanged JS | +0.002 [−0.151, +0.145] | +0.045 [−0.065, +0.164] |
| J real − J blank | −0.000 [−0.115, +0.119] | −0.033 [−0.122, +0.056] |
| J real − J donor | −0.076 [−0.217, +0.061] | −0.076 [−0.167, +0.008] |
| J real − matched native K | +0.034 [−0.042, +0.100] | +0.011 [−0.022, +0.043] |
| J real − matched frozen-body F | −0.022 [−0.098, +0.045] | +0.021 [−0.078, +0.109] |

The unchanged JS B2_pool endpoint scored 81.2355%. The real-history intervention changed 42 and 47 decisions, respectively, summed over the eight presentations. At the preregistered 10% sensitivity, it changed 94 and 93 decisions and scored −0.031 and −0.076 pp against unchanged JS. On the six-base mean, primary 5% real-history effects were −0.031 [−0.108,+0.040] and +0.003 [−0.055,+0.059] pp. On the frozen B2_pool branch they were −0.020 [−0.229,+0.170] and +0.078 [−0.086,+0.223] pp. Each seed shared source people, initialization, held people, and presentation orders. The replication changes the training episodes and device.

**Insertion and checks.** At each window, the full stack and a single added waveform model run independently. The intervention averages their current posteriors after the count filter and causal personalization. The original §382 reference archive retained the stack’s decisions, so the fusion reads verified §318 native scores. It preserves count-filter probabilities for the first 90 windows and converts the later combined log scores with a stable softmax. Original decisions and all age boundaries matched; zero weight exactly preserved class decisions on both branches, six bases, eight orders and all four people. Each actor bank matched its original prediction digest, split and canonical rows. Both 5% and 10% weights were fixed before fusion scoring. [Implementation](../../studies/18nov/fusion383/experiment.py)

Smoke 3123391 ran the replay and checked both seeds without held labels. Jobs 3123392_0 and 3123392_1 committed every decision before scoring, and 3123393 verified both completed reports. All jobs completed. [Galvani paired contrasts](fusion383_382_result.json) · [Ferranti paired contrasts](fusion383_1382_result.json) · [Galvani activation counts](fusion383_382_before_labels.json) · [Ferranti activation counts](fusion383_1382_before_labels.json) · [Registered decision rule](fusion383_result.json)

## Where the accuracy change went · §383b

With 5% real-history fusion, Galvani seed 382 corrected 19 wrong decisions, destroyed 19 right decisions, and changed four wrong decisions into other wrong decisions. The corresponding H100 counts were 22, 19, and six over the same eight deliveries. The official score gives cells and classes equal weight, so the weighted results differ slightly from simple counts.

| JS B2_pool, 5% J | Galvani seed 382 | H100 seed 1382 |
| --- | ---: | ---: |
| Accessible gain if every correction were kept (pp) | +0.307 [0.208, 0.414] | +0.340 [0.254, 0.426] |
| Loss from previously correct decisions (pp) | −0.305 | −0.295 |
| Corrections from age 0–89 / age 90+ (pp) | +0.208 / +0.099 | +0.241 / +0.099 |
| Losses from age 0–89 / age 90+ (pp) | −0.251 / −0.054 | −0.208 / −0.088 |

The lost decisions canceled 99.4% and 86.9% of the weighted corrections. Both registered predictions held. The perfect selector uses true labels; its result bounds this fixed intervention on these rows. A donor-history fusion offered a larger correction ceiling of +0.494/+0.493 pp but also lost −0.416/−0.373 pp. Each decomposed net matched the original official contrast, and rescoring the oracle matched the summed correctable mass per person and order. Smoke 3123400 and scorer 3123401 completed in 15 and 16 seconds. [Counts, intervals and age partitions](fusion383b_result.json)
