# Does robustness to task composition improve A6 session memory?

Four A6-initialized decoders used identical source people, current queries, prefix row pools and optimizer budgets. Two decoders saw disjoint prefixes with deliberately different task mixtures. Two decoders saw the same rows randomly repartitioned. Within each pair, one decoder trained on query cross-entropy (CE), and the other also trained to reduce the difference between centered logits on identical queries. That auxiliary gradient had 35% of the CE gradient norm. All models updated the spatial convolution, learned normalization and classifier. Only the training sampler used labels to form prefixes.

There were two seeds, 120 updates, two forward passes per update, and 24 class-balanced current queries per pass. At the first query the models had seen only the 24 manipulated prefix rows. Later queries also had the same preceding query rows in both passes. The two skewed prefixes shared one task and had a total variation distance of 0.667 between their task proportions. Random prefix partitions averaged 0.151 and 0.153. Source diagnostics used 24 fresh episodes sampled from the training sessions. They measure generalization across sampled rows and groupings within those sessions.

| Source diagnostic on skewed prefixes | Seed 382, matched CE → auxiliary | Seed 1382, matched CE → auxiliary |
|---|---:|---:|
| Mean squared centered-logit disagreement | 0.3029 → 0.0720 | 0.3254 → 0.0808 |
| Weighted query CE | 0.4161 → 0.4454 | 0.4112 → 0.4352 |
| Disagreement reduction | 76.2% | 75.2% |

The preregistration is in revision `bd59477` and the fit and evaluation code is in `dc5f26c`. Source-only Ferranti H100 smoke 509768 and array 509769 completed. The group gradients were nonzero, each trainable group changed, and frozen parameters retained their input hash. All eight checkpoints were locked before label-free held prediction.

## Four fit-excluded people, eight shuffled delivery orders

| INDIVIDUAL-MODEL contrast, percentage points | Seed 382, paired 80% interval | Seed 1382, paired 80% interval |
|---|---:|---:|
| Skew auxiliary minus skew CE | −2.104 [−3.123, −1.088] | −2.757 [−3.735, −1.714] |
| Random auxiliary minus random CE | −1.708 [−2.544, −0.873] | −2.654 [−3.573, −1.749] |
| Extra effect from using skewed partitions | −0.396 [−1.003, +0.236] | −0.103 [−0.757, +0.422] |
| Skew auxiliary genuine minus own blank | +2.952 [+1.381, +4.438] | +3.711 [+1.641, +5.722] |
| Skew auxiliary genuine minus matched-age donor | +3.078 [+2.162, +3.950] | +3.009 [+1.053, +4.956] |

All four people lost accuracy with the auxiliary objective at both seeds and under both partitions. Skew auxiliary scored 56.255% and 55.490% on the original held streams. Skew CE scored 58.359% and 58.246%. The original explicit A6 scored 60.785%, archived joint A6 scored 59.705% and 58.125%, and the frozen full stack scored 81.216%. The genuine prefix had value over another person's same-age prefix at both seeds, even after consistency training. The registered transfer benefit is refuted for this objective and strength. Whether task skew causes additional harm beyond random-partition consistency remains undecided. These four public people were previously examined. No returning-person or sealed improvement was tested.

Galvani CPU smoke 3125998 and three held actors 3126014 passed. Scorer 3126015 verified replay OK, native decisions and all three historical control means before producing the results. The paired intervals resample people and orders and weight the three delivery schemes equally. [Eight locked source fits](composition445_fits.json) and [held results with person-level contrasts](composition445_result.json).
