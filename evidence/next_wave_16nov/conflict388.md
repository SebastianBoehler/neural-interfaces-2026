# Do gradients from harder source people improve held-person A6 decoding?

Weighting the harder source person reduced cross-entropy on the hardest training person for both seeds. Held accuracy moved in opposite directions. Projecting conflicting gradients changed held accuracy by +0.134 and −0.087 percentage points relative to the matched harder-person objective. Both 80% paired intervals included zero. The registered cross-seed transfer condition failed.

| Whole decoder, four held people × eight orders | Seed 382 accuracy (%) | Seed 1382 accuracy (%) |
| --- | ---: | ---: |
| Unchanged full JS system | 81.236 | 81.236 |
| Unchanged full frozen system | 81.216 | 81.216 |
| Original A6, native k20 | 60.785 | 60.785 |
| N, equally weighted pair | 59.679 | 59.884 |
| H, harder-person weight | 59.776 | 59.736 |
| X, harder-person weight and symmetric projection | 59.911 | 59.649 |
| §382 ordinary session-uniform joint learned A6 | 59.705 | 58.125 |
| §382 frozen-body F | 59.911 | 59.550 |
| §382 jointly trained native K | 60.234 | 58.356 |

| Contrast, pp [paired 80% interval] | Seed 382 | Seed 1382 |
| --- | ---: | ---: |
| X − H | +0.134 [−0.158,+0.441] | −0.087 [−0.229,+0.067] |
| H − N | +0.097 [−0.025,+0.239] | −0.148 [−0.294,−0.003] |
| X − N | +0.231 [−0.034,+0.498] | −0.235 [−0.426,−0.029] |
| N − §382 ordinary J | −0.026 [−0.244,+0.185] | +1.759 [+1.179,+2.318] |
| X − original A6 | −0.875 [−1.778,+0.032] | −1.136 [−2.448,+0.153] |
| X − full JS | −21.325 [−23.507,−19.162] | −21.586 [−24.224,−19.031] |

**Mechanism.** The previous §93 compared participant-balanced and GroupDRO objectives on conventional SessionBN nets. Here all three arms start from the same source-distilled A6 session operator. Each update sees two independent 48-window episodes from different source people. A common schedule samples every unordered person pair eight times. H weights the two gradients by the softmax of their detached cross-entropies. X removes the component opposing the other person's original gradient when their dot product is negative, then applies the same weights. All three arms train the original convolution, learned session operator and classifier. They have identical per-seed episodes, initial parameters, 120 updates, group learning rates and frozen filterbank. The §382 historical comparison also changes the number and distribution of episodes per update. Its effect cannot be assigned to person balancing alone. [H100 training kernel](../../studies/18nov/conflict388/train388.py)

**Source-to-held comparison.** P04 had the largest initial source CE (0.6503). Under N/H/X the independent source-episode CE was 0.6243/0.6181/0.6177 (382) and 0.6114/0.6055/0.6072 (1382). The H−N P04 changes met the registered 0.005 CE reduction both times. X projected 66/120 and 53/120 conflicting pairs. The larger source CE typically received about 53.4% of the pair weight. These checks establish that the intervention acted at training time. Four held people contributed equal official cell counts, and intervals resampled presentation order and people. P04 and P10 were training people; genuine returning performance is unmeasured. [Source-person CE](conflict388_source.json) · [All fit proofs](conflict388_fits.json) · [Full held contrasts and individual people](conflict388_result.json)

**Execution.** Ferranti H100 509376/509377 ran six fits and the source smoke in 89 allocated GPU-seconds. The source diagnostic took another 11 GPU-seconds (509384/509385). The Galvani CPU2 actors reproduced the historical A6 k20/static outputs exactly on all 32 person/order cells, and checked independent single-step and prefix agreement. P09 failed the strict original parity check on cn212 and succeeded with identical inputs and code on cn102 (3123581). Original scorer 3123582 passed replay and locked all predictions, then called the metric module directly and failed before calculating an accuracy. Corrected scorer 3123583 called its official score function, retained all raw predictions and reproduced every historical outcome. Source revision d89c46a, diagnostic 70278a7. [Held prediction commitments](conflict388_before_labels.json)
