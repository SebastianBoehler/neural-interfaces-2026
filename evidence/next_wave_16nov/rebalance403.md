# §403 — Transport the session correction to the stock decoder

For any pair of classes, using a common multiplier adds the same log multiplier ratio to both arms' log odds. The learned-minus-native log-odds difference stays fixed. Which rows cross the decision boundary can change because both arms move relative to zero; a third class can also become the winner. A sign change in aggregate accuracy alone cannot identify which rows caused it.

The registered diagnosis partitions §400b's signed mix-to-rebalance accuracy change by whether the native winner flipped. Each contribution keeps the original equal-cell/class weights. The two contributions therefore add to the whole measured shift.

The intervention leaves the stock cap250 mixture and its preceding 30-window history in place. When the stock winner survives rebalancing, multiply its rebalanced probabilities class by class by the learned/native mixture ratio, then normalize. On a stock winner flip, forward stock probabilities. EM, TSEM, the joint count filter and causal personalization process the resulting probabilities. The ratio has no fitted coefficient. Both branch predictions use the existing source-trained models. A donor replaces only the learned numerator, while a native numerator gives an identity check.

Ten previously studied new-person folds supply the 2,655 rows on each of eight shuffled orders. The pooled six-seed base is B2_pool. Returning-person performance and the sealed endpoint require separate evidence.

## Results

Native-winner flips contributed −0.460 pp [80% −0.742, −0.148] of the original −0.631 pp reversal, or 72.8%. Rows where the native winner stayed the same contributed the other −0.171 pp [−0.338, +0.043]. This supports the registered majority prediction. Selecting on the *stock* trajectory then gave these complete endpoints.

| B2_pool full-system new-person dev | Accuracy | Learned minus comparator, pp [paired 80%] |
|---|---:|---:|
| Learned correction | 76.1165% | — |
| Stock cap250, no new branch | 76.2793% | −0.1628 [−0.3199, −0.0098] |
| Native identity correction | 76.2793% | −0.1628 [−0.3199, −0.0098] |
| Matched source donor | 76.1774% | −0.0609 [−0.2357, +0.1187] |
| Archived native-added §395c | 76.2325% | −0.1160 [−0.3715, +0.1181] |

Against stock the learned change was −0.527 pp on P04 and +0.062 on P10. The registered improvement over stock is refuted on this reused cohort; the learned–donor difference remains undecided. All eight native identity endpoints reproduced stock decisions exactly and probabilities to 2e−6. Fold-specific stock routing remains the existing new-person emulation. Some archived upstream members retained recording-order preprocessing, as in §395c and §400.

Galvani snapshot `ni26_rebalance403_v1_20261118`. Smoke 3124089 and array 3124090 completed. Collector 3124091 selected the wrong same-named score module and stopped before computing metrics. Corrected collector 3124199 scored the original endpoints. The verified §400 replay log ended in OK. [Paired scores and person results](rebalance403_result.json)
