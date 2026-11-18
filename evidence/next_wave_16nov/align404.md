# §404 Cross-interface task geometry

Graz and BrainHero both occur during session 1 for each source person. The paired observations identify effects associated with the interface, including its feedback, tasks and run conditions. Both use the same scalp montage.

We trained all five classifiers on 1,898 official windows from P04/P05/P06/P07/P08/P10. Only session 1 supplied the contrast directions. Within each person/interface, we subtracted the equal-class mean from the three task centroids. The ordinary classifier uses the original whitened distances. The first intervention shrinks distance along the three leading Graz-minus-BrainHero directions. The companion amplifies directions with positive cross-interface covariance after also removing the population task contrasts. Wrong-person pairings preserve the class, interface and number of source observations.

## Source confirmation

When each person was excluded from feature standardization, PCA, whitening and estimation of directions, shared-direction projections increased the held person's cross-interface task-pattern cosine by **0.262** versus cyclic wrong-person controls (80% person bootstrap 0.211–0.314, 6/6 positive). Difference directions captured 0.0163 more of the held person's interface-shift energy (0.0098–0.0228, 5/6 positive). The six held source folds used labels only for the descriptive task-centroid measurement. An unlabeled test window does not reveal those centroids. Jobs 3124114/15. [Fold measurements and source manifests](align404c_source.json)

The earlier descriptive audit fitted unsupervised whitening to all six source people. Difference-energy advantage averaged 0.0132, below the 0.02 prediction. The projected shared cosine advantage averaged 0.288. [Difference fit](align404_source.json) · [Shared fit](align404b_source.json)

## Full-system evaluation

The evaluation covers 757 held windows from P01/P02/P03/P09, using the eight canonical delivery orders. Each intervention enters the fixed B2_pool decoder through a 10% mix of conditional subtraction/word probabilities. Motor-imagery probability is preserved upstream of the original count filter, causal personalized classifier and final fusion. Comparisons are paired in the same exposed public people and order families. The protocol does not estimate returning-person or sealed performance.

| Difference in cell-balanced accuracy, pp (80% paired) | Shrink interface differences | Amplify person-specific shared directions |
|---|---:|---:|
| Whole classifier vs ordinary | +0.391 [-0.260, +1.042] | -0.134 [-0.790, +0.521] |
| Whole vs donor +1 | -0.803 [-1.596, -0.009] | -0.547 [-0.820, -0.274] |
| Whole vs donor +3 | -0.516 [-1.167, +0.135] | -0.265 [-0.790, +0.260] |
| Full B2_pool vs ordinary insertion | +0.001 [-0.043, +0.048] | -0.066 [-0.141, +0.010] |
| Full vs donor +1 | -0.090 [-0.184, -0.001] | -0.055 [-0.112, +0.000] |
| Full vs donor +3 | -0.133 [-0.228, -0.042] | -0.077 [-0.153, -0.010] |
| Full vs opposite intervention | +0.012 [-0.087, +0.119] | -0.145 [-0.282, -0.013] |
| Full vs frozen stock | -0.312 [-0.719, +0.031] | -0.379 [-0.743, -0.073] |

The genuine difference-shrink and shared-amplification models scored 41.975% and 41.450% alone. The ordinary classifier scored 41.585%. Full-system scores were 80.904% and 80.837%, versus 80.903% for ordinary insertion and 81.216% frozen stock. Against stock, cognitive-class recall on Graz/BrainHero changed by +0.000/−1.201 pp for shrinking differences and −0.163/−1.368 pp for amplifying shared directions. Neither intervention met its transfer criterion.

Galvani CPU jobs 3124097–3124101 and 3124102–3124106 completed. Both smoke runs passed the v3 replay, order reconstruction, zero change and prefix guards. Whole and full ordinary predictions matched between studies bitwise. Primary source at c06daf6, companion source at 9232ce7, fully excluded confirmation at 83de532. [Difference results](align404_result.json) · [Shared results](align404b_result.json)
