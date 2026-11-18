# §404 Cross-interface task geometry

Graz and BrainHero both occur during session 1 for each source person. The paired observations identify effects associated with the interface, including its feedback, tasks and run conditions. Both use the same scalp montage.

We trained all five classifiers on 1,898 official windows from P04/P05/P06/P07/P08/P10. Only session 1 supplied the contrast directions. Within each person/interface, we subtracted the equal-class mean from the three task centroids. The ordinary classifier uses the original whitened distances. The first intervention shrinks distance along the three leading Graz-minus-BrainHero directions. The companion amplifies directions with positive cross-interface covariance after also removing the population task contrasts. Wrong-person pairings preserve the class, interface and number of source observations.

## Source confirmation

When each person was excluded from feature standardization, PCA, whitening and estimation of directions, shared-direction projections increased the held person's cross-interface task-pattern cosine by **0.262** versus cyclic wrong-person controls (80% person bootstrap 0.211–0.314, 6/6 positive). Difference directions captured 0.0163 more of the held person's interface-shift energy (0.0098–0.0228, 5/6 positive). The six held source folds used labels only for the descriptive task-centroid measurement. An unlabeled test window does not reveal those centroids. Jobs 3124114/15. [Fold measurements and source manifests](align404c_source.json)

The earlier descriptive audit fitted unsupervised whitening to all six source people. Difference-energy advantage averaged 0.0132, below the 0.02 prediction. The projected shared cosine advantage averaged 0.288. [Difference fit](align404_source.json) · [Shared fit](align404b_source.json)

## Full-system evaluation

The evaluation covers 757 held windows from P01/P02/P03/P09, using the eight canonical delivery orders. Each intervention enters the fixed B2_pool decoder through a 10% mix of conditional subtraction/word probabilities. Motor-imagery probability is preserved upstream of the original count filter, causal personalized classifier and final fusion. Comparisons are paired in the same exposed public people and order families. The protocol does not estimate returning-person or sealed performance.
