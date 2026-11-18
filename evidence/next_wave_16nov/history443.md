# §443 Does a waveform decoder's session history improve the final decision?

Across ten held people, the complete frozen decoder scored 76.2793%. Adding the bounded genuine-history correction gave 76.3002% and 76.2812% for the two independent initializations. Both paired 80% intervals cross zero. This test provides no basis for changing the staged package.

| Historical B2_pool full-system dev contrast, pp | Seed 379 | Seed 1441 |
|---|---:|---:|
| Genuine history − frozen stock | +0.021 [−0.088, +0.120] | +0.002 [−0.138, +0.130] |
| Genuine history − donor | +0.122 [−0.022, +0.241] | +0.040 [−0.069, +0.139] |
| Genuine history − blank | +0.143 [−0.003, +0.318] | +0.053 [−0.089, +0.205] |
| 5% whole-model blend − frozen stock | −0.311 [−0.476, −0.096] | −0.242 [−0.402, −0.059] |
| Genuine history − whole-model blend | +0.332 [+0.043, +0.558] | +0.244 [−0.030, +0.483] |
| Genuine history − §318 JS stock | −0.083 [−0.197, +0.038] | +0.024 [−0.084, +0.146] |

The genuine-history correction changed 299 and 355 of 21,240 frozen-stock decisions across eight deliveries. On held P04/P10, changes were −0.083/+0.299 pp and −0.146/+0.361 pp. Both people were evaluated as new. The experiment has no returning-person folds or staged ZIP replay. The two initializations share the ten development people, source episodes, and delivery orders.

The actor subtracts its response to a fixed source-mean write at matched age from its response to actual session history. It centers the three logit differences, clips them to ±1, scales them by 0.10, and adds the result to the historical stock log posterior. Donor and blank use the same subtraction. Stock scores come from the complete §318 B2_pool cap250 archive, using native count-filter probabilities before age 90 and softmax of personalized scores thereafter. Metric weights match the 28 official development cells with equal weight for each of three order schemes. Paired intervals resample people and deliveries by scheme.

Galvani smoke 3125542 passed replay and all eight P01 orders. Arrays 3125544 and 3125546 restored all 20 original coherent checkpoints and reproduced all original genuine, donor and blank decisions. The second initialization also reproduced every §441 source-write decision. CPU jobs 3125545 and 3125547 verified both full-system stock branches, identity correction and pinned delivery order. Each job committed predictions before opening held labels. The registered two-seed support rule remains undecided. Source revision bc93cdd. [Seed 379 results](history443_379_result.json) · [Seed 1441 results](history443_1441_result.json)
