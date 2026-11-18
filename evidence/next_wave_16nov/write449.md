# Removing predicted task content from A6 memory writes

§445 penalized different predictions on identical queries after two different prefixes. This study instead changes only what the two frozen §382 J models retain from an earlier window. Their spatial filters, six write statistics, nonlinear memory reader and task classifier stay fixed. The original query and its contribution to the current normalization also stay fixed.

For each task we estimated a mean write in each of ten training sessions. We subtracted each session's equal-task mean, then averaged these contrasts across sessions. At inference the original population A6 classifier supplies three probabilities from that window alone. Their weighted sum of task contrasts is subtracted from the earlier write, preserving its original norm. The cyclic wrong-task control uses the same probabilities and exactly matches both write norm and edit distance. Both controls retain the same window count and mixing coefficient. The blank and same-age P04 donor controls test whether the effect depends on this person's previous windows.

## Did the intervention change task information?

| Source measurement | Seed 382 | Seed 1382 |
|---|---:|---:|
| Reduction in within-session task contrast energy, all ten source sessions used for prototypes | 17.42% | 17.45% |
| Reduction with the evaluated session excluded from prototype estimation | 9.34% | 9.35% |
| Reduction along transferred task-prototype directions, evaluated session excluded | 52.80% | 52.81% |

Energy is the sum of squared distances of the three task centers from their session's equal-task center, averaged over sessions. The wrong-task edits increased the corresponding total energy. The pooled estimate met the registered 10% reduction, while the stricter session-excluded estimate fell just below it. The classifiers had originally trained on these six people in both checks. Holding out a prototype session tests transfer of the write contrast across sessions within that training population.

The code appears in revision `3d18f67`. Ferranti H100 job 509792 ran a source smoke before computing prototypes from 1,898 source rows, using both original checkpoints. Independent Galvani source smoke and diagnostic 3126137 reproduced the prototypes. Galvani CPU smoke 3126133 and array 3126138 replayed P01/P02/P03/P09 across eight pinned shuffled deliveries. They reproduced historical population and explicit A6, plus genuine, blank and donor J decisions and probabilities. The semantic and wrong edits changed later logits with preserved first-query and blank outputs. Source and held row identifiers were disjoint and their hashes were checked before held scoring.

## Four held people and eight shuffled deliveries

| INDIVIDUAL-MODEL contrast in percentage points | Seed 382, paired 80% interval | Seed 1382, paired 80% interval |
|---|---:|---:|
| Semantic minus unchanged, genuine history | −3.479 [−5.751, −1.299] | −2.556 [−3.995, −1.205] |
| Semantic minus wrong task, genuine history | −4.206 [−6.853, −1.611] | −3.207 [−5.154, −1.257] |
| Semantic genuine minus same-age donor | +3.623 [+1.695, +5.572] | +2.984 [+0.690, +5.299] |
| Semantic genuine minus blank | +2.412 [+0.909, +3.926] | +2.736 [−0.111, +5.420] |
| Wrong task minus unchanged, genuine history | +0.727 [−0.635, +2.072] | +0.651 [−0.920, +2.287] |

Semantic subtraction reduced standalone accuracy from 59.705% to 56.226% and from 58.125% to 55.569%. P02, P03 and P09 lost accuracy at both seeds; P01 gained 0.451 and 0.195 points. Across the eight deliveries the semantic edit changed 637 and 609 decisions. The direction and size of the effect matter despite frozen query features and count weights. These data refute the registered transfer benefit for this edit. They establish that reducing task-linked write structure can hurt this frozen reader. They cannot distinguish useful task history from an input distribution shift induced by the edit.

The registered §449b incremental full-system comparison required a semantic advantage over the wrong-task control at both seeds and over unchanged at one. The experiment did not meet either condition. The original frozen historical full stack remained the reference at 81.216% and original explicit A6 at 60.785%. Scorer 3126139 passed replay, checked all six historical means per seed, then calculated equal-cell accuracy and intervals that resample both people and deliveries. Only these four previously examined public people were tested.

[Full scores and per-person contrasts](write449_result.json) · [Source intervention diagnostic](write449_diagnostic.json) · [Source data and checkpoint fingerprints](write449_source.json).
