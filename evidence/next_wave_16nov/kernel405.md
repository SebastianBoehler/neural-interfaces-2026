# §405 Whole-decoder spectral kernel learning

Learning kernel weights on held-source-person episodes reduced ten-person whole-decoder accuracy against uniform weights in both seeds. Neither decoder approached the frozen full stack. Ordinary full-stack LOSO traces could not supervise this experiment because other source folds included the outer evaluation person. The registered alternative tested learning a spectral metric.

| Whole decoder, official cell-balanced accuracy | Seed 405 | Seed 1405 |
|---|---:|---:|
| Frozen B2_pool | 76.279% | 76.279% |
| Frozen JS | 76.552% | 76.552% |
| Uniform kernels | 46.530% | 46.530% |
| Learned weights shifted two bands | 46.274% | 46.196% |
| Learned weights | 45.478% | 45.702% |

| Paired change in percentage points | Seed 405, official [80%] | Seed 1405, official [80%] |
|---|---:|---:|
| Learned − uniform | −1.052 [−1.641, −0.560] | −0.828 [−1.396, −0.298] |
| Learned − shifted | −0.796 [−1.392, −0.219] | −0.494 [−1.018, +0.153] |

Person-equal learned−uniform differences were −1.265 [−1.887, −0.650] and −0.943 [−1.572, −0.298]. Two and three of ten people improved. P04 lost 1.100/1.267 points and P10 gained 0.063/0.185 as new people. Both seeds refuted the registered uniform-weight advantage prediction for this learned decoder. The shifted-weight comparison was negative for the first seed and undecided for the replica.

The model used only the present EEG window at inference. Eight pinned delivery orders therefore produced identical kernel decisions while the frozen competitors could vary. Both seeds evaluated the same ten outer people and 28 official cells. The replica measures episode-seed sensitivity. It does not independently validate across people. No genuine returning-person or sealed measurement.

A100 smoke 3124213 passed. Both ten-fold arrays 3124214/3124215 and CPU replay/scorers 3124216/3124217 completed. Both replay guards printed OK. Independent NumPy/SciPy job 3124244 reproduced 190 P01 rows × three arms × two seeds × eight orders, zero decision differences. Source revisions and compact paired reports are in [the results](kernel405_result.json).
