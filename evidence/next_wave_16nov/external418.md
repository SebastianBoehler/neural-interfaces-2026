# Does learned A6 history transfer to external word generation?

On 26 Shin2018 participants, the A6-initialized decoder’s learned session memory changed word-task selectivity by −0.96 pp [80% −4.62,+2.56] for seed 382 and −1.03 pp [−4.68,+2.63] for seed 1382. Neither seed met the preregistered prediction of a positive lower bound. The intervals leave both gains and losses plausible.

| Complete model or release | Selectivity on 60-window sessions, pp |
|---|---:|
| A6 learned session memory, seeds 382 / 1382 | 10.83 / 11.28 |
| Same weights with blank history, seeds 382 / 1382 | 11.79 / 12.31 |
| A6 jointly trained explicit normalization, seeds 382 / 1382 | 10.83 / 12.95 |
| Original A6 with explicit normalization / population statistics | 10.58 / 11.03 |
| Complete v5.1b / cap250 archives | 14.81 / 20.13 |

Each whole decoder saw the same 60 true first-crop windows per person, in both §415 orders. An independent replay with additional resets after deliveries 20 and 40 reproduced every first-block label and both blank-history arms. The selectivity metric subtracts the change in SUB responses from the change in WORD responses between active and fixation trials. The measured package advantage is domain-specific. These estimates cannot be interpreted as official three-class balanced accuracy. [Task definition and package results](selectivity415.md)

The A6 source fits were restored from their original Galvani A100 and Ferranti H100 checkpoints. Only CPU inference and scoring ran for this study. The held public A6 experiment found a 5–6 pp benefit of real over blank history on the competition task. Carrying that finding to this external contrast was unsupported. The similar result for original explicit A6 versus population statistics (−0.45 pp [−3.59,+2.82]) also limits what this task says about which history mechanism is best. [Original A6 study](joint382.md) · [Full results](external418_result.json)

Initial smoke 3124602 failed on the historical fit-lock import before inference. The corrected smoke 3124605, 25 person jobs 3124606 and scorer 3124607 completed. Fit-lock, causal-prefix, pinned-order, matched-package endpoints and project replay checks passed. [Actor](../../studies/18nov/external418/worker.py) · [Scorer](../../studies/18nov/external418/score.py)
