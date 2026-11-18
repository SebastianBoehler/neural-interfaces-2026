# Paired prefix consistency for learned session memory (§438)

On ten held-out people, regularizing the waveform classifier to agree across two independent subsets of earlier session windows scored **42.072%**. Matched cross-entropy (CE) training scored **41.639%**. Both seeds improved under the official cell weighting, but only five people improved after averaging the seeds. The preregistered effect of consistency and the value of genuine session history both remain undecided.

| INDIVIDUAL-MODEL comparison | Official Δ, pp (80% interval) | Equal-person Δ, pp (80% interval) | People up | P04 / P10 Δ, pp |
|---|---:|---:|---:|---:|
| Consistency − CE, genuine history | +0.434 (−0.171, +0.860) | +0.083 (−0.443, +0.588) | 5/10 | +1.465 / +1.154 |
| Consistency, genuine − zero history | −0.735 (−1.984, +0.534) | −0.696 (−2.054, +0.680) | 5/10 | +1.313 / −2.980 |
| Consistency, genuine − source donor | +1.054 (−0.159, +2.513) | +1.826 (+0.845, +2.747) | 8/10 | +1.480 / −3.229 |

| Training seed | CE, genuine | Consistency, genuine | Consistency, zero | Consistency, donor |
|---|---:|---:|---:|---:|
| 438 | 40.373% | 40.659% | 41.815% | 40.518% |
| 439 | 42.904% | 43.485% | 43.800% | 41.517% |

Across 360 updates, each arm learned the spatial and temporal waveform convolutions, a 48-dimensional write vector and a history-conditioned decision head. Each update shuffled a complete source session. Two independently sampled subsets contained exactly half of each query’s earlier rows, rounded up. Both arms shared row draws, subset draws, initialization, loss weights and forward budget. The consistency arm penalized the squared difference between centered class logits. Evaluation supplied every earlier write, with a fresh state at each session. The age input was identical for genuine, zero-content and source-donor histories. A donor used the longest eligible source session in pinned delivery order, cycling its rows if needed.

The consistency model emitted all three classes at both seeds. Across 20 fits the mean write-coordinate standard deviation ranged from 0.271 to 0.555 on the first 12 held rows. Replacing genuine writes with zeros changed the logits by 0.142–0.612 RMS in that diagnostic. These checks establish use of memory. They do not establish that its contents improved decisions. Each model passed streaming step, fresh-state, prefix and earlier-write permutation checks. This study did not quantify changes in person-specific overfitting directly.

Twenty fold/seed jobs trained 40 models on Galvani A100s. Each fit excluded one complete person, and only the nine source people's labels entered training. The collector verified all 20 prediction archives before opening evaluation labels, replay OK, the pinned orders and both full-system reference scores. Scoring covered 2,655 rows, 28 participant/session/interface cells and eight orders. Each seed was scored separately, then paired deltas were averaged before resampling people and orders. This average describes training seeds separately, without combining their predictions into an ensemble. The frozen full system scored 76.279% and JS scored 76.552%. Neither integrated this decoder. Returning-person and sealed effects were unmeasured.

Run source revision `8c2c8db`. Source manifest SHA-256 `0acd11c1d9e37832bc34606f188435190d0dab17468c2af271b3066389526f09`. Galvani smoke 3125296, array 3125297, collector 3125298. [Results](invariant438_result.json) · [Prediction commitments](invariant438_before.json) · [Training](../../studies/18nov/invariant438/experiment.py)
