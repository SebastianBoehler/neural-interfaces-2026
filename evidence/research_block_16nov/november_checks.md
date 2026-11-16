# Additional November primary checks

Research only, 16 November 2026. These late checks broaden the source review; they do not introduce new runs or measured target gains.

## Trait/state hierarchy: a useful explanation, not a deployable normalization result

[Quantifying Trait-State Hierarchy in EEG Foundation Model and Signal Processing Embeddings](https://arxiv.org/html/2611.05652), arXiv v1 dated **3 November 2026**, examines whether trial embeddings cluster more strongly by person or task. It reports strong person structure in several frozen EEG embeddings, with task fine-tuning changing that organization. Its normalization experiment subtracts each person's mean embedding. The inspected method does not establish a strictly delivered-prefix estimate of that mean; its reported normalization gains therefore cannot be treated as streaming gains here.

This supports comparing task/style factorization and adaptation, but does not prove identity removal causes accuracy gains. Its hierarchy/accuracy correlation is not a causal intervention on our system. The 11 November [congruence-invariance paper](https://arxiv.org/html/2611.15204) provides a useful counterweight: removing identity and aligning class relationships are different measurements, and the metric can be invariant to the transformation being studied. Both sources argue for direct held-out-person classification checks alongside representation probes.

## LaBraM at Scale: adaptation and readout matter, not just parameter count

[LaBraM at Scale](https://arxiv.org/html/2611.03165), arXiv v1 dated **2 November 2026**, compares ten encoders from 5.8 million to 1.97 billion parameters and three pretraining collections. A 46.3-million-parameter checkpoint leads its mean full-fine-tuning comparison; frozen-probe and fine-tuning rankings differ. It also reports that decreasing pretraining loss can accompany worsening downstream performance, and that a larger language-model readout does not consistently help. These are useful counterexamples to “largest/latest encoder wins.”

The benchmark is not PROTEUS. Appendix C says seed-42 **test** scores select hyperparameters for replication, and combined linear-probe results choose the higher replicated test mean of two readouts. Appendix D removes known overlap sources but leaves some recording-level separation unverified. These qualifications limit interpretation as independent held-out generalization. No official ready-to-use checkpoint release or license was established in this check, and no download was made.

**Consequence:** frozen-probe failures do not close task-adapted models, and a good average benchmark does not select a competition model. Retain the broader scaling family in the solution map, but choose concrete local tests by task fit, independent controls, available artifacts, and worker cost. The above results do not justify a top-five forecast.

## Readiness versus novelty

LEAF, BrainPro, DEFT, SeeEEG and scaled LaBraM test different representation assumptions. They should not be counted as five independent chances of improvement: they can share training corpora, task splits, normalization advantages and failure modes. Prefer a portfolio with different mechanisms, while keeping matched heads, person splits and causal preprocessing fixed within each comparison. Source code observations identify reproduction work; they are not proof that the authors' published runs used a problematic default.
