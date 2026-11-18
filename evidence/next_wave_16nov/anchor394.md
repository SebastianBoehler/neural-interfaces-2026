# §394 — Preserve original causal A6 decisions during source training

For the learned normalization model, matching the original causal A6 predictions during training improved held accuracy over both parameter anchoring and unanchored training in both seeds. The learned model still scored 60.315% and 59.497%, below untouched causal A6 at 60.785% and the complete JS control at 81.236%.

| Held contrast, percentage points | Seed 382, paired 80% interval | Seed 1382, paired 80% interval |
|---|---:|---:|
| Learned normalization, teacher − parameter anchor | +1.145 [+0.620,+1.693] | +0.484 [+0.027,+0.933] |
| Learned normalization, teacher − unanchored | +0.610 [+0.364,+0.873] | +1.372 [+0.719,+2.046] |
| Explicit normalization, teacher − parameter anchor | −0.171 [−0.617,+0.268] | −0.598 [−1.226,+0.010] |
| Explicit normalization, teacher − unanchored | +0.217 [+0.043,+0.396] | +1.303 [+0.908,+1.693] |

Each model trained on the same six people and matched 120-episode schedule. The teacher predicted from the immutable explicit k20 network during training. Each noninitial update matched both penalties to 35% of the supervised gradient norm. All ten H100 fits passed parameter, gradient, input and checkpoint checks. The two same-H100 unanchored seed-382 fits reproduced both historical official scores exactly.

On 24 prespecified source probe episodes, learned teacher/parameter KL to k20 was 0.0063/0.0057 for seed 382 and 0.0138/0.0140 for seed 1382. The source KL prediction did not hold for both seeds. The teacher-anchored models had lower source cross-entropy in both seeds. Repeated source episodes sample training participants. [Source diagnostics](anchor394_source.json)

Ferranti 509398 → 509399 trained all ten checkpoints. Galvani 3123773/74 evaluated P01/P02/P03/P09 across eight pinned orders, after sealing all fits. CPU actors verified historical k20 and population probabilities to 1e−6 and exact labels, causal prefixes and reset. Score 3123775 stopped before evaluation because the pinned summarizer could not import its sibling. Job 3123807 set the historical directory on PYTHONPATH, verified unchanged source and replay, and ran the original scorer successfully. Source fit archive 8be310f01637857c9d625e7df613cb7466ca2999fe2a2ee29b3f3cf46bf7d3ac. Raw score SHA-256 34fddfe515fb793d2df6285285d3e913fa9d9d0835632d940f3f22bb83b8c696. [Paired metrics and participant results](anchor394_result.json)

## §394b complete selector

The standalone teacher-anchoring gain did not transfer to the full B2_pool selector. Each refit used the member's probabilities to choose historical training rows among equally sized confidence bands. Every alternative exchanged the same number per class. All four held people, both member seeds and eight delivery orders passed the original full-logit, native-label, near-random, prefix and recognition checks.

Teacher−stock was −0.012 [−0.144,+0.118] and −0.023 [−0.186,+0.140] pp. Teacher−permuted teacher was −0.020 [−0.200,+0.155] and −0.042 [−0.294,+0.208]. Teacher−parameter was +0.024/+0.011 pp. Teacher−unanchored was −0.022/+0.022 pp. Across this exposed cohort, the matched controls leave little evidence that preserving waveform logits helps select better CP training rows. Jobs 3123834/35/36 passed. [All paired metrics](rank394b_result.json)
