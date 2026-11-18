# §398 — Learn an affine correction inside the original causal normalization layer

The intervention keeps the pretrained spatial convolution, native k20 normalization and classifier frozen. A shared 7→32→2 network reads channel descriptors from normalized activations and adjusts gain and offset before activation. Its zero-initialized last layer reproduces the original waveform logits exactly. The prefix learner and current-only control have the same number of parameters and see the same native causal normalization.

Ferranti 509417 exercised the real original H100 source bundle and the prefix teacher, prefix parameter and current-only teacher paths. A node-affinity failure occurred before the first Python invocation. Slurm retried on mlcbm001 and the unchanged smoke passed. All eight H100 fits in 509418 completed. They used the same 120 source episodes per seed. On 24 separately seeded source episodes, the maximum logit shifts across arms ranged from 0.041 to 0.124. Both regularized arms applied 119 preservation gradients per fit at 35% of the supervised gradient norm. The historical k20 weights had the same digest before and after each fit. [Source probes and fit hashes](residual398_source.json)

Galvani actors 3123928 and 3123929 restore the original native CPU decoder. Each actor checks exact zero-adapter equality on every held order, historical probabilities within 1e−6, exact historical labels, causal prefixes and independent-session reset. Jobs 3123930 and 3123931 run the stock cap250 system with the member choosing which borderline historical rows enter personalization. The fixed comparison includes original k20, shuffled residual evidence, random near-cutoff selection and the stock selector.

## Held and full-system results

The prefix teacher-anchored residual scored 60.868% and 60.813% as a whole decoder. Untouched causal A6 scored 60.785%. The registered improvement and the benefit of using prefix descriptors remain undecided on these four held people. The complete JS comparison scored 81.236%.

| Contrast, official percentage points | Seed 382, paired 80% interval | Seed 1382, paired 80% interval |
|---|---:|---:|
| Whole prefix teacher − native k20 | +0.083 [−0.187,+0.382] | +0.028 [−0.402,+0.494] |
| Whole prefix teacher − current-only teacher | −0.015 [−0.187,+0.166] | −0.145 [−0.332,+0.055] |
| Whole prefix teacher − parameter anchor | −0.034 [−0.091,+0.022] | −0.033 [−0.131,+0.065] |
| Whole prefix teacher − unanchored | +0.048 [−0.214,+0.286] | −0.036 [−0.209,+0.123] |
| Complete selector, prefix teacher − stock | −0.036 [−0.158,+0.074] | −0.036 [−0.158,+0.074] |
| Complete selector, prefix teacher − permuted teacher | −0.087 [−0.269,+0.091] | −0.098 [−0.285,+0.087] |

At the selector, the residual changed 14 and 13 of the 14,630 retained training memberships relative to an original k20 member. Each seed changed exactly one final label, despite changing 45/69 standalone waveform decisions across the eight orders. On these data the selector rarely translated an active normalization correction into a different decision. The six source people trained the model and could not provide a separate held comparison. No lunch candidate arose. [Selection activity](residual398_activity.json)

All eight CPU full-system shards completed. Scorer 3123932 sealed actor and shard outputs before opening gold. Audit 3123934 reconstructed memberships without using labels. Raw result SHA-256 2b13f233ce5f634c97a24b1bc65e60c467a0b8274b61ceaffd26c4a0bf513cbd. [Complete metrics](residual398_result.json)
