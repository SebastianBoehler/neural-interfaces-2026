# Joint waveform and session decoder pilot (§373)

The experiment asks whether a waveform decoder can learn how much to trust previous windows from the session. Both arms start from the same A6 weights. A6 is an FBCNet trained on P04, P05, P06, P07, P08 and P10. The new fits use those same six people. The four evaluation people (P01, P02, P03, P09) are held out of the complete training pipeline and were exposed in earlier studies.

Both new models use the population normalization statistics and jointly fit the waveform body, classifier, item encoder and gated correction. A gate in [0,1] scales a bounded three-logit correction. The prefix arm receives the average of strictly earlier item embeddings. The matched static arm receives the current embedding in that position. Complete branches use more source people and different architectures. Their score indicates the development gap to replacement. The matched two-arm contrast isolates the changed training and inference recipe. The untouched causal A6 is a third reference.

The fixed pilot uses 64 common batches of 32 distinct windows from a single source session, with shuffled order and source cell/class weights. Evaluation uses eight pinned orders. Each person's two interface cells receive equal weight. The reported intervals resample people and orders within shuffle scheme. Four exposed people limit their interpretation. The source six include P04 and P10, so this experiment cannot estimate returning-person effects.

[Registration](../../docs/LEDGER.md) · [Joint training and scoring](../../studies/17nov/joint373/pilot.py) · [Frozen and JS control restoration](../../studies/17nov/joint373/controls.py) · [Original fixed-body comparator](causal_prefix359.md)

## Primary result

Job 3122960 completed in 50 seconds at 6 CPUs. The first job (3122958) stopped at an import collision before optimization. The repair checked the resolved module path. The rerun reproduced the untouched A6 reference of 60.7852% across all four people.

| Four-person development measure | Official cell BA | Prefix minus reference, pp (80% people/order) |
| --- | ---: | ---: |
| Joint prefix | 51.457% | — |
| Matched static | 50.389% | +1.068 [+0.899,+1.225] |
| Untouched causal A6 | 60.785% | −9.328 [−12.280,−6.334] |
| Frozen B2_pool complete chain | 81.216% | −29.759 [−33.613,−25.881] |
| JS B2_pool complete chain | 81.236% | −29.778 [−33.721,−25.837] |

The prefix effect against the trained static decoder was positive for all four people (+0.708, +1.302, +1.057, +1.203 pp in P01/P02/P03/P09 order). The whole decoder misses the replacement goal. The experiment changed native causal normalization to population statistics in both joint fits. This prevents attributing their deficit relative to A6 to fine-tuning alone. [Score record](joint373_result.json) · [Before-label proof](joint373_before_labels.json)

## Context controls at seed 374

Job 3122963 completed in 43 seconds. The prefix-trained model scored 51.991% against 50.290% for the matched static arm (+1.701 pp [+0.468,+2.902], three of four people up). Only the readout initialization and training-batch order changed from the primary comparison. The body initialization, 64 source batches and held people were shared.

With those same prefix-trained weights, replacing the real context with the same number of preceding P05 source items produced 52.434%. Real minus source donor was −0.443 pp [−5.961,+5.109]. A fixed source-population mean produced 51.105%; real minus population was +0.886 pp [−1.251,+3.029]. P09 gained 13.872 pp with real versus donor context while P02 lost 8.380 pp. This misses the preregistered condition for spending more on this session-specific estimator. The matched static comparison alone would have missed the donor result. [Context implementation](../../studies/17nov/joint373/context.py) · [Fixed-weight results](joint373_context_result.json) · [Before-label proof](joint373_context_before_labels.json)
