# F19 raw S1C hierarchy crossed with jitter

Adding the hierarchy to matched jitter reduced bounded WHOLE DEV by 1.1077 and 0.6277 pp at seeds 0 and 1. Both paired intervals were below zero, refuting the registered improvement over jitter-only. Hierarchy alone and the combined arm against hard CE remain undecided across seeds. [Preregistration](../../docs/LEDGER.md#f19--jointly-learned-raw-s1c-hierarchy-crossed-with-matched-jitter)

| Whole-decoder balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Exact hard CE | 58.5458% | 59.1613% | 58.3882% | 58.8654% |
| Exact F17 jitter | 58.7817% | 59.4005% | 59.2789% | 59.9529% |
| Hierarchical head | 58.0117% | 58.2544% | 58.3272% | 58.6987% |
| Hierarchical head and jitter | 58.1026% | 58.2928% | 58.8166% | 59.3252% |

| Paired contrast in percentage points [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded hierarchy minus hard CE | −0.9070 [−1.7199, −0.2164] | −0.1667 [−0.8449, +0.4998] |
| Bounded hierarchy with jitter minus jitter | −1.1077 [−1.4090, −0.7468] | −0.6277 [−1.1900, −0.1244] |
| Bounded hierarchy with jitter minus hard CE | −0.8685 [−1.4421, −0.3623] | +0.4598 [−0.3244, +1.1516] |
| Bounded factorial interaction | −0.2008 [−0.8210, +0.5880] | −0.4610 [−0.9805, +0.0592] |
| Ordinary hierarchy minus hard CE | −0.5341 [−1.4196, +0.2452] | −0.0610 [−0.7293, +0.5821] |
| Ordinary hierarchy with jitter minus jitter | −0.6791 [−1.0854, −0.3007] | −0.4623 [−1.0816, +0.1212] |
| Ordinary hierarchy with jitter minus hard CE | −0.4433 [−1.0839, +0.0479] | +0.4284 [−0.3686, +1.1234] |
| Ordinary factorial interaction | −0.1449 [−0.8518, +0.6023] | −0.4013 [−0.9246, +0.1128] |

Under bounded inference, the hierarchy-plus-jitter effect versus jitter-only was −1.6412/−0.9562 pp for P04 and −1.1642/−0.0858 pp for P10. Its effect versus hard CE was −0.0874/+0.0288 pp for P04 and −1.0888/+1.5874 pp for P10. These are fit-excluded whole-decoder results with people and delivery orders resampled in paired draws.

The hierarchy uses 345,889 nominal trainable parameters versus 347,903 for flat, a 0.58% reduction. Both heads have two identifiable affine waveform contrasts. The hierarchy trained both logits and the shared waveform body with categorical NLL. Its maximum numerical difference from the independently weighted binary decomposition was 1.193e−7. Motor-only labels gave zero gradient to the conditional logit, with active motor and conditional gradients during training. Independent inspection recovered the exact initial tensors and verified changes to actual body and head parameters in all 40 trees. Both flat states and all 169,920 flat held decisions matched sealed F17 exactly. All arms used unmodified held EEG and causal ordinary or bounded history. The eight orders yielded 896 exact first-window agreements, 1,792 exact restarts and maximum prefix posterior error 2.385e−6. [F17 matched controls](../fbc323_phase/result-18nov.md)

Ferranti 511265 passed the source smoke in 10 seconds. Galvani 3132601 passed the independent held smoke in 55 seconds. Ferranti 511266 trained all 80 fits, and Galvani 3132602 replayed all 20 held banks. Galvani 3132622 passed the parameter proof before collector 3132603 completed at 23:59:08 CET on 18 November 2026.

| Artifact in `ni26_friday_s1c_hierarchy_20261118` | SHA-256 |
|---|---|
| `actor.sha256` | `4770494bdeb0cef1fd767d2359759657a72c54032faf7c26c62e0493c702496e` |
| `score.sha256` | `0a3bc0114d4d2a01cdf39e4d1bd71a68c3f400760444ea278fd82437d2cfc061` |
| `weightproof.json` | `00f4248c7cb3149a299ecb11f13d2e6b706bc8fe8b9eeea987d52c69ffb47881` |
| `before_labels.sha256` | `30756ac9e5faa04d07a9271dcb388969de0616885569e8a60b9e0254bc2661a9` |
| `whole_results.json` | `f504fad70c3e8df062e434163fc838d06a999d69df310c694f9f2a7f7fc2ff2b` |
