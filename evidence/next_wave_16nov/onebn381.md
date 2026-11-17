# Learning the useful A6 normalization inside the waveform decoder (§381)

The original A6 FBCNet has one normalization layer. Holding its entire waveform body and classifier fixed, earlier replays measured 60.785% with causal session normalization and 53.830% with population statistics on four body-excluded people. This study asks a narrower question than §379, whose normalization control had no measurable benefit. Can source-only supervision recover an operation already known to improve this particular complete waveform decoder? [Positive control](joint373.md) · [Freshly trained decoder](wave379.md)

The replacement acts at the original pre-activation site. It summarizes each channel's signed and absolute standardized activations, plus four learned thresholds on absolute activation. A shared nonlinear readout uses the prefix summaries to predict a log-gain and offset per feature. The original temporal pooling and classifier receive those transformed activations. Inference runs one waveform decoder with an explicit current-session state. No teacher logits enter its prediction.

Three checkpoints separate training signals. C has 360 classification updates. D has 320 updates to predict the original layer's causal affine transform. DC continues D for 40 classification updates. All use the same six original source people, fixed A6 weights, source-only data scale and seed-381 schedule. Every checkpoint runs with real history, with the current window alone, and with the same current window preceded by an equal number of source-donor representations. For each order the donor sequence is P04 session 1. The scored folds P01, P02, P03 and P09 remain wholly outside fitting.

The source A100 replay uses the original CPU2 waveform front end with explicit device selection. An exact split at the normalization layer verifies the original tail. The affine supervision reproduces the native teacher's source logits to 9.54e−7 on its smoke. Held inference uses the original CPU2 whole-network entry point with the replacement installed physically at the normalization layer. Checks include the actual static and causal archived probabilities, explicit one-step state updates, truncation, reset and permutation of a fixed preceding set. [Training](../../studies/18nov/onebn381/train.py) · [Held waveform actor](../../studies/18nov/onebn381/evaluate.py) · [Registration](../../docs/LEDGER.md)

The independent, label-free normalization audit (3123312) compared applied log-gain and shift to the A6 operator on fresh source episodes and every held person/order. Static, C, D and DC had mean held squared errors 0.030193, 2.631320, 0.002018 and 0.871487, respectively. D recovered 93.32% of the static-to-teacher affine error and outperformed both classification-trained checkpoints. On the six source people the same errors were 0.008968, 1.255415, 0.000424 and 0.376393. Source-only classification updates moved the estimated affine away from the verified normalization target. [Affine audit](onebn381_affine.json)

## Held decision results

| Whole waveform decoder | Real session BA | Current window only | Source donor | Real minus static population, pp (80% paired interval) |
| --- | ---: | ---: | ---: | ---: |
| C, source classification | 59.911% | 53.431% | 50.645% | +6.081 [+2.849,+9.063] |
| D, source affine distillation | 59.530% | 53.960% | 51.289% | +5.700 [+3.372,+7.992] |
| DC, distillation then classification | 59.623% | 53.700% | 51.368% | +5.793 [+2.311,+9.448] |
| Original A6 with k20 | 60.785% | — | — | +6.955 [+4.131,+9.801] |
| Original static population | 53.830% | — | — | Reference |

D recovered 82.0% of the original decision gain, above the registered 50% target. Its positive lower bound versus static and real-minus-donor +8.241 pp [+5.625,+10.796] meet the other operator-recovery conditions. Real minus blank was +5.570 pp [+3.312,+7.745]. D still lost to untouched A6 on every held person, −1.255 pp [−1.878,−0.642].

C achieved a comparable decision gain using source class labels alone. Its real-minus-blank effect was +6.480 pp [+3.808,+9.215] and real minus donor was +9.266 pp [+4.896,+13.768]. D minus C was −0.381 pp [−1.692,+0.943]. C's poor affine MSE demonstrates that reproducing the native operator's exact scale and offset was unnecessary for this classification effect. DC minus D was +0.093 pp [−1.973,+2.108].

These comparisons use the original A6 classifier and four previously exposed, body-excluded people. Frozen B2_pool scored 81.216% and JS 81.236% on the same rows. D minus JS was −21.705 pp [−23.567,−19.868]. This supports learned session adaptation in a real active layer without supporting replacement of the complete system. No returning-person or sealed score was measured.

All four held actors reproduced the original causal and static probabilities exactly. Smoke 3123301 plus paired A100 fits 3123302/3123303 consumed 66 allocated A100-seconds after the initial failed restore. CPU actors 3123304/3123305 took 87–105 seconds each; 3123306 completed scoring in 91 seconds. [Full results](onebn381_result.json) · [Before-label evidence](onebn381_before_labels.json)
