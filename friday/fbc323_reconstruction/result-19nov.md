# F21 electrode-reconstruction consistency

The consistency loss reduced source disagreement between the two reconstructions by **14.94% and 14.65%** at seeds 0 and 1. The unchanged held PROTEUS recordings showed no established gain from the loss. Bounded whole-decoder accuracy was **58.4175% and 58.9710%**, with both registered contrasts undecided. [Preregistration and correction](../../docs/LEDGER.md#f21--source-electrode-reconstruction-consistency-for-raw-s1c)

| Held-new WHOLE DEV balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Exact clean hard CE | 58.5458% | 59.1613% | 58.3882% | 58.8654% |
| Alternating reconstruction CE | 58.4415% | 58.7959% | 58.4210% | 58.7749% |
| Reconstruction CE and consistency | 58.2223% | 58.4175% | 58.6561% | 58.9710% |

| Paired contrast in percentage points [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded consistency minus alternating CE | −0.3784 [−0.8629, +0.1087] | +0.1961 [−0.1615, +0.4834] |
| Bounded consistency minus clean CE | −0.7438 [−1.4689, +0.0876] | +0.1056 [−0.8479, +1.1125] |
| Bounded alternating minus clean CE | −0.3654 [−0.9955, +0.3740] | −0.0905 [−0.9855, +0.9240] |
| Ordinary consistency minus alternating CE | −0.2192 [−0.6572, +0.2228] | +0.2351 [−0.0762, +0.5084] |
| Ordinary consistency minus clean CE | −0.3235 [−1.2187, +0.7934] | +0.2679 [−0.7034, +1.3951] |
| Ordinary alternating minus clean CE | −0.1043 [−0.8657, +0.8700] | +0.0328 [−0.9213, +1.1830] |

Bounded consistency-minus-alternating was −0.6048/+0.2069 pp for P04 and −0.2250/+0.9429 pp for P10. Consistency-minus-clean was −1.3052/−1.8621 pp for P04 and −1.4893/+1.2545 pp for P10. All comparisons are fit-excluded whole-decoder results with people and delivery orders resampled in paired draws.

The maps used the authenticated Scherer montage names and the official PROTEUS channel order. The 19 preserved channels were P2, Pz, CPz, Cz, P1, P3, P5, CP3, C3, P6, P4, CP4, C4, FC4, F4, F3, Fz, FC3 and FCz. The spherical spline and nearest-three inverse-square rules reconstructed the other 22 channels from those 19. Historical Scherer processing used its 30 native inputs. F21 used only the common 19 PROTEUS source columns and geometry. The two active arms each used 244,800 source row appearances per primary reconstruction. Their isolated opposite views produced detached distributions without changing the primary SessionBN buffers or RNG. The actual KL gradient reached the raw body and affine classifier.

Mean source Jensen–Shannon divergence was 0.007275/0.006274 after alternating CE and 0.006188/0.005355 after consistency. The diagnostic used 64 fixed source appearances per fit, evaluated as two independent full source-session replays. It measures agreement on the source reconstructions.

Ferranti 511318 recovered the clean hard state, then rejected the extra gradient hook fired by the first-batch diagnostic. The v2 guard accounted for that hook separately from optimizer steps. With the same fitting recipe and map bytes, 511320 passed in 12 seconds and 511321 completed all 60 fits. Galvani 3132879 passed the independent smoke in 35 seconds. Galvani 3132880 sealed all 20 held banks and reproduced all 84,960 hard-control decisions, 672 exact first-window agreements and 1,344 exact resets. Maximum prefix error was 2.089e−6. Collector 3132881 completed at 01:22:43 CET on 19 November 2026.

| Artifact in `ni26_friday_s1c_reconstruction_v2_20261119` | SHA-256 |
|---|---|
| `maps.npz` | `b4128c37dacfef9ddb38d960b314f21ee98715b57c1120ddb60e98d9a80f4da4` |
| `actor.sha256` | `7db88bb4f87dc57e96b7def1b71bc34b9123b68d95c694ffa2598ec6431a4469` |
| `score.sha256` | `df899b65d4510929979d747c2b000e79ce8971a59a6f35ba104a228ee3134aa3` |
| `before_labels.sha256` | `a60091409710c0d616fd14603988c3354206dba02787c347f2ffb59bac25d69c` |
| `whole_results.json` | `fe563b10da46916df384e1e823a2dc418eb60b187bcc780ccf6617f42efd7d76` |
