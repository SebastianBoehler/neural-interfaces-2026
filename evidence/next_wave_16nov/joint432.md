# §432 Joint residual normalization in the A6 waveform decoder

The jointly trained residual scored 60.242% and 58.198% on four held people. Original explicit k20 scored 60.785% and beat R on every person at both seeds. The learned residual had an uncertain effect at seed 382 and reduced accuracy at seed 1382.

| Whole individual decoder comparison | Seed 382 | Seed 1382 |
|---|---:|---:|
| R / matched K, official % | 60.242 / 60.234 | 58.198 / 58.356 |
| R minus K, pp [paired 80%] | +0.007 [−0.098, +0.110] | −0.158 [−0.263, −0.065] |
| R minus same weights bypassing the residual | +0.026 [−0.101, +0.158] | −0.271 [−0.395, −0.146] |
| R minus untouched original k20 | −0.543 [−0.964, −0.141] | −2.588 [−3.335, −1.793] |
| R minus archived free-memory J | +0.537 [−0.114, +1.168] | +0.073 [−1.018, +1.151] |

The frozen full system scored 81.216% on these people. The matched K replicates the earlier explicit refit scores to the reported precision. Both primary residual comparisons refute a positive effect at seed 1382 and remain undecided at seed 382. The shared deficit against original k20 warrants examining supervised refit drift independently of the parameterization of memory.

R and K started from the original A6 waveform decoder with its explicit causal normalization (k20). R inserted the zero-initialized prefix-dependent affine from §398 at the output of that normalization. Both arms trained the spatial convolution, native affine and classifier. R additionally trained the residual. Six source people supplied all 120 shuffled, 48-window episodes per seed. Every gradient group changed. Bypassing the trained residual at the same R weights changed source-probe logits by up to 0.04784 and 0.04898. K had zero bypass effect. The analytic filterbank stayed fixed.

The four evaluation people P01/P02/P03/P09 were excluded from fitting but exposed during prior research. Each person received all eight pinned shuffled deliveries. The official metric balances classes within participant/session/interface cells. Paired 80% intervals resample people and orders within shuffle scheme. Both seeds share the same four people. Archived §382 J uses free learned memory. Seed 382 of that historical comparator ran on Galvani A100 and seed 1382 on Ferranti H100.

**Execution.** Source revision e288b09 plus the §432 scripts. Ferranti 509700 and 509707 failed initial bitwise guards. With the same decomposed operator path in R and K, smoke 509711 verified exact identity at the residual input/output. The decomposed output differed from the certified whole TorchScript by at most 4.77e-7, with identical labels. Job 509712 completed all four fits. Galvani's pinned cn102 had no spare cores, so the CPU smoke and actors used cn001. The eager actor requires bitwise whole-decoder identity at zero residual, unchanged original posteriors within 1e-6, identical original decisions and prefix consistency.

[Source-fit records](joint432_fits.json) · [Training](../../studies/18nov/joint432/train.py) · [Held replay](../../studies/18nov/joint432/actor.py)

Galvani CPU smoke 3125061 completed in 85 seconds. Jobs 3125062 produced the remaining three participant files in 77–79 seconds. Scorer 3125063 ran from 09:55:07 to 09:56:39 CET on 18 November. Historical references matched within 1e-9 percentage points and replay printed OK. [Paired score and all person-level contrasts](joint432_result.json) · [Prediction commitments](joint432_commitments.json)
