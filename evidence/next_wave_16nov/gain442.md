# Coherent electrode gains on an A6-initialized learned-memory decoder

Coherent electrode gains produced a small, unresolved change in held accuracy on both initializations. Genuine history beat a matched-age source donor by about seven points, including on the original clean training recipe. Untouched explicit A6 scored 60.785% and beat both fitted coherent models. The frozen full decoder scored 81.216% on these people.

| Individual model result | Seed 382 | Seed 1382 |
|---|---:|---:|
| Clean, real history | 59.705% | 58.125% |
| Coherent, real history | 59.796% | 58.154% |
| Independent, real history | 59.434% | 58.276% |
| Coherent − clean | +0.092 [−0.097, +0.288] | +0.029 [−0.280, +0.338] |
| Coherent − independent | +0.363 [−0.0004, +0.811] | −0.122 [−0.453, +0.218] |
| Coherent real − same-weights blank | +7.281 [+3.908, +10.814] | +5.560 [+2.911, +8.043] |
| Coherent real − same-weights donor | +7.088 [+4.052, +10.120] | +7.138 [+4.223, +9.989] |
| Coherent real − untouched explicit | −0.989 [−1.774, −0.135] | −2.631 [−4.031, −1.244] |

Both coherent-minus-clean intervals include zero. The coherent-minus-independent contrast was positive on all four people at seed 382, but P09 contributed +1.314 points and the other three contributed only +0.041 to +0.055. At seed 1382 the overall contrast reversed. These comparisons leave the transport of the §439 augmentation gain undecided. Coherent training increased the real-minus-blank contrast by +1.390 [+0.209,+2.637] and +0.267 [+0.025,+0.536] points. Blank accuracy fell on both seeds. Clean matched all 6,056 archived J decisions at each seed. [All arm and person scores](gain442_result.json).

## Method

We fitted the §382 A6 model on P04/05/06/07/08/10. All three arms started with the same filter bank, spatial convolution, learned normalization and classifier. The original filter bank stayed fixed. Each seed used the same 120 episodes of 48 same-session windows across conditions. Half the coherent episodes received one source-derived electrode gain vector throughout. Independent reassignment used exactly the same gain vectors and counts in each block of 12 episodes.

The A6 source preprocessing divides each electrode by one fixed scalar shared across electrodes. The source log-RMS difference between two sessions cancels that scalar. Applying the gain to these normalized waveforms therefore represents an electrode gain before the first model convolution. The GPU used only the exported source-six bundle. The evaluation used original EEG.

The same trained decoder saw the delivered prefix, the current window alone (blank), or an equal-length prefix from P04 session 1 (donor). Blank also resets the learned normalization's prefix length. Donor retains the true length, which separates session content from maturity. Each held person had one session. The scorer uses the official equal-interface/class measure with eight pinned delivery orders, balanced across three order schemes. The paired 80% intervals resample people and orders. All four people were previously examined.

## Execution

Code and pre-registration are in revision `a53412a`. Ferranti smoke 509758 and six H100 fits 509759 completed. Galvani CPU smoke 3125540 ran P01. Array 3125563 covered P02, P03 and P09. Job 3125564 printed replay OK and completed scoring. All actor prefix, donor, native, row and order guards passed. [Before-label actor hashes](gain442_before_labels.json). [Matched source fits](gain442_fits.json).
