# Does the waveform decoder benefit from the held person's memory content?

After changing only the decoder initialization and repeating all training, coherent augmentation raised the learned decoder from 45.190% to 45.629% on the same ten held people. The paired gain was +0.439 pp [80% −0.865,+2.171]. Coherent minus independently shuffled gains was +0.307 pp [−0.736,+1.495]. Both estimates had the same sign as the first initialization. Neither interval resolved an improvement in this replication.

| Prefix supplied to the same coherently trained weights | Official development accuracy |
|---|---:|
| Correct session and age | 45.629% |
| Zero write at the correct age | 41.926% |
| Mean write from nine source people’s sessions at the correct age | 45.773% |
| Actual earlier windows from a source session at the correct age | 46.381% |

Genuine history gained 3.703 pp [+1.679,+6.332] over zero at matched age. Against the fixed source-average write, the difference was −0.144 pp [−1.542,+1.300]. A plausible memory vector was valuable for these weights; using the individual held person to produce that vector yielded no resolved incremental gain. This directly motivates §443’s full-decoder test of the history-specific logits and §444’s readout anchored to a source-average prefix.

Thirty paired A100 fold jobs trained the three complete decoders under clean, coherent and independent training. All held-window checks passed for the first window, archived decisions, prediction commitments, and official frozen controls. Replay printed OK. Jobs 3125341–3125343 used 1,164 CPU-core-seconds and 485 A100-seconds. Public development only, same previously examined people and training schedule across initializations. [Scores and intervals](gain441_result.json) · [First initialization](gain439.md) · [Matched-age controls](../../studies/18nov/gain441/replicate.py)
