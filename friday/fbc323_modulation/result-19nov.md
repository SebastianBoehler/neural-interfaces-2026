# F25 · Multiresolution cross-band waveform modulation

19 November 2026, 04:50:20 CET. The multiresolution model reached **60.1431%** on seed 1, with positive paired intervals versus exact hard CE and the matched conventional arm. Seed 0 did not reproduce the gain, leaving both preregistered endpoints **undecided**. These are whole-decoder DEV results for ten fit-excluded public PROTEUS people across eight deliveries. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact hard CE | Conventional gate | Multiresolution gate | Multi − hard, pp [80%] | Multi − conventional, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 59.1613% | 59.3257% | 58.9364% | −0.225 [−0.793, +0.316] | −0.389 [−0.896, +0.158] |
| Bounded | 1 | 58.8654% | 59.6618% | 60.1431% | +1.278 [+0.886, +1.783] | +0.481 [+0.106, +0.802] |
| Ordinary | 0 | 58.5458% | 58.8671% | 58.5316% | −0.014 [−0.564, +0.474] | −0.335 [−0.802, +0.067] |
| Ordinary | 1 | 58.3882% | 59.2290% | 59.7973% | +1.409 [+1.100, +1.795] | +0.568 [+0.202, +0.837] |

Exact F17 bounded jitter scored 59.4005% and 59.9529% on these folds and orders. Bounded multiresolution minus hard for P04/P10 was +0.917/−0.936 pp on seed 0 and +1.035/+0.270 pp on seed 1. Against the conventional gate it was −0.640/−0.235 pp and +1.453/+0.240 pp. All roles here are fit-excluded.

Both learned arms added 4,032 trainable parameters. Native hard checkpoints and held decisions matched F16 exactly. The learned arms activated their cross-band weights and off-center temporal taps, and fixed dilation changes moved their source representations. Warm H100 32-window forwards took 1.735 ms hard, 2.119 ms conventional and 2.110 ms multiresolution. Held replay checked 672 exact first windows, 1,344 resets and maximum prefix drift 1.474e−6.

Ferranti smokes/fits `511434/511435` and Galvani smoke/held/collector `3133882/3133883/3133884` passed. Artifact SHA-256 values in `ni26_friday_s1c_modulation_20261119` follow.

| Artifact | SHA-256 |
|---|---|
| Source | `589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c` |
| Held scorer | `939a96514d432cedebf24a0c4500927f71401ed76117adc73fc76604453a779c` |
| Blind decisions | `d281d5ecaa0f15a2757d4997da57b16188aff10e0d70f643074cd899d2140005` |
| Results | `64d2182825146f30486781e7a847407556a6acb32cdec55e08ecfc35182d52af` |
