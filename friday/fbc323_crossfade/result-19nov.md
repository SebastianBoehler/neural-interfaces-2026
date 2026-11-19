# F36 · Whole raw decoder after reciprocal waveform crossfade

Completed 19 November 2026 at 10:30:15 CET. Cross-trial training improved bounded balanced accuracy by 0.619 percentage points in seed 1 and fell 0.094 pp in seed 0. Both registered comparisons are **undecided**. The matched same-trial control improved the point estimate over exact hard in both seeds, with paired intervals spanning zero. Scores are standalone DEV on ten fit-excluded PROTEUS participants, 28 official cells and eight deliveries. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact A100 hard | Cross-trial | Same-trial | Cross-trial − hard, pp [80%] | Cross-trial − same-trial, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 58.6790% | 58.5849% | 59.0675% | −0.094 [−0.580, +0.285] | −0.483 [−0.855, −0.044] |
| Bounded | 1 | 59.1089% | 59.7283% | 59.5533% | +0.619 [+0.155, +1.062] | +0.175 [−0.197, +0.532] |
| Ordinary | 0 | 58.1123% | 58.1521% | 58.4101% | +0.040 [−0.447, +0.443] | −0.258 [−0.873, +0.564] |
| Ordinary | 1 | 58.7062% | 59.2987% | 58.9643% | +0.593 [+0.084, +1.073] | +0.334 [+0.002, +0.663] |

Bounded same-trial minus hard was +0.389 [−0.072,+0.724] and +0.444 [−0.180,+1.048] pp. Cross-trial minus hard P04/P10 was +0.489/−0.010 and −0.139/+1.420 pp. Motor/subtraction/word recall changed +1.089/−1.204/−0.167 pp in seed 0 and −0.069/+1.344/+0.583 pp in seed 1. Cross-trial absolute gain and the contribution of each task varied across seeds.

Both active arms treated exactly 244,800 of 979,200 source appearances per seed. Across the 20 folds, the maximum paired raw RMS mismatch was 8.911e−9 relative to the native interior norm. Mean relative RMS displacement was 0.494 in both arms and both seeds. The mean analytic transition term was approximately 2.61% of the native interior first-difference norm on the cross-trial arm and 2.56% on the same-trial arm. The scorer rederived all same-task pairs from source labels after sealing held predictions. The same-trial control also preserves clean boundaries and changes interior timing. Phase and frequency content can differ between the treatments.

Galvani A100 source smokes and fits `3135613/3135630`, CPU held smoke and folds `3135629/3135631`, and independent score `3135632` all passed. The 20 paired fits reproduced exact prior A100 hard checkpoint bytes and 84,960 held hard decisions. All 672 first windows and 1,344 interleaved resets passed. Maximum prefix posterior drift was 1.722e−6. Same inference class, parameter count and raw input for all three arms.

| Artifact in `ni26_friday_s1c_crossfade_a100_20261119` | SHA-256 |
|---|---|
| Source actor | `baf7735eb9ca7f225b99ecbf58e6a16a8fea3a13813ae7c2e8434d0fff901510` |
| Held scorer | `2734eeee0e398a53ffc5900b303255a2ef193df31d2adfb448b59eb1aba67367` |
| Blind bank | `32bbcd0f1230c10ab88742ee6467a42b6bdef3788051644361d095093ed160f9` |
| Whole results | `86b65abf6273b296f5ef4fd8e4ce4712c8a1e7794aa16a3fb4c2bd31f6a7d08e` |
