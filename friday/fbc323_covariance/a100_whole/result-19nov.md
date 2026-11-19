# F33a · Independent A100 whole-decoder result

19 November 2026, 09:28:18 CET. The aligned spatial relation arm lost to the exact same-device hard actor on seed 0. Both registered bounded contrasts are **undecided** across seeds. This is standalone held-new DEV on ten PROTEUS people across 28 cells and eight deliveries. [Protocol and device scope](prereg-19nov.md)

| Inference | Seed | Exact A100 hard | Aligned | Permuted | Aligned − hard, pp [paired 80%] | Aligned − permuted, pp [paired 80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 58.6790% | 57.8089% | 57.9968% | −0.870 [−1.465, −0.429] | −0.188 [−0.688, +0.268] |
| Bounded | 1 | 59.1089% | 59.1970% | 58.9403% | +0.088 [−0.766, +1.080] | +0.257 [−0.665, +1.353] |
| Ordinary | 0 | 58.1123% | 57.6577% | 57.7178% | −0.455 [−1.040, +0.020] | −0.060 [−0.583, +0.429] |
| Ordinary | 1 | 58.7062% | 58.9318% | 58.3220% | +0.226 [−0.678, +1.240] | +0.610 [−0.198, +1.595] |

Bounded mean subtraction/word recall aligned minus permuted was +1.228 pp [−0.075, +2.332] and +1.334 pp [−0.644, +3.703]. Against hard, it was +0.330 and +0.997 pp, with both intervals crossing zero. Bounded P04/P10 aligned minus hard was +0.220/−0.218 and −0.516/−0.367 pp. The cognitive secondary agrees in sign while whole-accuracy and individual effects vary.

All twenty independently trained bare and wrapped A100 hard checkpoints matched byte for byte. Their separate held runs gave the same 84,960 predictions. Every fitted A100 hard fingerprint differed from its H100 counterpart, despite exact initial tensors, original source batches and equal recipe. The H100 metadata served ancestry only. This keeps hardware-dependent trained states outside the A100 fit and scoring comparison. The original Ferranti F33 wave remains separate.

Galvani source smoke `3135069` and held smoke `3135079` passed. Matrices `3135080/3135081` and scorer `3135082` completed. Held checks counted 896 exact first windows, 1,792 exact resets and maximum posterior prefix difference 1.722e−6. The scoring process sealed all held bank files before opening labels. Release `ni26_friday_s1c_covariance_a100_whole_20261119`.

| Signed artifact | SHA-256 |
|---|---|
| A100 source | `94598f4eb485b20ea40c0d7c13f4497ed07d04e9e9c062c72953e97460b782bc` |
| A100 held scorer | `6c8e27573d2b91176ec6d2e98068e94d8e671d878c92e515f6c58899ce558dbe` |
| Blind decision bank | `f996e3a55d376f6856bbd07425632033924a175636acc21b332d1a4d3f4f0525` |
| Results | `e7d27e64a5ba95a8d616a264d268b5d2c2af1612d449e93880695f588468fc10` |
