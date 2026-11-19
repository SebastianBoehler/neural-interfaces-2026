# F29 · Recurrent waveform feature readout

19 November 2026, 06:35:19 CET. Chronological order beats a matched interior permutation on both seeds. Both recurrent models score below exact plain S1C. The registered accuracy gain over hard CE is **refuted**. The order comparison is **supported** within this recurrent architecture. This is standalone DEV on ten fit-excluded PROTEUS people across eight deliveries. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact hard CE | Chronological | Scrambled | Chronological − hard, pp [80%] | Chronological − scrambled, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 59.1613% | 52.7632% | 49.6773% | −6.398 [−7.599, −5.347] | +3.086 [+1.904, +4.458] |
| Bounded | 1 | 58.8654% | 52.6079% | 49.7378% | −6.257 [−7.541, −5.009] | +2.870 [+1.717, +4.186] |
| Ordinary | 0 | 58.5458% | 52.6519% | 49.6381% | −5.894 [−7.135, −4.779] | +3.014 [+1.790, +4.408] |
| Ordinary | 1 | 58.3882% | 52.5494% | 49.6548% | −5.839 [−7.328, −4.413] | +2.895 [+1.596, +4.288] |

Bounded chronological-minus-hard for P04/P10 was −4.389/−7.071 pp on seed 0 and −3.904/−7.517 pp on seed 1. Its order-specific difference was +0.630/+3.317 pp and +2.774/+2.544 pp. P04/P10 are held-new roles.

Fitted recurring branches had matched 85,008 added parameters, nonzero gradients and exact native origin at zero. Chronological source peak correction averaged 0.2459/0.2491 of the 0.25 bound across folds. Swapping order at fixed weights moved source residuals by maxima averaging 0.3116/0.3795. Latent recurrent prefix comparisons were exact. The broader held prefix error was at most 1.464e−6, with exact first-window equality on 672 and 1,344 interleaved resets. All hard checkpoints and decisions matched F16 byte-for-byte. H100 smoke warm 32-window forward was 1.754 ms hard, 1.913 ms chronological and 1.906 ms scrambled.

Ferranti `511580/511581` and Galvani `3134335–3134337` passed. Neither recurrent arm advances to cap250 candidate testing.

| Artifact in `ni26_friday_s1c_recurrent_20261119` | SHA-256 |
|---|---|
| Source | `51461fb9084c99c0ac3f607debf21babeaac2667c88135173555e21df4b143ee` |
| Held scorer | `7f56f8906706c4e7990dad120d003b7518ae7f8d0612327cab7511ce9cc6ef04` |
| Blind decisions | `9616f52f139c579260d5128c8f6628e67dc536bef9e300bba78dd867040016bb` |
| Results | `285a0b6426557c39cdf16f1211a6b151c14587ed0c8382bfbef8afeaf4938979` |
