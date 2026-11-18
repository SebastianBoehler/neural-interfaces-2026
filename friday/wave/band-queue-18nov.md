# F12 filter-bank queue receipt

F12 compares spectral and random FIR initialization with the same 38,364 trainable parameters, source folds, training schedule and fixed causal readout. Both seeds passed spectral concentration and matched-norm initialization checks. A capped real-source CPU run completed four updates for each arm and passed the causal prefix, first-window and reset checks on 18 November at 16:56 CET. The CPU smoke contains no held scores.

| Cluster | Work | Slurm job |
| --- | --- | --- |
| Ferranti | Paired real H100 smoke | 510747 |
| Ferranti | Spectral and random arrays, each 20 fits at concurrency four | 510748 and 510749 after smoke |
| Galvani | Pinned stock validation and recording-order replay | 3128300 |
| Galvani | Paired first held person, seed 460 | 3128301 after preflight and verified transfer |
| Galvani | Whole raw held score, seeds 460 and 1460 | 3128302 and 3128303 after held smoke and verified transfer |

The 510747 request was reduced to two minutes while pending, after the local paired smoke passed in a 10-second capped allocation. Its live forecast at 16:59 CET was 18:47:51 CET. Production time limits and scientific settings remain as registered. The forecast can change.

From the Friday checkout, run `bash friday/wave/band_release.sh`. The relay transfers finished checkpoint/receipt pairs and authenticates actor source, file hashes, kind, seed and held fold. It releases the held smoke only when both four-step P01 actors are available. A seed score requires both complete sets of ten final checkpoints. Independent Galvani scoring checks that both arms used the same source episodes and non-FIR parameters at initialization, and verifies all official first windows before loading labels.

Ferranti release `~/ni26/friday_wave_band_20261118`. Galvani release `/mnt/lustre/work/gehler/gfh527/ni26_friday_wave_band_20261118`. Actor commit `138a5a9`, manifest SHA-256 `1d919642f3906b7e6d6bbd82c9273add5c96cc8161b786818cd315641d4a80a9`. Scorer commit `bbb5392`, manifest SHA-256 `c615f1555f9d194a2f3d51fac7b2968af57178d44f8e6fd08e6bba4468946cd5`.

## Active A100 route

Galvani A100 smoke 3128326 reached the existing causal comparison and failed before saving weights. The maximum one-window to batched logit difference was 0.0003211. Smoke 3128349 disabled cuDNN TF32 for both arms, retained the original tolerance and completed in 31 seconds. It verified active steps, causal behavior, matched episode rows, parameter count, identical non-FIR state and equal FIR norms. Ferranti jobs 510747–49 were all pending when canceled. The prior Galvani held jobs 3128301–03 were also canceled.

Active release `/mnt/lustre/work/gehler/gfh527/ni26_friday_wave_band_a100_20261118v2`. Both arms run entirely on Galvani A100 with FP32 convolution. The historical F7 and F9 reference actors trained on H100. The same-site checkpoint links remove the need for the Ferranti relay command.

| Work | Active job | Dependency |
| --- | --- | --- |
| Spectral fits | 3128355 | A100 smoke 3128349 |
| Random fits | 3128356 | A100 smoke 3128349 |
| Stock and order preflight | 3128357 | Scorer and input hashes |
| Independent held smoke | 3128358 | Preflight and source smoke |
| Seed 460 score | 3128359 | Held smoke and both complete production arrays |
| Seed 1460 score | 3128360 | Held smoke and both complete production arrays |

Actor commit `8d7d26a`, manifest SHA-256 `4385ab898b8091d1c3b47a9617897e2caf58c008c241ff6a64ad731c1457afe2`. Scorer manifest SHA-256 `e0a8c2df233087f31ffcddfc7aadae878ca2027343d1caa2859ad3b68e3d5a6e`. Galvani job receipts are `jobs.fit.tsv` and `jobs.score.tsv` inside the active release.
