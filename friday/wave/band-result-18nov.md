# F12 filter-bank initialization on held-new people

The spectral initialization scored 41.3716% and 41.1517% as a whole decoder. It changed accuracy by +0.1763 and −1.8469 percentage points against the matched random initialization. The two-seed rule leaves the initialization effect undecided. Both frontend variants trailed F7 and F9. The spectral variant was 17.674 and 17.894 points below the historical §323 S1C result.

| Held-new whole-model arm | Seed 460 | Seed 1460 |
| --- | ---: | ---: |
| Spectral initialization, genuine history | 41.3716% | 41.1517% |
| Random initialization, genuine history | 41.1953% | 42.9985% |
| Locked F7 supervised | 47.0967% | 47.2765% |
| Locked F9 fixed normalization | 48.0695% | 48.5907% |

| Paired contrast | Seed 460, pp [80%] | Seed 1460, pp [80%] |
| --- | ---: | ---: |
| Spectral − random | +0.1763 [−1.0622, +1.5386] | −1.8469 [−3.5430, −0.0796] |
| Spectral − F7 | −5.7252 [−8.3871, −3.7400] | −6.1248 [−8.8508, −3.7631] |
| Spectral − F9 | −6.6979 [−9.0212, −5.0399] | −7.4390 [−10.1564, −5.3183] |
| Spectral genuine − source donor | +0.5193 [−0.0522, +1.0273] | +0.0015 [−0.8154, +0.7592] |
| Spectral genuine − blank | +0.7294 [+0.0808, +1.3537] | −0.3192 [−1.0985, +0.4386] |

Ten people supplied 28 official metric cells and 21,240 decisions across eight pinned orders. Spectral minus random changed 9,771 and 10,130 decisions. P04 differences were −0.114 and −0.057 pp. P10 differences were −1.041 and −4.352 pp. Order-level differences ranged from −0.275 to +0.887 pp at seed 460 and −2.346 to −1.606 pp at seed 1460.

The F12 comparison matches architecture, 38,364 trainable parameters, episode schedule, optimizer and Galvani A100 training with full FP32 convolution. Initialized in-band energy was 94.8–99.3% for spectral FIRs and 1.8–48.0% for random FIRs. Each FIR updated during fitting. F7 and F9 used H100 training and a different frontend. The selected 59.0453% S1C aggregate uses the older FBCNet architecture and training recipe. Its gap is diagnostic. [Historical §323 method and results](../../../neural-interfaces-2026/evidence/next_wave_16nov/norm323.md)

A100 smoke 3128349 passed after disabling cuDNN TF32. The original unmodified tolerance had caught a batch versus one-window difference in smoke 3128326. All 40 final fits completed successfully. Spectral array 3128355 tasks took 11–31 seconds, and random array 3128356 tasks took 10–14 seconds. These are allocated fit times. Galvani preflight 3128357 and held smoke 3128358 passed. Seed scores 3128359/60 finished at 17:16:27 and 17:16:35 CET on 18 November. They authenticated source folds, canonical ages, causal prefix, reset, first-window and recorded stock labels before scoring the saved candidate predictions. No F12 model approached the historical anchor.

[Active queue and source manifests](band-queue-18nov.md). Galvani release `ni26_friday_wave_band_a100_20261118v2`.

| Seed | Decision SHA-256 | Result SHA-256 |
| ---: | --- | --- |
| 460 | `ca10d148458dc0b0601c85ed4e00184d77caec618c1108e09e5fa3647697f592` | `a6a832adc3379f4c42db232acc8848a10bb743702dd2fde1e77d224719579b69` |
| 1460 | `76e152cb517707c01e19cac7914de852759936f13a50c0f7f01dacb5b6c1a344` | `e03b4ff53bce649c0a8d973854bc52d816eac7543c60a56cd05f0e8ee5d18b3a` |
