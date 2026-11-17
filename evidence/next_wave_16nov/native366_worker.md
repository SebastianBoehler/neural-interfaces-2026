# §366 — Exact long-stream A100 worker

The measured package combines CUDA multi-tensor Adam, the rest decoder's session copy,
float32 filter reflection, the worker's DUCC Hilbert transform, NumPy-style complex64
centering and Euclidean alignment. The alignment uses the worker's existing OpenBLAS
library. The normal Solver adapter runs the C++ library with the original weights,
adaptation policy and prediction interface. [Worker source and package](../../native/worker-parity/README.md)

## Active adaptation

Job 3122783 completed in 12m23s on Galvani cn230. It ran C++ and freshly ran frozen
in clean processes on one A100 PCIe 40 GB. Both arms used public EEG in the same
shuffled order (seed 36017800), with recognition thresholds raised in both temporary
packages to exercise full adaptation. Each stream started from a fresh model copy.

| Stream | Matching labels | Frozen | C++ | Replay saving |
| --- | ---: | ---: | ---: | ---: |
| 378 windows | 378/378 | 157.846 s | 153.370 s | 2.84% |
| 800 windows | 800/800 | 156.822 s | 151.146 s | 3.62% |
| Both | 1,178/1,178 | 314.668 s | 304.516 s | 3.23% |

All work counters matched. Each stream made 24 neural refits and 480 optimizer
steps. Personalization made 18 and 39 refits and 288 and 710 fusions, respectively.
Soft time caps fired at 12 and 54 refits in the two streams. Neither arm logged a
stage error. Maximum host RSS was 3,188,768 KiB frozen and 2,990,608 KiB C++.
Peak reserved CUDA memory was 5.952 GB and 5.948 GB, respectively.

Wall time from public data loading through the final label was 319.640 s frozen
and 309.610 s C++. For an hour of identical work, the observed ratios correspond
to 116 seconds of solver replay and 113 seconds of measured wall time. The one
paired run used C++ first. Sealed session mix, setup and total evaluation time
remain unmeasured.

At the same replay mix, the ratio corresponds to 3.33% more solver work at
fixed time. Both arms still reached the 24-refit cap. Allocating saved time to
new refits or models requires a policy change and a paired accuracy and cost
measurement.

Galvani comparison `ni26_native366_exact_ea_v1_20261117/comparison.json`,
SHA-256 `627710e5279009869381ee5bb15111cf20fb51e129767f842325460800f830b6`.

An independent control (3122787, cn225) also matched all 1,178 labels and
counters when it additionally replaced the real branch's centering. Its paired
replay took 311.284 s frozen and 300.434 s C++. The selected build needs only
the complex centering change. The real-centering control had no ordinary pair.

## Ordinary recognition

Job 3122794 completed in 14m05s on cn230. The identical C++ build ran
first, followed by a fresh frozen replay on the same A100. All 2,655 labels
and work counts matched across 14 complete shuffled public sessions. Both arms
recognized each person at index 10 and made no adaptation refits. Stage logs
reported zero errors.

| Scope | Frozen | C++ | Saving |
| --- | ---: | ---: | ---: |
| Solver replay | 402.224 s | 397.991 s | 1.05% |
| Wall from public data load | 407.575 s | 402.354 s | 1.28% |
| Model load | 4.940 s | 3.982 s | 0.959 s |

The equivalent per hour of the same workload is 38 seconds of solver replay
or 46 seconds of wall time. Maximum host RSS was 2,334,416 KiB frozen and
2,206,632 KiB C++. Peak reserved CUDA memory was 52.4 MB and 46.1 MB, respectively.
The timings are a single pair for each workload. They give the speed of these
public replays in the replicated image, with actual sealed wall time still open.

Galvani comparison `ni26_native366_exact_ea_normal_v1_20261117/comparison.json`,
SHA-256 `a6fc874458418ca1e4460ff42395f3407f671286ada0364c56a047d19b9fd942`.
