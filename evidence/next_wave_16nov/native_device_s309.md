# §309 resident CUDA head benchmark

Completed on Galvani,16–17November2026, jobs3115174(smoke) and3115175(benchmark), both0:0.
Fresh source-locked workspace: `/mnt/lustre/work/gehler/gfh527/ni26_s309_device_20261116`.
The original§305 and all active scientific source snapshots were unchanged.

Three paired alternating full solves used one P01 fold's2,465source-only rows×44,239features,
float64, C=.3, zero initialization, the same L-BFGS options and independently checked loss/gradient.
No held labels or accuracy were used. Copies, allocation and final synchronization are inside time.

| Route | Three full-solve times | Median |
|---|---|---|
| Native CPU | 238.391s, 238.483s, 238.592s | 238.483s |
| Native A100-PCIE-40GB | 1.288s, 1.019s, 1.062s | 1.062s |

Median observed full-solve ratio CPU/A100 is224.5× on this exact workload.
CPU/GPU final objective gap is1.04e-10; independent max gradients are
3.38e−7/3.54e−7. CPU used384objective evaluations; A100used182. GPU peak allocated memory
was924,008,448bytes; whole-process peak RSS was4,081,392KiB. Effective CPU affinity was
[19,23,51,55], with OMP/MKL/OpenBLAS threads1. PyTorch2.14.1+cu126.

The ratio includes a different device and numerical optimization trajectory, not just faster kernels
or a programming-language change. It is a full-solve time, not an instrumented first-target time.
The earlier sklearn19–20s result came from another paired job and is not a simultaneous baseline here.
Whole held-person predictions, downstream accuracy, preprocessing costs, energy and package/worker
compatibility are not established by this training-only experiment. Input/scaler/source/library hashes
are in [the raw report](native_device_s309.json), SHA256 `04d698a0a7a88c70a8bb627785ec0e27c62a64314255e0ef1fa5105ed2ec0c00`.

This supports using a source-pinned resident GPU solver for subsequent matched research controls,
after checking their exact objective/data requirements. Do not substitute it silently for an existing
CPU-fitted scientific anchor or call matched loss proof of identical held predictions.
