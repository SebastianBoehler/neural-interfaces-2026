# §353 A100 worker prefix · 17 November 2026

Job 3122018 completed at 19:40:01 CET on galvani-cn223. Both packaged Solver
entry points ran inside the clean worker image replica with PyTorch 2.14.1+cu130,
driver 580.173.02 and one NVIDIA A100-PCIE-40GB. Each arm ran in a separate
process with a read-only package and two fresh-session replays of the same
31 real P01-1 windows. All labels matched across arms and repeats.

| Measurement | Frozen v6.6cap250 | Native v3 package |
| --- | ---: | ---: |
| Model load after CUDA initialization | 4.56 s | 3.94 s |
| First 31-window replay | 25.13 s | 22.68 s |
| Second 31-window replay | 4.50 s | 11.65 s |
| Process peak RSS | 2,117,764 KiB | 2,079,856 KiB |
| Peak live CUDA allocation | 45,752,320 B | 40,674,816 B |
| Peak reserved CUDA allocation | 52,428,800 B | 46,137,344 B |

Warm native replay took 2.59× as long in this pair. The short recognized
prefix exercises routing and session-copy isolation. The frozen reference
recognized at index 10, with zero FT or CP refits. The timings measure one
AB pair for this prefix. Full-shuffled, late-recognition and long unrecognized
streams, cap behavior, and complete evaluation timing require their own tests.

Native package manifest SHA-256 `4ccdcd24a7e3fd25c304bcd72aeb6feab6457ac1abeb32b7e57f20980877cd88`.
CPU reference fixture SHA `15f8080148292aa62c64f9cc21262ce6747ef81826fa061bbcb741f223541daa`.
Remote results and provenance are in `ni26_native353_worker_v2_20261117`.
Job cost was 200 CPU core-seconds and 100 A100 seconds. The later shared
Adam training repair used its own qualification artifact.

## Stage profile

Job 3122028 measured unmodified 12.01 s and clocked 11.89 s warm on a second
A100-PCIE-40GB. All 31 labels matched. Local clocks covered 99.6% of the
clocked run, with a clocked/unmodified ratio of 0.990.

The neural bank took 8.68 s, including 6.65 s at window index 1. The main
thread waited 1.40 s for the classical cores. The cores ran for 3.37 s
inside their asynchronous task and overlapped the bank. The serial timeline
uses the 1.40 s wait. Only 4.71 s of clocked step time
remained after excluding the single slow window.

Counterfactual 3122043 changes only the stateful neural bank's TorchScript
copy operation. The frozen Python solver uses deepcopy. LibTorch clone also
copies the class and methods. The one-TU A100 comparison reduced warm replay
from **11.77 s to 5.09 s**, a 56.7% reduction relative to unmodified native on
the same worker. Both arms matched the 31 frozen labels across both repeats.
This establishes the cost of the copy choice in this fixture. Complete
stream parity and timed comparisons with frozen remain open. Job 3122043
used 264 CPU core-seconds and 132 A100 seconds.
[LibTorch implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/csrc/jit/api/module.cpp)
