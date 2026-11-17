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
