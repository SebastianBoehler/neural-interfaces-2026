# Native iteration on Galvani

## Two independent replays on one A100 (§368)

Job 3122839 completed in 8m53s on cn214. It replayed the exact first 96
windows of the qualified public stress stream through the real Solver
in eight fresh worker processes. Each used eight neural refits, 160
optimizer steps, four personalization refits and six fusions. All
labels matched the pinned prefix, and all work counters matched.
The allocation had one 40 GiB A100 and two CPU cores. We made no
CUDA Multi-Process Service changes.

| Implementation | Pair serial | Pair overlapped | Overlap saving | Sampled device peak, overlapped |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 147.58 s | 116.32 s | 21.2% | 3,927 MiB |
| C++ | 142.50 s | 108.62 s | 23.8% | 3,039 MiB |

Pair wall includes process, container and model startup. A C++ replay
itself took about 92 s when overlapped and 57–59 s when alone. Overlap
improves batch throughput while increasing individual latency. C++
finished the overlapped batch 6.6% sooner than frozen. Both fit on
the GPU. Per-process allocator peaks were 2,114 MiB frozen and 2,110
MiB C++. Maximum per-process host RSS was about 2,845 and 2,602 MiB.
Device memory is a one-second sample, so allocation peaks can be missed.
The different device peaks cannot establish a change in maximum
concurrency. This is a short development workload, with competition
delivery and long-stream parallel capacity outside the measurement.

Galvani `ni26_concurrency368_v1_20261117/result.json`, SHA-256
`532c2f8f6fe6f4a38196b2760a3ed76c6363209650df7d5528b91ef9bf3b7a3b`.

## Native personalization schedule (§369)

Both experiments changed the single existing tensor `v6_cp_t0` in a copy of
the worker artifact. Both reused the same compiled libraries, filter and
estimator. The skip schedule starts at 80 and retains the stock cadence.
The window-90 schedule first fits at fusion onset and refits every 20 windows.

Job 3122845 completed in 10m42s. Its C++ personalization stage processed
raw EEG and the pinned frozen upstream scores and count-filter outputs for
eight B2_pool shuffled orders. Stock matched every frozen label and
the maximum score error was 1.41e-11. Every skip score matched native stock
bit for bit. All three arms used their registered schedules and fusion counts.

| Schedule | Public new-person DEV | CP time across eight orders | Refits |
| --- | ---: | ---: | ---: |
| Stock | 76.27934% | 149.43 s | 1,008 |
| Skip | 76.27934% | 143.30 s | 672 |
| Window 90 | 76.29211% | 140.66 s | 576 |

Window 90 changed 233 labels across the eight orders. Official delta
+0.0128 pp [80% −0.1079, +0.1350]. Paired people/order delta
+0.0433 pp [−0.0641,+0.1557]. P04/P10 new-person deltas were −0.2231
and +0.0959 pp. The quality result is undecided. Skip saved 4.1% of the
native personalization replay including raw filtering. Window 90 saved 5.9%
on that same scope. Full detector and GPU replay are measured separately.

Galvani `ni26_native_schedule369_dev_v1_20261117/score.json`, SHA-256
`f30cd5dacc622c7664966d90493579403870f3ca3446110fc02693bc8a1fc91a`.

The initial full 378-window A100 diagnostic 3122842 failed its label
comparison after all arms ran. The collector failed to preserve individual
label differences, and the stock timing bracket was 115.51 s to 75.40 s.

The 190-window repeat 3122874 completed in 5m47s. Both variants, stock and
the closing stock repeat matched every previously qualified label. Each
used 17 neural refits and 100 personalization fusions with zero errors.
Stock used nine personalization refits, skip six, window 90 five.
The stock timing bracket was 41.32–42.23 s and both variant replays
took about 54 s. Stock reused the warmed template. Both variants loaded
fresh templates with different warm-up histories. This run validates
the shorter worker behavior and gives no qualified whole-worker saving.

Galvani `ni26_native_schedule369_a100_v2_20261117/bench.json`, SHA-256
`fc68d6a046fb888440b7b4f8265b29c2c446930a5037823f0ba6fdca5647d0f8`.

## CUDA launch profile (§370)

Job 3122875 completed in 62 s. The untouched native worker matched every
label and activity counter in a fresh 48-window public ordinary prefix with
and without profiling. The profiler captured the last 16 calls after
recognition. Replay was 2.334 s unprofiled and 2.852 s profiled.

The trace captured 49,200 CUDA kernel launches (3,075 per window).
Host launch API calls account for 410.5 ms, or 14.4% of profiled wall.
The union of GPU kernel intervals is 434.2 ms, or 15.2%. Host copy API
calls account for 81.5 ms, with 28.7 ms in synchronization calls.
The repeated launch families include elementwise operations, cuDNN
convolutions and FFTs. Host API times are inclusive and subject to profiler
overhead. The measurements identify a candidate for fixed-shape graph
capture or selective fusion, without measuring any optimization speedup.
The trace did not cover active refits.

Galvani `ni26_cuda370_v1_20261117/result.json`, SHA-256
`dd6d2eb32415426b07ab3849fce2e2381af7d4a9504c67414cb4081e711d8d5b`.
