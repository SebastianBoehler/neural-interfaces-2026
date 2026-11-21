# CBraMod speed · 21 November

Both derived ZIPs retain the original hardcb assets and load-time self-test. The only changed entry is `submission.py`. The builder verifies the pinned parents, ZIP inventory, CRC and exact patched source. [Builder](../../submissions/final_stack/build_cb_speed.py) · [Parent hashes](final_r1a.md)

| Archive in `runs/sealed/final_zips/` | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hardcb_safe_st_cbgraph.zip` | 94,906,217 | `6ba4092d1f07e036cfb7e19f306eba993cfdea5b6d9cff779d03c32e847a39f5` |
| `final_stack_hardcb_safe_st_r1a_cbgraph.zip` | 94,906,596 | `7029bb25e0fac075f94c6e1982e98d08d2910e615f7666127e5fbbf6446b43cc` |

The three seeds share a whitening prior and use different fitted encoder and head weights. Keep the original NumPy float64 covariance and whitening, interpolate and scale the shared input once, then execute the three seed forwards. The CUDA graph replays those operations with a fixed device input, asynchronously staged from pinned memory, followed by one posterior transfer. The NumPy mean still uses float64. Capture and buffers belong to each session. Reset and deepcopy discard graph state. Capture failure warns and switches that session to eager execution. [Member code](../../submissions/final_stack/cb_graph.py.inc)

Local `replay/check.py` passed all six v3 references. Both exact ZIPs passed constructor self-test and matched their parent on the eight-window P01/1 prefix in recording order and shuffle 71000 on Mac CPU. Injected actor exceptions and rolled posteriors each caused the R1a ZIP's constructor to raise. [Fault-injection harness](../../submissions/final_stack/selftest_inject.py)

The A100 jobs use Leonardo Slurm under `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/cb-speed-21nov`. The terminal-label arrays compare each exact ZIP to its own parent through the official `fresh_model` and the full Solver. Galvani rejected noninteractive authentication, so no new RTX 2080 Ti timings were taken. Leonardo's R535 driver rejected the CUDA 13.0 SIF as configured. The retry binds NVIDIA's pinned CUDA 13.0 forward-compatibility libraries at process scope. The compatibility libraries are external to the two ZIPs. [Registration](../../docs/LEDGER.md#cbramod-cuda-graph--21-november) · [Slurm jobs](DISPATCH_21NOV.md) · [NVIDIA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html#use-the-right-cuda-forward-compatibility-package)

## Mechanism checks

Leonardo native A100, Torch 2.14.1+cu126 and Braindecode 1.8.1. Job 64996906 compared 160 real windows with the first 16 excluded from timing. All three fitted seeds use the same prior, so the interpolation can run once. Concatenating inputs to a single fitted encoder would apply the same weights to all three seeds. The measured vectorized arm instead batches their separate parameters with `torch.func`.

| Mechanism | Warm mean | Max absolute per-seed posterior error |
|---|---:|---:|
| Shared-input eager, including CPU whitening | 20.126 ms | Reference |
| Vectorized seeds, including CPU whitening | 22.786 ms | 8.34e-7 |
| GPU whitening and shared-input eager | 20.632 ms | 2.98e-8 |
| Causal CPU whitening alone | 0.475 ms | |
| Causal GPU whitening alone | 1.146 ms | |

Component profiling omits the final CPU posterior transfer. The current covariance depends on the delivered prefix, so the model recomputes the whitening transform for each incoming window. The production graph reuses resampling and transfers the posteriors together while leaving whitening on CPU.

## A100 parity and timing

Native smoke 64996944 and worker-image smoke 64997117 loaded both exact ZIPs through their self-test. After the first graph capture, the smoke made subsequent eager calls raise and still reproduced all reference posteriors. A copied model rebuilt the graph from a clean session state. A forced capture error raised one warning, used eager for the session and reproduced all posteriors. The graph and forced-eager maximum errors on the smoke were both zero.

Both benchmarks covered all 2,655 real float32 windows at shuffle 71000 with a session reset for each of the 14 sessions. After 100 warm-up windows, the benchmark counterbalanced original and candidate order by session and measured wall time through the CUDA synchronization and CPU posterior. Warm timings use 2,641 windows. The 14 first windows include graph capture or hard actor restoration. TF32 is disabled by the existing Solver.

| A100 environment and job | Original CBraMod warm mean | Shared eager warm mean | Graph warm mean (p95) | Graph first window mean | Hard warm mean | Max absolute CBraMod error, eager / graph |
|---|---:|---:|---:|---:|---:|---:|
| Worker SIF, torch 2.14.1+cu130, 64997118 | 22.727 ms | 22.487 ms | **6.792 ms** (6.941) | 93.044 ms | 12.153 ms | 0 / 0 |
| Native, torch 2.14.1+cu126, 64996948 | 20.930 ms | 20.731 ms | 6.710 ms (6.737) | 88.238 ms | 11.570 ms | 0 / 0 |

Both jobs ran the exact R1a ZIP pair on an NVIDIA A100-SXM-64GB with Braindecode 1.8.1. Both graph runs replayed on every scored window. The worker graph reduced mean CBraMod time by 70.1% versus the original member on the same GPU. The older 27.78 ms A100 result came from a different A100 model and run. [Previous timing](final_stack.md#review-fix-2--additive-runtime-and-load-time-self-test-20-nov-sat-0000-0100-cet)

## Additive sealed-time projection

**Hardcb + R1a, A100 worker-image proxy:** `2,555 + 20,083 × (0.00679214 + 0.01215284) + 46 × 3.18 = 3,081.8 s`. That leaves 518.2 s against 3,600 s. Amortizing both measured first-window costs at the public mean session length gives **3,093.5 s**. The corresponding native A100 projections are 3,068.4 s warm and 3,079.9 s with those first-window costs. The 3.18 s increment comes from the earlier exact-ZIP paired replay of recognised sessions. The 46 recognised sessions are an assumption. RTX PRO 4500 runtime remains unmeasured. [R1a session measurement](final_r1a.md#qualification-and-tape-checks)
