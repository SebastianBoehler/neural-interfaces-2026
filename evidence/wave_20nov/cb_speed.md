# CBraMod speed · 21 November

The candidate derives from the plain `final_stack_hardcb_safe_st.zip`, SHA-256 `47faa27b25f6fc1fcbcc1c74f2be4b8b2aeb2a1e02c6c957d342eb6c91b9c2ec`. The only changed entry is `submission.py`. The builder verifies the parent hash, ZIP inventory, CRC and exact source. The 16-order replay refuted the expected R1a gain, so the deliverable is the plain hardcb variant. [Builder](../../submissions/final_stack/build_cb_speed.py) · [R1a result](laneR1a_bases.md)

| Archive in `runs/sealed/final_zips/` | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hardcb_safe_st_cbgraph.zip` | 94,906,217 | `6ba4092d1f07e036cfb7e19f306eba993cfdea5b6d9cff779d03c32e847a39f5` |

The three seeds share a whitening prior and use different fitted encoder and head weights. Keep the original NumPy float64 covariance and whitening, interpolate and scale the shared input once, then execute the three seed forwards. The CUDA graph replays those operations with a fixed device input, asynchronously staged from pinned memory, followed by one posterior transfer. The NumPy mean still uses float64. Capture and buffers belong to each session. Reset and deepcopy discard graph state. Capture failure warns and switches that session to eager execution. [Member code](../../submissions/final_stack/cb_graph.py.inc)

Local `replay/check.py` passed all six v3 references. The exact plain ZIP passed the constructor self-test and matched its parent on the eight-window P01/1 prefix in recording order and shuffle 71000 on Mac CPU. [Fault-injection harness](../../submissions/final_stack/selftest_inject.py)

The A100 jobs use Leonardo Slurm under `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/cb-speed-21nov`. Galvani rejected noninteractive authentication, so RTX 2080 Ti timing is unavailable. Leonardo's R535 driver rejected the CUDA 13.0 SIF as configured. The retry binds NVIDIA's pinned CUDA 13.0 forward-compatibility libraries at process scope. Those libraries are external to the candidate ZIP. [Registration](../../docs/LEDGER.md#cbramod-cuda-graph--21-november) · [Slurm jobs](DISPATCH_21NOV.md) · [NVIDIA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html#use-the-right-cuda-forward-compatibility-package)

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

## A100 terminal labels and graph checks

The plain ZIP matched all 2,655 terminal labels in each delivery order, recording order and shuffle 71000, across all 14 public sessions. Recognition ages and refit counts also matched. Both check sets used the official `fresh_model` and full Solver, and the candidate performed one actual graph replay per window. Jobs 64996949_0–3 (native torch 2.14.1+cu126) and 64997119_0–3 (worker SIF, torch 2.14.1+cu130) completed.

Native smoke 64996944 and worker-image smoke 64997117 loaded the plain ZIP through its self-test. After the first graph capture, the smoke made subsequent eager calls raise and still reproduced all reference posteriors. A copied model rebuilt the graph from a clean session state. A forced capture error raised one warning, used eager for the session and reproduced all posteriors. The maximum graph and forced-eager errors on the smoke were both zero.
