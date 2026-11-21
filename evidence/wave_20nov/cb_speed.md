# CBraMod speed · 21 November

The plain derivative uses `final_stack_hardcb_safe_st.zip`, SHA-256 `47faa27b25f6fc1fcbcc1c74f2be4b8b2aeb2a1e02c6c957d342eb6c91b9c2ec`. The Team B derivative uses the pinned nine25 ZIP, SHA-256 `af40e37507996302e94f2d69df6beec20405cb6932965a5766352d3cb44b9c79`. Each builder verifies the parent hash, ZIP inventory, CRC and exact source. The only changed entry is `submission.py`. The 16-order replay refuted the expected R1a gain. The projections use the plain and nine25 bases. [Plain builder](../../submissions/final_stack/build_cb_speed.py) · [Team B builder](../../submissions/final_stack/build_team_b_speed.py) · [R1a result](laneR1a_bases.md) · [Team B blend](laneTB_blend.md)

| Archive in `runs/sealed/final_zips/` | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hardcb_safe_st_cbgraph.zip` | 94,906,217 | `6ba4092d1f07e036cfb7e19f306eba993cfdea5b6d9cff779d03c32e847a39f5` |
| `final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph.zip` | 100,404,159 | `911a31af329c3f38f75ae8a0f1331e18bc921e6c36b002538839ba04a2384c4e` |

The three seeds share a whitening prior and use different fitted encoder and head weights. Keep the original NumPy float64 covariance and whitening, interpolate and scale the shared input once, then execute the three seed forwards. The CUDA graph replays those operations with a fixed device input, asynchronously staged from pinned memory, followed by one posterior transfer. The NumPy mean still uses float64. Capture and buffers belong to each session. Reset and deepcopy discard graph state. Capture failure warns and switches that session to eager execution. [Member code](../../submissions/final_stack/cb_graph.py.inc)

Local `replay/check.py` passed all six v3 references. The exact plain ZIP passed the constructor self-test and matched its parent on the eight-window P01/1 prefix in recording order and shuffle 71000 on Mac CPU. Each injected actor exception or rolled posterior caused the constructor to raise. Rebuilding with the plain-only builder produced the same ZIP bytes. [Fault-injection harness](../../submissions/final_stack/selftest_inject.py)

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

## Plain hardcb timing

Worker-image smoke 64998037 and benchmark 64998049 used the exact plain archive and parent on NVIDIA A100-SXM-64GB, torch 2.14.1+cu130 and Braindecode 1.8.1. After 100 warm-up windows the benchmark counterbalanced order by session, synchronized CUDA and retained the CPU posterior transfer in wall time. Both CBraMod paths had zero maximum absolute error across all 2,655 real windows in shuffle 71000.

| Plain hardcb member | Warm mean, 2,641 windows | Warm p95 | First-window mean, 14 sessions |
|---|---:|---:|---:|
| Original CBraMod | 22.316 ms | 23.435 ms | 22.641 ms |
| Shared-input eager CBraMod | 22.069 ms | 23.039 ms | 22.325 ms |
| Graphed CBraMod | **6.659 ms** | 6.827 ms | 92.765 ms |
| Hard actors | 11.512 ms | 12.273 ms | 36.064 ms |

Plain hardcb projects to **2,919.9 s** on the A100 proxy from `2,555 + 20,083 × (0.00665927 + 0.01151183)`. Amortizing first-window costs at public session lengths gives **2,931.6 s**. The paired graph cut mean CBraMod latency by 70.2%.

The RTX 2080 Ti model scales the A100 graph time by the historical eager CBraMod ratio `32.75 / 27.78` and uses the historical 2080 Ti hard-actor cost of 14.3 ms. That gives 22.15 ms/window and **about 3,000 s** for plain hardcb. The model assumes graph execution on the RTX 2080 Ti. [Historical timing](final_stack.md#review-fix-2--additive-runtime-and-load-time-self-test-20-nov-sat-0000-0100-cet)

## Nine25 timing and graph checks

The nine25 derivatives inherit the plain ZIP's load-time self-test after verifying the two hard and three CBraMod weight files byte for byte. The Team B feature graph captures the three shallow, three slow, and three FBC feature forwards for input batches of one and 64 windows. Causal covariance, whitening, refresh ages, feature normalization, and all nine heads retain the original computations. A repeated 64-window chunk copies features before reusing graph storage. Session resets discard both graphs and their static buffers. [Team B feature code](../../submissions/final_stack/team_b/feature_graph.py.inc)

Slurm worker smokes 64998396 and 64998399 passed the constructor self-test and terminal prefix checks. The graph smoke verified actual replay on both Team B input shapes and injected one failure per shape. Both fallbacks warned, ran eager, and matched the graph posteriors exactly across 72 real windows. The candidate also rejected injected CBraMod and hard-actor exceptions and wrong posteriors on CPU.

The full worker benchmarks tested 2,655 windows over 14 sessions at shuffle 71000, each on its own A100-SXM-64GB with torch 2.14.1+cu130. The maximum absolute CBraMod and Team B posterior errors were zero for both derivatives. The benchmark alternated original and candidate order by session. Team B's warm mean excludes the two capture-eligible ages, 1 and 64, in each session. The full-session mean includes them.

| Nine25 job and derivative | CBraMod original → graph warm | Hard warm | Team B incremental original → candidate | Team B warm original → candidate | Team B full-session original → candidate |
|---|---:|---:|---:|---:|---:|
| 64998397, CBraMod graph | 24.654 → 6.810 ms | 12.885 ms | 5.307 → 5.344 ms | 8.680 → 8.707 ms | 8.685 → 8.711 ms |
| 64998401, CBraMod and Team B graphs | 21.582 → 6.594 ms | 11.192 ms | 4.970 → **3.346 ms** | 8.370 → **7.173 ms** | 8.375 → **7.414 ms** |

Team B's graph cut incremental latency by 32.7% and full-session latency by 11.5%, including capture. The candidate averaged 30.105 ms on the two capture-eligible ages versus 8.841 ms in the parent. The fresh per-device paired timing accounts for the faster path without assuming the historical 7.82 ms baseline applies on the new node.

Terminal arrays 64998398 and 64998402 completed in `cap250/runs/cb-speed-team-b-21nov` by 02:06 CET. Each derivative matched the exact nine25 parent on all 2,655 labels in recording order and on all 2,655 labels at shuffle 71000. Recognition ages and refit counts matched on all 14 sessions in both orders. The selected candidate uses both graphs after passing their timing and parity checks. It reproduced byte for byte in a second independent ZIP build.

## Additive projections

The formula is `2,555 + 20,083 × total_added_ms / 1,000`. On A100, the sum includes the measured hard actors, the graphed CBraMod member, and, for nine25, Team B. The amortized column includes measured graph capture and hard actor restoration at the public session lengths. RTX 2080 Ti uses the CBraMod historical ratio and hard cost above. For Team B it scales the A100 warm graph cost by the historical hard-actor ratio `14.30 / 11.80`. These are hardware proxies for the sealed limit of 3,600 s on RTX PRO 4500.

| Archive | A100 warm added ms/window | A100 warm projection | A100 with capture and restore | RTX 2080 Ti modeled added ms/window | RTX 2080 Ti modeled projection |
|---|---:|---:|---:|---:|---:|
| Plain hardcb | 18.171 | **2,919.9 s** | 2,931.6 s | 22.151 | **~3,000 s** |
| Nine25 with both graphs | 24.959 | **3,056.2 s** | 3,072.6 s | 30.766 | **~3,170 s** |
