# Team B overhead (fused-conv / channels-last) · 21 November 2026

Sections are appended by the Galvani and Leonardo collectors; merge, do not overwrite.

## Galvani identity and timing (collected 21 Nov 09:00 CET, `fc` = fused-conv ZIPs)

Galvani release `ni26_final_confirm_20261121`, script `teamb_overhead_submit_v2.sh`, collector `scripts/overnight21/teamb_overhead_collect.py`. The ZIPs scored here are the `_fc` fused-conv variants (not a separate channels-last `_cl` build; the e2e arms are `*_tbgraph` vs `*_tbgraph_fc`). Smokes 3146414/15 failed (fixed), reruns 3146424/25 passed.

**Label identity** (`confirm_gpu.sbatch ident`, parent vs `_fc`, shuffle 71000, all 14 sessions, taps compared per window): changed labels **0** in every job.

| Pair | Device | Windows | Changed | Max tap abs diff | Jobs |
|---|---|---|---|---|---|
| nine25 vs nine25_fc | 2080 Ti | 2655 (7 parts) | 0 | 7.5e-08 | 3146427,30,33,36,39,42,45 |
| nine15 vs nine15_fc | 2080 Ti | 2655 | 0 | 4.5e-08 | 3146428,31,34,37,40,43,46 |
| hardcb nine25 cbgraph vs _fc | 2080 Ti | 2655 | 0 | 7.5e-08 | 3146429,32,35,38,41,44,47 |
| nine25 vs nine25_fc | A100-PCIe | 2655 (4 parts) | 0 | 7.5e-08 | 3146448-51 |
| nine25 vs nine25_fc | CPU, stream `zero379` | 379 | 0 | 0.0 | 3146462 |

Taps are not bitwise equal on GPU (float-level 3e-08 to 7e-08), bitwise equal on CPU.

**End-to-end timing** (`e2e_timing.py`, arms alternating, Solver projection vs the 2555 s cap250 reference; limit 3,600 s). Jobs 3146452-61. Added ms/window over cap250 and projected seconds, range over repeats:

| Arm | 2080 Ti (n=3) | A100-PCIe (n=2) |
|---|---|---|
| nine25 | +51.8..52.8 ms, 3596-3615 s | +30.4 ms, 3165 s |
| nine25_fc | +18.8..25.4 ms, 2932-3064 s | +24.8..25.5 ms, 3054-3067 s |
| nine15 | +46.7..53.3 ms, 3493-3624 s | +28.1..29.7 ms, 3118-3152 s |
| nine15_fc | +20.0..26.7 ms, 2957-3092 s | +22.9..23.1 ms, 3014-3019 s |
| hardcb cbgraph | +16.6..29.5 ms, 2888-3148 s | +24.1..24.3 ms, 3039-3043 s |
| hardcb nine25 cbgraph | +56.1..61.7 ms, 3681-3793 s | +38.7..39.1 ms, 3333-3340 s |
| hardcb nine25 cbgraph_fc | +30.4..33.6 ms, 3165-3229 s | +33.6..35.8 ms, 3229-3274 s |

Reading: `fc` removes about 28 ms/window on the 2080 Ti (Team B inside time 36.5 → 9.0 ms) and projects below 3,600 s in all three repeats; the unpatched nine25/nine15 project 3,493-3,624 s there (at the limit; 2 of 6 repeats above 3,600 s, and 3,465-3,840 s over the 16 chained 2080 Ti runs of the chained table below). On the A100 the gain is small (15 → 10 ms). Label identity holds, so `fc` is the timing-eligible form.

**Chained e2e timing, unpatched ZIPs** (jobs 3146478-3146511, standing queue; per node, repeats in brackets, added ms / projected s; `fc` ZIPs not in these chains; remaining chain jobs still queued at 09:00):

| Arm | Node | Added ms (proj s) |
|---|---|---|
| nine25 | 2080 Ti cn111 | 52.7 (3614), 49.2 (3543), 49.3 (3545), 54.1 (3642) |
| nine25 | 2080 Ti cn116 | 46.0 (3480), 46.3 (3486), 61.1 (3782), 58.3 (3726) |
| nine15 | 2080 Ti cn111 | 52.3 (3605), 49.7 (3552), 50.4 (3566), 54.1 (3642) |
| nine15 | 2080 Ti cn116 | 46.7 (3492), 45.3 (3465), 64.0 (3840), 52.6 (3611) |
| hardcb cbgraph | 2080 Ti cn115 / cn116 / cn117 | 21.6 (2990), 26.4 (3086) / 19.4 (2944) / 21.7 (2991) |
| nine25 | A100 cn208 / cn220 / cn222 | 29.6 (3149) / 33.5 (3228) / 40.1 (3361) |
| nine15 | A100 cn208 / cn220 / cn222 | 29.7 (3151) / 31.1 (3179) / 40.8 (3374) |
| hardcb cbgraph | A100 cn220 / cn225 | 23.7 (3031) / 23.5 (3027) |

Unpatched Team B: 2080 Ti 3,465-3,840 s (8 runs per arm), 4 of 8 nine25 and 4 of 8 nine15 runs above 3,600 s; node cn116 is the slow one (up to 3,840 s). cbgraph alone: 2,944-3,086 s on every node. A100 nodes differ by 11 ms between cn208 and cn222.

## Qualification of _fc

**Gate 1 passed.** All three archives rebuilt byte for byte with `patch_tb_fused_conv.py` (source in `47da04c7`). Each rebuild passed CRC and preserved its parent's ordered inventory. Only `submission.py` changed, within the embedded Team B encoder.

| Arm | Parent ZIP SHA-256 | Fused ZIP SHA-256 | Entries |
|---|---|---|---:|
| nine25_fc | `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9` | `7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07` | 42 |
| nine15_fc | `9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b` | `6e6bbbb841db4ff1ea5e83e4bad75a090ab0a30455fa5ca570063b5ba828d1d4` | 42 |
| hardcb nine25_fc | `911a31af329c3f38f75ae8a0f1331e18bc921e6c36b002538839ba04a2384c4e` | `086c7600555efba5545c035e1ef3c4245ab036599b588400c34ac118753fcd73` | 45 |

**Gate 2 passed.** All four official warm-up pairs ran cap250, candidate, cap250, candidate. Each replay covered 755 windows in four v6 streams with zero errors. Both repeats per arm matched on all predictive scores and preserved package files. Cell-balanced accuracy was 0.9959229391 for both fused candidates and 0.9945788530 for cap250.

| GPU | Fused candidate | Job / node | cap250 ingestion, s | Candidate ingestion, s |
|---|---|---|---:|---:|
| 2080 Ti | nine25_fc | 3146870 / cn115 | 129.3 / 116.2 | 140.0 / 129.3 |
| 2080 Ti | nine15_fc | 3146871 / cn116 | 132.1 / 124.6 | 147.7 / 140.2 |
| A100-SXM-64GB | nine25_fc | 65013942 / lrdn0424 | 257.9 / 130.3 | 143.6 / 141.5 |
| A100-SXM-64GB | nine15_fc | 65013943 / lrdn3254 | 203.9 / 127.4 | 142.6 / 143.0 |

**Gate 3 failed for nine25_fc.** All six 2080 Ti and three A100-PCIe nodes completed the matched runs. The cn222 result exceeded 3,600 s by 36.1 s. The largest 2080 Ti projection was 3,169.1 s on cn117. The slow 2080 Ti node cn116 projected 3,125.9 s, compared with previous unpatched nine25 runs up to 3,782 s there.

| Galvani node | Job | cap250 ms/window | nine25_fc added ms/window | nine25_fc projected s | hardcb nine25_fc projected s |
|---|---:|---:|---:|---:|---:|
| 2080 Ti cn111 | 3146861 | 132.67 | +19.20 | 2,940.5 | — |
| 2080 Ti cn114 | 3146862 | 121.17 | +23.02 | 3,017.3 | 3,174.6 |
| 2080 Ti cn115 | 3146863 | 129.97 | +23.70 | 3,031.0 | — |
| 2080 Ti cn116 | 3146864 | 135.47 | +28.43 | 3,125.9 | 3,131.3 |
| 2080 Ti cn117 | 3146865 | 131.35 | +30.58 | 3,169.1 | — |
| 2080 Ti cn123 | 3146866 | 127.76 | +24.09 | 3,038.7 | 3,228.1 |
| A100-PCIe cn207 | 3146879 | 127.54 | +22.82 | 3,013.2 | — |
| A100-PCIe cn220 | 3146880 | 147.19 | +21.85 | 2,993.7 | — |
| A100-PCIe cn222 | 3146869 | 476.29 | +53.83 | **3,636.1** | — |

All three hardcb nine25_fc timings passed the 3,600 s comparison, with a maximum of 3,228.1 s in this wave and 3,274.4 s in the earlier A100 runs. Slurm forecast the original pending cn205/cn209 jobs (3146867/68) to start after 12:37 CET. The replacement jobs ran on cn207/cn220. The estimate adds paired Solver cost over 1,131 measured public windows to the sealed cap250 receipt. Load and the first warm-up session are excluded. The RTX PRO 4500 remains unmeasured.

The cn222 run was slow for both arms. Cap250 was 476.29 ms/window and nine25_fc 530.12 ms/window. Repeat 3146946 measures cap250 and both fused doses on cn222. The first result still counts against the registered limit.

**Gate 4 passed.** Grep of each packaged `submission.py` found the inherited hard-actor load self-test and the dead-session guard. Both hard-only archives also have the Team B load self-test. The hardcb archive preserves its parent's CBraMod load self-test and lacks a Team B-specific load self-test, as does its parent. Galvani jobs 3146872–74 on 2080 Ti cn116 observed six fused CUDA bodies per archive. Injecting capture failure at each Team B shape (1 and 64) produced two warnings, selected eager, and changed 0/72 labels per archive. The largest captured posterior delta was 1.57e−8. An injected live Team B failure marked the session dead and returned all eight labels in each archive on GPU and Mac CPU. The hard-only load tests rejected both injected capture failures, a posterior exception, and a rolled posterior. [Raw receipts](teamb_overhead/qualification/)

`python replay/check.py` printed `OK` locally. The qualification ran on public windows.
