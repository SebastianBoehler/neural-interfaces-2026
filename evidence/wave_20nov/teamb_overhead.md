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
