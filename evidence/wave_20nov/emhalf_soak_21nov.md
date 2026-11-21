Soak passed on Leonardo A100 with 20,083 calls per em_half replica, ratios 1.205/1.212, drift 1.023/0.984 and peak CUDA reserved 2.14 GiB.

| Arm | State · calls | Load / compute / wall (s) | Tenths (ms/window) | Recognised (later visits) | Reserved / allocated CUDA · RSS (GiB) | Ratio · projected s · drift | Node · GPU · PyTorch |
|---|---|---:|---|---:|---:|---|---|
| em_half A | finished · 20,083 | 3.50 / 2835.68 / 2846.32 | 144.94 139.50 139.36 139.63 139.43 140.43 140.97 143.81 141.25 142.66 | 106 (30) | 2.14 / 0.63 / 2.23 | 1.205 · 3,078 · 1.023 | lrdn0032.leonardo.local · NVIDIA A100-SXM-64GB · 2.14.1+cu126 |
| em_half B | finished · 20,083 | 38.23 / 2852.28 / 2928.71 | 149.11 141.65 141.76 142.22 141.27 141.36 141.41 142.30 139.73 139.44 | 106 (30) | 2.14 / 0.63 / 2.19 | 1.212 · 3,096 · 0.984 | lrdn0083.leonardo.local · NVIDIA A100-SXM-64GB · 2.14.1+cu126 |
| cap250 | finished · 20,083 | 15.16 / 2354.00 / 2406.08 | 131.53 118.23 116.56 116.25 114.86 113.66 112.78 115.00 117.13 116.15 | 106 (30) | 0.06 / 0.05 / 1.83 | 1.000 · 2,555 · 0.982 | lrdn2072.leonardo.local · NVIDIA A100-SXM-64GB · 2.14.1+cu126 |
| nine25_fc | finished · 20,083 | 33.62 / 2638.88 / 2709.88 | 139.48 130.35 129.82 130.51 130.73 130.40 131.15 130.61 130.61 130.34 | 106 (30) | 1.75 / 0.27 / 1.97 | 1.121 · 2,864 · 1.000 | lrdn3102.leonardo.local · NVIDIA A100-SXM-64GB · 2.14.1+cu126 |

Compute includes session copies, resets, transfers, prediction and cleanup. Wall includes ZIP verification, extraction, data loading and receipt writes. The first tenth includes cold start.
Ratios use the largest checkpoint shared with cap250. Sealed seconds are inferred as 2,555 s × ratio. Drift is last/second tenth of each observed prefix. The sealed RTX PRO 4500 was outside this measurement. Public sessions are repeated, with lengths at most 199 calls.
[Per-session seconds and recognition latches](emhalf_soak_21nov_sessions.tsv).
Smoke 65057671 · finished, 375 calls, 2 later visits recognised.

Bounds (1.15–1.30 ratio, ±10% drift, <11 GB reserved) · em_half A within ratio and within drift, below 11 GB reserved, finished without error. em_half B within ratio and within drift, below 11 GB reserved, finished without error. nine25_fc outside ratio and within drift, below 11 GB reserved, finished without error.

Leonardo Slurm smoke 65057671 and four-node step 65057672.0, one A100 per node. Staged source a85bc6edfab831f5de73d10541dc0bc563102ec1. Official stream code a66bc23d.
Order seed 71000, model seed 42. Raw receipts in cap250/runs/emhalf-soak-21nov/out/. SHA-256 values follow.
Public data SHA-256 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327. Fixed workload SHA-256 aefaff8ad2727486c945a5db95948ae8bc6e9386625ea2fa40420771b18e45c3.
Both em_half replicas recognised all 106 session copies. This workload stresses the recognised route with a fully public cohort.
Slurm allocation COMPLETED, exit 0:0, elapsed 00:49:56, 2026-11-21T20:41:16 to 2026-11-21T21:31:12 CET.
- `cap250-65057672.json` `43440bd0e10ec893ca3950f77b78c3a1f433864f58114b7a78081c002a14c55b` · ZIP `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`
- `em_half_a-65057672.json` `65066e7740e0f3cb46a147c463dbda615134e040f34495a1c0b882d82da6b083` · ZIP `833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d`
- `em_half_b-65057672.json` `844966f06a47d8bb22a24dd90587f05e5db26077c5ef890c36e4ec210362f2a8` · ZIP `833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d`
- `nine25_fc-65057672.json` `22cc6952e4634c0366b78935d7f3b52fa40336803de430df6b606d82a9ee2b6e` · ZIP `7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07`
- `smoke-65057671.json` `90a44cf36aab72e8401263e66c113b923e368bccd7cd27e123e9d249b91be5d3` · ZIP `833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d`
