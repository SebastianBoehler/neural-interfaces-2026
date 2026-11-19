# F-C21e independent whole-worker RTX timing

The two `3134283` timing jobs completed by **06:20:15 CET on 19 November**. Each used the same pinned worker image and original cap250 package on an RTX 2080 Ti. One process warmed on unmodified stock. It then loaded a fresh session model for each balanced block. The trial used the real native FBLight mixer and RouteB share. Timing excluded the prediction audit and did not open held labels.

| Stream and block | Warm stock seconds | Warm FBL2 seconds | FBL2/stock mean | Worst paired ratio |
|---|---:|---:|---:|---:|
| Scherer G day 1, 120 windows, ABBA | 26.561, 25.856 | 26.587, 26.563 | **1.01398×** | **1.02732×** |
| PROTEUS P10 visit 3, 31 windows, BAAB | 4.949, 4.809 | 4.821, 4.803 | **0.98632×** | **0.99877×** |

Both fixture means met the forecast of at most 1.015×. G passed the 1.03× mean and 1.10× worst-pair development criteria. G retained 200 optimizer steps and 30 CSP fusions per run. P10 recognized at age 10 and ran zero optimizer steps in every block. The separate returning-person replay covers a visit with real refits before recognition.

- On G, FBL2 first-frame latency was 147–148 ms and p95 was 648–650 ms. Stock was 148–149 ms first and 635–643 ms p95. FBL2 peak GPU reservation was 2.488–2.498 GiB, compared with 2.488–2.506 GiB for stock. The cold stock timing-block warm-up took 65.285 s, including a 6.977 s first frame. The cold archived-stock smoke took 65.691 s.
- On P10, FBL2 first-frame latency was 157–165 ms and p95 was 154–156 ms. Stock was 157–158 ms first and 155–156 ms p95. All timed P10 runs reserved 50 MiB. The cold stock warm-up took 23.093 s including a 6.835 s first frame.
- This is matched end-to-end development timing on two fixtures and RTX proxy hardware. The exact sealed worker and time limit remain unmeasured.

The two signed timing receipts in `ni26_friday_c21e_native_v2_20261119` are `timing_0.json` SHA-256 `0a7fba8325e681027bcaf3c93745c23f8a81835aa35c5a10a583e994189ef4f4` and `timing_1.json` SHA-256 `cdcbab61e76342541ca6deef8617991022868107553e592f9d21d50d4818d260`. Source manifest `61feef26593423464c1e3d8935c0ef43ce01d737df84fa6c73035d0b3b663841`. Actual pinned smoke `3134281` SHA-256 `98986d6606501f93b0cb6814fb37881e0d8be924ccd618315b16a58175d10bcd`.
