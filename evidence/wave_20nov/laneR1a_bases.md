# R1a on the hard-only and hardcb bases · 21 Nov 2026

Registered in `docs/LEDGER.md` before scoring ("R1a on the hard-only base…"). Exact final solver (pinned parent and R1a ZIPs), real `production-20261120-32a94ab7` fold packages, P04/P10 visit 1 enrolled, visits 2–3, full-session orders 71000–71015, equal-cell balanced accuracy, changes in pp. Paired 80 % intervals resample people, then orders (20,000 draws, seed 20261121). Two returning people is a thin sample; the 20/80 column is an unverified assumption. Run `cap250/runs/overnight-21nov-r1abases`, jobs 64996733–64997043 (24/24 completed); `r1a_bases_collect.py` output below.

**Result.** Neither base shows the H1 gain. hardcb + R1a at 71000–71003 reproduces H1 exactly (0 stored-tape mismatches, +0.20 earlier), but orders 71004–71015 give −0.26 [−0.53, +0.01] (P04 −0.51, P10 0.00), so 16 orders: −0.14 [−0.32, +0.02] equal cells, 20/80 −0.05. Hard + R1a: −0.11 [−0.31, +0.06] at 16 orders (−0.04 at 71000–71007, the item-1 window). Under the registered rule (mean ≤ 0 refutes) the +0.15 predictions are refuted on both bases; the intervals still reach zero, so this is "no positive effect", not a proven loss. The H1 P04 gain came from four orders (selection). Hard vs hardcb without R1a: −0.11 [−0.19, −0.01] at 16 orders (hardcb ahead, mainly P10). Held-new zero-change passed: P01–P03 visit 1, orders 71000–71007, both modes: recognition −1 and 0 labels changed on every stream. Every returning R1a stream recognised at age 10–47 with exactly one late refit at 100, the parent none. Runtime cost of R1a (+3.5 s per recognised session) buys nothing here.


### orders 71000-71007

| Arm − reference | orders | P04 Δ [80%] | P10 Δ [80%] | Equal cells Δ [80%, people-nested] | P(Δ>0) | 20/80 Δ* |
|---|---:|---|---|---|---:|---:|
| hard_r1a − hard | 8 | -0.14 [-0.34, +0.07] | +0.06 [-0.04, +0.16] | -0.04 [-0.19, +0.10] | 0.41 | +0.02 |
| hardcb_r1a − hardcb | 8 | -0.07 [-0.30, +0.17] | +0.06 [-0.00, +0.13] | -0.00 [-0.15, +0.11] | 0.54 | +0.04 |
| hard − hardcb | 8 | +0.00 [-0.20, +0.23] | -0.10 [-0.19, -0.00] | -0.05 [-0.16, +0.08] | 0.27 | -0.08 |
| hard_r1a − hardcb_r1a | 8 | -0.07 [-0.30, +0.17] | -0.10 [-0.20, -0.00] | -0.08 [-0.20, +0.04] | 0.19 | -0.09 |

### orders 71004-71015

| Arm − reference | orders | P04 Δ [80%] | P10 Δ [80%] | Equal cells Δ [80%, people-nested] | P(Δ>0) | 20/80 Δ* |
|---|---:|---|---|---|---:|---:|
| hard_r1a − hard | 12 | -0.38 [-0.62, -0.14] | +0.04 [-0.05, +0.13] | -0.17 [-0.42, +0.06] | 0.22 | -0.05 |
| hardcb_r1a − hardcb | 12 | -0.51 [-0.68, -0.35] | -0.00 [-0.08, +0.08] | -0.26 [-0.53, +0.01] | 0.12 | -0.10 |
| hard − hardcb | 12 | -0.04 [-0.24, +0.16] | -0.13 [-0.21, -0.04] | -0.08 [-0.19, +0.04] | 0.17 | -0.11 |
| hard_r1a − hardcb_r1a | 12 | +0.09 [-0.06, +0.25] | -0.09 [-0.16, -0.02] | +0.00 [-0.11, +0.13] | 0.48 | -0.05 |

### orders 71000-71015

| Arm − reference | orders | P04 Δ [80%] | P10 Δ [80%] | Equal cells Δ [80%, people-nested] | P(Δ>0) | 20/80 Δ* |
|---|---:|---|---|---|---:|---:|
| hard_r1a − hard | 16 | -0.27 [-0.46, -0.07] | +0.04 [-0.04, +0.12] | -0.11 [-0.31, +0.06] | 0.25 | -0.02 |
| hardcb_r1a − hardcb | 16 | -0.30 [-0.47, -0.13] | +0.01 [-0.05, +0.08] | -0.14 [-0.32, +0.02] | 0.17 | -0.05 |
| hard − hardcb | 16 | -0.08 [-0.23, +0.07] | -0.13 [-0.21, -0.06] | -0.11 [-0.19, -0.01] | 0.07 | -0.12 |
| hard_r1a − hardcb_r1a | 16 | -0.05 [-0.20, +0.10] | -0.10 [-0.17, -0.03] | -0.07 [-0.15, +0.01] | 0.13 | -0.09 |

### Absolute equal-cell BA (all collected orders)
- hard: P04 70.47, P10 88.23
- hard_r1a: P04 70.20, P10 88.27
- hardcb: P04 70.55, P10 88.36
- hardcb_r1a: P04 70.25, P10 88.37

### Run checks (info files)
- info_hard_new_P01_71000_71007: 8 streams, recognised [-1], late [()], labels changed 0, stored-tape mismatches n/a
- info_hard_new_P02_71000_71007: 8 streams, recognised [-1], late [()], labels changed 0, stored-tape mismatches n/a
- info_hard_new_P03_71000_71007: 8 streams, recognised [-1], late [()], labels changed 0, stored-tape mismatches n/a
- info_hard_ret_P04_71000_71003: 8 streams, recognised [10, 22, 29], late [(100,)], labels changed 34, stored-tape mismatches n/a
- info_hard_ret_P04_71004_71007: 8 streams, recognised [10, 27, 28, 30, 47], late [(100,)], labels changed 35, stored-tape mismatches n/a
- info_hard_ret_P04_71008_71011: 8 streams, recognised [10, 15], late [(100,)], labels changed 25, stored-tape mismatches n/a
- info_hard_ret_P04_71012_71015: 8 streams, recognised [10, 22, 32], late [(100,)], labels changed 43, stored-tape mismatches n/a
- info_hard_ret_P10_71000_71003: 8 streams, recognised [10], late [(100,)], labels changed 7, stored-tape mismatches n/a
- info_hard_ret_P10_71004_71007: 8 streams, recognised [10, 12], late [(100,)], labels changed 5, stored-tape mismatches n/a
- info_hard_ret_P10_71008_71011: 8 streams, recognised [10], late [(100,)], labels changed 2, stored-tape mismatches n/a
- info_hard_ret_P10_71012_71015: 8 streams, recognised [10], late [(100,)], labels changed 7, stored-tape mismatches n/a
- info_hardcb_new_P01_71000_71007: 8 streams, recognised [-1], late [()], labels changed 0, stored-tape mismatches n/a
- info_hardcb_new_P02_71000_71007: 8 streams, recognised [-1], late [()], labels changed 0, stored-tape mismatches n/a
- info_hardcb_new_P03_71000_71007: 8 streams, recognised [-1], late [()], labels changed 0, stored-tape mismatches n/a
- info_hardcb_ret_P04_71000_71003: 8 streams, recognised [10, 22, 29], late [(100,)], labels changed 33, stored-tape mismatches 0
- info_hardcb_ret_P04_71004_71007: 8 streams, recognised [10, 27, 28, 30, 47], late [(100,)], labels changed 43, stored-tape mismatches n/a
- info_hardcb_ret_P04_71008_71011: 8 streams, recognised [10, 15], late [(100,)], labels changed 25, stored-tape mismatches n/a
- info_hardcb_ret_P04_71012_71015: 8 streams, recognised [10, 22, 32], late [(100,)], labels changed 51, stored-tape mismatches n/a
- info_hardcb_ret_P10_71000_71003: 8 streams, recognised [10], late [(100,)], labels changed 6, stored-tape mismatches 0
- info_hardcb_ret_P10_71004_71007: 8 streams, recognised [10, 12], late [(100,)], labels changed 5, stored-tape mismatches n/a
- info_hardcb_ret_P10_71008_71011: 8 streams, recognised [10], late [(100,)], labels changed 2, stored-tape mismatches n/a
- info_hardcb_ret_P10_71012_71015: 8 streams, recognised [10], late [(100,)], labels changed 5, stored-tape mismatches n/a
