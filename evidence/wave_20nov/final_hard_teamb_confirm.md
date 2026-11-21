# Final hard-only Team B confirmation · 21 November 2026

Registered in [docs/LEDGER.md](../../docs/LEDGER.md#final-hard-only-team-b-confirmation--21-november-registered-0355-cet-before-any-timing-ni-v-or-confirmation-score-below-was-read) before any score below was read. Queue and job IDs: [OVERNIGHT_21NOV.md §4](OVERNIGHT_21NOV.md). Nothing was uploaded. Raw compact results: [final_hard_teamb_confirm/](final_hard_teamb_confirm/) (NI-V reports, confirmation collector JSON). The four ZIPs are those of [final_hard_teamb.md](final_hard_teamb.md) and [cb_speed.md](cb_speed.md), pinned by SHA-256 (`zips.sha256` in the Galvani release `ni26_final_confirm_20261121`).

**Headline.** (1) The CUDA graphs engage on the RTX 2080 Ti and change no label, but the isolated additive benches under-count Team B on this hardware: inside the real Solver the Team B encoder costs about 36 ms per window against 12 ms in the bench, so the production-like 2080 Ti projections are **3,520–3,657 s for hard + nine (nine25 3,544–3,657, nine15 3,520–3,624), 3,744–3,807 s for hardcb + nine25, 2,995–3,047 s for plain hardcb cbgraph**. On a Galvani A100-PCIe the same Solver runs give 3,132–3,160 s and 3,308 s. (2) NI-V Scherer: Team B changes nothing outside noise (no veto). (3) 16 fresh confirmation orders replicate the gain on the real fold packages, held-new and returning.

## 1 · RTX 2080 Ti timing (Galvani, worker SIF, torch 2.14.1+cu130, braindecode 1.8.1)

Projection `2,554.9 + 20,083 x added_ms / 1000`; added ms is relative to exact cap250 unless stated. The sealed cap250 took 127.2 ms/window (2,554.9 s over 20,083 windows). The RTX PRO 4500 and its host CPU are untested.

**Graph path engaged.** On the 2080 Ti, `check_hard_team_b.py timing` and `terminal` and `check_cb_speed.py bench` assert the expected graph replay counts (Team B batch 1 and 64, CBraMod) and raise otherwise; all runs passed. Graph and eager posteriors differ by 0.0 on all 2,655 windows, the injected capture failures fall back to eager with a warning, and the nine25 hard ZIP's graph path matched eager labels on all 14 sessions of shuffle 71000 (terminal jobs 3145949/3145950, 0 changed). Smokes 3145716, 3145717. Isolated CBraMod graph 8.2-9.0 ms against 27-28 ms original.

| 2080 Ti | A: isolated additive benches (2-3 runs / ZIP, ms/window → s) | B: official warm-up replay, cap250 / cand x 2 (min wall, gross +ms/window → s) | C: Solver end-to-end, same streams (+ms/window → s) | A100 projection, final_hard_teamb / cb_speed |
|---|---|---|---|---|
| hard + nine25 | 23.2-29.5 → **3,021-3,148** (typically 3,024-3,054; 3,113-3,148 on cn114/cn121) | 120.9 / 158.1 s, +49.3 → **3,544** | +49.3 to +54.9 (3 runs) → **3,545-3,657**; hard_safe_st alone +10 to +20 (2,759-2,957) | 2,930.7 |
| hard + nine15 | 23.7-29.1 → **3,032-3,139** | 121.0 / 159.8 s, +51.4 → **3,587** | +48.1 to +53.3 (3 runs) → **3,520-3,624** | 2,989.9 |
| hardcb nine25, both graphs | 32.2-38.1 → **3,202-3,321** | 124.3 / 169.1 s, +59.3 → **3,747** | +59.2, +62.4 → **3,744-3,807** | 3,056 warm / 3,073 amortised |
| hardcb cbgraph (plain) | 20.2-22.5 → **2,960-3,007** | 127.4 / 146.1 s, +24.5 → **3,047** | +21.9, +23.0 → **2,995-3,017** | 2,920 / 2,932 |

- A = `confirm_gpu.sbatch hard-timing|cb-bench` (jobs 3145945-48, 3145993-96, 3146067-74; nodes cn111, cn114-117, cn121-123; bench numbers include session start capture and actor restore, CBraMod amortised from first-window means). Team B graph per window was 11.7-12.2 ms on six nodes and 16.4-17.8 ms on cn114 and cn121, so the bench varies by node. The benches hold two models on an 11 GB card and print allocator OOM-retry warnings; results still passed.
- B = `friday/qualify_package.py pair` with `NI26_QUAL_GPU="2080 Ti"` (jobs 3145937-3145940). All four pairs passed: public score 0.9959229 for the three Team B archives and 0.9945789 for hardcb cbgraph (= cap250 0.9945789 for the latter), repeat scores equal, files unchanged, 4 v6 streams per replay without errors. The gross figure assigns the whole wall difference over 755 windows to the per-window cost, so it includes load and per-session overhead; but column C, which excludes load, gives the same number.
- C = `submissions/final_stack/e2e_timing.py` (new): official `fresh_model` Solver streams, 7 of 14 public sessions of shuffle 71000, one discarded warm-up session, arms alternating, CUDA synchronised, load excluded. Jobs 3146075 (hard family) and 3146077 (hardcb family), repeats 3146345 and 3146346 (hard family, cn122, cn115) and 3146347 (hardcb family, cn115), all on shared nodes (exclusive requests starved behind other users' arrays and were cancelled). 2080 Ti cap250 is 134-153 ms/window there (sealed: 127.2). Plain hardcb with the eager CBraMod member is +44.3 ms (3,445 s), the graph removes 22 ms of that.
- **Why A and C disagree.** For hard actors and CBraMod the Solver adds what the bench says (hardcb cbgraph +21.9 against 20-22), but Team B costs +32-37 ms per window in the Solver against 12 ms isolated (profile jobs 3146177, 3146190 CPU windows, 3146191 GPU windows: Team B posterior inside the Solver 36.2-36.5 ms; hard 12.8-13.7 ms). GPU utilisation is about 21 % with SM clocks averaging 1.5 GHz (min 300 MHz): the cost is launch and CPU bound, so it depends on the host. The time inside Team B includes any queued asynchronous GPU work of earlier components, so the 36 ms is an upper bound for Team B itself, not for the total; the total (+32 to +37 ms over hard_safe_st in the 2-arm runs, +49 to +55 over cap250 in the 4-arm runs) is measured directly. Root cause not isolated.
- A100-PCIe (Galvani, worker SIF; jobs 3146334 hard family, 3146335 hardcb family; shared nodes): cap250 129.6 and 147.9 ms/window; hard_safe_st +11.9 (2,793 s); nine25 +30.2 → **3,160 s**; nine15 +28.7 → **3,132 s**; hardcb cbgraph +23.7 → **3,031 s**; hardcb nine25 both graphs +37.5 → **3,308 s**. Team B inside the Solver 14.3-17.5 ms against 7.4 ms on the Leonardo A100-SXM bench. The Leonardo official pair (warm difference +16-19 s over 755 windows) is the lower end of this range.
## 2 · NI-V Scherer exact packages (Galvani 2080 Ti, exact ZIPs through the Solver)

Tags: `hss` = `final_stack_hard_safe_st`, `hss_n25` / `hss_n15` = the two Team B ZIPs (Team B bank is Scherer-free, PROTEUS only). Releases A (orders 71000-71007) and B (71008-71015), each: B-new = 9 people x both visits, full-session and within schemes; B-ret = enrolled visit 1, visit 2 streamed (16 orders: 8 full + 8 within). Paired deltas in pp against `hss`, people-resampled 80 % intervals as in `laneN_niv_final.md`; the `fa` arm was not run. Same instrument limits as before: 22 of 41 channels interpolated, absolute level about 48 % new / 56-57 % returning, magnitudes are weak evidence.

| Contrast | B-new full-session A / B (8 orders each) | B-new within A / B | B-ret full A / B | B-ret within A / B | sealed-weighted* A / B |
|---|---|---|---|---|---|
| nine25 − hss | +0.30 [−0.22, +0.81] / −0.01 [−0.47, +0.44] | −0.48 [−0.94, −0.02] / −0.44 [−1.03, +0.14] | −0.11 [−0.82, +0.61] / −0.00 [−0.56, +0.57] | −0.19 [−1.19, +0.82] / −0.19 [−1.09, +0.71] | +0.12 [−0.30, +0.54] / −0.01 [−0.36, +0.35] |
| nine15 − hss | +0.16 [−0.26, +0.58] / −0.05 [−0.43, +0.33] | −0.06 [−0.40, +0.27] / +0.02 [−0.35, +0.38] | +0.34 [−0.20, +0.87] / −0.15 [−0.59, +0.28] | −0.01 [−0.67, +0.65] / +0.05 [−0.50, +0.64] | +0.24 [−0.09, +0.57] / −0.09 [−0.37, +0.19] |

- 16-order means of the two releases (point only, intervals are not pooled): nine25 B-new full +0.15, B-ret full −0.06; nine15 B-new full +0.05, B-ret full +0.09. Absolute B-new full: hss 48.43 / 48.10, nine25 48.73 / 48.09, nine15 48.59 / 48.05.
- Registered rule: veto if the B-new full-session delta is below −0.3 pp or its upper 80 % bound below 0. Neither dose is close (smallest point −0.05, every upper bound > 0). Worst single person (B-new full): nine25 −1.09 / −1.72, nine15 −1.09 / −0.62; (B-ret full) nine25 −2.40 / −1.46, nine15 −1.46 / −1.07. Within-scheme B-new nine25 is −0.48 / −0.44 with the upper bound at −0.02 / +0.14 in the first release: a small out-of-lab negative on the secondary scheme, not on the headline. The prediction (about 0 [−0.4, +0.4] new, −0.1 [−0.7, +0.5] returning) held. Out of lab Team B neither helps nor hurts measurably; the in-lab gain of section 3 does not appear on Scherer.
**Releases C and D added (21 Nov 09:00 CET, 8 orders each, 71200-71207 / 71016-71023 per the queue; collectors 3146203/04, 3146476/77; E queued at the 300-task cap, not run).** Delta vs `hss`, pp (SE), people-resampled, 9 people:

| Contrast | Release | B-new full | B-new within | B-ret full | B-ret within | sealed-weighted* |
|---|---|---|---|---|---|---|
| nine25 | C | +0.13 (0.29) | −0.16 (0.43) | −0.07 (0.55) | −0.22 (0.60) | +0.04 |
| nine25 | D | +0.13 (0.31) | −0.44 (0.42) | −0.30 (0.57) | −0.15 (0.67) | −0.05 |
| nine15 | C | +0.34 (0.28) | −0.12 (0.25) | +0.17 (0.39) | +0.12 (0.49) | +0.27 |
| nine15 | D | +0.21 (0.29) | −0.12 (0.29) | −0.04 (0.35) | +0.24 (0.47) | +0.10 |

Pooled view, A-D (32 orders, equal 8-order weights, point means only; intervals are not pooled and releases share the same 9 people, so no SE is claimed): nine25 B-new full **+0.14** (A +0.30, B −0.01, C +0.13, D +0.13), B-new within −0.38, B-ret full −0.12, weighted +0.03. nine15 B-new full **+0.17** (+0.16, −0.05, +0.34, +0.21), B-new within −0.07, B-ret full +0.08, weighted +0.07. Veto rule (B-new full below −0.3 pp or upper 80 % bound below 0): not triggered in any of the 8 release/dose cells (smallest point −0.05; every upper 80 % bound > 0). Only B-ret full nine25 D (−0.30, SE 0.57) touches −0.3, on a secondary, noisy scheme. Out of lab Team B stays about zero to slightly positive on the full scheme and slightly negative within for nine25 (−0.16 to −0.48 in all four releases). *Sealed-weighted: UNVERIFIED 57/43 new/returning assumption.

- Not run: `hard.zip` vs `hss` identity on NI-V (the registered sanity), the `fa` arm.

## 3 · Confirmation orders on the real fold packages (Leonardo `laneTB-confirm-21nov`, verified Lane TB-verify code, hard base)

Orders 71156-71171 (16 fresh), 12 folds (10 held-new, ret_P04, ret_P10), base / nine15 / nine25. Smoke 65003908: base arm equals the stored hard tape on 190 windows, nine25 changed 10. Arrays 65003909-65003916 and collector 65003988 completed; collector self-check on the Lane TB-verify orders (65003984) reproduced +0.77 (nine15) held-new exactly. Paired 80 % intervals resample people, then orders, official cell weights; 57/43* is the UNVERIFIED sealed mix.

| Contrast | Held-new Δ pp [80%] | Returning Δ [80%] | P04 / P10 | J | 57/43* |
|---|---|---|---|---|---|
| hard nine25 − hard, 16 new orders | **+0.89 [+0.45, +1.26]** | **+1.21 [+0.91, +1.51]** | +1.19 / +1.23 | +1.05 [+0.79, +1.28] | +1.03 |
| hard nine15 − hard, 16 new orders | **+0.64 [+0.35, +0.89]** | **+0.78 [+0.54, +1.00]** | +0.77 / +0.79 | +0.71 [+0.52, +0.88] | +0.70 |
| hard nine25 − hard, pooled 24 (with 71124-71131) | +0.92 [+0.44, +1.31] | +1.00 [+0.75, +1.25] | +0.94 / +1.07 | +0.96 [+0.69, +1.18] | +0.97 |
| hard nine15 − hard, pooled 24 | +0.68 [+0.41, +0.92] | +0.68 [+0.49, +0.86] | +0.69 / +0.66 | +0.68 [+0.51, +0.83] | +0.68 |

- Held-new per order nine25: +0.07 to +1.89, all 16 positive; nine15 +0.16 to +1.51, all 16 positive. Held-new by person (16 orders): nine25 P05 +2.96, P04 +1.61, P10 +0.96, P06 +0.84, P03 +0.55, P07 +0.52, P09 +0.34, P08 +0.29, P01 −0.37, P02 −0.42; nine15 P05 +2.04, P04 +0.99, P10 +0.75, P06 +0.72, P09 +0.66, P07 +0.47, P03 +0.16, P01 +0.10, P08 −0.07, P02 −0.36. The gain is concentrated in P04 and P05; P01 and P02 are negative for nine25.
- Returning per order (16): nine25 15 of 16 positive (−0.16 once), nine15 14 of 16 (−0.01, −0.91). Returning is two people, so its interval understates person-level uncertainty.
- Registered rule: replicated (people-nested lower bound > 0 and mean ≥ +0.3) for both doses, held-new and returning. Prediction check: held-new nine25 +0.9 [+0.3, +1.4] ✓ (+0.89), nine15 +0.75 ✓ (+0.64); returning nine25 +0.6 predicted, **+1.21 came in higher** (earlier eight orders +0.59), nine15 +0.45 predicted, +0.78. The two returning measurements (8 orders +0.59 / 16 orders +1.21) differ by more than the intervals, so the returning level is not settled.
- Doses remain inseparable on held-new (nine25 +0.25 over nine15, intervals overlap); nine25 is ahead on returning in this sample. Neither is a gate.
