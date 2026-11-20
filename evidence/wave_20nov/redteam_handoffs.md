# RT-1 · clean returning-fold handoff

**H1–H5 and the NI-V confirmation completed by 00:00 CET on Saturday 21 November.** R1a was the only keep-adapting arm to clear the clean returning sign rule. It is a candidate for runtime qualification. Kappa40 alone cleared the data-only sign rule and remains undecided after NI-V.

## Instrument and checks

Real fold packages from `production-20261120-32a94ab7`, with P04 or P10 visit 1 enrolled and visits 2–3 held out. The real Solver emitted one label per delivered window on full-session shuffled orders 71000–71003. The v3 replay reproduced all six reference scores and printed `OK`. The paired scorer regenerated the four pinned deliveries and checked full cell coverage. PROTEUS data SHA-256 starts `2be4187fe33a`.

Results use balanced accuracy averaged equally over participant × visit × interface cells. Changes are percentage points. Paired 80% intervals resample people, then orders within people (20,000 draws, seed 20261121). Per-person intervals resample orders. **Two returning people is a thin sample.** *The 20/80 P04/P10 mix and 43% sealed returning share are unverified assumptions.* The share is reported as context.

| Arm | Primary reference | P04 BA | P10 BA | P04 Δ [80%] | P10 Δ [80%] | Equal cells Δ [80%] | 20/80 Δ* |
|---|---|---:|---:|---:|---:|---:|---:|
| stock | — | 70.20 | 89.22 | — | — | — | — |
| hardcb | stock | 70.06 | 89.22 | -0.14 [-0.48, +0.20] | +0.00 [-0.13, +0.13] | -0.07 [-0.28, +0.10] | -0.03 |
| hardcb_safe | stock | 70.06 | 89.22 | -0.14 [-0.48, +0.20] | +0.00 [-0.13, +0.13] | -0.07 [-0.28, +0.10] | -0.03 |
| R0z | hardcb | 70.06 | 89.22 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 |
| R1a | hardcb | 70.40 | 89.28 | +0.34 [+0.14, +0.54] | +0.07 [+0.00, +0.13] | +0.20 [+0.07, +0.37] | +0.12 |
| R5 | hardcb | 69.63 | 88.71 | -0.43 [-0.82, -0.04] | -0.51 [-0.90, -0.12] | -0.47 [-0.76, -0.18] | -0.49 |
| R2 | hardcb | 69.49 | 88.31 | -0.58 [-1.06, -0.09] | -0.91 [-1.17, -0.64] | -0.74 [-1.04, -0.41] | -0.84 |
| R4 | hardcb | 69.76 | 88.84 | -0.30 [-0.79, +0.18] | -0.38 [-0.56, -0.19] | -0.34 [-0.55, -0.09] | -0.36 |
| g6 | hardcb | 70.15 | 88.00 | +0.08 [-0.51, +0.67] | -1.21 [-1.66, -0.77] | -0.57 [-1.28, +0.14] | -0.96 |
| g12 | hardcb | 67.21 | 86.38 | -2.85 [-4.31, -1.40] | -2.84 [-3.23, -2.45] | -2.85 [-3.58, -2.10] | -2.84 |
| beta02 | hardcb | 72.59 | 87.91 | +2.53 [+2.00, +3.07] | -1.31 [-1.57, -1.05] | +0.61 [-1.34, +2.60] | -0.54 |
| beta04 | hardcb | 72.28 | 85.01 | +2.21 [+1.21, +3.21] | -4.21 [-4.46, -3.96] | -1.00 [-4.25, +2.34] | -2.93 |
| kappa40 | hardcb | 70.48 | 89.08 | +0.41 [-0.12, +0.95] | -0.14 [-0.20, -0.07] | +0.14 [-0.17, +0.54] | -0.03 |
| leak_all | stock | 87.77 | 95.07 | +17.57 [+16.83, +18.31] | +5.85 [+5.35, +6.35] | +11.71 [+5.82, +17.68] | +8.19 |
| noroute | stock | 61.06 | 89.72 | -9.14 [-10.32, -7.97] | +0.50 [+0.17, +0.83] | -4.32 [-9.23, +0.55] | -1.43 |

Order columns below follow seeds 71000 through 71003 and use the primary reference above.

| Arm | P04 Δ by order | P10 Δ by order |
|---|---|---|
| hardcb | +0.27/+0.01/-1.09/+0.26 | -0.26/+0.00/+0.00/+0.26 |
| hardcb_safe | +0.27/+0.01/-1.09/+0.26 | -0.26/+0.00/+0.00/+0.26 |
| R0z | +0.00/+0.00/+0.00/+0.00 | +0.00/+0.00/+0.00/+0.00 |
| R1a | +0.01/+0.27/+0.81/+0.27 | +0.26/+0.00/+0.00/+0.00 |
| R5 | -1.28/-0.25/+0.29/-0.49 | +0.27/-1.29/+0.03/-1.06 |
| R2 | +0.61/-1.07/-0.53/-1.33 | -1.29/-1.28/-0.24/-0.81 |
| R4 | +0.62/-0.26/-1.31/-0.26 | -0.24/-0.76/-0.49/-0.02 |
| g6 | -1.57/+0.79/+0.56/+0.55 | -1.04/-2.03/-0.24/-1.55 |
| g12 | -6.62/-1.02/-2.96/-0.81 | -2.08/-3.64/-2.28/-3.37 |
| beta02 | +1.34/+2.65/+2.67/+3.47 | -1.57/-1.04/-0.79/-1.84 |
| beta04 | +0.01/+2.92/+4.01/+1.92 | -4.17/-4.61/-3.60/-4.47 |
| kappa40 | -0.80/+1.35/+0.82/+0.28 | -0.01/-0.26/-0.02/-0.26 |
| leak_all | +16.48/+18.56/+16.14/+19.10 | +6.26/+4.49/+6.50/+6.15 |
| noroute | -11.62/-6.92/-10.91/-7.12 | +0.62/-0.24/+1.09/+0.54 |

| Arm | Δ P04 / P10 vs stock | Δ P04 / P10 vs hardcb_safe | Routed labels changed P04 / P10 |
|---|---|---|---|
| hardcb | -0.14 / +0.00 | +0.00 / +0.00 | — |
| hardcb_safe | -0.14 / +0.00 | +0.00 / +0.00 | — |
| R0z | -0.14 / +0.00 | +0.00 / +0.00 | — |
| R1a | +0.20 / +0.07 | +0.34 / +0.07 | — |
| R5 | -0.57 / -0.51 | -0.43 / -0.51 | — |
| R2 | -0.72 / -0.91 | -0.58 / -0.91 | — |
| R4 | -0.44 / -0.38 | -0.30 / -0.38 | — |
| g6 | -0.06 / -1.21 | +0.08 / -1.21 | 109 / 36 |
| g12 | -2.99 / -2.84 | -2.85 / -2.84 | 221 / 72 |
| beta02 | +2.39 / -1.31 | +2.53 / -1.31 | 100 / 36 |
| beta04 | +2.07 / -4.21 | +2.21 / -4.21 | 179 / 94 |
| kappa40 | +0.27 / -0.14 | +0.41 / -0.14 | 18 / 7 |
| leak_all | +17.57 / +5.85 | +17.71 / +5.85 | — |
| noroute | -9.14 / +0.50 | -9.00 / +0.50 | — |

## H1 · keep adapting after recognition

The R0z smoke matched 377/377 labels. Full R0z and hardcb_safe each matched hardcb on all 3,048 labels. R1a made one post-recognition refit at window 100 in each of 16 stream-orders. R5 made zero late refits and fused on 96–109 late windows per stream-order. R2 and R4 made 16–18 late refits and fused on 96–109 late windows each. R4 loaded `TTTFineTune`; the other keep arms loaded `FineTune`. The traced streams checked the real Solver after each prediction and rejected degraded paths.

**R1a** clears the P04 > 0 and P10 ≥ −0.3 pp rule. NETBCI was +0.36 pp and ds007221g was +0.24 pp BA2 (14 people). Scherer full B-ret was −0.12 pp [−0.28, +0.04] in Lane R4, and −0.05 pp [−0.12, +0.03] on the 16 Leonardo full shuffles (71008–71023, people and orders resampled). The original ~3,372 s projection used the earlier hardcb estimate. Adding the paired +3.3 to +3.5 s per recognised session to the newer hardware-specific baselines gives about 3,511–3,652 s at 46 recognised sessions. The 3,600 s budget remains unresolved on the RTX PRO 4500. R5, R2 and R4 miss the clean sign rule, with a negative equal-cell interval for each. None hit the registered P04 ≤ −0.3 pp on every order. [Paired timing and Scherer](laneR4_keepadapt.md) · [NETBCI](laneB10_returning_stress.md) · [ds007221g design](laneR4b_dose.md) · [Updated runtime](final_stack.md#review-fix-2--additive-runtime-and-load-time-self-test-20-nov-sat-0000-0100-cet)

## H2 · paired donor swap

The verified shipped ZIP (`9c4d5538…`) supplied all 14 requested arrays and six net files to each stock fold copy. With orders, source weights outside the swap, and the Solver held fixed, P04 rose +17.57 pp and P10 +5.85 pp. The +11.71 pp equal-cell uplift reproduces the scale of the earlier gap and clears the registered +8 pp P04 attribution threshold. The swap tests these donor components jointly.

## H3 · enrolled-visit sanity check

P04 stock visit 1 scored **98.96%** in each interface on order 71000 (job 64988100). This clears the registered ≥95% check.

## H4 · same-source route suppression

Lane RP jobs 64985932 (zero-change smoke) and 64986260 (full) held source weights fixed, disabled both recognition gates, and retained adaptation. All 16 stream-orders logged `recognised_at -1` and zero v6 errors. Each suppression tape matched its stock delivery. P04 fell on all four orders. P10 rose on three. The registered criterion that suppression meets or beats stock on both people failed. This contrast includes the adaptation that continues when recognition is suppressed. [Lane RP](laneRP_returning_person.md)

## H5 · data-only dose

All five variants reproduced 190/190 P01 labels with zero recognition and changed labels on routed returning windows. G12 lost at least 0.3 pp on P04 on every order. Beta02 and beta04 gained on P04 and lost more than 1 pp on P10. Kappa40 alone cleared the point-estimate rule. Its effect on P04 remains uncertain and its unverified 20/80 effect is −0.03 pp.

Kappa40 NI-V B-ret confirmation was registered before running (`9ff53b5a`). Smoke 64989686 paired all 120 windows and changed one label. Array 64989695 completed 9/9 people, with 16 full shuffles and 16 within-run shuffles on visit 2. The comparator was Scherer-free hardcbB enrolled on the same visit 1. All delivery rows matched, every stream was recognised, and the solver logged zero v6 errors.

| NI-V delivery | Kappa40 Δ vs hardcbB [paired 80%] | Changed labels |
|---|---:|---:|
| Full-session shuffle (primary) | +0.22 [−0.04, +0.50] | 352 / 16,800 |
| Within-run shuffle | +0.15 [−0.10, +0.41] | 347 / 16,800 |

The primary interval crosses zero, so kappa40 remains **undecided for packaging** under the pre-registered rule. Scherer data SHA-256 starts `a61887997532`. The scoring command is `python scripts/redteam/niv_dose.py score data/scherer2015/windows.npz NI_V_REFERENCE_DIR CANDIDATE_TAPE_DIR`.

## Reproduction

Registration `050198b1`. All 38 original Leonardo jobs completed. Run `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/redteam-21nov`; release `15a639718601e959015fd533fa9106476614aa8c`. Job IDs and dependencies are in `jobs.tsv`. H1 and H5 returning runs used the pinned harness with post-predict tracing to verify late adaptation and routed label changes. H2, H3, R0z and hardcb_safe used the pinned `stream_fold.py`. Score the copied tapes with `python scripts/redteam/report_ret.py data/proteus_windows.npz RECEIPT_DIR`. [Dispatch](DISPATCH_21NOV.md) · [H1–H5 specification](redteam_fable.md#handoff-checks-for-an-executor-register-the-block-below-in-docsledgermd-before-scoring-anything)
