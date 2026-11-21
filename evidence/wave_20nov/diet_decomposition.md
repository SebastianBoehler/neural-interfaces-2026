# Recognised adaptation diet decomposition

[Preregistration](../../docs/LEDGER.md#recognised-adaptation-diet-decomposition--21-november) · [Fable R6 PROTEUS](fable_r6_diet.md#held-orders-7180871815) · [Quarter Scherer transfer](DECISION_21NOV.md#quarter-final-checks) · [Delivery scheme audit](fable_r8_scheme.md)

Reference: signed production `em_half`, SHA-256 `833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d`. Scherer uses its Scherer-free equivalent, `376e0f26ac1660e8330df8a0100a74cc82b3f3b0858a7a77433de7149bac823b`. The signed source applies a 0.5 multiplier to recognised SessionEM and keeps native TSEM. Arm b uses the quarter ZIP's 0.25 multiplier on native TSEM. The four arms start at the native recognition latch. All four retain both EM fits.

| Arm | Change after recognition |
| --- | --- |
| a | Bypass CountFilter. Pass RouteB directly to CSP. |
| b | Apply 0.25 of native TSEM's log-likelihood correction. |
| c | Reduce the SessionEM multiplier from 0.5 to 0.25. |
| d | Bypass CountFilter and apply the quarter TSEM multiplier. |

## Dispatch

- **PROTEUS, Leonardo A100.** Run `cap250/runs/diet-decomposition-21nov`. The scorer uses the Fable R6 causal tape, `production-20261120-32a94ab7` folds, pinned recognition manifest `c4bbc41808d296aa4e2ee18151bf9680ed46f465b83462e94a6be632bed3dd3e` and full-session orders 71808–71815. The returning folds enrol visit 1 of P04/P10 and stream visits 2/3.
- The first Leonardo smoke `65057430` printed v3 `OK`. The CountFilter never latched on its chosen P04 visit 2 stream. The replacement smoke `65057831` uses P10 visit 3, order 71808, where R8's label-free detector recorded CountFilter activation at delivery 41. Replacement held array `65057833` (70/71/94/95) depends on that smoke. The original array `65057431` was cancelled before scoring.
- **Scherer, Galvani RTX 2080 Ti.** Run `ni26_diet_decomp_20261121`. Nine B-ret participants, same visit-1 enrolment overlays as the signed quarter experiment, visit 2 replayed in both schemes on 71864–71871. The four arms load the patched source in temporary worker directories. B-new 50/50 calculations use the authenticated zero-recognition bank and a separate live identity smoke.
- The first Galvani smokes `3150100/3150101` stopped at a broken relative archive symlink before replay. Corrected the link to the signed absolute path. Replacement smokes `3150107/3150108` gate arrays a–d `3150109–3150112`. Existing r8-repair jobs continued.
- Locally, the SHA-256 check and production patch anchor check passed for every arm. The Scherer reference bank reproduced B-ret **55.962% full-session, 58.229% within-run** and zero B-new recognition. Existing Scherer baseline report SHA-256 values matched the published receipts.
- Both instruments passed their active smoke. Leonardo reproduced 199 pinned labels on P10 visit 3, with recognition at delivery 11 and CountFilter active from delivery 41. All four live 84-window prefixes matched the tape, and the three-component recomposition matched the original quarter exactly. Scherer reproduced all pinned `em_half` labels and no-recognition identity (0/240 changed for each arm). On the recognised 120-window smoke a/b/c/d changed 5/6/1/8 labels, all from the latch onward.
- The 20:44 CET scheduler estimate put several independent 2080 Ti arrays after the cutoff. The remaining direct tasks were held while the single-capture NI-V replay was queued on RTX 2080 Ti (smoke `3150491`, array `3150492`). The smoke briefly showed a next-day estimate, then passed by 20:50 CET. It reproduced the exact em_half posteriors and all four live arm labels on the pinned 120-window return. Two independent live a shards provide a full-grid overlap check. Galvani also started a paired A100 smoke `3150535` with array `3150536` dependent on it. The A100 path uses native BF16 MANAS and recomputes `em_half` on every stream. The ledger records both amendments before opening accuracy.

## Results

Deltas are percentage points against `em_half`, with paired 80% intervals from 20,000 draws resampling people and then orders independently within each sampled person. As a collector check, the archived quarter result reproduced PROTEUS +0.191 pp at 50/50 and Scherer −0.673 pp full-session / −1.553 pp within-run returning. PROTEUS has two returning people. Scherer has nine and uses a projected channel layout. These follow-up contrasts reuse participants and orders.

Promotion requires PROTEUS 50/50 ≥ +0.10 pp with lower bound > 0 and Scherer returning ≥ −0.20 pp in both schemes. Any promoted production ZIP needs its SHA-256, an unchanged archive inventory outside the patch, 379-window held-new zero, an official warm-up pair, and one RTX 2080 Ti end-to-end timing by 22:30 CET.

### PROTEUS held 71808–71815

All four Leonardo shards completed by 20:53 CET. The collector reproduced the 32 recognised returning streams and zero false recognitions. Reference BA was 74.397% held-new, 81.599% returning and 77.998% at 50/50. Each arm changed 0/21,240 held-new labels. Receipt SHA-256 `ff5518ec424f93187bf2a58e8637e49fb6c2629134de10c4ac59b5782f6b9d50`.

| Arm | Returning Δ [80%] | 50/50 Δ [80%] | P04 / P10 Δ | Returning labels changed | PROTEUS gate |
| --- | ---: | ---: | ---: | ---: | --- |
| a | +0.002 [−0.081, +0.082] | +0.001 [−0.041, +0.041] | +0.069 / −0.065 | 34 / 6,096 | Miss |
| b | +0.135 [−0.082, +0.383] | +0.068 [−0.041, +0.191] | +0.258 / +0.013 | 224 / 6,096 | Miss |
| c | +0.228 [+0.095, +0.363] | +0.114 [+0.047, +0.182] | +0.255 / +0.201 | 75 / 6,096 | Pass |
| d | +0.170 [−0.099, +0.456] | +0.085 [−0.049, +0.228] | +0.330 / +0.011 | 253 / 6,096 | Miss |

Arm c was the only survivor of the PROTEUS gate. Both returning participants improved.

### Scherer B-ret 71864–71871

All nine RTX 2080 Ti shards in `3150492` completed by 21:05 CET. Each arm paired all 72 recognised B-ret streams and 8,400 windows per scheme. Reference B-ret BA was 55.962% full-session and 58.229% within-run. The shared replay matched all 3,840 decisions on the two independent live a shards and matched every pinned `em_half` label and recognition age on the full grid. Receipt SHA-256 `96d34f44aea80657977b1a3e5ed43026a5265c7128e26e6fb7ac3273d2d02de7`.

| Arm | B-ret full-session Δ [80%] | B-ret within-run Δ [80%] | Full-session 50/50 Δ [80%] | Within-run 50/50 Δ [80%] | Labels changed full / within |
| --- | ---: | ---: | ---: | ---: | ---: |
| a | −0.069 [−0.162, +0.012] | −0.608 [−0.977, −0.231] | −0.035 [−0.081, +0.006] | −0.304 [−0.489, −0.116] | 35 / 341 |
| b | −0.623 [−1.422, +0.179] | −1.202 [−1.893, −0.541] | −0.312 [−0.711, +0.089] | −0.601 [−0.947, −0.270] | 619 / 631 |
| c | +0.007 [−0.172, +0.182] | +0.271 [+0.086, +0.460] | +0.003 [−0.086, +0.091] | +0.136 [+0.043, +0.230] | 123 / 120 |
| d | −0.577 [−1.374, +0.230] | −1.576 [−2.341, −0.813] | −0.289 [−0.687, +0.115] | −0.788 [−1.171, −0.407] | 635 / 779 |

The Scherer 50/50 column uses zero B-new effect from the archived no-recognition bank. Each live B-new smoke confirmed the patch preserved labels with recognition absent. The external result suggests TSEM reduction explains most of the quarter diet's transfer loss. CountFilter bypass also costs accuracy within-run. The end-to-end contrasts include downstream interactions.

The nine paired Galvani A100/BF16 shards (`3150536`) agreed on all four transfer rulings. Each returning point differed from its 2080 Ti counterpart by at most 0.039 pp. For c, the A100 returning points were −0.031 [−0.202, +0.136] full-session and +0.270 [+0.093, +0.451] within-run. This checks precision sensitivity on the same participants and orders. Receipt SHA-256 `62ba4ff7d89845f80f898d8a03d139c90cf2daabbfba9f7fefcf1feafec000b4`.

### Promotion

Arm c passed all scoring thresholds. The other three arms missed the PROTEUS and Scherer gates.

Built `final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing_recognised_em_half_diet_c.zip` from the signed production parent. SHA-256 `cba9b208fbff7ce416efd8b5ce68ae1ad0771980d84e754433bd33e9996d0bca`. The builder checked the full member inventory and all payloads. Only `submission.py` differs. Galvani RTX 2080 Ti job `3150612` started at 21:06 CET for held-new parity. Official warm-up pair `3150613` and unprofiled Solver timing `3150614` depend on its success.

Held-new parity completed at **21:10:59 CET**, with 0/379 label changes and no recognition in either P01 or P02 on order 71800. The largest RouteB or terminal score difference was 5.14e−6. Receipt SHA-256 `2645692adb4e28cd625ef0699d98d765726621179d9cfffcf6bace872e4a4a58`. Both downstream checks started at 21:11 CET.

- **Official warm-up pair**, `3150613`, completed at **21:22:53 CET** on RTX 2080 Ti cn111. Both replays per ZIP scored 100.000% cell-balanced accuracy. All four streams per replay completed without errors and the extracted files were unchanged. Replay digests matched the signed ZIPs. Receipt SHA-256 `cad1758c1dae6b90c6fc98daec19b55d06f15354e459f90d02a819a89761395a`.
- **Unprofiled end-to-end timing**, `3150614`, completed at **21:22:22 CET** on RTX 2080 Ti cn122. One warm-up session was discarded. The six measured sessions supplied 1,131 windows per ZIP. Cap250 / em_half / c measured 138.24 / 166.76 / 167.34 ms/window. The candidate was 0.58 ms/window slower in this paired run. The established cap250 additive proxy projects 3,127.7 s for em_half and **3,139.3 s** for c. The sealed RTX PRO 4500 remains unmeasured. Receipt SHA-256 `b49f67703ffbd808f24f1e3453199688533cc9114329004af4d17f416f58a64a`.

**Arm c qualified at 21:22:53 CET**, before the 22:30 CET cutoff. The local ZIP's SHA-256 was verified again after both jobs completed. No Codabench upload was made.

## Galvani final fill

Registered before dispatch. All scoring and timing ran through Slurm. The unprofiled timing uses the same signed archives, warm-up and six measured sessions as job `3150614`. Nine node-pinned repeats were submitted at 21:26 CET:

| RTX 2080 Ti node | Slurm job |
| --- | ---: |
| cn102 (preemptable) | 3150646 |
| cn111 | 3150640 |
| cn114 | 3150641 |
| cn115 | 3150642 |
| cn116 | 3150638 |
| cn117 | 3150643 |
| cn121 | 3150644 |
| cn122 | 3150639 |
| cn123 | 3150645 |

Scherer NI-V B-ret uses a pinned 71840 A100/BF16 smoke (`3150647`) followed by nine paired shards (`3150648`) for orders 71872–71879. The collector computes cell-balanced accuracy and 80% intervals over people and orders.

### Completed timing repeats

Times are ms/window across the same 1,131 measured windows per archive. Projections are seconds using the paired increase over cap250 and its 2,554.9166 s sealed receipt.

| Node | Job | cap250 | em_half | c | c vs em_half | em_half projection | c projection | Receipt SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cn122 | 3150639 | 147.17 | 176.22 | 172.67 | -2.013% | 3,138.3 | 3,067.1 | `02f8c1d0f8d7a619b6b465ecc705f2200e1e6e90bb1a4c3f57813d1921b2bac8` |
| cn115 | 3150642 | 155.46 | 188.35 | 184.29 | -2.158% | 3,215.4 | 3,133.8 | `8350eb955e220c679813635e80436174deda9657b2009eeda5bb723b229c288a` |
| cn111 | 3150640 | 153.24 | 183.69 | 180.25 | -1.873% | 3,166.5 | 3,097.4 | `82bcf48f57a2e4963866a8cd88b8c8b9c7e663ad8f7f0b8ea8c2a8fa09dbf087` |
| cn114 | 3150641 | 158.14 | 187.89 | 188.86 | +0.515% | 3,152.3 | 3,171.8 | `deb0d065d7889182782014ce9608edc4e4ff572ab389663503037e9d08ed7508` |

Across four completed nodes, arm c ranged from −2.158% to +0.515% versus em_half. Three repeats were faster by more than the predicted ±1% band. The largest measured slowdown was below the +1% flag. The largest projected arm c runtime was **3,171.8 s** on cn114, 428.2 s below the 3,600 s compute budget. The largest em_half projection was **3,215.4 s** on cn115. These are paired RTX 2080 Ti projections for the sampled nodes.

The other five nodes had GPUs occupied beyond the cutoff. Slurm estimated 01:55 CET on 22 November for cn102 and 02:01 CET for cn116, cn117, cn121 and cn123. All five pending jobs were cancelled at 21:41:55 CET. No cn116 timing completed.

### Scherer NI-V B-ret, orders 71872–71879

Pinned smoke `3150647` passed on cn231. Live arm c matched all 120 causal replay labels. All nine A100/BF16 shards in `3150648` finished by 21:39:52 CET. The collector paired 72 recognised visit-2 streams and 8,400 windows per scheme. Independent row regeneration matched all 288 arm-stream row lists using manifest SHA-256 `8ea44213…`. Every fresh stream reproduced the em_half Solver's labels, RouteB output and terminal posterior exactly. Pre-latch labels matched between the arms.

| Delivery | em_half BA | Arm c BA | Returning Δ pp [80%] | Labels changed |
| --- | ---: | ---: | ---: | ---: |
| Full-session | 56.460% | 56.549% | +0.089 [−0.109, +0.293] | 115 / 8,400 |
| Within-run | 58.019% | 58.133% | +0.114 [−0.058, +0.289] | 138 / 8,400 |

Both points clear the −0.20 pp transfer flag. Within-run stays positive, below the +0.25 pp prediction, and both intervals cross zero. This release varies delivery order for the same nine participants. Execution sources are in `b531ed20`. Collector receipt SHA-256 `46596e4df5d5bd51e4ed159478623486eb183b910b04b1c6a6806fc05ede153f`.
