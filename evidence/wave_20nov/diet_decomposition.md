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

### Promotion

Arm c passed all scoring thresholds. Start exact production qualification for the SessionEM-only reduction. The other three arms missed the PROTEUS and Scherer gates.
