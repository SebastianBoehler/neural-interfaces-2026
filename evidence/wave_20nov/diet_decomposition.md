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

## Results

Pending cluster replays. Deltas are percentage points against `em_half`, with paired 80% intervals from 20,000 draws resampling people and then orders independently within each sampled person. As a collector check, the archived quarter result reproduced PROTEUS +0.191 pp at 50/50 and Scherer −0.673 pp full-session / −1.553 pp within-run returning. PROTEUS has two returning people. Scherer has nine and uses a projected channel layout. These are follow-up contrasts on previously used participants and orders.

Promotion requires PROTEUS 50/50 ≥ +0.10 pp with lower bound > 0 and Scherer returning ≥ −0.20 pp in both schemes. Any promoted production ZIP needs its SHA-256, an unchanged archive inventory outside the patch, 379-window held-new zero, an official warm-up pair, and one RTX 2080 Ti end-to-end timing by 22:30 CET.
