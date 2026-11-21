# Current state

## Final Track 2 submission

**Upload [submission.zip](../submission.zip) from the repository root.** It contains the qualified arm c decoder with a 0.25 SessionEM weight on recognised streams. The archive is 417,253,955 bytes.

SHA-256 `cba9b208fbff7ce416efd8b5ce68ae1ad0771980d84e754433bd33e9996d0bca`.

Run `shasum -a 256 submission.zip` from the repository root to verify the file. [Arm c qualification](../evidence/wave_20nov/arm_c_confirmation.md) · [Final sprint decision](../evidence/wave_20nov/arm_c_sprint_21nov.md)

## Track 2 contract

Task: one 41-channel EEG window at 120 Hz in, one of three tasks out, immediately. The model sees the delivered prefix,
no labels, no person ID and no recording-order guarantee. Cells (participant/session/interface) weigh equally in the
balanced accuracy. State resets between sessions. Rules, with sources: [OFFICIAL_CONTRACT.md](../OFFICIAL_CONTRACT.md)
(read upstream 2026-competition main for rules; the pin is for code only).

## Rules (upstream main @ 8447e181, 19 Nov)
- Sealed budget 3,600 s on one NVIDIA RTX PRO 4500; the 80 min worker timeout does not extend it. 4,800 s is wrong.
- Only the best eligible sealed score counts (Force_Best). Quota 7 per team, at most 1 per day, Failed counts for
  neither. **6 used, 1 left.** Deadline Sat 21 Nov 23:59 CET (22:59 UTC).
- **Organisers on Discord, 21 Nov:** the sealed score is computed for seen (returning) and unseen (new) participants
  separately, then averaged unweighted, so weight dev results 50/50 new/returning (this differs from the public
  `objective.py` plain cell mean; the organiser statement is the newer evidence). Returning share "not negligible", exact
  share private. Returning sessions are from different days. Runtime: 60 min of compute, model loading is extra, plus
  about 20 min Codabench slack; a timeout ends as Failed (no quota).
- Earlier candidates used a 3,060 s projection target and a 90 MB size target. The final selection uses the arm c qualification linked above.
- Never upload without the user's explicit authorisation; recheck live quota and phase first.

## Systems
| System | Evidence |
|---|---|
| v5.1b | Submission 1050185, 0.8197668759, 1,287 s. |
| **v6.6cap250, best sealed** | Submission 1058969, **0.8298729992**, 2,554.9 s, rank about 15-16 of the live table (19 Nov). Dev, corrected full system, shared fresh Sinc seed 0: +0.479 pp [paired 80% -0.222, +1.452] over v5.1b on ten held-new people (one order); within0 +2.546 pp [+1.641, +3.406] over P0r; matched full0 +0.667 pp [+0.033, +1.549]. [Sealed receipt](../friday/DECISION_18NOV.md#official-penultimate-receipt) · [Stage factorial](../friday/cap250c18/result-19nov.md) · [Within0](../friday/causal_replay/within0-overnight-queue.md#complete-d-eawcf-result--19-november-0224-cet) · [Full0](../friday/causal_replay/runs0_replay.md#common-source-full0-completed--19-november-0318-cet) |
| C++/CUDA worker | Reproduced cap250 labels and work counts on 2,655 ordinary and 1,178 forced-adaptation public windows (one timing pair per workload). Source fitting of the full production ensemble is incomplete. [Worker](../native/worker-parity/README.md) · [Timings](../evidence/next_wave_16nov/native366_worker.md) |

Exact cap250 is the reference for the historical comparisons below.
- **FBL2** (doubled FBLight): +0.383 within0, +0.488 full0 held-new; returning -0.130 equal cells / -0.208 at 20/80; Scherer -0.020. [External](../friday/cap250c21s/result-19nov.md) · [Corrected](../friday/cap250c21/result-19nov.md) · [Trial ZIP](../friday/cap250_fbl2_package/README.md)
- **Stock-protected FBLight**: +0.383 / +0.453 held-new; full runtime missed both RTX bounds, optimized archive missed the independent RTX posterior check. [Timing](../friday/cap250c21_count_view/README.md) · [Gate](../friday/cap250c21_gate/result-19nov.md)
- **F-C22 (multiresolution + FBL2)**: +0.538 / +0.778 held-new; external -0.017, did not qualify. [External](../friday/cap250c22_external/result-19nov.md) · [Corrected](../friday/cap250c22/result-19nov.md)
- **F-C20 (raw replacement)**: +0.305 / +0.331 held-new; Scherer -0.063; returning +0.130 / +0.052. [Returning](../friday/ea_returning/native_c20/README.md) · [Design](LEDGER.md#f-c20--replace-the-five-frozen-outer-raw-fbc-members-with-source-jitter-pupils)
- **F-C25 hard** (lead experimental alternate): +0.308 on two Scherer orders; A100 held-new +0.425/+0.374; pooled six orders +0.220 [+0.018, +0.432]; exact-archive external +0.050 [-0.370, +0.456]; hard+FBLight lost 0.579 externally; RTX conformance and runtime passed, G posterior drift exceeded 5e-5 in untouched stock too. [Pooled](../friday/cap250_hard_pooled/result-19nov.md) · [Exact archive](../friday/cap250_hard_exact/README.md#exact-archive-result--19-november-2026-100411-cet) · [Package](../friday/cap250_hard_package/README.md) · [Factorial](../friday/cap250_f32_cross/README.md#native-external-result)
- **Joint motor supervision**: +0.194/+0.066 over stock, trails hard; keep stock. [Held-new](../friday/cap250_f32/result-19nov.md) · [External](../friday/cap250_f32_external/result-19nov.md)
- **Gradient balance**: no promotion (within0 -0.272, full0 +0.146, external tie). [Standalone](../friday/fbc323_gradient_balance/result-19nov.md) · [Held-new](../friday/cap250_f35/result-19nov.md) · [External](../friday/cap250_balance_external/README.md)

## Validity caveat (18 Nov scoring audit)
Historical shuffled full-system comparisons keep recording-order transformer and shallow EA inputs (the `STRICT` guard
exempts them): treat them as full-system DEV with partial delivery replay, including the three-seed cap250 stock.
Repeated cohorts and new orders are not fresh-person confirmation. [Scoring audit](../evidence/next_wave_16nov/scoring_audit_18nov.md) ·
[Friday queue](../evidence/next_wave_16nov/FRIDAY_20NOV.md)

**Returning instrument (red team, 20 Nov).** The P04/P10 "visit 1 enrolled" packages used since 17 Nov (§321 bundle;
Lane A, R4, R4b tapes, D3, and the returning columns above scored through `friday/ea_returning`) keep the uploaded
package's Hilbert core, rest nets and count filter, which were fitted on the test visits. Their absolute level (90–97 %)
and the sign of returning levers measured on them are not trustworthy. Clean level from the real fold packages
(`production-20261120-32a94ab7`): P04 70.2 %, P10 89.2 %; hardcb − stock there −0.07 pp.
[Red team](../evidence/wave_20nov/redteam_fable.md)

## Inputs and checks
- Frozen ZIPs in `runs/sealed/final_zips/`; SHA-256 in [inputs.sha256](../friday/inputs.sha256). Local PROTEUS export: 2,655
  windows, ten people, 14 sessions (NEMAR nm000290 v1.0.0), made by [scripts/export_windows.py](../scripts/export_windows.py);
  [audit](../evidence/baseline/data_audit.json).
- `python scripts/friday_smoke.py` checks both ZIPs and loads each package; `python replay/check.py` checks six v3 scores in
  recording order. A shuffled scorer needs its own zero-change and pinned-order check. Neither measures private performance.
- Mac venv: `../neural-interfaces-2026/.venv/bin/python`. Galvani: native package and worker image from the pinned paths in the worker notes.
- Active files: [sparse-paths](../friday/sparse-paths); frozen history `git show cb3e05a:<path>`; replay inputs [replay/data.sha256](../replay/data.sha256).
