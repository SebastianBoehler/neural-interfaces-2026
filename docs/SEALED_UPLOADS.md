# Sealed-phase upload plan and log (3 Nov 2026)

Supersedes the upload sequence in docs/SEALED_PLAN.md (R19e / R15 order of 30 Oct). Every upload needs
the user's explicit authorisation for that upload, after a live check of phase and quota.

## Facts (live Codabench API, 3 Nov 21:53Z)

- Sealed Final Phase 30483: 2026-11-04 23:01Z → 2026-11-21 22:59Z (00:01 CET 5 Nov → 23:59 CET
  21 Nov), status "Next", **1 upload per day, 5 in total**, 0 submissions.
- Task 36592 "BCI decoding", created 5 Oct. The task's programs are not visible to participants (the
  API returns 404), so we cannot confirm the binding before the first run.
- Rules (docs/REVIEW_2NOV.md): a failed run uses no quota, a cancelled run does. Force_Best keeps
  the best sealed score, and logs are hidden. The daily reset is probably UTC midnight (unverified).
- Our leaderboard entry is our best upload, so **whichever arm scores best is the one audited**. Each
  arm must have a rebuild recipe before it goes up.

## Arms (all `submission.py` `25d51a6e…`, the frozen code; ZIPs in `runs/upload/`)

| Arm | ZIP (SHA-256) | sel Δ | dla Δ | ds Δ | sel without P10 | Real worker | Audit recipe |
|---|---|---|---|---|---|---|---|
| cand_nxif | `sealed_cand_nxif.zip` `0a562e62…` | +2.91 | +3.18 | +2.63 | +0.99 | code: yes (1024447); EEGNeX / IFNet layers replica only | committed, validated, incl. all-A100 |
| cand_nxif_emx | `sealed_cand_nxif_emx.zip` `98d39600…` | +2.85 | +3.92 | +1.77 | +1.34 | as cand_nxif | cand_nxif recipe + `set_params.py` |
| cand_m7r4 | `sealed_cand_m7r4.zip` `060850ee…` | +2.33 | +2.59 | +2.08 | — | sinc / fbcnet TorchScript ran (1026185) | docs/AUDIT_M7R4.md, validated 4 Nov (754 / 755, 99.81 %) |
| R19e (jit_g) | `sealed_r19e_jit_g.zip` `011e56bc…` | 0 | 0 | 0 | 0 | **this exact ZIP** (1024447) | R19e fresh-clone audit (30 Oct) |

Deltas are dev sel = mean(ds, dla) over R19e on pretraining seeds 0–11, never tuned on P04 / P09 or
the warm-up board. Each ZIP is byte-identical to its official A100 replay and worker-image replica
(files unchanged). The in-sample scores on the public windows only check agreement.

What the arms test:
- cand_nxif: best dev score.
- R19e: no teacher. It is insurance, and it measures the teacher's effect on sealed people. The only
  unseen held-out person we have, P09, lost 5.9 with the teacher.
- cand_nxif_emx: leans toward new people (57 % of the sealed set).
- cand_m7r4: a different teacher (four other architectures).

## Sequence (proposal; each day still needs authorisation)

| Day | Arm | Why |
|---|---|---|
| 1 (5 Nov) | cand_nxif | best expected score; first real-worker run of the EEGNeX / IFNet layers |
| 2 | R19e (jit_g) | insurance if the teacher fails on new people; S1 − S2 = the teacher's sealed effect Δ_T |
| 3 | rule below | |
| 4–5 | decided with the user from S1–S3 | |

**Day-3 rule (fixed before any sealed score):**
- Δ_T ≥ +0.5, so the teacher transfers: upload cand_nxif_emx.
- Δ_T < +0.5, so the teacher transfers weakly or not at all: upload a cand_nxif variant with teacher
  weight 3, built on demand (export + replays, about 1.5 h).
- Days 4–5 candidates:
  - the weight in the direction S1–S3 point (3 or 7);
  - cand_m7r4 (audit recipe validated 4 Nov);
  - the remaining arm.

Paired differences between arms on the same sealed windows are far less noisy than absolute scores,
but differences under about 0.5 should still be read as ties.

**If a run fails** (no quota used): the next upload is `sealed_r19e_jit_g.zip`. This exact ZIP
already ran on the real worker, so it secures a score. Then the failing package is compared with
its replica.

## Before each upload

1. Phase 30483 is "Current"; our submission count and the daily quota are read from the page; Chrome
   is logged in as `s_boehler`.
2. The ZIP's SHA-256 equals the table.
3. The user has authorised this upload by name.
4. Afterwards: record the submission id, status, score and duration in the log below.

## Log

- 3 Nov 23:00 CET: the user authorised day 1 ("Yes, authorise day 1": cand_nxif as the first sealed
  submission on 5 Nov). A one-shot session check-in is set for 5 Nov 00:17 CET to run the checks above
  and upload. Days 2–5 still need their own authorisation.

| Day | Date | Arm | Submission | Status | Cell-mean BA | Duration |
|---|---|---|---|---|---|---|
