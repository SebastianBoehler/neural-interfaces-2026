# Sealed-phase upload plan and log

> **Current facts (11 Nov 2026, 22:30 UTC).** Sealed phase 30483 closes 21 Nov 2026, 22:59 UTC. Allowance: 7 uploads
> per team, 1 per UTC day; **4 used, 3 left**. Best: submission 1048575, `cand_v41_bio`, 0.8161546323 (see "11 Nov,
> 18:31 UTC" below). A failed run uses no quota; a manually cancelled one does. The sections "Facts", "Arms" and
> "Sequence" below are the plan of 3 Nov and are history: the quota of 5, the arms and the planned order no longer
> apply, and `cand_m7r4` was never uploaded. "Before each upload" and the log are current.

Supersedes the upload sequence in docs/archive/SEALED_PLAN.md (R19e / R15 order of 30 Oct). Every upload needs
the user's explicit authorisation for that upload, after a live check of phase and quota.

## Facts (live Codabench API, 3 Nov 21:53Z)

- Sealed Final Phase 30483: 2026-11-04 23:01Z → 2026-11-21 22:59Z (00:01 CET 5 Nov → 23:59 CET
  21 Nov), status "Next", **1 upload per day, 5 in total**, 0 submissions.
- Task 36592 "BCI decoding", created 5 Oct. The task's programs are not visible to participants (the
  API returns 404), so we cannot confirm the binding before the first run.
- Rules (docs/archive/REVIEW_2NOV.md): a failed run uses no quota, a cancelled run does. Force_Best keeps
  the best sealed score, and logs are hidden. The daily reset is probably UTC midnight (unverified).
- Our leaderboard entry is our best upload, so **whichever arm scores best is the one audited**. Each
  arm must have a rebuild recipe before it goes up.

## Arms (all `submission.py` `25d51a6e…`, the frozen code; ZIPs in `runs/upload/`)

| Arm | ZIP (SHA-256) | sel Δ | dla Δ | ds Δ | sel without P10 | Real worker | Audit recipe |
|---|---|---|---|---|---|---|---|
| cand_nxif | `sealed_cand_nxif.zip` `0a562e62…` | +2.91 | +3.18 | +2.63 | +0.99 | code: yes (1024447); EEGNeX / IFNet layers replica only | committed, validated, incl. all-A100 |
| cand_nxif_emx | `sealed_cand_nxif_emx.zip` `98d39600…` | +2.85 | +3.92 | +1.77 | +1.34 | as cand_nxif | cand_nxif recipe + `set_params.py` |
| cand_m7r4 | `sealed_cand_m7r4.zip` `060850ee…` | +2.33 | +2.59 | +2.08 | — | sinc / fbcnet TorchScript ran (1026185) | docs/archive/AUDIT_M7R4.md, validated 4 Nov (754 / 755, 99.81 %) |
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
- 5 Nov 06:45 CET: the 00:17 check-in never uploaded (owner API: 0 submissions, 0/1 today, 0/5 total; phase
  "Current", image v4-sealed). The user re-authorised cand_nxif in this session ("Upload cand_nxif now"); ZIP
  SHA-256 checked (`0a562e62…`) and uploaded through Chrome as `s_boehler`.

| Day | Date | Arm | Submission | Status | Cell-mean BA | Duration |
|---|---|---|---|---|---|---|
| 1 | 5 Nov (UTC) | cand_nxif (`0a562e62…`) | 1034698 | Finished (v4-sealed: the TorchScript teacher loads) | **0.7794596196** (rank 9 of 46 at 07:54 CET; top 0.83) | 834.7 s |

**6 Nov: new arms (plan in docs/REVIEW_5NOV.md §5d/§6).** Live check 11:25 UTC: phase 30483 "Current", 1 of 5
used, 0 today; board top seohee / timothewt 0.825, 3rd 0.816, us 21st.

| Arm | ZIP (SHA-256) | Dev J vs packaged REC | Replays (A100 / worker replica) | Sealed runtime est. | Audit recipe |
|---|---|---|---|---|---|
| cand_rec_ts2 (REC + core1 + Hilbert core) | `sealed_cand_rec_ts2.zip` `e9f13567…` | +1.30 (≈ +0.8–1.1 after selection) | 211.3 s / 173.8 s, identical, files unchanged | 38–43 min (≤ ~47) | docs/AUDIT_SBN.md |
| cand_rec_ts2_tsem (cand_rec_ts2 + TSEM stage after SessionEM) | `sealed_cand_rec_ts2_tsem.zip` `9c31a167…` (36.1 MB) | TSEM +1.30 mean on fresh bases (≈ +1.2–1.4 after selection) | 218.9 s / 181.2 s, identical (0.974922 in-sample), files unchanged | ≈ 40–45 min (per-window scaling) | docs/AUDIT_SBN.md |
| cand_rec_ts2_tsem256 (hedge, alternate; TSEM PC256, λ 0.2; data-only, code `243c9901…`) | `sealed_cand_rec_ts2_tsem256.zip` `3799c0e0…` (57.1 MB) | vs cand_rec_ts2_tsem: ΔJ +0.30 (sd 0.21, B2 seeds 3–5; seen +0.53, new +0.07; picked after the confirmation) | 225.7 s / 188.9 s, identical (0.968190 in-sample), files unchanged | ≈ 42 min (paired ratio 1.00; ≤ 44) | docs/AUDIT_SBN.md |
| cand_rec_ts2_tsem_route (alternate; cand_rec_ts2_tsem + returning-person routing after TSEM; code `ab5fc0b9…`) | `sealed_cand_rec_ts2_tsem_route.zip` `34f60eca…` (57.3 MB) | vs cand_rec_ts2_tsem, pre-registered, 4 fresh bases: dJ +0.73 to +0.86 (new 0.00, seen +1.45 to +1.72, 0 impostor routings) | 211.0 s / 198.0 s, identical (0.982734 in-sample), files unchanged | ≈ 42 min (paired ratio 1.01; ≤ 47) | docs/AUDIT_SBN.md |
| cand_rec_ts2_tsem_route3 (alternate, supersedes v1 / v2; routing v3: gate on the unrouted tail, personal head in the core1 slot of a routed tail; code `3ef76b58…`) | `sealed_cand_rec_ts2_tsem_route3.zip` `9903c1ef…` (57.3 MB) | vs cand_rec_ts2_tsem, pre-registered, 4 fresh bases: dJ +0.99 to +1.12 (new 0.00, seen +1.98 to +2.23, P10 ≥ 0, 0 impostor routings; Stieger FAR 0.42 %) | 225.0 s / 188.3 s, identical (0.980088 in-sample), files unchanged | ≈ 42 min (paired ratio 1.01; ≤ 44) | docs/AUDIT_SBN.md |
| cand_rec_ts2_tsem_route3eb (alternate, PROVISIONAL, superseded by route3geb; route3 + early boost; code `db3aa8b6…`) | `sealed_cand_rec_ts2_tsem_route3eb.zip` `2f34907d…` (57.3 MB) | vs cand_rec_ts2_tsem (dev sim, 4 fresh bases): EB + routing +1.31 to +1.86 (EB + routing vs routing +0.28 to +0.74; new +0.53 to +1.09, P10 −0.5 to −1.2) | 222.9 s / 189.8 s, identical (0.973578 in-sample), files unchanged | ≈ 43 min (paired ratio 1.03; ≤ 45) | docs/AUDIT_SBN.md |
| cand_rec_ts2_tsem_route3geb (alternate, staging only; route3 + gated early boost; code `786bcab0…`) | `sealed_cand_rec_ts2_tsem_route3geb.zip` `74433d9b…` (57.3 MB) | vs cand_rec_ts2_tsem (dev sim, mean of 4 fresh bases): GEB + routing +1.80 (EB + routing +1.53, routing only +1.05) | 196.3 s / 183.2 s, identical (0.981432 in-sample), files unchanged | ≈ 42 min (paired ratio 1.00; ≤ 43) | docs/AUDIT_SBN.md |
| cand_rec_ts2_tsem_route3geb_fast (runtime insurance: route3geb's 32 data files + faster code `ccbce506…`; zero-weight nets skipped, shared FIR bank, tangent/net overlap) | `sealed_cand_rec_ts2_tsem_route3geb_fast.zip` `cfd1a4b4…` (57.3 MB) | same outputs as route3geb: every stage bitwise on all 2,655 public windows on A100 (worker image and venv) | 122.8 s / 111.5 s, identical (0.981432 in-sample), files unchanged | ≈ 19 min against 2,687 s, ≈ 26 min against the rerun's 3,732 s (solver ratio 0.41; ≤ 39 min without overlap gain) | docs/AUDIT_SBN.md |
| cand_rec_ts (REC + core1), fallback | `sealed_cand_rec_ts.zip` `1a4b48af…` | +0.65 | 196.1 s / 160.5 s | 37–39 min | docs/AUDIT_SBN.md |

- 6 Nov 12:45 UTC: **no upload today (user decision)**: keep iterating on ground-up solutions instead. Staged:
  cand_rec_ts2 (reviewed) and cand_rec_ts2_tsem (`sealed_cand_rec_ts2_tsem.zip` `9c31a167…`, replays identical
  218.6 s / 181.0 s, package == harness 2,655/2,655). Independent review (`scratchpad/review_tsem/`): safe to
  upload: only the inline TSEM stage and 8 new weight keys differ; future-perturbation, own-seed, reset and deepcopy
  tests bit-identical; packaged == batch tsem.py exactly; λ 0 byte-identical to cand_rec_ts2; basis independently
  recomputed (subspace cosine ≥ 0.999999998); robust to short sessions, missing seeds, singular covariance and bad
  windows; no ZIP size limit (36 MB fine); sealed runtime ≈ 40–45 min (≤ ~50). Code must be committed before audit.

- 6 Nov ~19:05 UTC: the user authorised uploading cand_rec_ts2_tsem in the 7 Nov UTC slot ("Yes, upload it"), as
  the transfer/compatibility signal; further uploads only for clearly better packages. The user then asked to upload
  immediately (6 Nov slot still unused); the scheduled check-in was cancelled.
- 6 Nov 19:20 UTC: checks passed (phase "Current", 0/1 today, 1/5 total, logged in as s_boehler, SHA-256 9c31a167…
  locally and re-checked in the browser on the uploaded bytes). The file is 36 MB, above the Chrome tool's 10 MB
  limit, so it was served once from 127.0.0.1 (`scratchpad/relay/serve_one.py`) into the page's own file input.
  Submission **1038262** created 19:20:15 UTC, status Running. Score to be read from the owner API (needs login).

| Day | Date | Arm | Submission | Status | Cell-mean BA | Duration |
|---|---|---|---|---|---|---|
| 2 | 6 Nov (UTC) | cand_rec_ts2_tsem (`9c31a167…`) | 1038262 | Finished (v4-sealed runs the Hilbert core and TSEM) | **0.8196888507** (+0.0402 over cand_nxif) | 2,497.2 s (41.6 of 60 min) |

Reading: S ≥ 0.812, so dev gains transfer about 1:1 (dev J +3.7 to +3.9 over cand_nxif → sealed +4.02; predicted
0.8165–0.8185 at full transfer). Keep TSEM; skip the paired ablation. 0.83 needs about +1 dev J more. Runtime margin
is now about 18 min, so any added stage must stay cheap.

Reading rule (fixed before the score): S ≥ 0.812 near-full transfer; 0.797 ≤ S < 0.812 partial (next: paired
ablation sealed_cand_rec_ts2.zip); S < 0.797 re-plan. Failed run: compare with the replica, upload nothing else
without authorisation.

- 7 Nov ~13:00 UTC: user decision: no further upload until the dev estimate reaches about 0.83 ("push to .83
  first").
- 7 Nov evening: independent review of `sealed_cand_rec_ts2_tsem_route3geb.zip` (`74433d9b…`):
  - **Verdict:** safe to upload, no blocking defect (`scratchpad/review_geb/`).
  - **Routing:** 0 impostor routings in every stress setting, including open-gate weak bases. GEB does not push new
    people toward routing (max impostor score 0.188 vs τ 0.22).
  - **Open-gate path:** bitwise equal to the reference, including open-and-routed cases.
  - **Causality, reset and wiring:** verified on the full package.
  - **Runtime:** sealed ≈ 42 min.
  - **Still needed:** the code (786bcab0 and the references) must be committed for the audit.
- Live board 7 Nov ~17:30 UTC: bryanyahir03 0.838, seohee 0.825, timothewt 0.825, xianglong_bao 0.823, us 0.820 (5th).
- 7 Nov 17:45 UTC: the user authorised commit, push and upload ("do that commit, push and upload"). Code committed and
  pushed as c4d600d (submission.py 786bcab0, identical to the ZIP's).
- Checks before upload: phase "Current", 2/5 used, 0 today, logged in as s_boehler, SHA-256 74433d9b… locally and
  re-checked in the browser.
- Uploaded through the local relay as submission **1040847**, created 17:49:21 UTC, status Running.

| Day | Date | Arm | Submission | Status | Cell-mean BA | Duration |
|---|---|---|---|---|---|---|
| 3 | 7 Nov (UTC) | cand_rec_ts2_tsem_route3geb (`74433d9b…`) | 1040847 | Finished | **0.8240266154** (+0.0043 over 0.8197) | 2,687.2 s (44.8 of 60 min) |

Expectation was about 0.827 central (0.822–0.835). The realised +0.43 points is about 0.6 of the honest dev estimate
(+0.75), whereas the large TSEM step transferred about 1:1, so small stacked gains transfer only partly.
Live board 7 Nov ~19:10 UTC: 0.838, 0.825, 0.825, **us 0.824 (4th)**, 0.823. **New baseline: route3geb.** 2 uploads
left. Staged candidates: routing v3 and routing v3 + EB (builds in progress), tsem256, route v1.

7 Nov evening cleanup (docs/archive/STATE_7NOV.md): `submissions/ensemble/submission.py` is now the lean file `dacee85a…`. It
reproduces the three uploaded packages byte for byte. The staged packages that were never uploaded (route v1, route3,
route3eb, tsem256) keep their ZIPs in `runs/upload/`. Their code is commit `c4d600d` (docs/AUDIT_SBN.md, top note).

Independent review of cand_rec_ts2 (6 Nov, `scratchpad/review_ts2/`): safe to upload. The ZIP differs from
cand_rec_ts only in submission.py and weights.npz as described; the new member is causal (future-perturbation and
reset tests bit-identical); weights equal dev arm4 to 3e-16; robust to zero/NaN/degenerate windows. Runtime estimates
were corrected: per-window scaling rather than total replay time (which includes a fixed overhead).

## 10–11 Nov: organiser rescoring of all Track 02 sealed submissions

Official commit 8734b2f6 (10 Nov 20:34 UTC): "Following a technical fix on the Track 02 sealed evaluation backend,
previously Finished Track 02 sealed submissions are being rerun automatically" (same IDs, no quota used). Scores are
provisional while rescoring continues. Allowance raised from 5 to **7** per team (1/day unchanged), so we have
**4 left**. The FAQ no longer states that windows are strictly chronological within a session ("the same evaluation
order is used for all submissions"); the public `streaming.py` still checks time order.

Public board 11 Nov 00:10 UTC (provisional):

| Rank | Team | Score | Duration (s) |
|---|---|---|---|
| 1 | sungjunwang | 0.830 | 1,222 |
| 2 | bryanyahir03 | 0.824 (was 0.838) | 968 |
| 3 | seohee | 0.821 (was 0.825) | 739 |
| 4 | liuyin | 0.819 | 2,220 |
| 5 | timothewt | 0.818 (was 0.825) | 2,909 |
| 6 | euntack_im | 0.814 | 682 |
| **7** | **us** | **0.813 (was 0.824)** | **3,732 (was 2,687)** |

Our displayed duration now exceeds the 3,600 s execution limit's nominal value, so runtime is a real risk for any
heavier package. Per-submission rescored values need the owner view (log in to Codabench).

Owner view after the rescoring (11 Nov, user logged in; allowance 7 confirmed by the user, 4 left):

| Submission | Package | Original | Rescored | Δ | Duration original → rescored |
|---|---|---|---|---|---|
| 1034698 | cand_nxif | 0.7794596196 | **0.7722919674** | −0.72 | 834.7 → 811.6 s |
| 1038262 | cand_rec_ts2_tsem | 0.8197 | **0.8080263921** | −1.17 | 2,496.9 → 2,496.9 s |
| 1040847 | cand_rec_ts2_tsem_route3geb | 0.8240266154 | **0.8134828319** | −1.05 | 2,687.2 → 3,732.2 s |

The ranking of our packages is unchanged. The step increments still transfer:
- TSEM step: +3.57 sealed for +3.8 dev (0.94×).
- Routing + GEB: +0.55 sealed for the honest +0.75 dev (0.73×; it was +0.43 before the fix).

This points to a uniform drop from the backend fix, not a penalty on our order-dependent stages. (12 Nov: the drop was not uniform across our packages (−0.72, −1.17, −1.05). Ten stream cuts and orders were fingerprinted against these numbers and none fits; the cause is still unknown. On 10 Nov the organisers also removed the chronological-order guarantee from the FAQ. Ledger §62.) route3geb's rescored
duration rose by 1,045 s while the TSEM package's did not. The run is above the nominal 3,600 s limit yet finished,
so worker speed varies and runtime headroom matters.

11 Nov: **cand_rec_ts2_tsem_route3geb_fast** staged as runtime insurance. It is route3geb's data files unchanged plus
Codex's runtime code (from the lean `dacee85a…`), packaged as one `submission.py` `ccbce506…`. Upload ZIP
`runs/upload/sealed_cand_rec_ts2_tsem_route3geb_fast.zip` `cfd1a4b452b6a840a71626447abe405bbcceaaf41f64a0eb04c0f7c2d43f417d`
(57,280,590 B).
- **Parity:** every stage is bitwise equal to the shipped ZIP on all 2,655 public windows (A100, worker image and venv).
- **CUDA nuance:** one cold-vs-warm first stream differs by up to 1.5e-8 (labels identical; docs/AUDIT_SBN.md).
- **Runtime:** A100 solver time is 0.41× route3geb's, so about 19–26 min sealed.
- Not uploaded; the user authorises each upload.

## 11 Nov: routing v4 checkpoint (staged, not uploaded)

**cand_v4_bio** = the fast package (route3geb outputs) plus a label-free identity gate in front of the personal heads.
Upload ZIP `runs/upload/sealed_cand_v4_bio.zip` `94185c2fd5f2e4de0d8107809fe51653213c15073156c54017a8149f824a9722`
(57,739,427 B); code `submissions/ensemble_v4/submission.py` `cd593655…`, committed.

| Check | Result |
|---|---|
| Dev against route3geb, six bases | dJ +0.45 to +0.72 (mean +0.59); seen +0.89 to +1.45; new 0.00 to +0.02; P10 −0.24 to +0.52 |
| External identification (67 held-out people, session level, τ 0.25) | 1.9 % false accepts, 86.9 % correct accepts (route3geb's gate: 10–42 % on Lee2019 / Stieger2021) |
| Code parity with the gate off | byte-identical to the fast package, all stages, 2,655 public windows |
| Integration (gate on) | gate decisions equal the stand-alone class; outputs equal the fast package before the first routed window of every session |
| A100 replay 3087470 / worker-image replay 3087471 | identical scores (0.9748375896 in-sample on the public test split; the fast package: 0.9814320116), same ZIP, files unchanged; 157 s / 95 s |
| Paired timing (ABBA 3087472) | 95.9 / 94.7 s against 93.2 / 93.5 s: 1.02× the fast package, so about 20–36 min sealed |
| Stress | noise, zeros, mains hum, offsets never route; amplitude scaling, time reversal and a flat channel do not break identification |

**Open before an upload:** an independent review of the package (as for earlier uploads) and the user's go-ahead after a
live quota and phase check. **Known limits:** one new-person dev session is routed at negligible weight (the gate
accepts about 2 % of impostors by design); the in-sample score on public people drops because personal heads replace an
in-sample core, which says nothing about sealed data; the dev gain is P04-driven; the main expected sealed effect
(more returning sessions accepted) is not measurable on dev.

**Independent review of cand_v4_bio (11 Nov, report `runs/bio1/review_v4.md`; the reviewer had the code and the
contract, not our conclusions): no blocking defect.**
- **Verified:** causal and reset-safe (changing later windows leaves earlier scores bit-identical; deep copy plus reset
  replays like a fresh model); without `bio_*` keys the new routing step equals the old one on 432 fuzz cases including
  every exception path; the package merge rule equals the dev replay's with no off-by-one; `Bio.scores` equals
  `TwoCov.llr` to 1e-15; no input the caller can pass raises or yields a non-finite ramp; about 0.85 ms per window.
- **Calibration caveat (medium, the user's call):** the threshold was calibrated scoring every 5th window with
  one-session galleries and a centring vector that included the probe; the package scores every window, P04 and P10 have
  three gallery sessions, and the centring never contains the probe. The 1.9 % false-accept rate is therefore a lower
  bound. A package-faithful recalibration is running (Galvani 3088231: every window, probe excluded, one- and
  three-session galleries). The earlier calibration did cover windows 10–200 (it ran with `BIO_KEEP=200`).
- **Hygiene, fixed in the builder:** the channel-order assertion never ran (now checked against the submission's channel
  list); the threshold default was 0.23 while the package uses 0.25 (the builder now requires `tau`).
- **Known and left in the staged code:** an out-of-range person index from the gate would raise an uncaught
  `IndexError` in the routed block; unreachable here because gallery and heads are both built from `route_names` (the
  builder now asserts equal lengths). To be closed with an assertion in `Route3.__init__` at the next code rebuild.
- **Weak spot of our own check:** `check_bio_gate.py` compares only the windows before the first routed one (140 of
  2,655, since every public session routes at window 10); full-length parity rests on the gate-off replay.

## 11 Nov, evening: routing v4.1 checkpoint (staged, not uploaded; supersedes cand_v4_bio)

**cand_v41_bio** = cand_v4_bio with the gate threshold moved from 0.25 to 0.26 (package-faithful recalibration, ledger
§14, second amendment) and one assertion from the review. Upload ZIP `runs/upload/sealed_cand_v41_bio.zip`
`53fc16797f541a81c5bf96f2c1bdbc2356d270ee8f38f978b9ba672b19da14be` (57,739,465 B); code
`submissions/ensemble_v4/submission.py` `45f9f2fd…`, committed. The first build's ZIP (`94185c2f…`) was deleted.

| Check | Result |
|---|---|
| External identification, scored as the package scores (every window, probe excluded from the centring) | 1.9 % false accepts, 86.7 % correct accepts at τ 0.26 |
| Dev against route3geb, six bases | dJ +0.45 to +0.66 (mean +0.58); seen +0.90 to +1.32; P10 −0.25 to +0.52; new 0.00 with 0 of 14 new-person sessions routed |
| Code parity with the gate off | byte-identical to the fast package, all stages, 2,655 public windows |
| Integration (gate on) | gate decisions equal the stand-alone class; outputs equal the fast package before the first routed window of every session |
| A100 replay 3088399 / worker-image replay 3088400 | identical scores (0.9748375896 in-sample on the public test split), same ZIP, files unchanged; 154 s / 109 s |
| Paired timing (ABBA 3088401) | 109.0 / 108.7 s against 107.0 / 107.9 s: 1.01× the fast package |
| Independent review | no blocking finding (of the first build; the only code change since is the assertion it asked for) |

**Before an upload:** your go-ahead after a live quota and phase check. **Known limits:** unchanged from the first
build (the in-sample drop is expected; the dev gain is P04-driven; three-session galleries were never tested externally;
the main expected sealed effect is not measurable on dev).

## 11 Nov, 18:31 UTC: upload 4, cand_v41_bio (routing v4.1)

- **Authorisation:** the user, 11 Nov evening ("if ... a rank improvement and valuable signal out of that submission then
  submit already"; then opened Codabench in Chrome for the upload). Our stated expectation: +0.2 to +0.8 sealed points,
  not top 3; a rank improvement is likely, not certain.
- **Checks before upload (live, in the logged-in browser):** phase 30483 "Sealed Final Phase" = Current; 0 of 1 used
  today, 3 of 7 in total; `can_make_submission` true; logged in as s_boehler; SHA-256 `53fc1679…da14be` and 57,739,465 B
  verified locally and again in the browser on the bytes that were attached.
- **Upload:** through a one-shot relay on 127.0.0.1 into the page's own file input (the Chrome tool cannot attach more
  than 10 MB). Two earlier attempts in a tab that was being reloaded created no submission; the third ran in its own tab.
- **Submission 1048575**, created 2026-11-11T18:31:09Z, status "Submitting" at creation.
- **Note:** the user deleted the first sealed submission (1034698, cand_nxif, 0.7723) from Codabench before this upload
  to free storage; 1038262 and 1040847 remain.

| Day | Date | Arm | Submission | Status | Cell-mean BA | Duration |
|---|---|---|---|---|---|---|
| 4 | 11 Nov (UTC) | cand_v41_bio (`53fc1679…`) | 1048575 | Finished | **0.8161546323** (+0.0026718 over route3geb's 0.8134828319) | 1,128.6 s (18.8 of 60 min) |

Reading rule (fixed before the score): against route3geb's rescored 0.8134828319, a paired gain ≥ +0.3 means
returning-person routing transfers and the personal template prior and trust weights are worth building; between −0.1
and +0.3 it is neutral and the returning-person line stops; below −0.1 the gate is routing new people and v4 is dropped.
A failed run uses no quota: compare with the replica, upload nothing else without authorisation. Never cancel it.

**Result of upload 4 (read 11 Nov ~19:00 UTC, owner API).** Cell-mean BA 0.8161546323; pooled BA 0.8159187836; accuracy
0.8159139571; duration 1,128.6 s.
- **Paired gain over route3geb: +0.267 points.** By the rule fixed beforehand this is the neutral band (−0.1 to +0.3),
  just below its upper edge: the gate does no harm and the gain is real, but returning-person routing does not transfer
  strongly, so the personal template prior and trust weights are not built.
- **Transfer:** the dev gain was +0.58 J; +0.267 sealed is 0.46× of it, a little below the 0.6–0.7× seen for small
  stacked gains. The hoped-for extra from accepting more returning sessions (87 % against 10–42 %) did not show up as a
  visible addition: the eight single-session public people gain little from a personal head, as P10 does.
- **Runtime:** 1,128.6 s against 3,732 s for the rescored route3geb: the fast runtime code works on the real sealed
  worker (0.30×), which frees about 40 minutes of budget.
- **Board, 11 Nov ~19:00 UTC (191 entries):** seohee 0.846; sungjunwang 0.830; xmqxmq 0.826; timothewt 0.825;
  bryanyahir03 0.824; navattack 0.821; liuyin 0.819; kyalcin 0.817; **us 0.816 (9th)**; kang5554 0.815. The field moved
  during the day (three entries are new or improved above us), so the rank is unchanged. Third place needs +1.0.
- **New baseline: cand_v41_bio.** 3 uploads left.

## 12 Nov: v5 checkpoint, cand_v5_own (built on Galvani, not staged as a ZIP, not uploaded)

**cand_v5_own** = cand_v41_bio plus the own-space personal stage (ledger §37, §41–§45): on windows the v4.1 gate
routes to public person j with ramp r, the routed output is multiplied by exp(0.10 · r · mean clipped log posterior of
three members fitted on j's public sessions in j's own space). Nothing else changes; 10 new arrays (`own_*`, 5.2 MB).

| Item | Value |
|---|---|
| Code | `submissions/ensemble_v5/submission.py`, SHA-256 `c3d3706e188f22d2aba2a324906338e379d799b0eef03702c8124a1dd2473d73` (v4.1 code + class `OwnStage`, 78 added lines) |
| Package | Galvani `runs/sealed/cand_v5_own/` (33 files); `weights.npz` `e88bc3b3936c2ce7d4f61c0e5ddd14f8b479737a937633c1da577cebd0108abf` (62,841,447 B) |
| Build | `python scripts/add_own_stage.py runs/sealed/cand_v41_bio runs/sealed/cand_v5_own beta=0.10` (job 3090163) |
| Dev replay of the package form (§45), six bases | dJ +0.89 (+0.66 to +1.00); seen +1.79 (P04 +3.51, P10 +0.07); new 0.000, no new-person session routed |
| A100 replay 3090169 / worker-image replay 3090170 | identical scores (0.9932767697 in-sample on the public split; v4.1: 0.9748375896), files unchanged; 166.2 s / 121.8 s |
| Runtime against v4.1 (154 s / 109 s on the same replays, different nodes; no paired timing yet) | about 1.08–1.12×, so roughly 21 min sealed against v4.1's 18.8 |
| Integration check `scripts/check_own_stage.py` (job 3090168, 35 min, all 2,655 public windows) | **passed**: outputs equal to cand_v41_bio on every window without a stage; the package's member log posteriors equal the study class (max difference 1.6e-12) and the staged output equals base × exp(β r log members) exactly (max difference 0.0); every public session is staged from window 10 to its own person (95 % of its windows); 27 labels change, in-sample accuracy 0.9917 against 0.9819 |

**Open before any upload:** a paired timing (job 3090327, running); an independent review of the code (running); the user's go-ahead after a live quota and phase check. **Known limits:** the in-sample rise is expected
(the stage's members are fitted on the public sessions) and says nothing about sealed data; the dev gain is P04's
(P10: +0.1); the eight single-session people are measured only within a session (§41).

### 12 Nov, later: review of cand_v5_own and the v5.1 rebuild (supersedes the first build; nothing staged or uploaded)

- **Paired timing of the first build (ABBA, job 3090327, one A100 node, worker image):** cand_v41_bio 114.8 / 113.0 s,
  cand_v5_own 119.6 / 121.9 s: **1.06×**, about 20 min sealed against v4.1's 18.8.
- **Independent review** (a reviewer given the code and the contract only): **no blocking defect.** Confirmed by read
  and unit tests: causal (the stage changes only the returned output; nothing it writes is read later); reset and deep
  copy clean; bitwise equal to v4 without the `own_*` keys and with the stage present but never triggered; members
  equal the study class to 1.2e-14 on real sessions; sklearn's `predict_proba` reproduced exactly; flat cost per
  window. Not covered by the review: an end-to-end run of the real package (done separately: integration check and
  both replays above), the montage-reordering path with the stage, Linux/A100 numerics.
- **Low-severity findings, fixed in v5.1** (`submissions/ensemble_v5/submission.py`, SHA-256 prefix `34eba3b0f3eb5a3d`;
  the first build's `c3d3706e…` is superseded):
  1. the stage caught only three exception types; now any failure inside it leaves the v4.1 output of that window,
     and numpy warnings are silenced inside the stage;
  2. a window with a flat or non-finite channel was staged from filter noise; such a window is now neither added to
     the session sums nor staged (the public data have none, so parity with the study code is unchanged);
  3. `Route3.stage` is now cleared on reset;
  4. checker: a NaN difference could pass and a stage that never ran could pass; both are now asserted;
  5. builder: channel names are now checked in order, not by membership.
  Not changed: no outlier guard on the running sums (one extreme window dominates them for the session, as in the
  study code and in v4.1's own stages); a rebuild will not reproduce the weights' SHA byte for byte (timestamps,
  BLAS), so the replayed build is the one to stage.
- **v5.1 results (12 Nov 02:10 UTC; package `runs/sealed/cand_v51_own` on Galvani; `submission.py` `34eba3b0f3eb5a3d…`,
  `weights.npz` `e88bc3b3936c2ce7…`, identical to the first build's weights):**
  - build + integration check 3090346: **passed** (equal to cand_v41_bio on unstaged windows; members equal the study
    class to 1.6e-12; staged output exact; 27 labels changed; the stage ran on 95 % of the public windows);
  - A100 replay 3090347 and worker-image replay 3090348: identical scores (0.9932767697 in-sample, the same as the
    first build), files unchanged; 204.4 s and 103.7 s (different nodes);
  - paired timing 3090349 (one A100 node, worker image): cand_v41_bio 118.5 s, cand_v51_own
    123.7 / 122.0 s, cand_v41_bio again 109.7 s (all four runs in): **1.08×**, so roughly 20 min sealed.
  - montage-reordering check 3090406 (`scripts/check_perm_stage.py`, the path the review could not cover): four
    sessions, 758 windows, 718 with the stage active, streamed once in training channel order and once with the
    channels permuted and announced through `ch_names`: labels equal, staged windows equal, maximum output
    difference 0.0. **Passed.**
  The candidate is replayed, checked, reviewed and timed. Not staged as a ZIP, not uploaded; an upload needs the user's
  go-ahead and a live phase and quota check.
