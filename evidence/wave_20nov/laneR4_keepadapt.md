# Lane R4 · keep adapting after recognition (registered Fri 20 Nov 2026 ~13:05 CET, before any arm was scored)

Hypothesis (never tested): cap250 and the final stack throw away in-session adaptation for recognised returning people. The latch `recB`
(first positive ramp, irreversible) retires FT and CSP; FT ran in only 10 of 32 returning cells (laneT) and CSP stops at recognition. About
43 % of sealed cells are returning, 87 % of those eight people with ONE public session (`evidence/audit_20nov/sealed_composition.md`,
medium-low confidence). Team C's default TTT gave +4.49 on returning P04/P10 against the same stream without TTT (`laneT4_valid.md`, two people).

## Arms (all on the final stack = hardcb; code `laneR4/build_keep.py`: only submission.py patched, anchors asserted once each)
- R0z all switches off: must equal hardcb bit for bit (code-path control).
- R1 FT stays active after recognition (stock objective; labels = v5.1b routed output, stored labels already relabelled at recognition).
- R2 R1 + CSP stays active (its history is not cleared, its fusion keeps running on the routed posterior).
- R3 R2 with the lite5 TTT objective after recognition only, stock FT before (laneT form: per-window views, Sinkhorn targets, IM 0.3, anchor 1e-2, 5 steps).
- R4 R3 where the FT pseudo-label contains the routed person's head: label = (1 - 0.5 r) out_v5 + 0.5 r h (r routed ramp, h head posterior of the routed person; before recognition r = 0, unchanged).
- Runtime-fit forms, registered now: R2s (post-recognition refits only every 20 windows), R2q (every 40). Scored on all instruments like the rest.
Silent-no-op guards in the worker: recognition ages and v5 tapes identical in all arms; every keep arm must have a refit after recognition and move the scores vs hardcb where the grid allows;
R2+ must have CSP not retired; an unrecognised session must be bitwise unchanged.

## Instruments
1. Returning P04/P10, Lane A instrument (visit 1 enrolment, visits 2-3, 8 orders x 4 cells, real fold packages, jobs `ni26_laneR4_20261120/ret`), paired against hardcb. Two people: intervals are order-only per person and nested over people x orders for equal cells; descriptive.
2. NI-V Scherer B-ret (9 people, visit 2, 8 full + 8 within orders, exact packages `hardcbB_<arm>` vs `hardcbB`), plus B-new as a false-recognition veto (Scherer recognises other-person streams, so keep-adapting acts there too).
3. Held-new full0 + within0: the chain freezes recognition and the FT targets; held-new people have zero routed windows (LOPO receipts), so the recB branch is never taken and the arms equal the final stack by construction. Real-package check: `laneR4/check_zero.py`, unrecognised noise stream, bitwise scores for every arm (R0z also on a recognised session). Stated limit: no real held-new package exists, a real held-new person is not re-run.
4. Runtime on recognised sessions: `laneR4/time_rec.py`, real public sessions (recognised), RTX 2080 Ti and A100, arm vs final stack in the same job.

## Predictions (pp over hardcb; BA)
- Returning P04/P10, equal cells (stock is 90-97 %, near ceiling): R1 +0.3 [-0.6, +1.2], R2 +0.3 [-0.6, +1.2], R3 +0.4 [-0.7, +1.5], R4 +0.3 [-1.0, +1.3]; R2s/R2q about R2 minus 0.1. Team C's +4.5 will NOT reproduce at 90-97 % (it was measured against a no-adaptation A1 at 66 %); P04 is the one at risk (heavy forms hurt it).
- Scherer B-ret (stock about 57): R1 +0.8 [-1.5, +3], R2 +0.8, R3 +1.0 [-2, +4], R4 +0.8 [-2, +3]. Scherer B-new: -0.1 [-0.5, +0.3] each.
- Held-new: exactly 0 (bitwise) for every arm; R0z equals hardcb on every tape.
- Runtime: FT stock cost 7.6-12.2 s per session (17 refits); a recognised session now runs about 13 more refits, +6 to +9 s; about 46 recognised sessions in the sealed set:
  R1/R2/R3/R4 projected sealed 3,211 + 46 x (6 to 9) = 3,490-3,625 s, i.e. mostly above the 3,450 s bound; R2s about 3,400; R2q about 3,320. Prediction: R1-R4 fail 3,450 (P ~ 65 %), R2q fits.

## Rules (fixed now)
- Per arm: SUPPORTED if (a) returning equal-cell gain >= +0.3 with P04 and P10 each >= 0, and (b) Scherer B-ret 80 % lower bound > 0 (people-resampled, full and within pooled over orders); REFUTED if P04, P10 and Scherer B-ret point deltas are all <= -0.2 or Scherer B-ret upper bound < 0; else UNDECIDED. Bars are reading aids inside +-0.5 noise, P04 and P10 reported separately, neither decides.
- Adoption for the fallback package (nothing is uploaded from this lane): supported or undecided-positive on both returning instruments, B-new >= -0.3, held-new bitwise zero change, projected runtime <= 3,450 s (46 recognised sessions x measured delta added to 3,211 s; if the full form fails, the strongest fitting form among R2s/R2q is the candidate).
- Sealed weight shown next to every returning number, labelled UNVERIFIED: 0.43 x (0.87 x Scherer B-ret proxy + 0.13 x mean(P04, P10)); and the old 0.2/0.8 P04/P10.

## Corrections found at smoke, before any arm was scored (Fri ~13:45 CET)
- First R3/R4 build applied the lite5 objective to ALL refits, so unrecognised sessions changed (the worker's unrecognised-session guard raised). Fixed: lite5 acts on refits after recognition only, the stock FT objective before; unrecognised sessions are then bitwise the final stack for every arm. Smoke/timing rows for R3/R4 above that guard were built before this fix (timing of recognised sessions is unaffected: no refit runs before window 20 and recognition came at age 10).
- The zero-change check counted 5 refits in 60 windows; the schedule gives 4 (windows 20, 30, 40, 50). Assertion corrected.

## Addendum, Fri ~14:30 CET (after the timing of R0z/R1/R2 on recognised sessions, before any returning or NI-V arm was scored)
Timing, recognised public sessions (recognition at age 10 in all four), one session mean, paired in one job, `laneR4/time_rec.py`: RTX 2080 Ti R0z (= final stack) 32.5 s, R1 50.7 s (+18.2), R2 54.0 s (+21.5);
A100 31.5 / 42.5 (+11.0) / 47.4 (+15.9). The registered prediction (+6 to +9 s) was too low: FT refits are 12.2 s (RTX) / 7.1 s (A100), the fine-tuned nets' per-window forward adds about 6 s, CSP about +3.3 s (RTX) / +4.9 s (A100).
With 46 recognised sessions: R1 +840 s (RTX) -> about 4,050 s, R2 +990 s -> about 4,200 s; both far above 3,450 s. R2s and R2q (already queued) cannot fit either (FT forward + CSP alone are about 9 s). The fit budget is 239 s / 46 = 5.2 s per recognised session.
New arms, registered now, both inside the budget by construction: **R5** CSP stays active, FT retired as stock (cost about +3.3 s RTX / +4.9 s A100 -> about 3,363 s); **R1a** FT after recognition = one refit at age 100 (stock objective, 5 steps) and no FT forward before it
(predicted cost +4.5 s [3, 6] RTX -> about 3,418 s; measured in the same timing job). Predictions pp over hardcb: returning P04/P10 equal cells R5 +0.1 [-0.5, +0.6], R1a +0.1 [-0.6, +0.7]; Scherer B-ret R5 +0.3 [-1, +1.5], R1a +0.3 [-1.5, +2]; B-new -0.1 [-0.5, +0.3]; held-new exactly 0.
R2s/R2q stay in the queue as dose information (how much FT is worth), not as candidates. Same reading rule and adoption rule as above.
