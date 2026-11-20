# Lane R4 · keep adapting after recognition (registered Fri 20 Nov 2026 ~13:30 CET, before any arm was scored)

Hypothesis (never tested): cap250 and the final stack throw away in-session adaptation for recognised returning people. The latch `recB`
(first positive ramp, irreversible) retires FT and CSP; FT ran in only 10 of 32 returning cells (laneT) and CSP stops at recognition. About
43 % of sealed cells are returning, 87 % of those eight people with ONE public session (`evidence/audit_20nov/sealed_composition.md`,
medium-low confidence). Team C's default TTT gave +4.49 on returning P04/P10 against the same stream without TTT (`laneT4_valid.md`, two people).

## Arms (all on the final stack = hardcb; code `laneR4/build_keep.py`: only submission.py patched, anchors asserted once each)
- R0z all switches off: must equal hardcb bit for bit (code-path control).
- R1 FT stays active after recognition (stock objective; labels = v5.1b routed output, stored labels already relabelled at recognition).
- R2 R1 + CSP stays active (its history is not cleared, its fusion keeps running on the routed posterior).
- R3 R2 with the lite5 TTT objective (laneT form: per-window views, Sinkhorn targets, IM 0.3, anchor 1e-2, 5 steps).
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
