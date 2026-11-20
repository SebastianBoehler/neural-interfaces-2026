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

## Result (Fri 20 Nov, ~21:10 CET; recovered after the Mac outage; code `laneR4/combine.py`, `laneR4/ret_worker_norm.py`, tables in `work/r4/` not committed)
Run history and one silent-failure finding: R1 (job 3143013) and R1a (3143218 reruns) completed on the first pass. For R2, R3, R4, R5 every task failed (jobs 3143007/3143020/3143150/3143172, 64 failed tasks) with "invalid native posterior tape scores". Diagnosis (debug job 3144031): from window 90 on, when CSP is kept active and fuses,
the package's `scores` are log-scores (count-filter log posterior + cap x log CSP posterior; the label is their argmax), rows sum to up to ~17, not 1. The pinned inner worker's normalisation check raised; the labels were never wrong, and the NI-V and B10 runs of R2-R5 (labels only) are unaffected. Fix: `ret_worker_norm.py` re-compiles the inner worker source with one inserted
softmax of the score rows, applied only to a session whose CSP fused windows (`model.cp.fused > 0`, not dead); all other checks stay, any other bad tape still raises (printed per session `[r4-norm]`). Smoke 3144282 (R5, RET_N 130) passed, R5 refits after recognition 0 as required. Full reruns 3144384/-86/-88/-90 (R5/R2/R3/R4, 16 tasks each, 64/64 COMPLETED) with afterok collectors 3144385/-87/-89/-91.
Late refits per cell (guard): R2/R3/R4 12-18 after recognition (R1 same), R5 0, R1a 1 (age 100); recognition ages identical across arms (worker asserts).

Returning P04/P10 (Lane A instrument, 8 orders x 4 cells, gain in BA pp over hardcb; 80 % intervals over orders per person; equal cells nested over people x orders, two people: descriptive):
| arm | P04 | P10 | equal cells | 0.2 P04 + 0.8 P10 (UNVERIFIED weighting) |
|---|---|---|---|---|
| R1 | -0.72 [-1.12, -0.32] | -0.13 [-0.24, -0.03] | -0.43 [-0.79, -0.10] | -0.25 |
| R1a | -0.46 [-0.69, -0.23] | -0.07 [-0.20, +0.07] | -0.26 [-0.50, -0.05] | -0.14 |
| R2 | -1.05 [-1.51, -0.59] | -0.13 [-0.36, +0.11] | -0.59 [-1.12, -0.08] | -0.31 |
| R3 | -1.05 [-1.28, -0.81] | -0.03 [-0.33, +0.24] | -0.54 [-1.10, +0.01] | -0.24 |
| R4 | -0.72 [-0.98, -0.45] | -0.10 [-0.37, +0.16] | -0.41 [-0.75, -0.05] | -0.22 |
| R5 | -0.72 [-1.02, -0.43] | +0.06 [0.00, +0.12] | -0.33 [-0.76, +0.06] | -0.10 |
Scherer B-ret (9 people, paired, people-resampled 80 %), full / within: R1 -0.41 [-0.97, +0.17] / +0.47 [-0.21, +1.17]; R1a -0.12 [-0.28, +0.04] / -0.11 [-0.27, +0.05]; R2 -0.46 [-1.04, +0.12] / +0.52 [-0.14, +1.20]; R3 +0.06 [-0.45, +0.57] / +0.37 [-0.03, +0.77]; R4 +0.06 [-0.62, +0.74] / +0.76 [+0.22, +1.32]; R5 +0.05 [-0.09, +0.19] / +0.01 [-0.10, +0.12].
Scherer B-new (false-recognition veto): +0.000 full and within for every arm (nothing moved).
Sealed proxy (UNVERIFIED composition, 0.43 x (0.87 x B-ret full + 0.13 x mean P04/P10)): R1 -0.18, R1a -0.06, R2 -0.20, R3 -0.01, R4 0.00, R5 0.00 (pp); with the old 0.2/0.8 weighting (also UNVERIFIED): -0.11, -0.06, -0.13, -0.10, -0.10, -0.04.
Held-new: `check_zero.py` (job 3142333) printed ZERO-CHANGE OK (unrecognised noise stream bitwise for every arm, R0z bitwise on a recognised session); B-new moved by exactly 0. Stated limit unchanged: no real held-new package exists.
Runtime (A100, paired in one job, mean of 4 recognised sessions, `time_rec.py`; base final stack 31.4-31.5 s): R1 42.5 (+11.0), R2 47.4 (+15.9), R3 50.9 (+19.4), R4 47.1 (+15.6), R1a 34.9 (+3.5), R5 34.2 (+2.8). RTX 2080 Ti (addendum, earlier job): R1 +18.2, R2 +21.5; R1a +3.3 and R5 +0.7 to +2.6 (bracketing R0z drifted 32.5-36.4 s there: noisy). Sealed projection 3,211 + 46 x delta (A100): R1 3,717, R2 3,942, R3 4,103, R4 3,929 (all > 3,450, as predicted; RTX would be higher for R1/R2: 4,048 / 4,200); R1a 3,372 and R5 3,340 fit. The sealed GPU is an RTX PRO 4500, not measured here.
Rules applied: no arm SUPPORTED (equal-cell gain >= +0.3 never reached; every returning point delta on P04 is negative). Not REFUTED by the letter either (the rule needs P04, P10 and B-ret all <= -0.2, or a B-ret upper bound < 0; P10 is -0.13 to +0.06 and R3/R4/R5 B-ret are >= 0): UNDECIDED for all six, leaning negative on the in-lab returning instrument (all equal-cell deltas negative, R1 and R2 intervals exclude 0 on P04 and nested). Within-order B-ret is positive for R2/R3/R4 (R4 +0.76 [+0.22, +1.32]) but the full-order B-ret is not, and +-0.5 is the noise.
Adoption (fallback package): none. The returning instruments are not positive (the Team C +4.5 did not reproduce at 90-97 %, as registered), and the arms that fit the runtime (R1a, R5) are negative on P04 (-0.46, -0.72). Opposite regime for the record: B10 NETBCI foreign-lab stress, same arms, +0.4 to +3.7 over hardcbB (`laneB10_returning_stress.md`): keep-adapting helps where the routed model is far from ceiling. Nothing is closed; this is "undecided, leaning negative for the in-lab sealed-like returning case".
