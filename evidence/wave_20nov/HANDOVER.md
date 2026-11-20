# Handover for a fresh thread · Fri 20 Nov 2026, ~22:10 CET

Deadline: **Sat 21 Nov 2026, 23:59 CET (German time; 22:59 UTC)**. One sealed upload left; only the team's best sealed
score counts; a Failed run costs no quota, a finished or self-cancelled one does. 3,600 s on one RTX PRO 4500.
Best sealed so far: cap250 0.8299 (rank ~15–16). Board (19 Nov): 1st 0.859 (tie), 3rd 0.848, 10th 0.837.

## Upload candidate (unless a lane below beats it)
`runs/sealed/final_zips/final_stack_hardcb.zip` (cap250 + F-C25 hard + CBraMod 1/6 slot), sha 958fd3f3…782705.
- Real per-fold packages (cap250 pipeline, Leonardo run production-20261120-32a94ab7, orders 71000–71003):
  new people hardcb − stock +0.19/+0.49/+0.30/+0.34 (4/4 orders, mean +0.33); returning P04/P10 +0.00/+0.00/−0.55/+0.26.
- Replay chain: +0.39 [+0.28,+0.52] new people; NI-V (Scherer, 8 orders) ≈ cap250; runtime projected ~3,130–3,150 s.
- Expected sealed ≈ 0.830–0.832 (rank ~13–15). Not top 3.
- Ties: dsu_l5 package (lite5 + correction on unrecognised sessions) ≈ +0.01 sealed-weighted vs hardcb.

## Still running (results land on the clusters; check cards)
- R4b confidence-gated returning adaptation, Leonardo r4b-run → `laneR4b_dose.md` (due Sat 08:00).
- D3 drift shift: Galvani done (P04 −1.34, wrong-enrolment −0.15, not a candidate); NETBCI finishing → `laneD3_drift.md`.
- K2 GPT ideas 4–6 on Galvani → `laneK2_ideas456.md` (due Sat 08:00).
- B1 leftover sparse20 returning (minor).
- The hourly compute keeper was a session cron of the old thread: recreate it in the new thread (CronCreate) and
  keep Galvani ≥ 25 and Leonardo ≥ 15 of our jobs.

## What is settled (20 Nov)
No data-path bug; rules re-read upstream (8447e181); sealed ≈ 106 visits, ~60 new / ~46 returning, ~87 % of returning
from the eight one-session public people. Foundation models, extra data, virtual subjects, clean-slate systems (6–15 pp
below), mixer re-weighting (R1), session-stage re-tuning (S2), EM template coupling (G1, −5 pp), widened post-mixer
corrections (W), vote calibration (C2v), alignment ideas (Y), keep-adapting for returning people in-lab (R4) — all ≤ 0
or undecided in the full system. Evidence: evidence/wave_20nov/*.md, evidence/council_20nov/*.md, evidence/audit_20nov/*.md.

## Bottlenecks (lane Z profile + lanes P2, B10, R4, Y)
1. Mixer dilution + redundancy: a member's gain reaches the output only where the whole mix is wrong (slot ≈ 1/16;
   real member gains mostly land where others are already right). Needs NON-redundant per-window information.
2. New people's class map: a per-person rotation worth +3–5 pp exists (oracle, Y #5; P04 50 % → 85 % with one labelled
   session), but no label-free estimator recovers it yet.
3. Session stages (rebalance, EM, TSEM) carry cap250 and are at a local optimum (S2); pulling them toward source
   templates hurts (G1).
4. Returning people: adapting after recognition helps far from the ceiling (NETBCI +3.7) and hurts near it (P04/P10);
   the sealed one-session people are unmeasured; a reliable gate is the open piece (R4b).
   **Red team, 20 Nov 22:50:** the P04/P10 half of this is unsound. The Lane A / R4 / D3 returning packages carry the
   uploaded package's Hilbert core, rest nets and count filter, fitted on the test visits. Clean level (real fold
   packages): P04 70 %, P10 89 %, not 90–97 %. Re-score in-lab on the real fold packages: `redteam_fable.md` F1, H1, H5.

## Decision Sat ~10:00
Upload hardcb unless R4b (or another lane) shows a packaged, runtime-safe gain on returning without in-lab loss.
Read "in-lab loss" on the real fold packages (`redteam_fable.md` H1), not on the Lane A tapes.
Upload Sat ~11:00–14:00 with the user's explicit go-ahead; check live quota first.
