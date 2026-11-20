# Council · Fable · Fri 20 Nov 2026, ~12:00 CET

**Answer.** Nothing we measured supports a ≥1 pp sealed gain by Saturday; I put it below 10 %. Upload the final stack
(cap250 + F-C25 hard + CBraMod) on Saturday by 14:00. Expected sealed gain about +0.15 pp (range −0.3 to +0.6), which
moves the rank by one place at most. Top 3 (+1.7 pp) is out of reach with what exists.

All numbers are DEV on the ten public people unless marked sealed. Sealed weighting (57 % new) is an unverified assumption.

## 1. New bank test: does an independent whole system repair cap250's errors?

Team C's system shares no member, trunk or stage with cap250. Its order 0 is the order of cap250's full0 bank, so the
two can be fused per window, causally. Run locally on `work/teamC/C1`, `work/teamC/C3e` (A1, A2 posteriors) and
`work/teamC/cap/full0.npz`; official cell metric, 28 cells, people-resampled paired 80 %. Scripts not committed.
Not pre-registered: I decided the arms before scoring but wrote no ledger entry, so read it as a diagnostic.

| Quantity (held-new, full0, one order) | Value |
|---|---|
| cap250 terminal / raw stage / team C A2 | 73.74 / 73.94 / 65.18 |
| cap250 errors that C-A2 gets right | 32.8 % (inside the cap250 family: about 10 %) |
| Label oracle "either is right", window accuracy | 82.3 % against 73.7 % |
| When the two disagree (30 % of windows) | cap250 right 57.8 %, C right 29.0 % |
| cap250's lowest-margin third of windows | cap250 55.0 %, C 52.2 % |
| Per-person accuracy, cap250 against C, correlation | 0.86 (P04: 50 and 46) |
| Log-linear fusion, raw stage × C1-A2, weight 0.05 / 0.1 / 0.2 / 0.3 | +0.07 / +0.30 [+0.06, +0.61] / +0.19 / −0.11 |
| Same, weight chosen on nine people, scored on the tenth | C1-A2 −0.16 [−0.54, +0.35]; C3e-A2 +0.26 [−0.04, +0.66] |
| Leave-person-out logistic stacker on both log posteriors | −1.2 to −1.8 (raw-only control −0.34) |

Reading: an independent system fixes a third of cap250's errors on paper, but it is no better than cap250 exactly
where cap250 is unsure, and both fail on the same people. Fusion is worth 0 to +0.4 pp and a learned gate loses.
A cap250 + clean-slate ensemble is not a route. Limits: one order, ten reused people, weights chosen on the same
people (the nested rows correct for that). My first pass fused cap250's terminal scores as if they were
probabilities and gave −35 pp; that was a scaling bug of mine, fixed by using the raw-stage posterior.

## 2. Why the chain adds 14 pp and still absorbs member and data gains

- **The 14 pp is the mix, not the tail.** cap250's raw stage (the mixed posterior of about 90 members, with SessionBN,
  EM, fine-tuning inside) is already 73.94; count filter and tail add −0.2 under full shuffle. Clean systems built
  from one to three families with their own adaptation land at 59–70.
- **Absorption is dilution times redundancy.** A family swap moves about 5/90 of the mix. A member that is +4.6 pp
  better standalone gains on windows where the other families were already right; the mix's errors sit on windows
  and people where every family fails. Expected transfer is slot weight × non-redundant share: about 0.05–0.1,
  which is the "×10 shrink" in the calibration table. Section 1 shows redundancy alone is enough: a whole extra
  system at its best weight adds +0.3.
- **Adaptation is saturated inside cap250, not inert in general.** Gradient TTT is +2.95 [+1.8, +4.1] on a base
  with no adaptation (8 orders) and +0.08 / +0.34 in cap250, which already holds SessionBN, EM, FT and CSP.
- **What B5 did not run:** only the raw family (5/6 of one slot) was varied; every other member kept nine people.
  B5 shows one family's N does not matter given the rest, not that the system is flat in N. The all-members curve
  is unmeasured. It does not matter for Saturday: no further same-task people exist.

## 3. What information limits new-person accuracy

The person's own class-to-pattern map. Evidence, all pointing the same way:
- P04 is 46–50 % in every label-free system (cap250, teams A, B, C, M1) and 85–90 % once one labelled session is
  enrolled. Independent systems rank people the same (r = 0.86).
- The signal holds the map: tangent features + the session's own labels reach 72 % where cross-person reach 50 %.
- Cross-person nets discard it: team B's own-label centroid in its learned projection scores what its classifier
  scores (66.2 against 65.4). T-A's label gain on a frozen embedding was +4 to +6 pp, half my forecast.
- Without labels the session does not reveal it: class variance is small against within-session nuisance variance
  (run clustering ARI 0.05), so EM, Sinkhorn, self-training, cluster-then-name and in-context attention recover
  0 to +3 pp on weak bases and about 0 on cap250.

So new-person error is not a compute, capacity or data-volume gap that Saturday can close. It needs either labels
from the person (not available) or source people who resemble them (not available publicly).

## 4. My audit of 19 Nov against the overnight evidence

| Earlier conclusion | Verdict | Evidence |
|---|---|---|
| R1: coverage binds; curve still steep at nine | **Held standalone, failed as a lever.** | B5 +1.15 pp/person standalone (I forecast 1.0), −0.07 in the system; B6 81 people +0.23; Scherer as people −4.3 (C3s); lane V 0 or harmful |
| "More same-task people in the members is the one change with a ceiling above 1 pp" | **Wrong at system level** in every tested form | same |
| R2: label-free adaptation nearly inert, cause open | **Refined.** Real on weak bases, saturated in cap250; cause now leans "capped" | T2, lane T, team B R1/R2 |
| R3: search ran below instrument resolution | **Held, reinforced** | T2: one person-order cell swings several pp; A2h +8 on two people became −0.2 on ten |
| Simple design refuted | **Held** | clean-slate 44–70 against 73.7 |
| No posterior-space operation helps | **Held** | A1b Sinkhorn −0.45; team B balance 0.00 |
| 89.8 % of errors shared, so new members cannot repair | **Too strong as stated.** An outside system fixes 33 %; it still cannot be harvested | section 1 |
| Composition sum 82.78 ≈ sealed 82.99 "consistent" | **Weaker than I wrote.** One equation, two unknowns; my returning term used P04/P10, about 13 % of returning cells | sealed_composition |

## 5. Ranked recommendations

**1. Upload the final stack (hardcb).** Dev +0.39 [+0.28, +0.52] over two orders; returning +0.01; NI-V no veto.
Sealed forecast: 0.57 × 0.39 × 0.5 transfer + 0 ≈ +0.1 to +0.2; I use +0.15, sd about 0.35. Since only the best
score counts, the value is E[max(0, Δ)] ≈ +0.23 pp, and the cost of a miss is zero.

**2. Fix the runtime projection before the upload (decisive, cheap, tonight).** The 3,211 s figure multiplies
cap250's sealed time by a ratio (1.257) measured on public replay, where recognition fires at window 10–60 and
retires fine-tuning. Lane T says that replay is uninformative about FT cost. On unrecognised sessions the stack is
278 ms/window (2080 Ti); cap250's number on that path was never taken. If the stack's extra cost is additive
(about 16 ms/window from the replay: (208.4 − 165.8) s / 2,655), the sealed projection is about 2,555 + 320 ≈
2,900 s, not 3,211. Test: `laneT/time_ft.py` on the stock cap250 ZIP and the stack ZIP, same card, white-noise and
real held-new streams. Forecast: path ratio 1.06 [1.03, 1.15]. If the ratio is above 1.30 on that path, the stack
projects past 3,500 s and the fallback is `hard` alone (×1.06). Rule stated now.

**3. Use any confirmed headroom only for arms already scored jointly.** Candidates: lane T `sparse20` / `lite5`
(registered, unscored; forecast +0.2, undecided at best) and B6 arm X (+0.23 [+0.10, +0.37], one order, and it
occupies the same raw slot as `hard`, so it needs a joint arm, not addition). Take one only if its joint arm with
hardcb is ≥ hardcb on both orders by Sat 10:00. Do not add FBL2: +0.55 on dev with hard, −0.58 on Scherer with
hard, −0.2 returning.

**4. Backlog B1 (two more full-shuffle orders for hardcb vs hard vs cap250).** Narrows the choice between hardcb
and hard; cannot change the direction. Run it because compute is free.

**Set aside for this deadline:** cap250 + clean-slate fusion (section 1), heavy TTT (exceeds 3,600 s, −3.0 on
Scherer), any clean-slate system as the upload (−4 to −8.5 pp new, −23 pp returning), more people or corpora.

## 6. Where our measurements could be wrong

- **Returning one-session people are 38 % of sealed cells and unmeasured.** P04/P10 with visit-1-only enrolment
  score 89.6 / 95.9, which is the same situation on two people. If the eight behave like P04, 4 pp of sealed error
  sits there. No instrument exists; no Saturday action follows.
- **Sealed delivery.** If it is run-contiguous after all, within0 applies: hardcb +0.27 there, still positive.
- **Transfer 0.5** comes from four uploads; cap250 itself transferred at 0.9 to 2.
- **Upload timing.** A Failed run costs no quota, so upload by 14:00 to leave time for a timeout retry with `hard`.
