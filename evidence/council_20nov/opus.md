# Council · Opus (skeptical adversary) · Fri 20 Nov 2026

Read: BRIEF, STATE, SYNTHESIS, sealed_composition, assumptions_rules_opus, forecast_calibration §B, final_stack,
laneT_ttt_cap250, laneT4_valid, laneR_onesession (+ jsons), laneB6, laneE_capacity, CLEANSLATE, cleanslate_teamC, and
`submissions/lean_v66/submission.py:1040-1160`, `laneT/ttt_core.py`. pp = balanced accuracy points; "dev" = ten reused
held-new people. Numbers below are from those files; my own estimates are marked *est*.

## 1. Bottom line
- No measured route reaches +1 pp sealed by Saturday. The largest plausible one is more session adaptation for new
  people (the only lever sealed has confirmed: v5.1b→cap250 +1.01). Members, data and clean-slate systems are done.
- Final upload: the final stack (cap250 + F-C25 hard + CBraMod, "hardcb") is the floor candidate. Add TTT-light only if
  test T1 below passes. Swap CBraMod for a fitted heavier TTT only if T2 and T3 pass by Sat 10:00.
- Upload by Sat ~11:00, not 14:00–16:00 (§4). Clean-slate systems should not enter the final choice (§3e).

## 2. Measurement problems that could hide or invent a gain (checked in code and data)
a. **The runtime projection uses the wrong mix of paths.** 3,211 s = 2,555 s × 1.257, a ratio from public replay. In
   that replay every person is recognised at window 10–60, so FT and CSP retire almost at once (laneT card says this for
   the TTT ratio, and it applies to the stack ratio too). About 57 % of sealed sessions are unrecognised and run FT for
   the whole session. The only unrecognised-path timing is `time_ft.py`: stack 278 ms/window on the 2080 Ti, and cap250
   was never timed on the same fixture. If the stack/cap250 ratio on unrecognised sessions is above 1.257, the 389 s
   margin is too optimistic. This is the cheapest decisive check (T0) and it also governs every TTT arm.
b. **Frozen label loop.** In every chain arm (stack, TTT light/heavy, B6), the FT targets are the stock run's relabelled
   posteriors, and recognition, the inner route and the EM/TSEM gates are frozen. In the package, the FT families'
   changed posteriors feed back into the next refit's targets. T4 found that heavy TTT fails by drift, not collapse
   (Scherer A2h −3.0 [−4.7, −1.5]; quarter-by-quarter decline). The frozen chain removes exactly that feedback, so it
   probably overstates heavy forms (+0.34 both orders, +0.65 full0) more than light ones. Light uses the stock step
   count, so I expect less drift, but it is unmeasured on new people with live labels. The only live-label
   new-person instrument we have is NI-V B-new on real packages (Scherer, used as a veto only).
c. **Order coverage.** Every held-new number from the stack chain covers full0 plus within0. full0 is one
   full-shuffle order, and within0 has a different structure. Organisers say chronology is removed, so full0 is the
   closer stand-in for sealed, but it is still one order. The two orders often disagree by more than the stated ±0.3:
   TTT-light +0.49 vs −0.34, heavy +0.65 vs +0.03, cap250-vs-P0r +0.67 vs +2.55. Each order has roughly ±0.4 of its own
   noise. Treat both-order means as the estimate, not full0 alone.
d. **The cap250 reference in the clean-slate harness was not re-streamed.** It is the archived `c25d_full0` bank: one
   order, unpaired with 71000–71007 (CLEANSLATE:75-79). This does not change the verdict, because team C is
   −8.6 [−12.4, −4.5] held-new and −23 pp returning.
e. **"The returning arm measured the wrong people" is only half right.** Lane A enrols P04/P10 on visit 1 only and
   streams visits 2–3. That is the sealed one-session protocol, just on two people. cap250 scores 89–97 there, with
   8/8 recognition. Back-solving sealed 82.99 with dev new at 76.0 gives returning ≈ 92 (*est*, with an unverified
   57/43 split). So returning headroom is small unless the eight are not recognised. Recognition false-accepts 144/144
   Scherer streams but 0/14 PROTEUS held-new sessions. With about 60 sealed new sessions, a few false accepts would
   each retire FT. *est* cost ≤ 0.2 pp. Unmeasured, but not worth a lane now.
f. **Transfer calibration splits by lever.** Member and stage uploads transferred 0.41–0.57× (v4.1, v5.1b,
   route3geb). The adaptation upload (cap250, FT dose) gave sealed +1.01 against dev full0 +0.67 and within0 +2.55.
   That is 0.63× the both-order mean overall, and about 1.1× per new person if all of it came from unrecognised
   sessions (FT retires on recognition). Use ≈0.5× of dev for member arms (hard, CBraMod, B6) and ≈0.57 × 1.0 × the
   both-order dev mean for adaptation arms. The current "+0.23 sealed-weighted" for hardcb is on the high side:
   0.392 × 0.5 ≈ +0.20 (*est*).
g. **B6 (+0.23 at MASS 5/6) is selection-inflated and conflicts with hard.** It is the best of five corpus arms on one
   order: the controls and siblings ran from −0.15 to +0.08, and G, N and M alone are flat. It also swaps the same
   raw-family 5/6 slot that hard replaces, and it was never scored with hard. Do not stack it without a joint run, and
   a joint run does not fit before Sat 10:00 together with T1–T3.
h. **Silent no-op hunt.** I found none in what I read. Lane T's ctl reproduced the released banks bitwise (FBL within
   2.5e-4). The stack's identity arm equals stock bitwise, and the CBraMod package parity is 3.1e-6 to 3.8e-5 (fp16
   2.3e-4). One inert option remains: `v6_ft_session_s` is 1e9 in cap250 and the stack (`submission.py:1049,1157`),
   so nothing bounds a session's FT time. Lane E tested setting it (150 s guard, 2 s demo: graceful, 99/100 labels
   agree). It should be set in the final package (§4).

## 3. Ranked recommendations

**R1 · hardcb as the floor candidate (ready).** Dev +0.392 [+0.280, +0.519] both orders. Returning +0.011, NI-V
+0.015. Sealed *est* +0.20 (σ ≈ 0.35), so E[max(0, Δ)] ≈ +0.25. Runtime blocker: T0 must confirm ≤ ~3,450 s with the
unrecognised path counted.

**R2 · hardcb + TTT-light (zero added runtime; decide on T1).** Dev over the stack: +0.08 [−0.17, +0.31] both orders,
full0 +0.49 [+0.12, +0.80], within0 −0.34. Returning −0.04 on real packages. It adds mean only if full0 is the right
stand-in for sealed. Under Force_Best, a mean-neutral increment with extra variance still raises the option value
slightly (*est* +0.02 to +0.05 on E[max]). The risk is live-label drift (§2b).
T1 (Galvani, ~2 h): NI-V B-new and B-ret exact real packages, stack+light vs stack, 8 full-shuffle orders.
Adopt if B-new ≥ −0.15 and no person is below −3 pp. Otherwise ship R1.

**R3 · fund heavier session adaptation by dropping CBraMod, if T2 and T3 pass.** This is the only lever with a
sealed-confirmed mechanism and a ceiling of about +0.3 to +0.6 sealed (*est*). CBraMod is member-type
(+0.13 both orders, worth about +0.07 sealed) and costs about 500–650 s. In the stack, heavy TTT earned
+0.34 / +0.65 (full0), about 2.5× more per order, but it needs about +1,400–2,200 s. Stated plainly: no fitted heavy
form has been scored yet.
- T2 (Galvani, chain, ~2–3 h): the registered `lite5` and `sparse20` arms (laneT card, ~09:40, not yet run because 2FA
  is needed), plus one new arm `hard+sparse20, no CBraMod`, registered before scoring.
  Prediction: sparse20 +0.20 [−0.45, +0.85] over the stack (card) and *est* +0.15 [−0.5, +0.8] for the no-CBraMod
  form vs hardcb.
- T3: the same arms as real packages on NI-V B-new, with drift and the veto. Plus `time_ft.py` on the unrecognised
  fixture for the cap250, stack and arm packages.
- Adopt over R1/R2 only if the both-order mean is ≥ +0.2 vs hardcb, NI-V B-new ≥ −0.15, the projection with the
  unrecognised path counted is ≤ 3,450 s, and the package's `v6_ft_session_s` is set (§4).
- If T2/T3 are not back by Sat 10:00, ship R1 or R2.

**R4 · FP64 static members to FP32 (speed only, for budget).** The static FBCNet members run as float64 on the GPU
(`lean_v66 :287-296`), and Blackwell workstation FP64 is slow. That could free hundreds of seconds for R3. It needs the
same-input label check on an RTX card plus a real-card timing. It does not fit with T1–T3 by Sat 10:00. Run it only if
Galvani has an idle GPU after T0–T3.

**R5 · do not pursue for this deadline.** Clean-slate systems: −8.6 held-new and −23 returning vs cap250, and they have
no personal route. Lane V/S: ≈0 or negative. B6 in the stack: §2g. More personal head weight: refuted (dose −1.7 to
−12.4). Heavy forms of team C's TTT: Scherer −3.0. Re-uploading stock: worth 0.

**Possible ≥1 pp route, not reachable by Saturday.** Session adaptation that improves its targets from its own
outputs without drifting: TTT whose relabel targets come from the live mixture, with anchoring, a step budget and
Sinkhorn balance. It needs a live-label held-new instrument, meaning cap250 LOPO real packages, which do not exist
(Lane A built them only for P04/P10). Build those 10 fold packages after the deadline first; every adaptation claim
since 18 Nov rests on frozen-label chains.

## 4. Upload decision (decision theory and rules)
- Force_Best, one slot left, Failed is free and counts toward neither the daily nor the total limit
  (`terms.md@8447e181:25`). So the criterion is E[max(0, S − 0.82987)], and the "do no harm" gates do not apply. Prefer
  the higher both-order mean and accept variance. Do not require lower bounds above 0.
- **Sequence for the free retry.** A timeout can stay "Running" for up to 80 min, and the queue was long on 18 Nov.
  Uploading at 14:00–16:00 leaves room for at most one retry before 23:59, and only if Codabench marks a timeout as
  "Failed". That is unverified: ask on Discord today. Upload the chosen package by Sat ~11:00. If it Fails, upload R1
  the same day. If the bold arm (R3) was chosen, upload it first, since its failure is free.
- **Set `v6_ft_session_s` in the final package** (key-only, as in Lane E; for example 60 s for the stack, 25–30 s per
  session for R3 arms). This bounds the cost of any pathologically slow session. It does not bound the run
  (106 × limit), so T0 still decides. It changes outputs only on sessions that exceed the limit. Check it on the
  2 s guard demo and the public replay (labels identical).
- The package is 90.5 MB. The 90 MB cap is ours, not official, so keep it.
- My pick if nothing new lands: **R1 hardcb with the guard key**. With T1 passed: R2. With T2 and T3 passed
  (≥ +0.2 over hardcb both orders, NI-V veto clear, ≤ 3,450 s): the R3 arm.

## 5. Decisive tests, ordered (all fit before Sat 10:00 if 2FA is available today)
| id | what | where | cost | decides |
|---|---|---|---|---|
| T0 | `laneT/time_ft.py` on the unrecognised fixture: cap250 vs stack vs light vs sparse20, RTX 2080 Ti + A100, ABBA | Galvani | <1 h | runtime of every candidate |
| T1 | NI-V B-new/B-ret exact packages: stack+light vs stack, 8 orders, drift by quarter | Galvani | ~2 h | R2 |
| T2 | chain full0+within0: lite5, sparse20, hard+sparse20 without CBraMod (register first) | Galvani | 2–3 h | R3 |
| T3 | NI-V real packages for the best T2 arm + `v6_ft_session_s` guard check | Galvani | ~2 h | R3 veto |
| T4 | Discord: "is a sealed run that hits the 3,600 s limit marked Failed?" | user | min | sequencing |
