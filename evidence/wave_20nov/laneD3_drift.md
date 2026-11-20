# Lane D3 · cross-day drift estimated from one labelled enrolment (registered Fri 20 Nov 2026 ~21:15 CET, before any D3 stream was scored)

Idea 3 of `evidence/council_20nov/research2_codex.md` (§3), implemented as specified there. Code `laneD3/` (`drift.py` the estimator, `build_drift.py` patches only
`submission.py`, anchors asserted once each; `run.py`/`run.sbatch` NETBCI worker; `niv/enrol.py` extended, opt-in `d3=True`). Never edits `submissions/*`.

## Mechanism (as built)
- Enrolment writes per enrolled person (class means, within-class variance, class counts) of the first 12 coordinates of the 256-PC view `u` (raw view coordinates,
  the same causal `TangentCore` extraction on enrolment and query). Public people carry NaN rows: a window routed to a person without moments gets no correction (counted).
  Deviation, stated: no run-split uncertainty is stored (the `enrol()` API gets no run ids); centroid uncertainty enters the noise term as var_c / n_c.
- Per window, AFTER RouteB, only if a gate routes it (ramp r > 0 to person j, i.e. after the recognition latch): from STRICTLY EARLIER windows (running sums, O(1) per window; the
  stock population shadow posterior p_i(c) of v5.1b is the class weight) two interleaved halves (parity of the window index) each average u_i - sum_c p_i(c) mu_j,c. Both halves need
  >= 20 windows. Gated arm: the two half offsets need a positive dot product in enrolment-variance units (pooled within-class variance); pooled standardised offset Delta (n-weighted),
  r = clip(1 - tr(V_noise)/||Delta||^2, 0, 0.5), V_noise = residual variance of u - p mu about the common offset / n + centroid variance. Centroids shift by r Delta; each class logit moves
  by the diagonal-Gaussian log-likelihood difference (shifted minus original centroid, class variances), clipped to +-0.3, then softmax of log(RouteB posterior) + move.
  The corrected posterior is the model's output (count filter and cp read it); contrasts, personal head, identity, fine-tuning are untouched and never see it.
- Arms: `hardcbB` (exact, enrolled package, unmodified submission), `D0z` (code in, mode 0, smoke only: must equal hardcbB label for label), `D3f` fixed half-shift (r = 0.5 whenever both halves
  have >= 20 windows; no corroboration, no shrinkage), `D3g` uncertainty-gated half-shift. R4's recorded matched streams are read from the finished B10 run (`laneB10_returning_stress.md`; same
  orders full:72000/72001), not rerun.

## Instruments
1. NETBCI (B10 design: 19 people, day 1 enrolled as hardcbB with placeholder rest classes, days 2-4 streamed, one window per call, fresh model per (arm, order, day)), orders full:72000, full:72001,
   Leonardo AIFAC_P02_1675. All arms in one job per person on the identical tapes (paired). BA2 as in B10.
2. P04/P10 (Lane A instrument, Galvani CPU SIF, fold packages with visit 1 enrolled, visits 2-3, orders full0, full1; hardcb final stack vs D3f/D3g on `pkg_ret/{P04,P10}_*`). Ceiling stress.
3. NI-V wrong-enrolment harm check (Galvani, `fa` mode: package k holds person k's visit 1; the 8 other Scherer people stream through it, both visits, order full:0; every window a gate routes to
   k is a wrong recognition; hardcbB enrolled vs D3f vs D3g), plus B-ret as a by-product if time allows.
Checks that fail loudly: D0z == hardcbB bit for bit; a D3 arm may differ from hardcbB only after the recognition window and only with drift activity (worker raises otherwise); enrolled d3 row finite,
public rows NaN (worker asserts); drift stats (routed, active, short, uncorroborated, mean r, max |logit move|) stored per stream; timing paired in the same job (ms/window).

## Predictions (80 % intervals)
- NETBCI BA2 over hardcbB: D3g +1.2 [-0.5, +2.9] (the §3 forecast); D3f +0.9 [-1.5, +3.0]; gated minus fixed +0.3 [-0.5, +1.0]. Drift active in 60 % [30, 90] of routed windows for D3f, mean r of D3g 0.15 [0.0, 0.4].
  Enrolled-person recognition identical in all arms (same gate).
- P04/P10 (BA pp over hardcb, equal cells): D3g 0.00 [-0.45, +0.45]; D3f -0.1 [-0.8, +0.4]; gated is expected to be closer to 0 than fixed.
- NI-V wrong-enrolment: D3g 0.00 [-0.3, +0.2], D3f -0.05 [-0.5, +0.2]; routed windows exist only where hardcbB routes already.
- Cost: < 0.5 ms/window for the drift stage (two accumulators); measured as paired time per stream arm minus hardcbB.

## Rules (fixed now; paired per person, people-resampled 80 % intervals pooled over days and the two orders; bars are reading aids, P04 and P10 reported separately)
- NETBCI per arm: SUPPORTED if the 80 % lower bound > 0, REFUTED if the upper bound < 0, else UNDECIDED. The uncertainty step is credited ONLY if gated minus fixed has a positive point value AND lower bound > 0
  (the §3 requirement is a positive contrast; with 19 people the interval is the rule). Otherwise "uncertainty step not credited" whatever the sign of D3g alone.
- D3g overall SUPPORTED (build a full-chain candidate) only if: NETBCI supported, gated > fixed credited, P04 and P10 each >= -0.45 pp (inside the +-0.5 noise), NI-V wrong-enrolment upper bound >= 0 and worst person >= -1.0 pp.
  Undecided otherwise; nothing here is closed on P04 or P10 alone. Sealed weighting is UNVERIFIED and not applied.

## Run log
Smoke (before any scoring) Leonardo 64980637 (NETBCI k0, full:72000, day 2; hardcbB, D0z, D3f, D3g): D0z equals hardcbB label for label; recognised at window 10, routed 169/179; D3f/D3g differ from hardcbB in 8 windows, all
after recognition, drift active in 139 routed windows (30 short of the 20-per-half minimum), no NaN row; enrolled d3 row finite (class counts 90/45/45), public rows NaN. Paired time per stream D0z 38.2 s, D3f 38.3 s, D3g 38.2 s (drift stage < 0.2 ms/window).
Observation at smoke, stated now: D3g's mean r was exactly 0.5 in every active window (the offset is large against its noise in enrolment-variance units), so on this stream gated = fixed by construction; if that holds across people the
gated-minus-fixed contrast is exactly 0 and the uncertainty step is "not credited" by the rule above (not a result about the idea, a result about this noise rule at these sample sizes).
Galvani P04/P10 packages: `laneD3/add_d3.py` recomputes the moments with the fold package's own TangentCore; the stored fold gallery row is reproduced only to correlation 0.99 (relative max error 0.32 P04 / 0.29 P10: the fold row was not built with the package's own tangent parameters), so the exact-reproduction
assert was relaxed to correlation > 0.97 (the moments are in the package's own query-side coordinates, which the drift stage needs). Setup job 3144615 (packages pkg_ret/{P04,P10}_{hardcb,D0z,D3f,D3g}), NI-V setup 3144618.
NETBCI full array 64981129 (19 people, hardcbB + D3f + D3g, orders full:72000, full:72001, days 2-4) submitted Fri 20 Nov ~22:05 CET.
Galvani returning smoke 3144620 (P04 full0 visit 2, 120 windows, hardcb + D0z + D3f + D3g): D0z bitwise = hardcb on every tape key; D3f/D3g first change at window 40 (recognised at 10), 80 active windows, 109 routed windows seen by the drift stage = RouteB's 109, no NaN row;
D3g r = 0.5 in every active window again (gated = fixed on this stream too). Paired time per session D0z 58.9 s, D3f 59.6 s, D3g 61.1 s, hardcb 59.9 s (noise level; CPU SIF).
Queued Fri ~22:50 CET: NETBCI array 64981129 (Leonardo, submitted 22:05, pending); P04/P10 returning array 3144678 (full0/full1 x P04/P10, visits 2-3, arms hardcb + D3f + D3g); NI-V wrong-enrolment `fa` arrays 3144621 (hb), 3144622 (D3f), 3144623 (D3g) and B-ret `ret` 3144624/25/26 (hb/D3f/D3g), all 9 people.
