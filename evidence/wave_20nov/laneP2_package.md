# Lane P2 · package the dispersion correction with a real member (registered Fri 20 Nov 2026 ~12:55 CET, before any arm below was scored)

Parent: `laneP_stack.md` section 5 (ds_l5 +0.134 / ds_all +0.469 vs final stack, stand-in member, stock-raw pseudo-labels, per-session reset). Code `laneP2/`; Galvani release `ni26_laneP2_20261120`.
Never uploaded from this lane; the user decides.

## Decisions fixed before scoring
- **Member (decided now)**: package net 47, `sbn_fbcnet_raw_k20_s0.pt`, the live cap250 raw FBCNet SessionBN seed 0 (already run every window as the FT raw member; the five outer raw members are replaced
  by hard+cb in the stack, member 47 is not). Same recipe as the lane P stand-in (`fbcnet_raw_sbn`, k20, seed 0). Segment contributions come from the package class `SharedFIRSJit` (hook after
  `flatten_layer`, float64 einsum with the final Linear, recovered-logit check, `None` on mismatch = window skipped and counted in a `[disp]` stderr line, never reused).
- **Pseudo-labels**: the stock-raw argmax of lane P cannot exist inside a package (no parallel stock chain). Packaged form: argmax of the package's own UNCORRECTED routed output (RouteB) of each window,
  history of the same session only (reset in `reset_state`). Consequently the chain is re-scored with this label form (`ds*`) and with stock labels (`ds*_stk`, = lane P form) so the label change is measured, not assumed.
- **Real-member dev instrument**: leave-one-person-out fits of the same recipe (job array over P01-P10), exported as stateful TorchScript traces and streamed through the package class with the hook
  (`laneP2/seg_member.py`; preprocessing = eager causal alignment, whose package parity is the cap250 SBN build card). Not covered: the package InputStream path itself, all-public member weights (the held person is in the package member's training set, so only LOPO fits can score it).
- **Forms built** (on `final_stack_hardcb_tttlite5.zip`, only `submission.py` differs): `ds_l5` = hardcb + lite5 + dispersion; `ds_all` = + FBL2 (friday/cap250_fbl2_package logic composed on the stack mixer; FBL2 veto: external hard+FBL2 -0.579, returning -0.2 at 20/80). Switches `DispEnsemble.DISP_ON`, `FBL2_ON` exist for the zero-change check only.
  NI-V copies are built on the Scherer-free base (`hardcbB_tttlite5.zip`).

## Checks, all chained afterok after the build (smoke first)
1. Package unit checks (CPU): `zero` (DISP_ON and FBL2_ON False equals the lite5 package bitwise, steps and labels), `disp` (package output equals `segcorr.correct` on the tapped posteriors/segments to 1e-12, skipped 0), `fbl2` (composed FBL2 on a zeroed stack equals the friday FBL2 trial ZIP).
2. Full chain, held-new full0 + within0, real member (arms: P set controls + ds, ds_l5, ds_all self-label, `_stk` stock-label controls).
3. Returning P04/P10 (fold packages, 8 orders x 4 cells, lane T worker; correction acts on every routed window including recognised ones, so a returning effect is possible).
4. NI-V Scherer B-new / B-ret (8 orders, exact packages, `score_candidate.sh`).
5. Paired A/B/B/A runtime vs cap250 on unrecognised sessions, RTX 2080 Ti and A100, projection 2,555 s x ratio.

## Predictions (pp, both-order mean unless stated; 80 % half-width about +-0.7 per forecast_calibration)
- Unit checks PASS; real-member posterior vs eager < 1e-4; `ds_l5_stk` (real member, stock labels) within +-0.10 of the lane P `ds_l5` (+0.526 vs cap250, +0.134 vs final stack).
- Label change (self minus stock labels): ds_l5 0.00 [-0.15,+0.15]; ds_all 0.00 [-0.15,+0.15].
- vs cap250: ds_l5 +0.50 [-0.2,+1.2]; ds_all +0.80 [-0.3,+1.9]. vs final stack: ds_l5 +0.10 [-0.5,+0.7]; ds_all +0.40 [-0.4,+1.2] (haircut from the +0.47 / +0.13 stand-in values: selection among ten ranked arms and the unexplained three-way interaction).
- Returning P04/P10 w20/80 vs final stack: ds_l5 -0.05 [-0.25,+0.15] (lite5 -0.04 included); ds_all -0.20 [-0.45,+0.05].
- NI-V vs hardcbB B-new: ds_l5 -0.05 [-0.4,+0.3]; ds_all -0.05 [-0.4,+0.3]; B-ret equal within +-0.3.
- Runtime ratio vs cap250: lite5 stock 1.18 RTX / 1.23 A100; correction adds < 1 %; FBL2 x1.014 -> ds_l5 <= 1.24, ds_all <= 1.26; projected 3,150-3,230 s, bound 3,450 s: PASS.

## Reporting rules (fixed now)
Adoption suggestion over the final stack (user decides; bars are reading aids): both-order mean vs final stack >= +0.2, returning w20/80 >= -0.1, NI-V no veto (upper 80 % bound of B-new or B-ret vs hardcbB < 0, or worst person < -1.0 pp), projected runtime <= 3,450 s, all unit checks PASS. ds_all is additionally flagged for the FBL2 external veto whatever the numbers. Intervals with 0 inside are "undecided", not closed. Anything that silently does not apply (skipped windows, switch not applied, bank mismatch) is reported first.

## Progress 20 Nov evening (restart after the Mac outage; results read from Galvani `ni26_laneP2_20261120`, nothing re-scored)
Note on provenance: the header time of this file was edited 13:10 -> 12:55 by the cut-off agent without a record; the first scored chain job (3142150/51) started about 13:09 Galvani time, the unit checks 3142129-39 ran 13:00-13:22. Read the registration as "before the arm scores", not to the minute.
Built (12:49): `zips/ds_l5.zip` 89be4c71..., `ds_all.zip` 923db84b..., Scherer-free `hardcbB_ds_l5.zip` c2c30840..., `hardcbB_ds_all.zip` 26ab53eaf... Unit checks PASS (zero bitwise on 180 and 568 windows for ds_l5/ds_all; disp streaming-vs-batch 0.00; fbl2 bitwise vs the friday trial ZIP).
Chain, real member (3142140/50/51/52; full0 + within0; `laneP2/chain/score_P.json`): vs final stack ds_l5 +0.133 [-0.18,+0.44], ds_all +0.343 [-0.07,+0.73]; vs cap250 ds_l5 +0.525 [+0.18,+0.86], ds_all +0.735 [+0.28,+1.18]; ds (no lite5) -0.193 [-0.40,-0.04]; label form (self minus stock labels) ds_l5 -0.017, ds_all -0.050 (inside the registered +-0.15); ftctl equals hardcb bitwise.
Returning P04/P10 (32 cells, 3142669/70, collector 3142671), vs final stack: lite5 w20/80 -0.087; **ds_l5 -0.595 [-0.69,-0.50], ds_all -0.781 [-0.92,-0.63]; all 8 orders negative.** The registered returning bar (>= -0.1) FAILS for both ungated forms (registered -0.05 / -0.20). Silent-no-op check: the correction does move returning cells (P04 full0 ds_l5 85.67 vs ftlite5 87.79 in cell 2).
Timing (A/B/B/A, unrecognised, 3142157-60): see below.

## Lane P2-U · correction only on unrecognised sessions (registered Fri 20 Nov ~18:50 CET, before any gated arm was scored)
Mechanism: the returning loss is the correction acting on windows of RECOGNISED sessions (RouteB output of a personal head, pseudo-label history built on it). Forms `dsu_l5`, `dsu_all` = ds_l5 / ds_all with `DispEnsemble.UNREC_ONLY` True: after the window's own step, `recB` True -> output returned uncorrected (`[disp] ... gated N`). Unrecognised sessions (all held-new chain windows, NI-V B-new, sealed new people) are unchanged by construction; the gate is the only difference.
Predictions: (a) unit `gate`: dsu on recognised sessions equals the lite5 package bitwise, gated > 0; gate switched off equals the ungated ds zip bitwise. (b) returning w20/80 vs final stack: dsu_l5 -0.09 [-0.2,0.0] (= lite5), dsu_all -0.3 [-0.6,-0.1] (FBL2's own -0.2 plus residual). (c) NI-V B-ret dsu_l5 equals hardcbB_tttlite5 exactly on recognised sessions; B-new equals the ungated ds form. (d) held-new chain: not re-run; the scores of ds_l5 / ds_all stand for unrecognised sessions (chain windows are unrecognised; a recognition event in the chain would make the gate matter: reported as an assumption). (e) runtime = ds forms (gate is free).
Rule: dsu forms are the adoption candidates only if returning w20/80 >= -0.1 (dsu_l5) and every other bar of the registration above holds; the ungated forms are reported but fail the returning bar.
