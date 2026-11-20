# B1 · evidence for the final upload choice (registered Fri 20 Nov 2026 ~09:55 CET, before any arm below was scored)

Candidate = final stack (`final_stack.md`, hardcb, projected 3,211 s RTX). Everything is scored against it and against exact cap250.
Release `ni26_b1_20261120` (Galvani), chain code `laneB1/chain/` (copy of the hash-pinned lane T chain with a selectable arm set, `B1_SET`).

## What the instruments can and cannot do (stated now)
- **More full-shuffle orders: not available in the full system.** The corrected Friday chain pins two deliveries (full0, within0); laneC/H `full_replay`
  and the chain consumers assert `ctx.Pperm == full0 delivery`. The X bank holds eight orders but only as standalone posteriors. So "more orders" is
  replaced by: seed spread (S set), and the eight-order Scherer instrument (NI-V exact packages, 8 fresh orders) for packages.
- **Seeds**: CBraMod has three seeds (S set scores each alone in the 0.2 share); hard pair, FT stream and cap250 are single-seed (no spread available);
  X has three seeds (X set).
- NI-V new / B-ret for final stack and hard are already in `final_stack.md`; this card adds NI-V for the new lane T packages only.

## Arms and rules
S set (full0, within0; hardcb with ONE CBraMod seed in the 0.2 share): predictions each hcb_sk - hardcb in [-0.15, +0.15] pp, range across seeds <= 0.3 pp,
all three above cap250 in the mean of both orders. Reading: if any seed arm is below cap250 in both orders the claimed hardcb gain is seed-fragile.
T set (lane T fitting forms; the card in `laneT_ttt_cap250.md` holds the arms, predictions and rules): ftctl (zero-change), ftlite5, ftsparse20 in the
final stack, held-new full0 and within0, returning P04/P10 (real fold packages as lane T), NI-V Scherer (RTX 2080 Ti exact packages, 8 orders),
runtime on unrecognised sessions (laneT/time_ft.py, RTX 2080 Ti and A100). Extra prediction now: NI-V Scherer delta vs final stack lite5 -0.05, sparse20 -0.10
[-0.5, +0.3] (the heavy objective with channel drop and noise views is an interpolated-montage risk; lane T4 refuted a different heavy TTT at -3.0 there).
Runtime: lite5 = stock FT cost (3,211 s); sparse20 about 18 % more FT step-prefix work (5 refits x 20 steps at prefix 20..180 vs 17 x 5), +2 s per unrecognised
session on RTX, projected 3,340 s [3,250, 3,450] (the 3,300 s bound may fail).
X set (B6 arm X raw family, 3-seed bank, **full0 only** because the bank has no within0 and no returning fits; stated, returning counted unknown):
**mixing rule fixed now**: the raw slot proposal of the final stack is 0.8 hard + 0.2 cb. Primary `hcX` = X **shares the slot with the hard pair**: 0.4 hard + 0.4 X + 0.2 cb
(CBraMod share unchanged). Secondary `hXc` = X **replaces CBraMod**: 0.8 hard + 0.2 X. Both at MASS 5/6, same chain tail. Plus `hcX_sk` with one X seed.
Predictions vs final stack (full0, paired people 80 %): hcX -0.05 [-0.5, +0.4]; hXc -0.05 [-0.5, +0.4] (F-C22: raw-slot edits interact negatively; X vs stock +0.23 alone,
hard and cb already hold +0.51). Supported if lower bound > 0, refuted if upper < 0, else undecided (bars are reading aids, +-0.5 noise).
Adoption rule for the final package: T forms as in the lane T card (both-order mean >= +0.2 vs final stack, returning >= -0.1, NI-V >= -0.3, runtime <= 3,300 s, zero-change).
X forms cannot be adopted tonight on this evidence (full0 only, no returning, no package): reported as direction only unless full0 >= +0.4 with lower bound > 0.

## Zero-change (pkg_zero.py, 200 windows per session, two people)
`final_stack_hardcb.zip` with CB_MIX = 0 and HARD_FILES = () must reproduce `v66_cp1_cap250.zip` step posteriors to < 1e-9 and labels exactly (the script's own
assert) on P01 (new) and P04 (returning), visit-1 and visit-2 sessions, CPU. (The script previously only zeroed CB_MIX, which is stock only for the cb-only zip;
it now also clears HARD_FILES and takes `PKG_ZERO_WINDOWS`.) Prediction: PASS.

## Stale Galvani jobs cancelled (20 Nov ~09:30 CET)
3136655_[0-17] / 3136656 / 3136657 (lane B shuffle external smoke chain, ni26-lane-b): head job 3136654 failed on a staging path (`make.py`), dependents could never run;
the corrected public result (`out/held.json`, 3136661) is final and no card waits on the external half (audit_20nov names it as a possible re-stage, not queued).
3137806 (`wrap`, cpu, priority) -> 3137807/3137808 (ni26-laneE-time, A100 / RTX): Lane E capacity timing; lane E is refuted/undecided on Scherer and its timing question is superseded by
`laneT/time_ft.py`. Nothing waits on them.
