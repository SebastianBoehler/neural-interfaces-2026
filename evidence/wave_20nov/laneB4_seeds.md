# Lane B4 · More seeds for the neural family with the largest seed variance (cap250 SessionBN families)

Registered Thu 19 Nov 2026 ~23:10 CET (22:10 UTC), before any lane-B4 posterior was scored. Code `laneB4/` (lopo_b4.py, collect_b4.py, timing.py, leo_fit.sh);
`laneH/full_replay.py` (Galvani, pinned Friday producers) unchanged. Reads: AGENTS.md, audit_20nov/SYNTHESIS.md, assumptions_model_codex.md (active weights).

## Context (what the weights are)
cap250 SessionBN nets 35-64 = five groups x six seeds, 1/90 each = 1/15 per group: raw FBC (k20), EA FBC (exported twice, k2/k5, so two groups from one set of six fits),
FBLight (k5), Sinc (m7r trunk, k20). FT updates seed 0 of each family only. Extra seeds beyond 6 are therefore frozen forward-only members.
Sinc seeds are tied to pretrained m7r trunks (`runs/pretrain/sincshallow_ea-m7r-s0..3` exist; more seeds need new trunks) - Sinc is measured on 4 seeds only.
The held-new full-system route (laneH/full_replay.py, outer.replace) exists **only for the raw FBC slot** (`sc_fbraw`): the bank mean replaces five frozen seeds (MASS 5/6) or one (1/6).
Other families have no full-system swap route in this lane; they are measured standalone, and a full-system claim for them is not made here.

## Compute
Ferranti was not usable: no repo/data staging (lane V cancelled there), no `/mnt/lustre/work/gehler` mount, queue about two days deep (20+ pending jobs).
Training on Leonardo (AIFAC_P02_1675, release `/leonardo_work/AIFAC_P02_1675/ni26_laneB4_20261120`, a100 boost nodes); full-system scoring on Galvani cpu (lane H route) after copying the small banks;
timing on a Galvani RTX 2080 Ti (RTX class; the sealed worker is an RTX PRO 4500, faster, so the number is conservative).

## Design
1. Standalone LOPO, shipped recipes (public PROTEUS only, 40 epochs, receipts require epochs = shipped): raw `fbcnet_raw_sbn`, EA FBC `fb_sbn_rot05` (k5 at test), FBLight `fblight_ea_rot05_sbn` (k5),
   Sinc `sincshallow_ea_ptm7r_rot05_sbn` (k20). Seeds 0-11 (raw, EA FBC, FBLight), 0-3 (Sinc); ten held people plus the deployment fit on all ten ("all").
   Eight pinned full orders. Seed variance = SD across seeds of cell-weighted held-new BA (mean over orders); K-seed-mean curve K = 1..max (60 subsets per K).
2. Full system (raw family only, lane H route, order full0): arms R3 (seeds 0-2 = lane H arm A control form), R6 (0-5), R12 (0-11) at MASS 5/6 and 1/6, plus single-seed banks S0..S5 at MASS 5/6 for per-seed full-system spread.
   Paired against exact cap250 (Friday c25d full0, `--cap250` check, stock 73.74), and R6 / R12 against R3 and R12 against R6, people-resampled 80 %, cell-weighted, 10,000 draws.
3. Runtime: per-window forward time of one StreamBN-traced net per family on the 2080 Ti (batch 1, 1000 calls), x 20,083 sealed windows = added seconds per extra seed; compared with 3,600 - 2,555 = 1,045 s and the self-imposed 3,060 s projection (505 s).

## Predictions
- Seed SD of single-seed held-new BA (pp): raw 0.35 [0.2, 0.6], EA FBC 0.30, FBLight 0.40, Sinc 0.45 (four trunk-tied seeds; large uncertainty). Ranking is a guess; no family is expected to stand out by more than 2x.
- Standalone K-curve (raw): +0.34 pp from K = 1 to 3 (lane H: 58.97 -> 59.31), +0.12 from 3 to 6, +0.06 from 6 to 12.
- Full system vs exact cap250 (mass 1/15 of the mix, so about 0.1 x standalone, widened to the instrument): R3 -0.10 [-0.45, +0.25] (lane H A control: -0.15 at 5/6, +- 0.2); R6 -0.08; R12 -0.05 [-0.40, +0.30].
  R12 - R3 +0.04 [-0.10, +0.18]; R12 - R6 +0.02 [-0.08, +0.12]. Per-seed full-system SD about 0.15 pp. The user-level prior for the whole lane stays +0.0 to +0.2 pp.
- Runtime: about 5 ms per extra seed-window on the 2080 Ti, so about 100 s per extra seed over 20,083 windows (six extra: ~600 s, over the 505 s margin; three extra: ~300 s).

## Rules (stated before scoring)
- **Refuted** (more seeds of the raw family): full-system upper 80 % bound of R12 minus cap250 <= 0 at 5/6 AND at 1/6 (the lane brief), and R12 - R6 point <= 0.
- **Supported** (build candidate): R12 - cap250 lower 80 % bound > 0 at 5/6 AND R12 - R6 point > 0. Otherwise **undecided** (instrument +-0.5 across people; neither P04 nor P10 decides).
- Family choice: the family with the largest single-seed BA SD is reported as "most variable"; raw is trained and scored in the full system regardless because only it has a route.
  A non-raw family being most variable gets its K-curve standalone and the note "no full-system route here".
- Runtime fits if the added seconds for the proposed extra seeds <= 505 s (self-imposed) and are stated against 1,045 s (official); final-stack additions (F-C25 hard, CBraMod) are not subtracted here.
- Not covered: returning people, orders other than full0 in the full system, FT interaction (extra seeds are frozen), Sinc beyond 4 seeds, the sealed worker's own timing.

## Addendum A (Thu 19 Nov ~23:45 CET): standalone seed variance and raw full-system results seen; Sinc/FBLight/EA FBC route registered before any of those was scored
Seen before this addendum (standalone, no full-system number of Sinc, FBLight or EA FBC yet): SD of single-seed held-new BA, mean over 8 orders: **Sinc 0.80** (seeds 0-5), raw 0.62, FBLight 0.57, EA FBC 0.26 (12 seeds each except Sinc).
K-seed-mean BA: raw 58.58 (K=1) / 58.91 (3) / 59.06 (6) / 58.92 (12), EA FBC 61.83 / 62.31 / 62.52 / 62.54, FBLight 56.45 / 56.86 / 57.07 / 57.14, **Sinc 58.35 / 59.81 (3) / 60.39 (6), still rising** (+0.17 from 5 to 6).
Raw full-system (lane H route, full0, 5/6 swap) was already scored: R3 -0.151, R6 0.000, R12 -0.154 pp vs cap250; single seeds -0.34 to +0.22 (see results below).
Sinc is therefore the highest-variance family and the only unsaturated one; its extra seeds need new m7r trunks (cap250 holds trunks s0-s5): trunks s6-s11 (jobs 64886838) and Sinc LOPO fits s6-s11 (64886839), same recipe (30 epochs, rotation 0.05, seven MI corpora).
Family route: `laneB4/galv/outer_fam.py` = laneH/outer.py with `B4_FAMILY` (fbl, sinc, fb) replacing the family slot; zero-change, stage-movement and warm-state asserts unchanged; the slot-5 assert is kept for raw only. If an assert fails the family is "no route", not a result.
Arms per family F in {sinc, fbl, fb}: F6 (seeds 0-5) and F12 (0-11; Sinc after the new trunks) at MASS 5/6 and 1/6, against exact cap250 and F12 against F6, people-resampled 80 %.
Predictions (full-system pp): Sinc12 vs cap250 +0.05 [-0.25, +0.35] (standalone gain 6 -> 12 about +0.3, x0.1 plus instrument); Sinc12 - Sinc6 +0.04 [-0.10, +0.20]; Sinc6 vs cap250 -0.05 [-0.35, +0.25] (the swap form carries its own control offset, as raw R3 -0.15);
FBLight12 - FBLight6 +0.01; EA FBC12 - EA FBC6 0.00. Sinc seed 0 is the lowest (56.95) and is the one FT updates; it is kept in this form (1/6), unchanged.
Rules: same as above, per family. Sinc extension is called supported only if Sinc12 - cap250 lower 80 % > 0 at 5/6; refuted if upper <= 0 at 5/6 AND 1/6; otherwise undecided. Runtime from timing.py.
