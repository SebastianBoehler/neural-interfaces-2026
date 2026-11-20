# Lane O · proxy instrument for the final choice on eight full orders (registered Fri 20 Nov 2026 ~12:30 CET, before any arm was scored)

Why a proxy: the corrected Friday chain can only be run on full0 and within0 (it pins the delivery and needs per-order frozen upstream artifacts: archived
control, gates, route, relab, FT/bridge releases, hard/CBraMod banks); regenerating them for seeds 71001-71005 is a multi-day rebuild. The coordinator chose
option 2: a DIFFERENT, weaker instrument over all orders the banks have. Everything below is labelled PROXY and is not the full-system chain.

## Instrument (what it is and is not)
- Ten held-new people x eight pinned full orders (`runs/shuffle_perms.npz` `full|k|all_loso`, k=0..7 = seeds 71000+k; same file as lane B4/B6 banks).
- Scored object: the **raw-slot proposal as a standalone posterior** (official cell-mean BA, cell weights). No mixing with the other 14/15 of cap250, no EM/TSEM, count/CSP,
  recognition, FT. Arms: stock = R6 six-seed raw-FBC bank mean (the cap250 raw group; not bitwise the package slot), hard = mean of the two F32c pair seeds
  (new streams, see below), cb = CBraMod-C 3-seed mean (`ensemble_C_held_new.npz`), hardcb = 0.8 hard + 0.2 cb, X (lane B6 3-seed bank), hcX = 0.4h+0.4X+0.2cb, hXc = 0.8h+0.2X.
- hard on orders 1-7 is NEW: `laneO/hard_orders.py` runs the pinned F32c fold actors (same `paired`/`frozen_member`/`do_order` path as `friday/f32c_v2/produce32.py`)
  on full0..7 for 10 people x 2 seeds (Slurm array 3140891, smoke 3140866); zero-change: full0 must equal the archived `hard|full0` fold bitwise (asserted in the script).
- NOT covered, stated now: T-light (FT streams exist only for full0 and within0; hardcb+T-light stays a two-order chain result from `laneT_ttt_cap250.md`), the chain's
  nonlinearity (slot movement is diluted ~15x and re-gated in the chain; standalone slot deltas are NOT pp-comparable to chain deltas), returning people, NI-V (lane N).
- Calibration: on full0 the same standalone deltas are printed next to the chain's measured full0 deltas (hard +0.331, cb +0.263, hardcb +0.514); agreement of sign/rank on
  full0 is the only validity check available.

## Predictions (pooled over 8 orders, pp, standalone slot BA, vs stock)
hard +1.0 [-1, +3] (the hard actors beat the raw FBC members standalone; F-C25 was selected on this); cb +0.5 [-2, +3] (CBraMod 1/6 share, weak standalone); hardcb > hard by +0.3 [-0.5,+1.1] (a 20 % CBraMod share
helps a posterior average via diversity); hcX - hardcb 0.0 [-1, +1]. Sign of hardcb-hard positive in >= 5/8 orders. Full0 standalone ranking hardcb > hard > cb (as in the chain).

## Rules (fixed now)
Intervals (80 %, 10,000 draws) resample people and orders jointly (also people-only and orders-only reported). Supported if the people+orders lower bound > 0, refuted if the
upper bound < 0, else undecided. Reading aid for the upload choice only: hardcb vs hard decision stands on the chain (two orders) unless this proxy sign disagrees with the chain
on full0, in which case the proxy is declared uninformative for that contrast. No gate; neither instrument alone decides.

## Jobs (Galvani, release `/mnt/lustre/work/gehler/gfh527/ni26_laneO_20261120`; the account cap of 36 running jobs is shared with other lanes, so the array queues)
smoke 3140866 (array task 4 = P05 seed 0, full0 only) -> array 3140891 (20 tasks, afterok smoke) -> `score_o.py` 3140892 (afterok array) writes `score_o.json` / `score_o.md`.
