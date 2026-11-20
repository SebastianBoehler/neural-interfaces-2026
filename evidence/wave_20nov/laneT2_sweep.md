# Lane T2 · wide TTT recipe sweep on team C's base ensemble C1 (E6) · 20 Nov

Question: which test-time-training recipe gives the best held-new BA on team C's fixed base, and within the sealed
time budget? The base is not retrained: ten LOPO packages `work/C1/pkg_P01..P10` from team C's Galvani release
(copied 20 Nov 07:43 CET; weights sha256 prefixes: P01 ea8956e3, P02 9edd9ed4, P03 383187b8, P04 4a7e6756, P05 f825d8ac,
P06 27a32448, P07 f2712e09, P08 20106aa9, P09 7bf8d1da, P10 4aa88cfd). Held-new LOPO, official cell-mean BA via harness
`data.cell_table`, harness delivery (`D.plans`, full-session shuffle, orders 71000/71001; official `fresh_model`).
Code: `cleanslate/ttt_sweep/` (`ttt_model.py` = copy of team C `submission.py` at 95b4405b + options target/aug/params,
all validated; `cfgs.py` design; `run_sweep.py`; `collect.py`; `sweep.sh`). Leonardo `/leonardo_work/AIFAC_P02_1675/ni26_ttt_sweep`.

## Pre-registration (before any sweep posterior was scored; only seen: team C's P07/P09 partial A1 73.0/68.8, A2h 81.5/77.3)
Factors (levels): steps/refresh {4,8,16,32}; lr {3e-4,1e-3,3e-3}; refresh stride {5,10,20} (first refresh n=16, then every
stride; dense_until = 16 so stride is the only schedule knob; team C's own schedule is every 4, dense to 48); target
{Sinkhorn-balanced, plain ensemble, hard one-hot with max posterior >= 0.5}; aug {weak: shift 8, noise 0.03, no channel drop,
scale 0.05; strong: team C default shift 24, noise 0.1, drop 0.1, scale 0.15}; IM weight {0, 0.5, 1}; adapted params {BN affine,
last block (head + EEGNet block 2 + BN), all}; L2 anchor {on 1e-2, off}. Prefix-BN refresh is on in every arm. Centre = steps 8,
lr 3e-4, stride 10, Sinkhorn, strong, IM 0.5, all, anchor on. Reference arms on the same code: A0, A1, A2, A2h (team C schedule).
Design: stage 1 = 4 refs + centre + 15 one-factor-at-a-time + 288 balanced-random configs (296 unique), 4 people
(P01,P04,P07,P10, fixed before scoring, every third person) x order 71000. Stage 2 = top 20 % by stage-1 BA (+ refs A1/A2h)
on all 10 people x orders 71000,71001. Stage 3 (if time) = top 8 on orders 71002-71007.
Predictions (BA pp, all-ten-people scale): A1 ~57; best stage-2 recipe - A2h = +1.5 (range 0 to +4); steps 16/32 >= 8 (+0.5);
lr 3e-3 < 1e-3 (-1, drift); Sinkhorn >= plain (+0.5); hard ~ Sinkhorn (+/-0.5); weak aug < strong (-0.5); IM 0/0.5/1 flat (+/-0.3);
params all > last > BN only (+1 / +0.5); anchor off ~ on (+/-0.5).
Rules: promotion is by rank, not a gate. A factor effect (marginal mean over the random design, people-bootstrap 80 %) is
"supported" if the interval excludes 0, "refuted" if both ends lie inside +/-0.3, else undecided (dev noise +/-0.5 across
people). Successive halving is trusted only if stage-2 BA of the promoted recipes has Spearman > 0.3 with their stage-1 BA.
Runtime: ms/window = stream wall per window incl. Python on one A100 (Leonardo); sealed-card estimate = x2.5 (team C
convention, unmeasured ratio); usable if x2.5 <= 170 ms, i.e. A100 <= 68 ms. Slower recipes are listed but flagged.
Dev weighting like the sealed set is not applied here (held-new only, 10 equal people).

## Log
- 20 Nov 07:5x CET: smoke 64916675 (P07, 4 configs) and equivalence check 64917241 passed. Equivalence vs team C stream_eval on their
  order 0 (window accuracy, P07/P09): A1 73.02/68.78 (team C 73.0/68.8, identical); A2h 78.84/77.25 (team C 81.5/77.3). So the frame
  matches exactly without gradient TTT; gradient TTT is not bit-reproducible on GPU (P07 A2h differs 2.6 pp for the same code and
  order) and strongly order-dependent (P07 A2h: 78.8 / 64.5 / 77.3 on orders 0 / 71000 / 71001, A1 73.0 / 71.4 / 69.3). Run-to-run noise of
  a single person-order TTT cell is therefore several pp; ranking needs many person-orders. Deviation from the pre-registered
  design (made before scoring): stage 1 uses TWO orders (71000, 71001) on the 4 people instead of one, since the cost is small.
- Stage 1 submitted: Leonardo job 64917505 (296 array tasks = 4 people x 74 config chunks, 296 configs x 4 people x 2 orders).
- Stage 1 done (job 64917505, 296/296 tasks; collect 64919264): 296 configs, P01/P04/P07/P10 x orders 71000,71001, A100 ms/window with warm-up.
  References on these 4 people: A0 59.61, A1 60.88 (7 ms), A2 64.20 (56 ms), A2h 60.58 (104 ms; same code as team C's A2h, i.e. the "heavier" recipe
  is not better than A1 here). Leaderboard `laneT2_s1_leaderboard.json`; marginal effects (random design, bootstrap over configs, 80 %):
  target sink +2.96 [2.5,3.4] / plain +0.91 / hard -3.68 [-4.5,-2.9]; lr 3e-4 +1.47 [1.0,2.0] / 1e-3 +0.44 / 3e-3 -1.89 [-2.7,-1.1];
  stride 20 +0.77 [0.2,1.4], 5 -0.76; params all -1.41 [-2.2,-0.7], last +0.92, bn +0.44; IM 1.0 +0.70 [0.1,1.3]; steps 4 +0.54, 32 -0.63;
  aug and anchor within +/-0.2 (undecided -> flat). The config bootstrap ignores person noise (4 people), so intervals are optimistic.
  Predictions that failed: params all > last > BN (reverse), IM flat (1.0 slightly better), steps>=16 not better (worse), A2h not better than A2.
  Stage 2 submitted: job 64919524 (top 58 of 292 + A0/A1/A2/A2h = 62 configs, 10 people x 2 orders, 210 tasks).
