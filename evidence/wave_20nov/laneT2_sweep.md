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
