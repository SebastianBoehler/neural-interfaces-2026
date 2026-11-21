# Lane TB-verify NI-V: Team B nine members at 0.25 on hardcbB, Scherer (21 Nov, Leonardo, account 1636)
Registered in docs/LEDGER.md ("Lane TB-verify NI-V") before any score. Rule: veto signal if B-new delta < -0.3 pp.
What ran: hardcbB_TB25 = R4 hardcbB.zip with build.revised(src, 0.25, 3, bank) (dead-session guard included) + team_b.pt from the Scherer-free
bank (laneTB-21nov weights.pt; asserted people P01..P10, first_only [], ext ''); hardcbB_TB0 = same code, blend 0.0. Builder laneTB/build_niv.py
(job 64998390, also enrols hardcbB_TB25_k0..8 with the unchanged niv/enrol.py: it only rewrites weights.npz and copies other members, so no
string patch of submission.py is involved). Smoke 64998410 (k=5, full:s71008, visit 1, 120 windows): TB0 labels == hardcbB (120/120),
TB25 changes 12/120. Full run array 64998411 (27 tasks, all COMPLETED): B-new = 9 people x (8 full + 8 within seeds s71008-71015) x visits 1,2,
hardcbB and TB25 on the same orders; B-ret = enrolled visit 1, visit 2 streamed, 16 full + 16 within orders s71008-71023, TB25 vs the existing
Lane R4 hardcbB outputs (same orders, same exact ZIP lineage). Scoring laneTB/score_new.py and laneR4/leo/score.py; deltas in balanced accuracy pp,
paired, people-resampled 80 % intervals (9 people).
B-new (hardcbB absolute 48.6, vs B-ret 56-58): full-session (both visits) +0.008 [-0.29,+0.33]; within -0.19 [-0.71,+0.34]; pooled -0.09 [-0.41,+0.22];
full visit 1 +0.45 [-0.13,+1.01], full visit 2 -0.44 [-1.00,+0.15] (opposite signs: noise).
B-ret: full -0.23 [-0.74,+0.27]; within -0.10 [-0.86,+0.66]; pooled -0.17 [-0.74,+0.43]; abs 56.4 / 57.9.
Against the rule: B-new full-session +0.008 is above -0.3, so no veto; the prediction (+0.3 [-0.5,+1.0]) is not confirmed either, every interval contains 0.
Limits: 9 people; 22 of 41 channels interpolated; B-new uses 8 orders per scheme; hardcbB B-new rerun on Leonardo (not the Galvani 2080 Ti numbers
in final_stack.md), so absolute values differ from that table. TB25 vs hardcbB differ also by the dead-session guard (TB0 check only on one stream).
Raw npz kept on Leonardo (ni26_laneTB_20261121/out, ni26_laneR4_20261120/out/hardcbB_TB25_k*).
