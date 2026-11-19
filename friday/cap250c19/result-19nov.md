# F-C19 · Cross-trajectory CSP evidence at complete-system disagreements

19 November 2026, 03:51:45 CET · FULL-SYSTEM DEV · Corrected D-EAwcf within0 and D-EAfcf full0 · Same ten held-new PROTEUS people

The fixed 1.5× CSP odds rule lost **0.1515 pp** versus the exact stock trajectory within0 and gained **0.0332 pp** full0. The within0 interval refutes a gain. The full0 interval spans zero. The previous-window control lost more within0 and slightly outscored the rule full0. The registered external test condition failed. [Preregistration](../../docs/LEDGER.md#f-c19--cross-trajectory-csp-evidence-at-complete-system-disagreements)

| Complete trajectory or terminal rule | Within0 balanced accuracy | Within0 change vs stock, pp [paired 80%] | Full0 balanced accuracy | Full0 change vs stock, pp [paired 80%] |
|---|---:|---:|---:|---:|
| Stock with warm adaptation and CSP | 78.3241% | — | 73.7390% | — |
| Frozen waveform, with its own CSP | 77.4299% | −0.8943 [−1.7125, −0.0045] | 73.8977% | +0.1587 [−0.3770, +0.7063] |
| Current-window CSP odds | 78.1727% | −0.1515 [−0.2669, −0.0479] | 73.7721% | +0.0332 [−0.1320, +0.2278] |
| Previous-window sham | 77.7221% | −0.6021 [−0.7639, −0.4433] | 73.8167% | +0.0777 [−0.2191, +0.3715] |
| Equal log blend on eligible rows | 78.0282% | −0.2959 [−0.4864, −0.0882] | 74.2155% | +0.4766 [+0.2669, +0.7758] |
| Gold oracle on eligible rows | 79.5317% | +1.2075 [+0.7043, +1.7009] | 75.5312% | +1.7922 [+1.5044, +2.0850] |
| Gold oracle across all rows | 80.2520% | +1.9279 [+1.1265, +2.6874] | 76.4354% | +2.6965 [+2.0913, +3.2197] |

Current-window odds minus the sham was **+0.4506 pp** [+0.3052, +0.5974] within0 and **−0.0445 pp** [−0.3141, +0.2620] full0. Versus the blend it was **+0.1445 pp** [−0.1201, +0.3778] and **−0.4434 pp** [−0.6445, −0.2958]. The separate 4,000 paired draws resample people over 28 official interface/visit cells in each delivery. Both measurements use the same people.

The eligible oracle found 32 rescues within0 and 48 full0. The unrestricted oracle found 51 and 72. The eligibility mask itself retains about two-thirds of the tested alternatives' avoidable errors, so the alternatives have measurable complementarity. The current CSP cue used little of it. The gold oracles are retrospective choice ceilings for these two trajectories. They do not estimate expected sealed improvement.

## Fixes and regressions against stock

The fixed odds switched on 29 of 93 eligible within0 rows and 34 of 103 full0 rows. Three swaps in each delivery changed one wrong label to another. The previous-window cue switched on 39 and 38 rows. The fixed-odds rule produced zero motor imagery rescues, with six and two regressions. Mental subtraction had 2 fixes / 7 regressions within0 and 6 / 6 full0. Word generation had 9 / 2 and 10 / 7.

| Person | Within0 fixed odds fixes / losses / net | Full0 fixed odds fixes / losses / net | Within0 eligible / all oracle fixes | Full0 eligible / all oracle fixes |
|---|---:|---:|---:|---:|
| P01 | 1 / 1 / 0 | 2 / 1 / +1 | 5 / 6 | 4 / 6 |
| P02 | 0 / 0 / 0 | 1 / 1 / 0 | 1 / 2 | 3 / 3 |
| P03 | 0 / 0 / 0 | 2 / 1 / +1 | 0 / 2 | 2 / 5 |
| P04 | 6 / 6 / 0 | 4 / 4 / 0 | 14 / 23 | 13 / 24 |
| P05 | 2 / 2 / 0 | 1 / 2 / −1 | 5 / 8 | 6 / 9 |
| P06 | 0 / 2 / −2 | 2 / 0 / +2 | 1 / 2 | 5 / 5 |
| P07 | 2 / 2 / 0 | 1 / 0 / +1 | 2 / 3 | 4 / 4 |
| P08 | 0 / 0 / 0 | 1 / 3 / −2 | 2 / 2 | 2 / 2 |
| P09 | 0 / 1 / −1 | 1 / 1 / 0 | 1 / 1 | 1 / 2 |
| P10 | 0 / 1 / −1 | 1 / 2 / −1 | 1 / 2 | 8 / 12 |
| **All** | **11 / 15 / −4** | **16 / 15 / +1** | **32 / 51** | **48 / 72** |

## Verification and receipts

Each complete replay used 2,655 windows across 14 sessions. Stock raw, count-filter and final scores were bitwise equal to the authenticated parent. The challenger used V6r with warm reads forbidden and matched V6Fr with literal zero deltas. Each native CSP branch processed the same covariance values, 126 refits and 1,395 fusions. Its refit sampled only strictly prior rows and that branch's own raw scores. The selector never changed those inputs or states. Both native likelihoods reconstructed the original terminal values exactly. The original count filter moved 2,005 rows per branch within0 and 308 full0.

The first real smoke `3133786` found active CSP and exact stock but chose two naturally unlatched count sessions. The fixed smoke kept P01, chose the interleaved peer from the actual count latch before age 120 and passed. P01 had six eligible real disagreements, with three current-window switches and one sham switch. The within0 and full0 blind replays took 2m32 and 2m55 on Galvani CPU2. These are replay times. Both prediction banks were locked before the separate scorer opened labels. No further external run qualified.

Remote release `ni26_friday_causal_rebuild_20261118/fc19_csp_gate_v2`. Producer commits `6b7c660c` and `5afe4d89`.

| Artifact | Job | SHA-256 |
|---|---|---|
| `source.sha256` | — | `0154ffdc8358f6a7f7865f855f70203400cda95ca5b7924478f57467a1fc8cd7` |
| `smoke.json` | 3133796 | `a1d3d3afcaf4204a667c3a372b6f059cf0352bf66ce737bec037ca83869bc680` |
| `within0.json` | 3133798 | `5b3601bb18ca911a49daac9b8b8bf27d0894f1bdc2fd60570de9f6ff09ffef71` |
| `within0.npz` | 3133798 | `950cda1f806543693ecd88f0029ca3e589ea44ff101a42bf493c47db2ac4559e` |
| `full0.json` | 3133797 | `c6e48aacfe076995f6f39e2acc5fe57086aa22177a217488d210e10f3f2b49bb` |
| `full0.npz` | 3133797 | `794fc90d70f419d9ffe1563a179442c900b5528318b333e1fcb1c57b02d41162` |
| `scoring.sha256` | — | `9bee3c72d88968706af0855a427a5213cf6f8092115b09ac217ba5d297634635` |
| `score.json` | 3133799 | `675c23bf75c75b25a793e8b96abf6bfce37673509dd1d7a82dc64baaeddc7707` |
