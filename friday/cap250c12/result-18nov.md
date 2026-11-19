# F-C12 directional Euclidean alignment pilot

F-C12 passed its single-session engineering smoke. The exact cap250 worker used both active covariance histories on Scherer G visit 1, order 411. Each changed the same one prediction at age 89. The changes reached the live model, and the choice between directional and scalar clipping left this session's decisions identical. The run used no target labels.

## What ran

Galvani job `3131628` completed at 22:01:51 CET on 18 November, with 7m18s allocated wall time on one A100 PCIe 40 GB. It authenticated the original ZIP, the extracted members, the official evaluator, Scherer inputs and the previously sealed stock row map and work counters. The historical replay printed `OK`. The source and the checksum-pinned frozen package stayed in separate mounts.

The current 41×480 waveform and its stock covariance went through the original input stream first. The adapter changed only the covariance retained for later windows in the 32-member Euclidean alignment group. The fit prior's minimum-to-maximum eigenvalue ratio was 0.00035787, above the registered 1e−10 requirement. The maximum uncapped eigenvalue in the fit-prior basis was 45.35. Both active arms marked the same three updates at ages 48, 86 and 90. Their retained raw traces matched within 1.58e−16 relative error.

## End-to-end checks

| Arm | Later aligned inputs changed vs stock | Maximum terminal probability change vs stock | Decision changes | Warm time / stock, passes 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Zero | 0 | 0.00004952 | 0 | 0.979× / 1.033× |
| Directional | 71 | 0.22539 | 1 | 0.971× / 1.026× |
| Scalar | 71 | 0.11489 | 1 | 1.001× / 1.010× |

Both active arms passed the 1.05× median and 1.10× maximum warm runtime limits. The raw and terminal posterior vectors showed small drift between independent stock and zero workers, peaking at 1.52e−5 and 4.95e−5 respectively. Their actual aligned inputs matched bit for bit, and both reproduced all 120 archived decisions and all counters. These controls distinguish the observed probability movement from independent-worker replay drift. Both active arms preserved first-window outputs and all labels and aligned inputs in repeated independent 31-window prefixes after replaying a prefix of visit 2. All full passes ran ten neural refits, 200 Adam steps and five covariance personalization refits, with zero route recognitions or errors.

The one-session result measures dependency and runtime on A100. Scored external evidence and the registered warm RTX proxy check are still pending.

The release is `/mnt/lustre/work/gehler/gfh527/ni26_friday_c12_directional_ea_20261118/`. The source manifest SHA-256 is `097cb68fa55db2f87cefacac54afa2a2d32f264ce3dc7f543b65ce0de46e04da`. The `ea_smoke.json` receipt SHA-256 is `d0da4bcaba5ad6acb1ac283cf530ee7cf6513bab0603d83336f8769d43c51d52`. An independent reader verified the source binding, both cap masks, equal retained traces and the exact aligned-input and label controls. [Preregistration](../../docs/LEDGER.md#f-c12--limit-directional-influence-in-the-live-euclidean-alignment-history)
