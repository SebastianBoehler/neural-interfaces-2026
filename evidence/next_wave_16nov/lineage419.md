# §419 · Public development comparison of v5.1b and cap250

The historical fold models favor cap250 on both cohorts. The exact ZIPs were fitted on the public data, including the evaluation people and sessions. Running those ZIPs on PROTEUS would measure in-sample performance. The strongest public comparison available pairs the historical fold outputs below. The strongest literal ZIP comparison measuring the competition classes on external people remains [Scherer](lunch_qualification385.md).

| Historical full-system development comparison | v5.1b (%) | cap250 form (%) | Paired change (pp) | 80% people/order interval |
|---|---:|---:|---:|---:|
| New, official cells · ten people, 14 sessions, 2,655 windows, 28 cells | 74.724 | 76.279 | +1.556 | +1.009 to +2.071 |
| New, equal people | 78.178 | 79.642 | +1.463 | +0.935 to +2.000 |
| Returning, official cells · two people, four sessions, 762 windows, eight cells | 79.882 | 81.127 | +1.245 | +0.643 to +1.862 |
| Returning P04 | 69.989 | 71.798 | +1.808 | |
| Returning P10 | 89.774 | 90.456 | +0.682 | |
| Returning, assumed 20% P04 / 80% P10 | 85.817 | 86.724 | +0.907 | +0.598 to +1.862 |

Nine of ten new people improved, and all eight new-person orders favored cap250. The two-person intervals describe only variation among these two people and orders. Combining the official new-person effect with the assumed returning mixture at 56.6% new gives **+1.274 pp**. This sensitivity calculation has no independent private-cohort calibration. Public and private people, private delivery pattern, recognition rate and length distribution can differ. The 250-window limit never binds in these public sessions (maximum 199 windows). The +0.272 pp JS result compares a separate experimental system to this same 76.279% control.

## Which calculations support the comparison

- **New.** B2_pool uses all six main seeds. The old output is the fully staged P0r inside the authenticated §318 control input. It agrees bitwise with the earlier §257 reference, as do V6Fr and V6r. The candidate output is the authenticated §318 control **after** the detector-gated count filter and capped causal personalization. All eight final controls agree bitwise with the separate zero318 outputs. The §201 route receipts cover 14 sessions on each order with zero routed windows. The original §318 archive checksum manifest is pinned; the existing scorer had already verified its full 128-file identity against the original store. [§318 qualification](coral_fbl_s318.md)
- **Returning.** Historical seen_session keeps session 1 of P04 or P10 and holds out that person's sessions 2 and 3. The old output is the routed base of retv6. The candidate is §245's T arm with the personal TSEM prior (strength 20), eye/rest branch and the held-person detector-gated count filter. Its intermediate ER probabilities equal those from the old run bitwise on every order. Disabling every latch reproduced unfiltered outputs bitwise. The study uses the old shadow before recognition. The ZIP runs the separate RouteB tail and consumes the inner gate decisions. CP stops at recognition. The historical reconstruction does not provide complete prefix or gate parity with either archive. The §321 returning bridge saved base, disabled-JS and JS predictions, without the v5.1b inner predictions needed for a second direct pair. Its shared production coefficients, auxiliary nets and old gate calibration also limit held-session claims. [§321 evidence](js321_returning.md)
- **Deployment differences.** The historical shuffled eye and rest branches average nine seed streams in the recorded runs; the staged ZIP includes three eye and three rest nets. Historical new-person logs also retain recording-order transformer and shallow EA slots. Pinning an order verifies which simulation ran; archive-wide equivalence has not been established. Public folds informed development and model selection. These differences preclude a literal release-gain estimate.
- **Clock and switches.** In the staged weights, FT starts at 20 with spacing 10, uses five steps, seed 0, old-chain labels and r15 relabeling. Route mode is 1; refits stop on recognition. The window cap is 250. The per-session time budget is 1,000,000,000 seconds, effectively disabled. No historical process-wide timeout is part of this comparison. Count filter, personal prior and CP are enabled. CP refits on a 20-window schedule and begins fusion at window 90. The v5.1b archive has none of those v6 switches.
- **Weighting.** Each scheme (within/full/runs) gets one third. Within uses four orders; full and runs use two each. The paired bootstrap resamples people and orders within schemes 4,000 times, seed 0. Official results weight each person by their number of official cells. The two returning people also appear as new-person folds under separate training exclusions. The mixture does not supply independent replication. §249 used r15 raw outputs, different order counts, six bases and an approximate baseline. The table recomputes both levels from paired arrays.

## Reproduction

The tracked [scorer](../../studies/18nov/lineage419/score.py) reads historical predictions only. Stage the following from Galvani root `/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117` into local `runs/lineage419/`, preserving each basename. Use `runs/coral304_inputs/control-original.sha256`, all eight `runs/chainout/s281_v66p_control_<order>.npz`, plus `runs/s281_v66_cp/{control,zero318}_<order>_B2_pool.npz`. The scorer also requires the original local §245, §257 and §281 stores. Run from the repository root.

```sh
.venv/bin/python replay/check.py > runs/lineage419/replay.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python studies/18nov/lineage419/score.py
```

The [result and input hashes](lineage419_result.json) record both score matrices summarized by person and order. Preregistration revision `e43a107`. Archive SHA-256 values are `4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef` (v5.1b) and `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` (cap250). Their respective in-archive submission.py SHA-256 values are `0259d1eabaf747ac85b178f01832e5f37ea25bddebbebb865d04ec321f498534` and `37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f`. Both predictions remained positive, satisfying the registered directional condition.
