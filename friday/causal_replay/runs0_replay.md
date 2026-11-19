# Shared-source held-new runs0

D-EArcf is preregistered in `ced2652f`. In both EA conventions it supplies the actual runs0 waveform replay from the same ten source-excluded FBLight and Sinc checkpoints used for D-EAwcf. The matched systems each use three Eye and three Rest members, fixed recording calibration, and four independently regenerated seed-0 refits. The exact original archive control uses its owning runs0 inventory of nine Eye and three Rest. Within0's original archive used nine Eye and nine Rest. Compare the paired arms inside runs0. Cross-instrument differences include different original auxiliary inventories and historical anchors.

The real RTX P01 frozen smoke `3133373` completed at 02:52:29 CET. All 190 archived decisions matched, with 4.17161e−7 maximum posterior difference and exact aligned first window, truncated prefix and interleaved reset. Its source state hash `2aa00a45…b223` is the same state used in within0. The scheduler waits on each successful independent predecessor.

| Galvani jobs | Proof and immutable release |
| --- | --- |
| `3133373 → 3133374 → 3133375` | P01 smoke, other nine RTX folds and complete FBLight bank, `runs0_fbl_v1`, source `a433c3fd…a528`. |
| `3133395` | Original archive control and three exact count/CSP prefixes, `runs0_zero_v4`, source `973c312d…c360`. |
| `3133380` | Pinned common Sinc and FBLight, recognition and both strict-past target banks, `runs0_shared_bridge_v1`, source `a52f2835…8d41`. |
| `3133397→98`, `3133399→3400`, `3133401→02`, `3133403→04` | Genuine P01 prefix smoke then full FB raw, FB EA, FBLight and Sinc seed-0 refits. `runs0_matched_refit_v1`, source `2163dd1e…4d3a`. |
| `3133405 → 3133406` | Complete blind predictions then separate authorized gold score, `runs0_matched_terminal_v1`, source `76e56764…54d4`. |

The first exact-zero job `3133376` stopped at a missing script dependency. The second `3133381` got through the original incumbent and raw cap250 identity, then stopped at the inherited nine-Rest assertion. The owning §430 producer `ni26_aux430_20261118v2/run.py` sets three Rest members for runs0, and its job `3125008_6` records exact zero Rest change. A stale staging submission `3133385` was canceled. The queued v4 replay verifies this source inventory before comparisons. These earlier attempts produced no held result.

```bash
ssh mlcloud-galvani 'sacct -X -j 3133373,3133374,3133375,3133380,3133395,3133397,3133398,3133399,3133400,3133401,3133402,3133403,3133404,3133405,3133406 -o JobID,State,Elapsed,Start,End,NodeList'
ssh mlcloud-galvani 'cd /mnt/lustre/work/gehler/gfh527/ni26_friday_causal_rebuild_20261118 && find runs0_matched_terminal_v1 -maxdepth 1 -name "matched_runs0*" -exec sha256sum {} +'
```
