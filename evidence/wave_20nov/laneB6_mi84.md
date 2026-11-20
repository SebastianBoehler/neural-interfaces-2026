# Lane B6 · ds007221 (OpenNeuro v1.0.3, CC0) and NETBCI (NEMAR nm000305 v1.0.1, CC BY 4.0) as extra own-head corpora in the Lane H arm-B trunk

Registered Thu 19 Nov 2026 ~23:30 CET on Leonardo (account AIFAC_P02_1675, release root `/leonardo_work/AIFAC_P02_1675/ni26_laneB6_20261120`), before any B6 posterior was scored.
Read before registering: Lane H card and result (A 59.31, B 59.81 standalone; B vs stock +0.04 full system), Lane S card, the staging smoke and a mu-band ERD check
(`scripts/mi84_erd.py`, Graz sub-01..04, 6 runs each; no PROTEUS label or decoder output involved).

## Data staged (Slurm, serial partition with internet; raw kept under `data/mi84_raw`, never committed)
- **ds007221g**: Graz group, sub-01..14, 2 scenes x 6 runs x 40 trials, left_hand / right_hand / feet / rest. Source `s3://openneuro.org/ds007221` (public, v1.0.3, CC0).
- **ds007221lr**: Hybrid (sub-37..73) + Hybrid-online (sub-74..84), only the `acq-graz` offline condition (left_hand / right_hand). The flicker (SSMVEP), video, SSVideo and online-feedback conditions
  and the whole SSMVEP-MI group (sub-15..36, action-observation trials) are NOT staged: stimulus-locked SSVEP / video drive is not motor imagery, and the audit rated them contamination, not coverage.
  So "MI84" here is 14 + 48 = 62 ds007221 people, not 84. Stated, not silent.
- **netbci**: sub-1..19, 4 sessions x 6 runs, right_hand vs rest (online cursor task; 250 Hz EDF from `data.nemar.org/nm000305/v1.0.1`, BIDS derivative of the Dataverse original).
- Export `scripts/export_mi84.py`: `export_external.preprocess` (notch at 50 / 60 Hz harmonics, 0.1-75 Hz zero-phase FIR at native rate, resample 120 Hz, microvolts) and `to_proteus`
  (our 41 channels, spherical splines for absent ones, no re-reference: same as `export_moabb.py` for the m7 corpora). Scalp channels are renamed to standard_1005 spelling; HEO, VEO, EKG, EMG, Trigger, CB1, CB2 dropped.
  Windows: one 4 s window per event. **Offset: 0 s after the marker for ds007221, 1 s for NETBCI** (rule of `export_moabb`: max(0, min(1, task length - 4))). The brief said "1 s after cue";
  the ERD check shows the ds007221 marker is the task onset (mu ERD from 0 to 4 s, rebound at 4-5 s, 8 s trial period = 4 s task + 4 s rest), so 1 s would cut the last task second and add one rebound second.
  NETBCI's target lasts 5 s (cursor feedback from 3 s), so 1 s as in the brief.
- Class maps: `experiments/class_map.OWN_HEAD` (name -> local index; feet 0, left_hand 1, rest 2, right_hand 3 for ds007221g; left_hand 0, right_hand 1 for lr; rest 0, right_hand 1 for netbci). `own_head_labels`
  raises on any other name; `events.tsv` value<->name pairing and trial counts (recordings.tsv) are asserted per recording; the pretrain wrapper asserts `classes == OWN_HEAD[corpus]` and every class present.
  These corpora never enter the shared 4-way block, so no label can reach a PROTEUS task index (the Shin 2018 loader bug cannot recur here); `experiments/test_class_map.py::test_own_head_map`.

## Arms (identical to Lane H arm B except the corpus list; 40-epoch SessionBN k20 fine-tune, seeds 0-2, all_loso P01-P10, eight pinned orders)
- A, B: Lane H banks reused unchanged (A public-only; B = hp7, 7 corpora, one head each, rot 0.05, 30 ep).
- **G** `hbg`: B + ds007221g. **N** `hbn`: B + netbci. **M** `hbm`: B + ds007221g + netbci. **X** `hbx`: B + ds007221g + ds007221lr + netbci (everything staged; **registered primary**).
Caveat stated now: `pretrain.py` draws batch // n_corpora rows per corpus, so adding 1-3 corpora lowers each older corpus's share by 1/8 to 3/10 (Lane S caveat, same). Returning people: not scored (held-new only; returning counted 0).
Code `laneB6/` (`pretrain_b6.py`, `lopo.py`, `collect_b6.py`, `paired.py`), launcher `scripts/cloud/leo_lane_b6.sh`; full system by `laneH/full_replay.py` on Galvani (full0, MASS 5/6 raw-family swap; secondary MASS 1/6).

## Predictions (balanced accuracy, held-new, 3-seed ensemble)
Standalone (paired, people+orders 80 %): X-B **+0.3 pp [-1.0, +1.5]**; M-B +0.2; G-B +0.2; N-B +0.1 (Lane H: B-A +0.5; unified Scherer+Shin hurt -1.8; here heads are own, so no label coupling).
Full cap250 (full0, paired people-only 80 %): **X vs stock +0.05 pp [-0.5, +0.6]**, X vs B in the same slot +0.05; B vs stock +0.04 is the Lane H measurement. Sealed-weighted claim = 0.57 x held-new delta.
Reason for the small prior: the fine-tune overwrites most of the trunk, and the new people share neither the PROTEUS montage, task mix nor session structure.

## Rules (stated before scoring)
- **Refuted for this form / dose**: X vs stock full-system (MASS 5/6, full0) paired 80 % upper bound <= 0. **Supported**: lower bound > 0 AND X-B standalone point > 0 AND X vs B in the slot point > 0.
  Anything else **undecided** (instrument +-0.5 across people; P04 / P10 reported, neither decides). M, G, N are reported on the same rules as isolating arms, not as selections.
- Controls: A vs stock within +-0.4 (Lane H: -0.15 at 5/6), B vs stock reproduces +0.04.
- Not covered: fb / fblight families, k other than 20, MASS other than 5/6 and 1/6, orders other than full0 in the full system, SSMVEP-MI / flicker / video conditions, other window offsets, returning people.

## Result (Fri 20 Nov 2026; Leonardo staging 64884xxx-64886974, pretrain 64887597, fits 64887599, collect 64916397; Galvani full0 consumer 3140348-53, 3140356-57, dir `ni26_laneB6_20261120`; files `evidence/wave_20nov/laneB6/`)
Staged: ds007221g 14 people / 6800 windows; ds007221lr 48 people / 7679 (Hybrid sub-37..73 and Hybrid-online sub-74..84, Graz condition); netbci 19 people / 14429; 81 people in all. Smoke (1-epoch pretrain + one fold) passed first.
Hybrid-online `events.tsv` lists each marker twice (80 rows for 40 trials); the exporter first failed loudly on the trial count, then collapses exact adjacent duplicates and still raises on anything else.
All guards passed (trunk keys, per-corpus class lists, banks complete, orders pinned). Stock/zero equality held for every arm.

| Standalone held-new BA (full0), 3 seeds | A 59.31 | B 59.81 | G 60.27 | N 60.08 | M 60.32 | **X 60.06** |
|---|---|---|---|---|---|---|
| vs B, people+orders 80 % | | | +0.49 [-0.15, +1.08] | +0.19 [-0.48, +0.80] | +0.16 [-0.49, +0.83] | **+0.28 [-0.40, +0.92]** |

| Full cap250, full0, paired people-only 80 % (P04 / P10 delta alone) | MASS 5/6 | MASS 1/6 |
|---|---|---|
| **X vs stock** | **+0.228 [+0.095, +0.373]** (P04 +0.01, P10 +0.34) | -0.077 [-0.180, +0.055] |
| X vs B | +0.190 [-0.004, +0.445] (P04 -0.35, P10 +0.16) | -0.037 [-0.149, +0.074] |
| M / G / N vs stock | +0.077 / -0.079 / -0.039 | -0.077 / -0.001 / +0.036 |
| controls: A vs stock, B vs stock | -0.151 [-0.368, +0.074], +0.038 | 0.000, -0.040 |

Reading against the rules above: **supported at MASS 5/6, full0** (X vs stock lower bound +0.095 > 0; standalone X-B +0.28 > 0; X vs B +0.19 > 0). Not refuted. Controls hold (A -0.15 within +-0.4; B +0.04 reproduces Lane H).
Reading aids, stated plainly: one order (full0) and one seed set; MASS 1/6 is slightly negative (undecided); people-only interval, instrument +-0.5 across people; the standalone intervals all cross zero; P04 under X-B is -0.35 and P10 +0.16, neither decides.
The gain does not appear in the single-corpus arms (G, N, M are at or below B in the system), so it comes from the full 81-person mix or is noise; M-to-X (+0.15 [0.00, +0.34]) is the only step that isolates the 48 Hybrid people.
Sealed-weighted claim = 0.57 x 0.228 = about +0.13 pp (returning people not scored, counted 0). Predictions: standalone X-B +0.3 (seen +0.28); full-system X vs stock +0.05 (seen +0.23, optimistic side for the data).
Not run: other orders in the full system, more seeds, returning people, fb / fblight families, SSMVEP-MI / flicker / video conditions, other window offsets, a candidate ZIP.
