# Lane P · combined candidates of every positive component (registered Fri 20 Nov 2026 ~11:15 CET, before any arm below was scored)

Why: Force_Best makes the last upload a free option (audit_20nov row 6): choose by expected sealed gain that fits 3,600 s, not by interval thresholds.
Everything is against exact cap250 and against hardcb (final stack), full current system, shuffled delivery, ten reused held-new people (instrument: corrected
Friday chain, orders full0 and within0 only; paired people-resampled 80 % intervals, 10,000 draws, as `score_b.py`). Code `laneP/` (chain copy of the B1 chain,
arm set P), release `ni26_laneP_20261120` on Galvani (chain replays exist only there).

## 1. Table of positive signals (point estimate > 0 in the full system; numbers from the cards named)
| component | slot | pp (80 %) | instrument, orders | runtime | conflicts / status |
|---|---|---|---|---|---|
| hard (F-C25 hard pair) | 5 outer raw FBC members, MASS 5/6 | vs cap250 +0.318 [+0.178,+0.469] (full0 +0.331, within0 +0.305) | chain, 2 orders; NI-V -0.047; returning -0.020; pooled external +0.050 | x1.06 warm | raw slot |
| CBraMod-C 3 seeds | 0.2 of raw proposal (hardcb) | alone +0.130 [0.000,+0.302]; hardcb - hard +0.074 | chain, 2 orders; returning +0.050; NI-V +0.107 B-ret | +13 % | raw slot |
| **hardcb (final stack)** | raw slot 0.8 hard + 0.2 cb | **+0.392 [+0.280,+0.519]** (full0 +0.514, within0 +0.270) | chain, 2 orders; returning +0.011; NI-V +0.015 | 3,211 s (replay ratio 1.257 RTX); unrecognised-only paired ratio 1.200 RTX / 1.239 A100 (B1 timing pairs) | base of every arm below |
| TTT lite5 (B1 T set) | FT streams of 4 families | vs hardcb +0.074 [-0.289,+0.480] (full0 +0.187, within0 -0.039) | chain, 2 orders; returning not separately scored | ratio 1.182 RTX / 1.233 A100 (same as stock FT) | FT slot; fits |
| TTT light / heavy (lane T) | FT streams | +0.079 / +0.343 vs hardcb | chain, 2 orders | light = stock; heavy est. 4,600-5,400 s | heavy does not fit; light dominated by lite5 form |
| whole-window / dispersion correction (lane K test 2) | after RouteB on the raw slot output | +0.227 [-0.008,+0.430] / +0.334 [-0.205,+0.773] vs hardcb | chain full0 only; stand-in member (N9 raw FBCNet seed 0, not a package member); history crossed sessions | < 2 ms/window (untimed) | post-hoc control reading for whole; needs a packaged segment export |
| FBL2 (doubled FBLight) | FBLight member share 1/16 -> 1/17 | vs stock +0.383 / +0.488 (within0 / full0); hard+FBL2 vs stock +0.574 / +0.553, i.e. +0.27 / +0.22 over hard | chain, 2 orders; returning -0.130 equal cells, -0.208 at 20/80; external hard+FBL2 -0.579, FBL2 Scherer -0.020 | x1.014 (F-C21e) | FBLight family shares the FT stream with lite5; external negative |
| X raw family (B6) | raw slot | alone +0.228 [+0.095,+0.373] vs stock; **in the slot with hardcb: hcX -0.064 [-0.286,+0.130], hXc -0.001** (B1 X set) | chain full0 | x1 | shares the raw slot with hard: NOT positive with hardcb, not combined |
| sparse20, n5 subsets, alpha-peak, TTT sparse | | -0.227 / -0.218 / -0.109 | | | non-positive, dropped |

## 2. Combinations (fixed now) and the one combination rule
Rule: slot proposal stays 0.8 hard + 0.2 cb at MASS 5/6 (hcX forms are not combined: negative point). Each component acts at its own site, applied once, in this order:
FT family banks swapped (lite5) -> FBL2 doubling of the live FBLight member -> raw slot replacement -> segment correction on the raw slot output (w 0.10, clip 5,
prior 20, start at 8 per class, frozen stock-raw argmax pseudo-labels, constants unchanged from lane K) -> count/CSP tail. **Correction history resets at every session start**
(the package cannot carry state across sessions; lane K's person-level history crossed sessions and is kept only as control `wp`/`dp`). No weights are tuned.
| id | arm | content |
|---|---|---|
| P1 | `ftlite5` | hardcb + lite5 (re-scored here, must equal B1 T5 bitwise) |
| P2 | `ws` | hardcb + whole-window correction, per-session history |
| P3 | `ds` | hardcb + dispersion correction, per-session history |
| P4 | `ds_l5` | hardcb + lite5 + dispersion correction |
| P5 | `ds_all` | hardcb + lite5 + FBL2 + dispersion correction (all compatible positives) |
Secondary (reported, not candidates): `ws_l5`, `hcF` (hardcb + FBL2), `hcF_l5`, controls `ftctl` (zero-change, equals hardcb terminal bitwise), `wp`/`dp` (lane K form, must reproduce full0 74.48/74.59 within 0.02),
`stock`, `identity` (bitwise equal to stock), `hard`, `hardcb` (reproduce the 19 Nov receipt: full0 +0.514, hard +0.331).
The segment export is rerun for both orders from one fit per person (`laneP/seg_export.py`, Galvani 3141400); stand-in limit stated: it is not a package member.

## 3. Predictions (pp, held-new, vs hardcb, mean of both orders; 80 % half-width +-0.7 per forecast_calibration rule 2)
P1 +0.07 (measured B1, not a prediction); P2 +0.10 [-0.4,+0.6] (lane K person-history +0.23 full0; per-session reset starts later: haircut); P3 +0.15 [-0.4,+0.7];
P4 +0.20 [-0.5,+0.9]; P5 +0.30 [-0.6,+1.2]. Interactions (joint minus sum of parts, vs hardcb): P4 0.00 [-0.3,+0.3]; P5 (three-way) -0.05 [-0.4,+0.3]
(F-C22 saw a negative interaction between two edits sharing a slot; here the sites differ). Controls pass as stated above. FBL2 (hcF) +0.20 [-0.4,+0.8] vs hardcb.
Returning P04/P10: corrections and FBL2 not scored (counted unknown; FBL2 is -0.2 at 20/80 in its own card); NI-V: not run for corrections (no package).

## 4. Ranking and upload rule (fixed now)
Expected sealed gain over cap250, shrunk as forecast_calibration rules 2-3: S(c) = 0.5 x [0.57 x dnew(c) + 0.43 x dret(c)], dnew = both-order mean of c minus stock; dret = returning
P04/P10 w20/80 where measured (hardcb +0.011, lite5 -0.040 on top), else 0 (unknown, not positive). Intervals report spread, they do not gate. Eligibility (all required): projected
runtime <= 3,300 s (paired warm ratio x 2,555 s, B1 timing; replay-based 3,211 s shown as the conservative figure), a buildable and parity-checkable package tonight, no component
with an unresolved veto (NI-V B-new/B-ret upper bound < 0 or Scherer-free worst person < -1 pp). A combination that needs a component without a package (the segment correction needs a
packaged raw member exporting segment contributions) is ranked but flagged NOT PACKAGEABLE TONIGHT; the upload recommendation is the best packageable arm. Ties within 0.03 pp go to
fewer components. Interaction is reported next to each joint number; a negative interaction lowers S (joint, not sum, is used).
