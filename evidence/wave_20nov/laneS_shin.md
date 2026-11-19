# Lane S · Shin 2017 (MI / SUB) and Shin 2018 (WORD) as correctly mapped training people for the raw FBCNet SessionBN member

Registered Thu 19 Nov 2026 ~21:45 CET, before any lane S posterior was scored. Plan source: `evidence/audit_20nov/same_task_people_codex.md` (Friday plan).

## Loader bug (audit bugs_datapath 3) and fix
`protocol.load` copied the file's `y` and dropped `classes`: shin2018 (`baseline=0, WG=1`) entered every fold as MI / SUB (1560/1560/0). Fix: `experiments/class_map.py`
maps class *names* to task indices (MI 0, SUB 1, WORD 2), `protocol.load` uses it for every external row, drops rest/baseline rows (AUX, never a task class)
and raises on any unmapped or unexpected name or a classless file without a pinned entry (only scherer2015 is pinned). `experiments/test_class_map.py` reproduces the
old counts [1560,1560,0] and passes with the map (shin2018 -> [0,0,1560], scherer [710,710,710], unknown name raises). Direct effect on cap250: 0 (not in the package).
The Lane H pooled pretrain (`laneH/pretrain_h.py`) already mapped by name; its dict lookup raised on unknown names. Lane S `pretrain_s.py` additionally asserts the per-corpus name set.

## Arms (same net, SessionBN k 20, 40 epochs, seeds 0-2, all_loso P01-P10, eight pinned orders, as Lane H; only the trunk differs)
- A, B: Lane H banks reused unchanged (A public-only; B = hp7, 7 corpora, one head each; shin2017a/b there have their own two-class heads).
- **P** `hsp`: B trunk recipe, but shin2017a (left/right -> MI 0) and shin2017b (subtraction -> SUB 1, rest -> aux 3) share the 4-way block.
- **W** `hsw`: B recipe + shin2018 (WG -> WORD 2, baseline -> aux 3) in the block; shin2017 as in B (own heads).
- **D** `hsd`: B recipe + shin2017a/b + shin2018 all in the block (combined Shin arm; the registered primary).
Caveats stated now: pretrain draws batch//n_corpora rows per corpus, so W/D lower each old corpus share by 1/8; Shin cohorts carry 35 reconstructed of 41 inputs
(audit montage risk); shin2017 uses both crops of a trial, shin2018 the first window (as Lane H). Returning people not scored (held-new only; returning counted 0).

## Predictions (balanced accuracy, held-new, 3-seed ensemble)
Standalone (people+orders 80 %): D-B **+0.5 pp [-1.0, +1.5]**; P-B +0.3, W-B +0.2 (Lane H B-A measured +0.5 with Scherer/Shin-free trunk; unified Scherer+Shin hurt -1.8).
Full cap250 (laneH/full_replay.py, full0, MASS 5/6 raw-family swap; secondary MASS 1/6), paired people-only 80 %: **D vs stock +0.15 pp [-0.6, +0.8]**;
P +0.10, W +0.05. Sealed-weighted claim = 0.57 x held-new delta (returning counted 0), i.e. +0.09 for D.

## Rules (stated before scoring)
- **Refuted for this form**: D vs stock full-system (MASS 5/6, full0) 80 % upper bound <= 0. **Supported**: lower bound > 0 and D-B standalone point > 0 and D vs the matched
  B in the same slot point > 0. Anything else **undecided** (instrument +-0.5 across people; P04 and P10 reported, neither decides). Interval crossing zero = direction undecided.
- Controls to read with it: A vs stock within +-0.4 (Lane H: A -0.15 at 5/6), B vs stock (+0.04), D-B and P/W-B isolate the Shin people from the Lane H trunk.
- Not covered: fb/fblight families, k other than 20, other MASS, orders other than full0 in the full system, Ajra, late-minus-early, source-label permutation, returning people.

## Result (Thu 19 Nov ~22:35 CET; Leonardo 64882940/41 smoke, 64883272 pretrain x9 (8 min each), 64883273 fits (99 banks per new arm set, all guards passed); Galvani consumers 3138891-3138900, dir `ni26_laneS_20261120`; code `laneS/`, commit see git log; evidence `evidence/wave_20nov/laneS/`)
A and B are the Lane H banks reused unchanged (stock cross-check 73.739 equals cap250 `c25d`, cap250_checked true in all ten runs). Held-new P01-P10, 3-seed ensemble, full0.
Standalone BA (full0): A 59.31, B 59.81, P 59.40, W 59.49, D 58.93. Paired people+orders 80 %: D-B **-0.64 [-1.71, +0.22]**, P-B -0.42 [-1.55, +0.39], W-B -0.61 [-1.51, +0.15], D-A -0.13 [-0.97, +0.81].

| Full cap250, paired people-only 80 % | MASS 5/6 | MASS 1/6 |
|---|---|---|
| D vs stock (primary) | **-0.190 [-0.326, -0.038]** (P04 -0.54, P10 -0.17) | -0.040 [-0.156, +0.071] |
| P vs stock | -0.078 [-0.218, +0.033] | -0.229 [-0.349, -0.108] |
| W vs stock | +0.072 [-0.133, +0.335] | -0.040 [-0.135, +0.074] |
| D vs B (matched slot) | -0.228 [-0.433, +0.042] | 0.000 [-0.177, +0.134] |
| P vs B / W vs B | -0.116 [-0.303, +0.109] / +0.034 [-0.227, +0.391] | -0.190 [-0.312, -0.087] / 0.000 [-0.149, +0.150] |
| controls: A vs stock / B vs stock | -0.151 / +0.038 | 0.000 / -0.040 |

Reading against the rules: D vs stock upper 80 % bound -0.038 <= 0 at MASS 5/6, so **refuted for this form** (correctly mapped Shin 2017 + 2018 in the m7r pretrain block, raw FBCNet SessionBN family, full0, held-new). Standalone D-B is also negative.
Sealed-weighted claim 0.57 x (-0.19) = -0.11 pp; do not stack. W alone (Shin 2018 WORD) is +0.07 [-0.13, +0.34] at 5/6 and 0 at 1/6: undecided, below the +0.05 planned size; P (Shin 2017 pair) is negative at 1/6. Prediction was too optimistic (+0.15 predicted, -0.19 seen; standalone +0.5 predicted, -0.64 seen).
Not run: other orders in the full system, returning people, fb/fblight, late-minus-early, permutation and montage controls (not needed once the main contrast is refuted), Ajra.
