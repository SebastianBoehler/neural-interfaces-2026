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
