# Lane P3 · dsu_all vs hardcb(_safe) vs cap250 on one comparator (20/21 Nov 2026)

Pre-registered in docs/LEDGER.md (Lane P3) before scoring. Gap audit row 3. Nothing closed; not uploaded. Re-analysis of stored lane P2 per-window outputs (Galvani `ni26_laneP2_20261120`), script `scripts/overnight21/laneP3_compare.py`, numbers `laneP3_dsu_vs_hardcb.json`. Official-style cells (participant x session x interface, equal cells); paired 80 % intervals resample people (held-new) and people then orders (returning).

Instrument limits: held-new dsu_all is the UNGATED ds_all chain output (0 recognised held-new streams, so the gate is a no-op; assumption, not re-run); hardcb_safe = hardcb when no v6 error (checked below); returning = P04/P10 visits 2-3 only, 2 people, tape replay without state feedback; 57/43 and 20/80 are UNVERIFIED. Reproduces the stored numbers (+0.735 new vs cap250, -0.421/-0.423 returning vs hardcb).

## Per-cohort deltas (pp)
| | dsu_all - hardcb | dsu_all - cap250 | hardcb - cap250 |
|---|---|---|---|
| held-new, full0 | +0.386 [-0.205,+0.848] | +0.900 [+0.229,+1.425] | +0.514 [+0.333,+0.695] |
| held-new, within0 | +0.300 [-0.178,+0.821] | +0.570 [-0.029,+1.191] | +0.270 [+0.108,+0.432] |
| **held-new, both** (people 80 %) | **+0.343 [-0.074,+0.730]** | +0.735 [+0.274,+1.169] | +0.392 [+0.278,+0.520] |
| **returning, 8 orders** (people x orders 80 %) | **-0.422 [-0.57,-0.27]**, 7/8 orders < 0 | -0.488 [-0.68,-0.29] | -0.066 [-0.23,+0.09] |
| returning within (4 orders) | -0.654 [-0.85,-0.46] | -0.750 [-1.01,-0.52] | -0.096 [-0.20,0.00] |
| returning full (2) | -0.183 [-0.45,+0.15] | -0.254 | -0.071 |
| returning runs (2) | -0.198 [-0.53,+0.19] | -0.198 | 0.000 |

Returning dsu_all - hardcb: P04 -0.42, P10 -0.42 (20/80 weighting -0.423, UNVERIFIED). Neither person positive on any scheme except P04 full +0.02. Held-new per person dsu_all - hardcb: P02 -1.06, P03 -1.05, P05 -0.25, P10 -0.19 negative; P04 +1.34, P06 +0.83, P08 +1.32, P09 +1.56 positive; P01 0, P07 -0.01 (4 of 10 negative; the P04 sign differs between held-new and returning use). Per-cell returning deltas in the JSON (`returning_cells_all8`): 6 of 8 cells negative, worst P04 v2 Graz -1.30.
Mixed 57/43 (unverified): 0.57 x 0.343 + 0.43 x (-0.422) = +0.01 pp, i.e. a tie with hardcb; versus cap250 +0.42 - 0.21 = +0.21 and hardcb gives +0.22 - 0.03 = +0.20. The held-new gain over hardcb is undecided (interval contains 0), the returning loss is not.

## Changed predictions (dsu_all vs hardcb, mean per 95-window cell)
- Returning P04/P10 cells: 0.81 windows change (0.9 %); 0.19 fixed, 0.59 broken. Ceiling (every changed label that hardcb got wrong becomes right, class-recall weighted) **+0.23 pp**; floor -0.62 pp. Against lite5 alone (FBL2 effect only) 0.69 change, ceiling +0.20.
- Held-new cells (ungated): 4.4 / 4.1 change (full0/within0), 2.1 fixed vs 1.8 broken, ceiling +2.75 / +2.53, floor -1.88 / -1.85.
- Therefore, if the eight one-session returning people behave like P04/P10 (recognised, gate active), +1.54 pp is **not possible** for dsu_all: its changed-window ceiling there is about +0.23 pp. It could only occur if those sessions are not recognised, and then dsu_all acts as on held-new, where the measured gain over hardcb is +0.34 (ceiling +2.6, floor -1.9). +1.54 pp as an average over the eight would need nearly the best held-new person (P09 +1.56) from every one of them; only 1 of 10 held-new people reaches it. Podium needs +3.70 pp on the eight: above every measured per-person value.

## Dead-session patch (job 3144791, CPU; `laneP3_dead_check.log`)
Guard present in dsu_all.zip (SHA 03572497...4aed): **no** (unpatched `raise ValueError(...did not consume...)`). `patch_dead.py` gives `dsu_all_safe.zip`, SHA f4aa7e1a9a2afcced9c767b1c4115defd8cc00a8ae4dddcdca8187854e6abf0e (only submission.py differs; kept on Galvani `ni26_laneP2_20261120/zips/`, not committed). Zero-change: 379 windows (P01/1 190, P04/2 189) labels equal, max |step posterior diff| 0.0. Injected eye error at window 4: unpatched crashes (ValueError), patched finishes the session. Not rerun: RTX/A100 qualification, exact-package replay on the dev tape (code path unchanged without an error).
