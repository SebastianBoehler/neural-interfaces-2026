# Wave 10 (3 Nov): the teacher's data, two untested directions

cand_nxif stays the lead (docs/OVERNIGHT_2NOV.md; audit recipe docs/AUDIT_NXIF.md). Waves 5–9 closed
the teacher's architecture, recipe, C / w, seeds, fusion, map edits and arithmetic heads. Two rows of
the allowed space (docs/REVIEW_2NOV.md §4, §7) are still untested. Both follow the independent-errors
account: raise the teacher's accuracy on PROTEUS without giving it PROTEUS labels.

1. **10u, in-domain adaptation without labels.** The teacher trunk never sees PROTEUS. `pretrain.py
   --uda LAM` adds a quarter batch of unlabelled PROTEUS windows (dev participants, three window starts,
   per-session EA; never P04 / P09) to every batch. A gradient-reversal adversary (`dg.Adversary`) then
   makes the trunk features invariant between corpora and PROTEUS; BatchNorm statistics also see PROTEUS.
2. **10m, more MI brains.** MOABB 1.7.2 has many MI corpora that did not exist at R10 (Stieger2021,
   Yang2025, Wang2026, Zhou2020, Chang2025, ...). Only licence-clean ones (CC0 / CC BY / ODC-By, healthy
   participants) are used, checked on the data host.

`scripts/export_moabb.py` now places windows by the task interval's length. Every corpus used so far
starts at 0 s, so their exports are unchanged; Yang2025, Weibo2014 and BNCI2014_001 start later.

## 10u (`runs/diag/wave10u.sh`, jobs 3044759–62, screen 3044763)

EEGNeX (2080-galvani, as the originals) and IFNet, m7r recipe, seeds 0–5, `--uda` 0.1 (m7ru1) and 0.5
(m7ru5). Smoke test 3044755: both paths run; `--uda 0` is the old code path.

**Transductive by design:** the unlabelled pool holds all 8 dev participants, so each LOSO fold's
held-out person was seen without labels. That makes this an upper bound for unseen people.

**Pre-registered:**
- A UDA pair must beat the base m7r pair by ≥ +0.5 sel on s05, with both seed halves (s02, s35) above
  the base at the same C / w.
- Then the honest version: cross-fitted trunks (pool = G1 or G2 only; each group scored with the
  trunks that did not see it) must keep ≥ +0.3.
- Then pretraining seeds 6–11.
- Before staging: the mean without P10 and the worst participant.
- If the upper bound fails, the lever is closed.

**Result (screen 3044763; all 24 trunks completed, EEGNeX 33–35 min, IFNet 9–10 min per seed).** sel Δ
over R19e, C 0.01:

| pair | s05 w3 | w4 | w5 | w6 | w7 | best s05 | vs base best |
|---|---|---|---|---|---|---|---|
| m7r (base) | +2.11 | +2.35 | **+2.65** | +2.33 | +2.24 | +2.65 | — |
| m7ru1 (λ 0.1) | +2.21 | +2.67 | +2.75 | +2.29 | +2.31 | +2.75 | +0.10 |
| m7ru5 (λ 0.5) | +2.27 | +2.87 | +2.56 | +2.22 | +1.95 | +2.87 | +0.22 |
| EEGNeX m7ru1 + IFNet m7r | +2.09 | +2.73 | +2.54 | +2.20 | +2.49 | +2.73 | +0.08 |
| EEGNeX m7r + IFNet m7ru1 | +2.12 | +2.44 | +2.55 | +2.51 | +2.02 | +2.55 | −0.10 |

The convention since wave 5 compares each pair's best with the base's best (+2.65 at its production
w 5). On that reading, no pair reaches +0.5. The largest gap is m7ru5 at +0.22. Read strictly per w,
one cell crosses the bar: m7ru5 at w 4 is +0.52 over the base at w 4, with halves +0.61 / +0.27. But
that is the maximum of 20 cells, and at the base's production w 5 the same pair is below the base
(−0.09). The teacher-alone dla barely moves (base 77.75 at w 5; UDA pairs 77.4–77.9).

**Verdict: closed.** Even the transductive upper bound, which saw every dev participant's unlabelled
windows, adds no more than seed noise. Making the trunk invariant between corpora and PROTEUS, with
BatchNorm statistics that include PROTEUS, does not make the teacher's errors more independent of the
members'. This fits wave 9's profiling: the teacher's per-person bias lies on the MI-vs-cognitive axis
and rebalancing already absorbs it. The cross-fitted honest version is not run.

## 10m (`runs/diag/wave10m_export.sh`, then `wave10m_train.sh`)

**Licences** (subagent, read on the data hosts; we confirmed every one ourselves via the
figshare / Zenodo / Dataverse APIs):

| Corpus | Licence | People | Note |
|---|---|---:|---|
| Stieger2021 | CC BY 4.0 | 62 | 377 GB in all: sessions 1–3 only |
| Yang2025 | CC BY 4.0 | 51 | one 65.6 GB archive, 3 sessions |
| Zhou2020 | CC BY 4.0 | 20 | 7 sessions; 26–41 channels |
| Chang2025 | CC BY 4.0 | 28 | 4 sessions |
| GuttmannFlury2025_MI | CC0 | 31 | 3 sessions |
| Kumar2024 | CC BY 4.0 | 18 | 6 sessions, 22 channels |
| Weibo2014 | CC0 | 10 | 5 single-limb classes kept |
| Rozado2015 | CC0 | 30 | left hand vs rest |

Excluded:
- Wang2026: CC BY-NC.
- BNCI2014_001: CC BY-ND.
- Brandl2020: CC BY-NC-ND.
- HefmiIch2025: patients and a no-commercial clause.
- Forenzo2023: CC BY, but 94 GB of 60 s cursor-pursuit trials, a different paradigm.

**Jobs:** exports 3044839–45 (one CPU job per corpus) and 3044846 (Stieger, 31 tasks).

**Exports:**
- Rozado2015 dropped: MOABB needs a RAR extractor (unrar / unar / 7z with RAR), and Galvani has none.
  `7za` lacks the codec, and conda-forge `unrar` is only a Python binding.
- Zhou2020: MOABB names the 41 channels of subjects 1–12 `EEG1…EEG41`, with no positions (the
  repackaged files carry no channel names). Only subjects 13–20 (26 named channels) are exported
  (3045020).
- Stieger2021: about 700 windows per participant over sessions 1–3.

Exported windows (41 channels × 480 samples):

| Corpus | Windows | People | Classes |
|---|---:|---:|---:|
| Stieger2021 | 40,793 | 62 | 4 |
| Yang2025 | 30,591 | 51 | 2 |
| Zhou2020 (13–20) | 13,335 | 8 | 4 |
| Kumar2024 | 7,156 | 18 | 2 |
| Weibo2014 | 3,950 | 10 | 5 |
| Chang2025 | 2,670 | 22 | 3 |
| GuttmannFlury2025_MI | 2,520 | 31 | 2 |

Chang2025 has 22 of 28 people, because 6 are missing upstream. Yang2025 (3044839) took 3 h 6 min, and
pretraining 3045827–30 waited on it (`afterok`). The screen is job 3045831. In all, m13r adds 60,222
windows from 140 people to m7r, and m14r adds 101,015 from 202.

**Pretraining:** EEGNeX + IFNet, m7r recipe, seeds 0–5, two variants:
- m13r = the 7 m7r corpora + the 6 smaller new ones;
- m14r = m13r + Stieger2021.

**Screen:** against the base m7r pair, scored three ways:
- C 0.01 with all heads;
- C 0.003 with all heads;
- C 0.01 with the 19 original heads only (`NI26_NCOL=19`: the trunk learns from more brains while
  the map keeps its features).

**Pre-registered:**
- A variant must beat the base pair by ≥ +0.5 sel on s05, with both seed halves above the base at the
  same C / w.
- Six comparisons run, so a pass must then hold ≥ +0.3 on pretraining seeds 6–11.
- Also reported:
  - the teacher-alone dla;
  - P(right | R19e wrong), the independence the gain depends on.
- Before staging:
  - the mean without P10 and the worst participant;
  - the teacher exporter must support the chosen head set.

**Result (screen 3045831; all 24 trunks completed).** Each m13r EEGNeX seed took 50 min, m14r 63 min,
IFNet 13 / 34 min. Best s05 sel Δ over R19e, w 3–7, vs the base pair's +2.65:

| pair | C 0.01 | C 0.003 | C 0.01, 19 m7 heads |
|---|---|---|---|
| m7r (base) | +2.65 | +2.63 | +2.65 |
| m13r | +2.25 | +2.57 | +2.42 |
| m14r | +1.80 | +2.07 | +2.69 |
| EEGNeX m14r alone | +1.91 | +1.97 | +1.61 |
| IFNet m14r alone | +1.39 | +1.70 | +2.33 |

No variant reaches +0.5; the best is +0.04 (m14r with the 19 m7 heads). The single-net rows have no
m7r single-net control in this screen, so they only show that neither net carries the pair. The per-family profile
(`screenW10m_heads.log`, s05) shows why:

| trunk | teacher-alone dla | P(right \| R19e wrong) |
|---|---|---|
| EEGNeX m7r / m13r / m14r | 51.4 / 56.9 / 56.7 | 0.50 / 0.47 / 0.45 |
| IFNet m7r / m13r / m14r | 54.8 / 58.7 / 60.0 | 0.48 / 0.51 / 0.48 |

**Verdict: closed.** More MI brains make the teacher better alone, by 5 points of dla. But its errors do
not become more independent of the members'. For EEGNeX they become less independent (0.50 → 0.45), and
the gain in the ensemble disappears. Extra heads at C 0.01 also dilute the map. This is the clearest
evidence yet for the independent-errors account: accuracy alone is not the lever.

## Explorer arm staged: `runs/sealed/cand_nxif_emx`

Wave 9's explorer core (docs/OVERNIGHT_2NOV.md, round 3) is now a data-only package. It is cand_nxif
with `rb_alpha` 1.15, `em_shrink` 0.85 and `em_n0` 40 (`scripts/set_params.py`, job 3044931). Other
files are unchanged, and `submission.py` is still `25d51a6e…`.

- Harness agreement 1.0 for both the source and the new parameters. 45 of 2,655 labels differ from
  cand_nxif.
- `weights.npz` `e7b85706…`, ZIP `98d39600…`.
- Official A100 replay (3044932) 118 s and worker-image replica (3044933) 72 s. Files are unchanged
  and both give the same scores.
- In-sample on the official test windows: 0.9488 vs cand_nxif's 0.9646. These are trained-on
  participants, so this only checks agreement, not generalisation.
- Dev evidence (from wave 9's hidden check): s011 sel +2.85 vs +2.91, dla +3.92 vs +3.18, ds +1.77 vs
  +2.63. It is better for new people and worse for returning ones, about neutral at the sealed mix.

It is a candidate for a later sealed upload under Force_Best, not a replacement for the lead.

## 10h: hardware sensitivity of the teacher (`runs/diag/wave10h.sh`, 3045837, screen 3045838)

This closes the audit recipe's open gap: an auditor who rebuilds everything on A100. EEGNeX m7r seeds 0–9
train on `a100-galvani` (tag `m7ra100`), and seeds 10–11 are the audit workspace's A100 rebuilds. The
screen scores the teacher from all-A100 EEGNeX plus the original IFNet against the staged teacher at C
0.01, w 3–7, on s05 / s611 / s011.

This is not a selection arm. Expected: s011 sel within seed noise (±0.7) of the staged +2.91.

**Result (screen 3045838; EEGNeX seeds 0–9 on A100, 24 min each).** sel Δ over R19e, C 0.01, w 5
(the staged configuration):

| teacher | s05 | s611 | s011 | s011 dla |
|---|---|---|---|---|
| staged: EEGNeX (RTX 2080 Ti) + IFNet | +2.65 | +2.92 | **+2.91** | 78.01 |
| all-A100 EEGNeX + IFNet | +2.52 | +2.76 | **+2.80** | 77.79 |
| EEGNeX alone (RTX 2080 Ti) | +2.17 | +2.70 | +2.45 | 77.65 |
| EEGNeX alone (A100) | +2.28 | +2.41 | +2.62 | 77.70 |

The staged row reproduces wave 5 exactly (+2.91), which confirms the screen. An all-A100 rebuild moves
sel by −0.11 on s011 and −0.13 / −0.16 on the halves, well inside seed noise (±0.7). EEGNeX alone
moves both ways (+0.11 on s05 and −0.29 on s611). At other w, the all-A100 pair's best s011 is also
+2.80 (w 5), so the staged weight still fits.

**Verdict:** an all-A100 audit rebuild changes the dev score by about a tenth of a point, smaller
than the gap between two seed sets (+2.91 vs +2.83).

**Package (`runs/diag/wave10h_pkg.sh`).** Export 3046292 built `runs/sealed/cand_nxif_a100`, exactly as
the staged teacher was built but from the A100 trunks: harness 1.0, 36 labels differ from R19e-g
(staged: 40), `teacher.pt` `ccc8e2f7…`, and `weights.npz` the same as the staged one. Full-path
agreement with the staged cand_nxif (3046293) is **752 / 755** official-test windows (99.60 %) and
**2,649 / 2,655** overall (99.77 %). In-sample, 0.9659 vs 0.9646. An all-A100 audit rebuild therefore
meets the ≥ 99 % expectation.

## 10p: EEGNeX m7r + IFNet m13r (post hoc, pre-registered before its scores)

10m's per-family profile picks one pair that was not screened. IFNet m13r is the only trunk whose
accuracy alone (58.7 vs 54.8) and independence (P(right | R19e wrong) 0.51 vs 0.48) both rise. The pick
comes from s05, so s05 cannot confirm it.

**Pre-registered:**
- Screen `nx7_if13` = EEGNeX m7r + IFNet m13r on the cached s05 heads (C 0.01, C 0.003, and the
  19 m7 heads).
- It must reach ≥ +0.5 over the base's +2.65 on s05 with both halves.
- It must then hold ≥ +0.3 over the base on the hidden seeds 6–11 (s611, same C / w as the s05 pick).
  IFNet m13r seeds 6–11 are trained now, before any score is seen.
- Only the s611 check counts as evidence; s05 is the gate.

**Result (gate 3046418).** Best s05 sel Δ, w 3–7, vs the base's +2.65:

| | C 0.01 | C 0.003 | C 0.01, 19 m7 heads |
|---|---|---|---|
| base | +2.65 | +2.63 | +2.65 |
| EEGNeX m7r + IFNet m13r | +2.14 | +2.71 (w 6) | +2.64 |

**Verdict: closed at the gate** (best +0.06). The hidden-seed training (3046417) was cancelled after
42 s, before any s611 score existed. Higher independence for one net alone (0.51) does not survive the
pair map. Wave 10 closes all three teacher-data levers (in-domain adaptation, more MI corpora and the
post-hoc pair) within the current teacher design: a logistic map on seed-averaged head log-probs.
