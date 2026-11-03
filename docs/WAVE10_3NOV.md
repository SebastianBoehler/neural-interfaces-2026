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

**Pretraining:** EEGNeX + IFNet, m7r recipe, seeds 0–5, two variants:
- m14r = the 7 m7r corpora + the 7 smaller new ones;
- m15r = m14r + Stieger2021.

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
