# Public data hunt for Track 2 (14 Nov 2026)

Web search only. No dataset or large file was downloaded. "Confirmed" means read on the source page today.
"Memory" means not verified today. HAL and gitlab.inria.fr blocked automated reads (Anubis), so the
Inria papers there were not read.

## Verdict on the belief "no such data exists"

The belief is **mostly right**. No public EEG dataset has all three PROTEUS classes (hand MI, mental
subtraction, word generation) apart from BNCI 2015-004, which the team already knows about. The closest
new finds are:

- an **MEG** dataset with all three classes (Rathee 2021, 17 participants, 2 sessions);
- a 3-subject **EEG** set with hand MI and word generation (BCI Competition III dataset V);
- two **unreleased** EEG studies with two of the classes: Jeunet 2015 (Inria Bordeaux, the same lab as
  PROTEUS) and Banville 2017 (INRS).

Raw EEG from the PROTEUS programme beyond the ten public participants is not public anywhere I could find.

## PROTEUS origin

PROTEUS BCI Bordeaux is published on NEMAR as **nm000290** (v1.0.0, 1 Oct 2026, DOI
10.82901/nemar.nm000290, **CC-BY-4.0**).

- **Who and why:** the authors are Dreyer, Bourdil, Bechon, Kojima, Velut, Rimbert, Roy and Lotte. The
  data come from the ANR project **PROTEUS** (ANR-22-CE33-0015, 2023–2026). It is led by Fabien Lotte at
  Inria Bordeaux, with partners ISAE-SUPAERO (Roy), LAMSADE / INSA Rouen (Yger) and Wisear. The project's
  stated aim is to measure BCI variability across contexts "while sharing the collected databases". Its
  two use cases are tetraplegic BCI user training and auditory attention monitoring.
- **What the NEMAR README says:** 10 participants; 14 sessions (1 or 3 per participant) in chronological
  order; up to 8 runs per session (112 runs in all):
  - BaselineOE and BaselineCE;
  - AcquisitionGraz and AcquisitionBH ("calibration runs with **sham feedback**");
  - OnlineRawGraz and OnlineRawBH (real feedback, two runs each).

  Tasks are MI of the **non-dominant hand**, mental subtraction, and word generation from a given letter.
  Cue codes are OpenViBE 33024, 33025 and 33026. The README states that "13 cued runs have fewer trials".
- **Recording:** 41 of 64 channels of a Brain Products actiCAP slim / actiCHamp system, **500 Hz** (not
  120 Hz), reference M1, ground FpZ, EDF+.
- **Competition tracks page:** "two with their first three sessions and eight with their first session".
  The sealed set "combines later sessions from those participants with recordings from entirely new
  participants". Its size and composition are not disclosed.
- **Competition ethics page:** the release "comes from Inria Bordeaux and the NEARBY/PROTEUS projects".
  I found nothing public about NEARBY.
- **BrainHero:** a Unity rhythm game (Guitar Hero style) in Inria's OpenViBE GitLab group. It talks to
  OpenViBE over LSL and is meant to "gamify Mental Task BCI during research protocol" (from the search
  snippet; the page itself was blocked).
- **No publication:** NEMAR lists no paper citations, and the PROTEUS results page lists no dataset. The
  nearest papers on that page are:
  - Dreyer et al., "EOG markers of fatigue during MI BCI use", Graz BCI Conference 2024;
  - Trocellier et al., "Visual cues can bias EEG deep learning models", Neuroergonomics 2024;
  - Dreyer et al., taxonomy of variability factors, BCI Meeting 2025.

  Any of these may describe PROTEUS sessions or screens. None could be read because HAL blocked the fetch.
- **Inferred:** the "tetraplegic user training" use case suggests the sealed "new participants" may
  include end-users with disability. This is not stated anywhere.

## Candidate table (new datasets only, ranked by participants × matching classes × montage overlap)

| # | Dataset | Matching classes | Participants × sessions | Signal / montage | Trials and timing | Licence and use for an entry | Public? | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Rathee, Raza, Roy, Prasad 2021, "A magnetoencephalography dataset for motor and cognitive imagery-based BCI" (Essex / Ulster), figshare 10.6084/m9.figshare.c.5101544 | 3 of 3 (MI is **both hands**; also feet) | 17 × 2 days | **MEG only**, 306 channels, 1 kHz. No EEG. | 50 per class per session; 2 s rest, then 5 s imagery | Article CC BY 4.0, metadata CC0. The data-file licence on figshare was not checked. Usable if CC BY. | Yes | Confirmed |
| 2 | BCI Competition III dataset V (IDIAP; Millán, Chiappa) | 2 of 3 (left and right hand MI, word generation from a random letter) | 3 × 4 sessions on one day | EEG, 32 channels (10-20, includes Fp), 512 Hz | No trials: about 15 s self-paced blocks, switched by the operator | Terms not stated on the page. Competition data is usually free for research with citation (memory). | Yes | Confirmed |
| 3 | Jeunet et al. 2015, PLoS ONE 10(12) e0143962 (Inria Bordeaux / Potioc, same lab as PROTEUS) | 2 of 3 (left-hand MI, mental subtraction; plus mental rotation) | 18 (17 analysed) × 6 days, 5 runs each | EEG, about 30 channels from F3 to PO8, 256 Hz, g.USBamp. Close to the PROTEUS montage. | 15 per task per run; Graz-like 8 s trial, cue at 3 s, feedback from 4.25 to 8.25 s | Data statement: "All the data (BCI performance, scores … predictors)" in an annex at hal-01177685. This reads as tables, not raw EEG. | **Raw EEG not found public** | Confirmed (paper); annex unread |
| 4 | Banville, Gupta, Falk 2017, "Mental Task Evaluation for Hybrid NIRS-EEG BCIs" (INRS Montréal) | 3 of 3 (MI, word generation, subtraction; plus rotation, singing, navigation, face imagery) | 9 × 3 sessions | EEG, 62 channels BioSemi, 512 Hz, plus NIRS | Not read | No data statement found | **Not found public** | Confirmed (paper) |
| 5 | Friedrich, Scherer, Neuper (Graz, about 2012–2013), seven mental tasks | 3 of 3 | About 9 (memory) | EEG, Graz montage | Not read | None found | **Not found public** | Citation confirmed; data not |
| 6 | Wang et al. 2022, "A test-retest resting, and cognitive state EEG dataset", OpenNeuro ds004148 | 1 of 3 (subtraction: counting back by 7 from 5000, a block of about 5 min) | 60 × 3 sessions | EEG, 62 channels, 500 Hz | Blocks, not cued trials | Article CC BY 4.0. OpenNeuro is normally CC0 (memory, not checked). | Yes | Confirmed |
| 7 | CNBI EPFL Brain Tweakers, Cybathlon 2016, Zenodo 841764 | 0–1 ("3 mental commands"; the classes are not named on the page) | 2 pilots, many sessions | EEG, 16 channels from Fz to CP4 (FC, C, CP), 512 Hz | Race and training logs | CC BY 4.0. Usable, but 4.7 GB. | Yes | Confirmed |
| 8 | BNCI Horizon 003-2014 "Mental arithmetic" | 1 of 3 | 8 | Description PDF not read | Not read | Not read | Yes | Title confirmed only |

Checked and not matching:

- **COG-BCI** (ISAE-SUPAERO): 29 participants × 3 sessions; MATB, n-back, PVT and flanker. No arithmetic,
  word or MI task.
- **Neuroergonomics 2021 MATB** (Zenodo 5055046): workload only.
- **Leeuwis 2021** (DataverseNL, 55 participants): left/right MI only.
- **Yang 2025 multi-day MI** (51 participants): MI only.
- **neuroTUM Cybathlon 2024** (arXiv 2511.23384): 3 classes, but no data released.
- **Graz Mirage91 / Statthaler 2017**: four mental tasks, CC0 notice in the paper. No repository was found
  in the search snippets; check the paper.
- **Essex Cybathlon pilot data**: nothing found.

## Corrections and confirmations of the team's known list

- **BNCI 2015-004:** confirmed: 9 users with CNS damage, 2 days, 30 channels, 256 Hz, 7 s imagery. The
  NEMAR mirror is nm000144. Licence is **CC-BY-NC-ND-4.0** ("No derivative works permitted"). Training on
  it may count as a derivative, so ask the organisers. They cleared it on 11 Nov per team memory.
- **Shin 2018 (Scientific Data 10.1038/sdata.2018.3):** confirmed: 26 participants; dataset A is n-back,
  B is DSR, C is word generation. It is EEG+NIRS.
- **Zyma 2019 / PhysioNet EEGMAT:** confirmed: 36 subjects, 23 channels; licence is the Open Data Commons
  Attribution License v1.0. Rest versus serial subtraction, not cued trials.
- **Track page source list:** Stieger 2021 (62), Dreyer 2023 (87, "legacy warm-up"), Zyma 2019 (36),
  Scherer 2015 (9).

## Three most promising leads

1. **More PROTEUS from the source:**
   - Find out whether NEMAR or OpenNeuro will release later PROTEUS sessions or more participants after
     the competition.
   - Read the HAL PDFs in a browser: Dreyer Graz 2024 (hal-04706552), Trocellier Neuroergonomics 2024
     (hal-04651337) and Dreyer BCI Meeting 2025 (hal-05151145). Check participant counts, whether the
     users are tetraplegic, and what BrainHero shows during the task period.
   - Ask on Discord whether 500 Hz raw is allowed (NEMAR has 500 Hz) and what NEARBY is.
2. **Jeunet 2015 raw EEG (Inria Bordeaux):** 17 participants × 6 sessions of left-hand MI and subtraction,
   recorded with a Graz-like protocol in the same lab.
   - Open the hal-01177685 annex in a browser to see whether raw EEG is attached.
   - If it is not, the lab is the only route. That route counts as non-public data, so check the
     competition rules on permitted data first.
3. **Rathee 2021 MEG (all three classes, 17 × 2):**
   - Check the figshare file licence (likely CC BY 4.0).
   - Decide whether MEG can help at all. It cannot be used as EEG directly. At most it could serve as a
     class-structure or source-space prior, so the ceiling is likely low.
   - BCI Competition III dataset V is the only EEG set with hand MI and word generation, but it is tiny
     (3 subjects, block design).

## Sources

- NEMAR nm000290: https://nemar.org/dataset/nm000290
- NEMAR nm000290 repository: https://github.com/nemarDatasets/nm000290
- Facebook NeuroAI PR 294: https://github.com/facebookresearch/neuroai/pull/294
- Competition ethics page: https://neural-interfaces26.github.io/ethics.html
- Competition tracks page: https://neural-interfaces26.github.io/tracks.html
- ANR project page: https://anr.fr/Project-ANR-22-CE33-0015
- PROTEUS project site: https://project.inria.fr/proteus
- PROTEUS results page: https://project.inria.fr/proteus/results/
- BrainHero repository: https://gitlab.inria.fr/openvibe/unity-games/brain-hero
- Rathee 2021 MEG paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8085139/
- Rathee 2021 figshare collection: https://springernature.figshare.com/collections/A_magnetoencephalography_dataset_for_motor_and_cognitive_imagery_BCI/5101544
- BCI Competition III dataset V: https://www.bbci.de/competition/iii/desc_V.html
- Jeunet 2015: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0143962
- Banville 2017: https://pmc.ncbi.nlm.nih.gov/articles/PMC5664195
- Wang 2022 (ds004148): https://pmc.ncbi.nlm.nih.gov/articles/PMC9470564
- EPFL Cybathlon dataset: https://zenodo.org/records/841764
- BNCI Horizon dataset list: http://bnci-horizon-2020.eu/database/data-sets
- BNCI 2015-004 on NEMAR: https://nemar-website.pages.dev/dataset/nm000144
- PhysioNet EEGMAT: https://physionet.org/content/eegmat/
- neuroTUM Cybathlon 2024: https://arxiv.org/abs/2511.23384
- Leeuwis 2021: https://doi.org/10.34894/Z7ZVOD
- COG-BCI: https://zenodo.org/records/6919170
