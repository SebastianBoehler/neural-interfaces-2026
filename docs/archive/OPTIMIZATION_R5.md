# Optimization round 5: organizer-recommended external data

Run 19 October 2026 (Europe/Berlin). Reference: the round-3 rebalanced
ensemble (`ensemble_all_r3`; v2 + aligned and raw ShallowConvNets ×5 +
FBCNet ×5, α = 0.5, k = 5): `dev_loso` 65.1, `dev_session` 77.3,
`all_loso` 62.8, `seen_session` 66.6. Selection on `dev_loso`/`dev_session`
only; `all_loso`/`seen_session` evaluate.

## Licences and provenance (checked 19 October)

| Dataset | Source | Licence | Training / shipping a model |
|---|---|---|---|
| Dreyer 2023 (87 participants, 27 EEG ch., 512 Hz, L/R-hand MI, Graz protocol) | [Zenodo 8089820](https://doi.org/10.5281/zenodo.8089820) (27.5 GB); per-participant ZIPs on [OSF 8tdk5](https://osf.io/8tdk5/) as used by MOABB 1.7.2; paper [10.1038/s41597-023-02445-z](https://doi.org/10.1038/s41597-023-02445-z) | CC BY 4.0 | Allowed with attribution |
| Scherer 2015, BNCI 2015-004 (9 participants with stroke/SCI, 30 ch., 256 Hz, 2 days, WORD/SUB/NAV/HAND/FEET) | [BNCI Horizon 2020](https://bnci-horizon-2020.eu/database/data-sets) entry 004-2015 ("Individual imagery"), licensor TU Graz; NEMAR nm000144 v1.0.2; paper [10.1371/journal.pone.0123727](https://doi.org/10.1371/journal.pone.0123727) | **CC BY-NC-ND 4.0** | Unclear: NC (cash prizes) and ND (whether a trained model is an adaptation) |

Competition Terms, Rule 4: pre-training on "publicly available,
redistributable" data is allowed, original licences stay in force, and every
external corpus must be declared with a compute estimate. The organisers'
dataset directory lists both corpora without licences.

**Decision (user, 19 October, relayed by the coordinating session):** use
Scherer 2015. "Don't count the competition as commercial use", and since
the organisers recommend it, download it for the comparison even if it is
later dropped. Shipping a Scherer-trained model is allowed in principle
(declared in the method description), but any package containing it is
flagged for a final user decision.

## Preprocessing (`scripts/export_external.py`)

Mirrors the NeuralBench extractor used for PROTEUS: notch at every 50 and
60 Hz harmonic below min(Nyquist, 301 Hz) (zero-phase, `filter_length
"auto"`), FIR band-pass 0.1–75 Hz (`picks="all"`), resampling to 120 Hz,
volts → µV, one 4 s window from 1 s after each cue. The 41 PROTEUS channels
are copied when recorded and otherwise interpolated from all recorded EEG
channels with MNE spherical splines (`standard_1005` positions). Native
references are kept, as in the official pipeline. Downloads and exports ran
on the Galvani login node (OSF/BNCI at about 11 MB/s there, about
0.1 MB/s on the Mac); raw files stay under `data/<name>/raw/` (gitignored).

| Dataset | Cue | Labels | Windows | Interpolated channels |
|---|---|---|---:|---|
| Dreyer 2023, participants 1–20 | annotation 769/770 (arrow cue) | L/R-hand MI → 0 | 4,800 | AF3 AF4 F1 F2 F5 F6 P1 P2 P5 P6 PO7 PO8 TP7 TP8 (14) |
| Scherer 2015, all 9, both days | trial start + 3 s (description Fig. 1) | HAND → 0, SUB → 1, WORD → 2 | 2,130 (710 per class) | AF3 AF4 C1 C2 C5 C6 CP1 CP2 CP5 CP6 F1 F2 F5 F6 FC1 FC2 FC5 FC6 PO7 PO8 TP7 TP8 (22) |

MOABB's BNCI2015-004 event names (`math`, `letter`, …) do not match the
data set description (1 WORD, 2 SUB, 3 NAV, 4 HAND, 5 FEET); the exporter
reads the `.mat` files directly with the description's codes. Two Scherer
sessions (E, H day 2) have 35 trials per class. Median window SD: PROTEUS
15.3 µV, Dreyer 13.0 µV, Scherer 7.6 µV. Scherer windows SHA-256
`a6188799…2701e91` (identical on Galvani and the Mac).

## Harness

`protocol.load(external=...)` appends `data/<name>/windows.npz` after the
PROTEUS windows with split `external`; `folds` adds them to every training
fold, asserts they never enter a test fold, and leaves fold names and
PROTEUS indices unchanged, so saved probabilities mix with the reference.
External sessions are their own alignment streams. `run_cv.py NAME@TAG`
selects them, with options appended to the data set name:

- `~n` fills the interpolation null space (`protocol.fill_null`): per
  session, every eigen-direction of the mean covariance below 2e-3 of the
  mean eigenvalue is topped up to it with seeded white noise (2.5% of the
  signal RMS for Scherer); stronger, recorded directions are unchanged.
- `~w0.1` sets the loss weight of external windows (PROTEUS 1).
- `~m` treats external windows as carrying nothing below 4 Hz: v2 zeroes
  their delta-band tangent block and slow-waveform block (no gradient on
  those weights) and builds the global delta reference from PROTEUS only.

With external data, logistic regression uses class-balanced sample weights
and the networks a class-balanced, sample-weighted loss. Without external
data nothing changes: v2 `dev_session` reproduces 68.27, and the refactored
`AlignedNet` matches HEAD bit for bit (CPU, 2 epochs). `*_ft` configs
fine-tune 10 epochs on PROTEUS only (lr/3) after joint training; `v2_ws`
fits the scaler with the sample weights and the global references on
PROTEUS only. `scripts/external_probs.py` scores the round-3 mix with
members swapped; it reproduces the reference (65.07 / 77.31 / 62.83 /
66.64).

## Why external data hurt, and what fixed it

Member results, `dev_loso` / `dev_session`, deep members seeds 0–2 (Galvani
A100 jobs 2977845, 2977868, 2977874, 2977905, 2977968), v2 one
deterministic run (Mac CPU):

| Member | PROTEUS only | + Dreyer | + Scherer | `~n` | `~n~w0.5` | `~n~w0.25` | `~n~w0.1` | `_ft` `~n` |
|---|---|---|---|---|---|---|---|---|
| v2 | 56.2 / 68.3 | 44.9 / 49.7 | 55.1 / 56.3 | 55.4 / 55.3 | 55.6 / 56.0 | 55.9 / 57.5 | 56.9 / 60.4 | – |
| aligned ShallowConvNet | 53.5 / 58.3 | 50.6 / 56.2 | 48.2 / 53.9 | 48.0 / 55.3 | 49.1 / 54.8 | 49.7 / 57.4 | – | 49.7 / 58.1 |
| raw ShallowConvNet | 52.6 / 61.1 | 48.2 / 51.9 | 53.5 / 60.5 | 53.9 / 60.2 | 54.3 / 61.7 | 54.2 / 61.4 | 54.1 / **64.0** | **56.4** / 61.9 |
| FBCNet | 59.2 / 67.8 | 52.8 / 58.5 | 56.8 / 63.3 | 57.5 / 66.1 | 59.3 / 68.5 | 59.9 / **70.1** | **60.1** / 69.8 | 59.0 / 70.0 |

Dreyer: one seed only; every member collapses, so it was stopped.

Diagnosis:

1. **Single-class sessions (Dreyer).** Every Dreyer session is motor
   imagery only. Session alignment recentres it on its own mean, so aligned
   MI windows sit at the identity, where PROTEUS puts the three-class
   session average: the model learns "centred = MI". Not fixable by
   reweighting; Dreyer is rejected for these members.
2. **Interpolation null space.** 41 channels interpolated from 30 (27)
   span 30 (27) dimensions. After alignment PROTEUS sessions have unit-scale
   energy in all directions (smallest eigenvalue 0.35–0.72); Scherer
   sessions had zero in 11 (smallest 0.000, after `~n` 0.66). Filling
   helped FBCNet (+0.7 / +2.8) but not ShallowConvNet or v2.
3. **Missing slow band.** The BNCI Scherer files are high-passed at about
   4 Hz at source (no power below 3.5 Hz in the raw `.mat`, despite the
   "0.5–100 Hz" in the description), while PROTEUS has most of its power
   and much of its class information below 4 Hz (v2's delta band and 1 Hz
   slow waveform; full-band ShallowConvNet). Scherer windows teach that slow
   activity is uninformative. This explains why aligned full-band
   ShallowConvNet loses most, FBCNet (filter bank from 4 Hz) least, and v2
   loses mainly on P10's later sessions.
4. **Weak, mismatched labels.** v2 trained on Scherer alone reaches
   47.2% leave-one-participant-out within Scherer and 41.3% on PROTEUS
   (motor imagery transfers worst: 35% recall). Equal-weight Scherer (2,130
   windows against about 1,700 PROTEUS windows per fold) dilutes PROTEUS.
   Lower weights help every deep member.
5. **Statistics leakage into v2 (minor).** At weight 0, v2 still lost 2.7
   points `dev_session`: the global reference prior (and, slightly, the
   scaler) included Scherer windows. With PROTEUS-only statistics
   (`v2_ws`) weight 0 reproduces 68.27 exactly, but weight 0.1 still gives
   59.4: the Scherer labels themselves conflict with v2's P10 decision
   rule, even with the sub-4 Hz features masked (`~m`: 59.0 / 59.4).
   v2 stays PROTEUS-only.

## Ensemble selection (`dev_loso` / `dev_session`, rebalanced)

Swapped members use seeds 0–2; unswapped members keep their five round-3
seeds (`runs/experiments/r5/ens_*.json`).

| Swap into the round-3 mix | dev_loso | dev_session |
|---|---:|---:|
| reference | 65.1 | 77.3 |
| all four members + Dreyer (seed 0) | 58.1 | 61.6 |
| all four + Scherer, unfixed | 63.5 | 69.1 |
| raw ShallowConvNet + FBCNet + aligned ShallowConvNet, `~n~w0.25` | 64.8 | 76.0 |
| v2 `~n~m~w0.5` | 65.0 | 71.6 |
| raw ShallowConvNet `_ft` + FBCNet `_ft` + aligned `_ft` | 64.3 | 74.9 |
| raw ShallowConvNet `_ft` + FBCNet `~n~w0.1` | 66.3 | 76.8 |
| **raw ShallowConvNet + FBCNet, both `~n~w0.1` (chosen)** | **66.0** | **77.3** |

"Both `~n~w0.1`" was chosen: the best `dev_loso` that does not lower
`dev_session`, one setting for both members. It is the best of about ten
compared variants, so part of the +0.9 may be selection noise.

## Evaluation (`all_loso`, `seen_session`; Galvani 2979584, seeds 0–4)

Five seeds per swapped member, as in the round-3 reference:

| | all_loso | seen_session |
|---|---:|---:|
| raw ShallowConvNet ×5: PROTEUS only → `~n~w0.1` | 53.1 → 54.8 | 53.7 → 54.7 |
| FBCNet ×5: PROTEUS only → `~n~w0.1` | 55.2 → 56.2 | 54.9 → 57.0 |
| ensemble, no rebalancing | 61.3 → 61.7 | 60.1 → 61.1 |
| **ensemble, rebalanced** | **62.8 → 63.5** | **66.6 → 66.8** |

`all_loso` per participant (rebalanced ensemble, points): P01 −1.6,
P02 +0.6, P03 +1.1, P04 −0.5, P05 +0.1, P06 −1.2, P07 +3.7, P08 +1.5,
P09 +3.1, P10 +1.4 (7 of 10 up). `seen_session`: P04 +0.5, P10 −0.3.
The gain is small and agrees with `dev_loso` (+0.9) in sign and size; the
seen-participant protocol is flat within its 8-cell noise.

With dev seeds 3–4 added (Galvani 2979672; five seeds per swapped member)
the chosen swap gives `dev_loso` **65.6** (+0.5; the three-seed +0.9 was
partly selection noise) and `dev_session` 77.3 (±0); members: raw
ShallowConvNet 54.6 → 56.0 / 61.5 → 65.2, FBCNet 59.7 → 60.2 /
68.9 → 69.4 (seed-averaged probabilities).

## Package (`scripts/swap_external.py`, Galvani 2979634)

`ensemble_all_r5` = `ensemble_all_r3` with the five raw ShallowConvNets
(slots 5–9) and five FBCNets (slots 10–14) retrained on all 2,655 PROTEUS
windows plus 2,130 Scherer windows (`~n~w0.1`, seeds 0–4, A100, 185 s);
v2, the aligned ShallowConvNets, member weights and rebalancing are copied
byte for byte, `submission.py` is unchanged (SHA-256 `567199ec…`).

| Check | Result |
|---|---|
| package vs harness, all 755 official-test windows | 100% agreement |
| official ingestion + scoring, Mac CPU | 1.0 (leakage), files unchanged, 84 s |
| official ingestion + scoring, Linux A100-node CPU | 1.0 (leakage), files unchanged, 95 s |
| CPU time per window, same Mac, back to back | r3 64.0 ms, r5 61.1 ms |

The retrained FBCNets' alignment prior includes the Scherer windows, so
they no longer share the aligned ShallowConvNets' input stream (3 instead
of 2 causal streams); per-window time is unchanged within noise. Staged:
`runs/upload/sealed_candidate_ensemble_all_r5_scherer.zip` (SHA-256
`42005d95…40c4`, identical to the Linux replay ZIP). Not uploaded.

## Decision

Dreyer 2023 is rejected for all members (single-class sessions). Scherer
2015 helps only the two members whose inputs it can inform without
contradicting PROTEUS, and only at a low weight with the null space filled:
raw ShallowConvNet and FBCNet `~n~w0.1`. The rebalanced ensemble gains
+0.5 `dev_loso`, ±0 `dev_session`, **+0.7 `all_loso`** (62.8 → 63.5,
7 of 10 participants up) and +0.1 `seen_session`. The gain is small,
consistent in sign, and of the size of the round-3 FBCNet step.

`ensemble_all_r5` is staged as the new sealed-phase candidate **subject to
the user's final decision on shipping a model trained on CC BY-NC-ND data**.
If it is shipped, the method description must declare BNCI 2015-004
(Scherer et al. 2015, licensor TU Graz) as external training data, with the
compute estimate (Galvani A100: 2 × 5 nets, about 3 min). If not,
`ensemble_all_r3` remains the candidate; the expected cost is about
0.7 points `all_loso`.

Not done: Dreyer as pretraining only (it cannot inform 3-class aligned
members directly), Scherer for v2 or the aligned ShallowConvNet (every
variant lost on `dev_session`), Zyma 2019 and Stieger 2021.
