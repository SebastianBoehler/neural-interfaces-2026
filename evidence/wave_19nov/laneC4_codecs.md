# Lane C4 · BrainCodes and NeuroRVQ-EEG as frozen encoders with a small head, in the Lane C / C3 pipeline

Registered 19 November 2026 (~19:25 CET) before any C4 fit or score. Ranks 1 and 3 of `evidence/review_19nov/research_codex.md`. Same protocol as
[Lane C3](laneC3_pratibha.md): pooled-C stage A (six corpora incl. Scherer), PROTEUS stage B (25 epochs), seeds 0-2, ten LOPO folds plus returning plus
all-public (36 fits), eight pinned orders, Leonardo account AIFAC_P02_1675. Difference from C2/C3: the encoder is **frozen** (no gradient, eval mode, forward
under `no_grad`); only a LayerNorm over tokens and the learned-query attention-pooling head (score, task 4, motor 5) train (lr 1e-4 LayerNorm / 1e-3 heads,
weight decay 0.05, Lane C's cosine and OneCycle unchanged; Lane C's source-only augmentation unchanged). Per-fold head fitting on public source people only.

## Provenance, licences, audit

| Model | Pin | Licence | Notes |
| --- | --- | --- | --- |
| BrainCodes encoder | `microsoft/BrainCodes` commit `b075ec094633f0cc5293681030d1e00dc1cb9e18`; `weights/braincodes.pt` SHA-256 `f173669f...` (7.5 MB) | MIT | released classes imported as they are from the hash-checked pinned copy; `torch.load(weights_only=True)` |
| BioCodec tokenizer (frozen, needed at inference) | `klean2050/BioCodec-release` commit `575bf6ea7ae14a34754b217d9b940597a16951bd`; `biocodec_ckpt.pt` SHA-256 `b8b5c864...` (38 MB) | MIT | python files read: no network/subprocess/eval/exec; same loader |
| NeuroRVQ-EEG encoder | `braindecode/neurorvq-eeg-pretrained` revision `56ce74ce061428ec7829a648c29a462e8ca254ff`; `model.safetensors` SHA-256 `2e53d170...` (= HF LFS oid, 23.5 MB, 5.87 M) | **CC BY-NC 4.0** (non-commercial; the author code and checkpoint share it) | torch-only copy of braindecode's `NeuroRVQ` encoder (commit 26ae8b4b, read in full); no `trust_remote_code`, no braindecode import in the fit path |

All 18 BrainCodes/BioCodec files are hash-checked at every model construction (`laneC4/braincodes_member.py`); NeuroRVQ weights and config likewise. Fetched on a
Leonardo login node by commit-pinned URL; nothing ran on login nodes except hashing. Rights: BrainCodes/BioCodec MIT are packageable; NeuroRVQ's non-commercial
licence is a rights question to settle with the organisers before any packaging claim (Lane C3 declared corpus and licence the same way); it does not affect the
scoring here. Pretraining data: BrainCodes uses Healthy Brain Network (R3/R4/R7/R8), NeuroRVQ the 12 public sets in its card (incl. PhysioNet MI, which is also one of our
stage-A corpora: overlap with our external training rows, none with PROTEUS) plus about 235 h private motor EEG.

## Input forms (first-form choices, none selected on held scores)

- BrainCodes: our 120 Hz windows linearly resampled to 250 Hz (1000 samples), per channel and window z-score (as `BrainCodesModel.tokenize`), tokenised in fp32 without
  TF32, positions from MNE `standard_1005` via the released `coords_from_names` (all 41 names resolve), head on the bottleneck grid `[C, T_b, 128]`
  (README: the tap of all reported results). `set_coords` rebinds the montage per batch (channel subsets under augmentation).
- NeuroRVQ: 120 -> 200 Hz linear resample (800 samples = 4 patches), per channel and window z-score (the patch embedding's GroupNorm makes the scale largely moot; the authors
  clip +/-500 uV on raw data, which does not apply to aligned unit-scale input), channel slots by lowercase name (all 41 names are in the 104-name vocabulary), head on the
  `[C x patches, 4 scales x 200]` tokens.
- Not covered: 0.5-40 Hz codec band-pass beyond what our pipeline already applied, other taps (`bottleneck_per_chan`, flattened channel-time head), fine-tuning, other heads.

## Controls

Same head, same pipeline, same seeds on a **frozen random-init encoder of the same architecture**: NeuroRVQ random = trunc-normal(0.02) weights with the released layer-scale
1e-5 (near-identity residual stream: the control is a conv-patch-embedding + position probe); BrainCodes random = the released BioCodec tokenizer and its codebook-seeded
code embeddings kept, every backbone weight re-drawn. This isolates what the *backbone's pretraining* adds, not what discrete codec tokens carry. The random bank is a
Lane C-sized wave (36 fits), scored standalone and in the mixer exactly like the pretrained one.

## Predictions (forecast rules of `evidence/review_19nov/forecast_calibration.md`: FM lever ratio 0.42, standalone to full system x0.1 with sign flips allowed, dev gains inside noise are undecided)

- Standalone held-new BA, mean of 8 orders, 3 seed pooled: **BrainCodes-pretrained 49 % [43, 54]**, **NeuroRVQ-pretrained 49 % [43, 54]** (frozen body, small head; below
  Pratibha-C 55.97 and REVE-C 55.22; the paper retention of frozen over fine-tuned is about 96 % on a time-global task and lower on time-localised ones). Returning cells about +5 pp above held-new.
- Random-init controls: **NeuroRVQ-random 46 % [41, 51]**, **BrainCodes-random 46 % [41, 51]** (codec tokens and conv/position features already carry band-power content). Pretrained minus random
  **+3 pp [-1, +7]**.
- In the cap250 1/6 mixer slot (`laneC/full_replay.py`, full0, people-only paired 80 %, one order): BrainCodes **+0.1 pp [-0.7, +0.8]**, NeuroRVQ **+0.1 pp [-0.7, +0.8]** (research prior +0.30 / +0.20
  shrunk by the FM ratio 0.42 to +0.13 / +0.08); mean slot with CBraMod-C **+0.2 [-0.6, +0.9]**; random-control slots **0.0 [-0.7, +0.7]**. Fixed rows about equal to regressed rows.
- Zero-weight arm must equal stock bitwise (guard in `full_replay.py`); stock terminal argmax must equal the c25d `full0.npz`.

## Refutation and reading rules

- Packaging rule (task): in-mixer 80 % upper bound **< +0.5 pp** for the model alone and as the mean slot with CBraMod -> that model is not worth packaging in this slot form. The CBraMod-alone and
  Pratibha+CBraMod comparators (+0.263 [0.001, 0.527], +0.256) are the bar to beat for a mean-slot claim; an interval inside the +/-0.5 dev noise reads undecided, not refuted.
- A positive in-mixer delta counts for the pretrained encoder only if the matching random-encoder slot is lower by a margin outside noise; otherwise the lift is the head/slot form, not the backbone.
- Standalone: pretrained below random by more than 1 pp, or standalone < 45 % with the canary's trainability check flat, marks this head/input form as failed, not the backbone; one documented single fix
  is allowed before the full wave, never after held scores.
- Card rule (Lane C): in-mixer upper bound < +1.00 pp refutes the step change at 1/6; nonpositive refutes benefit. Not covered: other doses, returning inside the mixer, orders 1-7 for cap250.
- Drop rule (task): a model that cannot be loaded and verified within about 2 h is dropped and the reason recorded here.

## Gates before the wave

1. `smoke_c4.py`: strict load and hash pins, frozen body, shapes, batch vs singleton (5e-4) and bf16 vs fp32, pretrained vs random difference, timing, and NeuroRVQ copy vs braindecode's own class on the same
   weights (best effort; reported if it cannot run). 2. One-step canary per model through Lane C's trainer (prefix and singleton guards). Code: `laneC4/`, launcher `scripts/cloud/leo_lane_c4.sh`.
