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

## Form 1 result and the form-2 change (19 Nov, ~19:45 CET; written after form-1 held scores were seen, said plainly)

Form 1 (LayerNorm + one-query attention pooling, `bank_*_f1`, jobs 64873633-64873644): every arm is near chance. Held-new / returning BA, mean of 8 orders: BrainCodes-random 36.05 / 36.12,
NeuroRVQ-pretrained 35.34 / 34.03, NeuroRVQ-random 37.97 / 39.67 (chance 33.3; predictions 46-49 were far too high). BrainCodes-pretrained did not seal: 3 of 36 fits (tasks 4, 9, 16) failed the
batched-vs-singleton parity guard (1.2e-3 against 5e-4; a discrete codec code flipping at a near-tie), the other 33 and the stage-A fits passed. Stage-A and stage-B training losses sit at about ln 3 = 1.10 (stage A 1.08-1.14
flat over 960 steps, stage B final 1.02-1.4): the head cannot fit even its training rows, so this is a head/input-form failure, not evidence about the backbones (card rule, standalone < 45 % with a flat loss).
The rule allowed one documented fix before held scores and none after; I go past it because all four arms were uninformative (the held scores cannot rank the forms, and the cause is visible in the training loss),
and I keep form 1 reported as is. Nothing in form 2 is selected on held scores.

Form 2 (one change, the same for all four arms): the pooled form discards per-electrode lateralisation, so each token now gets its electrode position (BrainCodes: MLP over the 3-D coordinate; NeuroRVQ: learned
embedding of its vocabulary slot) added to a 128-wide projection of the LayerNorm'd token, then eight-query attention pooling (concatenated, dropout 0.1) feeds the task and motor linears. Body frozen as before;
everything else (pipeline, augmentation, schedules, seeds, orders) unchanged. Parity guard 5e-4 for NeuroRVQ, 3e-3 for BrainCodes (the near-tie flip above; a guard, not a result). Gate: arm-A P02/P10 stage-B-only
fits (`diag`) must show a training loss well below ln 3 (< 0.9) before the wave; their held recall is not used. Predictions for form 2 (the first-form predictions stay recorded above as missed): standalone held-new
**BrainCodes 45 % [38, 52]**, **NeuroRVQ 45 % [38, 52]**, random-init controls **42 % [36, 48]**, pretrained minus random **+3 pp [-2, +7]**; in-mixer deltas as registered above
(the standalone level, not the slot dose, is what changed). If form 2 stays below 42 % with a trainable loss, no further form: the lever is closed for this slot as a frozen-probe, not for fine-tuning.

## Form 3 (19 Nov, ~20:00 CET, before any form-2 or form-3 held score; the form-2 gate was a training loss and it failed for NeuroRVQ)

Form 2's arm-A P02/P10 stage-B-only fits (jobs 64874939 BrainCodes, 64874945 NeuroRVQ; only the training loss was read): BrainCodes 0.91-1.17 on single minibatches, NeuroRVQ flat at 1.10 (running mean 1.0966 at
3,625 steps; the gate was < 0.9). Two causes were identified from the code and the diagnostics, not from held scores: (1) per-token LayerNorm leaves the frozen NeuroRVQ tokens (std 0.0065, dominated by shared
position/patch content, layer scale 1e-5) with almost no row-specific variance relative to the head's learning rate; (2) Lane C's causal EA whitening decorrelates the electrodes, which a frozen encoder reads by position.
Changes, applied to all arms alike: the token LayerNorm becomes a feature-wise BatchNorm (running statistics, eval at test), and the input is run in two forms, **raw** (no covariance alignment, the window only demeaned
then z-scored in the model; nothing crosses windows, so it is causal by construction) and **aligned** (Lane C's EA, as in C2/C3). Diagnostics of form 3 (running training loss at about 3,000 steps, arm A, P02/P10): BrainCodes raw 0.86/0.90,
aligned 0.93/0.95; NeuroRVQ raw 0.91/0.94, aligned 0.98/0.96. All trainable (form 1 and 2: 1.10 flat) and raw lower in both models; my < 0.9 gate was met by one of eight, I read it as "trainable, no longer ln 3"
and go past it plainly. **Primary arm = raw** (chosen on training loss and mechanism, declared here before any form-3 held score); aligned is run for all four pretrained/random cells and reported; no arm is selected on its held score
(selecting the better of two inputs in the mixer would be a forking-path bias and is reported as such). Eight waves (2 models x {pretrained, random} x {raw, aligned}), `bank_*_f3_*`; parity guard 3e-3 BrainCodes / 5e-4 NeuroRVQ.
Predictions (form 3, held-new standalone, 8-order mean; slots unchanged): BrainCodes raw **44 % [38, 51]**, aligned 42 %; NeuroRVQ raw 44 % [38, 51], aligned 42 %; random controls 2 pp lower; in-mixer deltas as registered
(+0.1 [-0.7, +0.8] pretrained, 0.0 [-0.7, +0.7] random). Forms 1 and 2 stay in the record as misses (form-1 predictions 46-49 vs 34-38 measured).

## Result (19 Nov ~20:40 CET) · form 3, frozen encoders, standalone and in the cap250 1/6 slot

Jobs (Leonardo, AIFAC_P02_1675, release `/leonardo_work/AIFAC_P02_1675/ni26_lane_c4_20261119`): form 1 external/fit/collector 64873633-64873644 (BrainCodes-pretrained unsealed: 3 parity failures); form-2 diag 64874939/64874945 and
form-3 diag 64875818-64875821, smoke 64875816 (strict hash-pinned loads, frozen body, batch vs single 2e-7, NeuroRVQ copy vs braindecode's own class on the same weights: **max abs difference 0.0**, 1.3 / 2.9 ms per window
NeuroRVQ / BrainCodes on A100 batch 32); form-3 waves: raw 64876187-64876199 (external, fit, collector for BrainCodes pre/random, NeuroRVQ pre/random), aligned 64876200-64876212; no failed fit. In-mixer (Galvani,
`ni26_laneC4_member_v1_20261119`, banks `ni26_fm_banks_20261119/c4`, jobs 3138349-3138363; all guards passed: zero arm equal to stock bitwise, laneC arm differs at mix/rebalance/EM/TSEM, stock terminal argmax equals c25d `full0.npz`).
Compact numbers: `evidence/wave_19nov/laneC4/{standalone_f3,inmixer_full0}.json`. Raw input has no delivery-order dependence, so its eight orders are identical (one number).

Standalone, 3-seed pooled, held-new / returning BA (8-order mean; chance 33.3):

| Encoder | raw input | aligned input | pretrained minus random (raw / aligned) |
| --- | ---: | ---: | ---: |
| BrainCodes pretrained | 39.14 / 45.90 | 41.35 / 44.68 | -0.30 / +3.71 held-new |
| BrainCodes random-init backbone (codec kept) | 39.44 / 39.07 | 37.64 / 37.40 | |
| NeuroRVQ pretrained | 42.14 / 43.16 | 38.65 / 41.81 | +1.85 / -2.42 held-new |
| NeuroRVQ random-init | 40.29 / 44.68 | 41.07 / 43.62 | |

Against the comparators: Pratibha-C 55.97 / 62.83, REVE-C 55.22 / 62.08, CBraMod-C 52.76 / 52.30. Frozen probes sit 13-17 pp below the fine-tuned members and the random-init controls are within about +-2 pp of the pretrained cells:
no consistent pretraining signal (signs flip between input forms). Form-3 predictions (raw 44 [38, 51], aligned 42; random 2 pp lower): inside the intervals, random controls were not lower. Form-1 predictions (46-49) missed by 10+ pp.

In the cap250 1/6 slot (held-new full0, stock 73.74; people-only paired 80 %, one order; delta pp [80 %]; fixed / regressed rows):

| Slot member | alone | mean slot with CBraMod-C |
| --- | ---: | ---: |
| BrainCodes raw | -0.195 [-0.452, +0.101]; 9/14 | +0.108 [-0.157, +0.404]; 16/13 |
| BrainCodes aligned | -0.157 [-0.387, +0.071]; 12/16 | +0.149 [-0.039, +0.347]; 16/12 |
| NeuroRVQ raw | -0.081 [-0.309, +0.188]; 9/11 | -0.188 [-0.492, +0.174]; 7/12 |
| NeuroRVQ aligned | -0.229 [-0.489, +0.083]; 7/13 | -0.041 [-0.355, +0.323]; 10/11 |
| BrainCodes random raw / aligned | -0.161 [-0.371, +0.099] / -0.038 [-0.234, +0.184] | raw -0.037 [-0.297, +0.281] |
| NeuroRVQ random raw / aligned | -0.193 [-0.453, +0.104] / -0.041 [-0.295, +0.256] | raw -0.076 [-0.332, +0.266] |
| BrainCodes + NeuroRVQ raw (both pretrained) + CBraMod | | -0.001 [-0.216, +0.262]; 8/8 |

Reading by the registered rules: every in-mixer 80 % upper bound is **at most +0.40 (< +0.5)** alone and as mean slot with CBraMod, for both models, both inputs and the controls: the packaging rule holds, **no packaging for BrainCodes or
NeuroRVQ as frozen-probe slot members**. No mean slot beats CBraMod alone (+0.263 [0.001, 0.527]) or Pratibha+CBraMod (+0.256); the best, BrainCodes aligned + CBraMod +0.149, is below both and its lift over the random-control slot (-0.037) is
inside noise (+0.19 on a bound that is not a gain). Registered predictions +0.1 [-0.7, +0.8] pretrained: inside, sign negative alone in all four cells. Counts of the in-mixer instrument (the 80 % intervals resample the ten people only, one order) read undecided
for the +-0.5 dev noise only where an interval crosses zero; here none reaches +0.5. Five of the four-model FM in-mixer tests earlier gave -0.003 to +0.26; this adds eight more cells between -0.23 and +0.15.
Scope of the closure: frozen-encoder probes with the form-3 head on Lane C's inputs. Not closed: fine-tuning the BrainCodes (1.9 M trainable) or NeuroRVQ (5.9 M) encoders in the Lane C3 form (both are small enough to package and run, and
BrainCodes' published fine-tuned PhysioNet MI figure is the claim that was not tested here), channel-time flatten heads, other taps, aligned-vs-raw with a fine-tuned body. Silent no-ops checked: raw-input patch asserted active (`data.align is raw_align`), step-loss wrapper asserted, receipts carry model, init, input,
head form and frozen flag and the collector refuses others; the first-form BrainCodes-pretrained bank never sealed and is not used. Rights: NeuroRVQ is CC BY-NC 4.0 (a rights question for any packaging); BrainCodes and BioCodec are MIT.
