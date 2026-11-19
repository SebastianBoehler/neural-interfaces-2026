# Lane C3 · Pratibha-2 (MANAS-2) as the pretrained backbone of the Lane C / C2 pipeline

Registered 19 November 2026 (~18:55 CET) before any C3 fit or score. Same protocol as [Lane C2](laneC2_backbone.md): pooled-C stage A (six
corpora including Scherer), PROTEUS stage B, seeds 0-2, ten LOPO folds plus the returning fit plus the all-public fit (36 fits), eight pinned
orders, Leonardo account AIFAC_P02_1675.

## Provenance and security audit

`MannasAI/pratibha-2`, Apache-2.0, pinned **revision `863edc940d82a02744776a7820641bf1bc32310b`** (last modified 15 Nov 2026), fetched on a Leonardo
login node by hash-pinned URL (`resolve/<rev>/...`; files only, nothing executed): `model.safetensors` SHA-256 `5c7fb0a1de75bf01c179fdc416a4e4af3ad4ead7895ac03b7e357adb587cfa64`
(equals the HF LFS oid), `positions.json` `e7fb3cbd...`, `model.py` `a637e414...`, `config.json` `7f72afaf...`. `model.py` was read in full: it imports only
json, math, pathlib, torch, transformers (and safetensors lazily), runs no subprocess/network/eval and only reads `positions.json` next to itself.
Because the Leonardo venv has no `transformers` and compute nodes have no network, `laneC3/pratibha.py` is a torch-only copy of those classes (GEGLU
with both gate terms kept, RMSNorm, 22 blocks d=512, 8 heads, patch 200/stride 180, Fourier+learned 4-D position encoding); no `trust_remote_code`.
Weights/table are loaded from the pinned local copy with all three hashes checked; the load is strict (missing or unexpected body keys, or any key outside
the known training-only prefixes, fail). Nothing was run on the Mac. Not verifiable here: equality of our copy's outputs with the official class
(transformers unavailable on the cluster); the copy is line-for-line, with strict key and shape loading as the check, plus the smoke below.

## Pipeline differences from C2 (first-form choices, none selected on held scores)

Input: our causally aligned windows, 120->200 Hz linear resample, then per-channel per-window z-score with `correction=1` (README of the release; replaces
C2's clamp and amplitude constant). Positions: exact-name lookup in the pinned table (metres) x 100 -> cm; all 41 PROTEUS, Shin2018, Lee, Scherer,
PhysioNet and Schirrmeister channel names resolve (checked locally); a missing name fails. Lane C's source-only augmentation unchanged (channel dropout
subsets the tokens and positions). The position encoder runs in fp32 inside bf16 autocast (phases reach ~1e2). Head: learned-query attention pooling over all
(patch, channel) tokens + `Linear(512,4)` task + `Linear(512,5)` motor head (REVE-like; the pooled-mean head collapsed for CBraMod, so the smoke includes a
stage-B-only trainability check). Optimiser lrs 1e-4 body / 1e-3 head, stage A 960 steps batch 16, stage B 25 epochs OneCycle, bf16, parity guard 5e-4.

## Predictions

- INDIVIDUAL-MODEL held-new, 3-seed pooled, eight-order mean: **Pratibha-C 53 % [47, 58]**, i.e. at or below REVE-C 55.2 (the claimed NeuralBench rank-3.94 does not
  cover this 3-class/41-channel/120 Hz-resampled setting, and CBraMod, 3.5 ranks worse, scored 52.8). Returning cells 58 % [48, 66].
- In the cap250 1/6 mixer slot (`laneC/full_replay.py`, full0, people-only paired 80 %): Pratibha **+0.2 pp [-0.8, +1.0]**; CBraMod+Pratibha mean slot +0.2 [-0.8, +1.0].
- Zero-weight arm must equal stock bitwise (guard in `full_replay.py`); stock terminal argmax must equal the c25d `full0.npz`.

## Refutation and reading rules

- Packaging rule (task): in-mixer 80 % upper bound < **+0.5 pp** -> Pratibha is not worth packaging for this slot form (consistent with the C card, which keeps the in-mixer form open only up to that bound).
- Card rule (Lane C): upper bound < +1.00 pp refutes the step change at 1/6; a nonpositive bound refutes benefit; an interval crossing +1.00 stays undecided. Bars inside the dev noise (~+-0.5) read as undecided.
- Standalone < 48 % with the smoke's stage-B-only check flat (posterior spread ~0): this head/input form failed (not evidence about the backbone); a documented single fix is allowed before the wave, never after held scores.
- Not covered: other slot doses, returning inside the mixer (instrument does not support P04/P10 visits 2-3), orders 1-7 for cap250 (one order exists).

## Gates before the wave

1. `smoke_pratibha.py` (strict load, cm positions, shapes, bf16 vs fp32), 2. Lane C canary (one supervised step on each corpus, one PROTEUS step, 189 held-P02 posteriors, prefix and singleton guards), 3. arm-A P02 and P10 trainability check (non-constant posteriors only; its balanced recall is not used to tune).
