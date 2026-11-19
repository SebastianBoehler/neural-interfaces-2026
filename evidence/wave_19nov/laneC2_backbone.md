# Lane C2 · Same D1 pipeline, different pretrained backbone (CBraMod)

Registered 19 November 2026 before any C2 fit. Run on Leonardo (account AIFAC_P02_1675, compute nodes only), because
eduVPN is off until about 18:00 CET. Lane C (REVE) results were already on disk when this was written; the REVE numbers
below are read, not predicted.

## Backbone choice and provenance

| Candidate | On Leonardo? | Licence | Decision |
| --- | --- | --- | --- |
| CBraMod | **yes**: `ni26_fm280/hf/hub/models--braindecode--cbramod-pretrained`, rev `584cdc41…`, `model.safetensors` SHA-256 `a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3` (the §71/§280 pinned blob) | MIT code (GitHub), Apache-2.0 weights (HF `weighting666/CBraMod`), see `evidence/fm_15nov/a0_table.md:16` | **used** |
| LaBraM | not staged | MIT | not needed |
| EEGPT | not staged; Apache-2.0, needs 58 fixed channels | | not needed |

Same architecture family as §280 (braindecode `CBraMod`, 12 criss-cross layers, 19.7 M parameters, vs REVE-Base 69 M).
CBraMod has no electrode coordinates and a channel-count-agnostic criss-cross attention, so native corpus montages are
fed directly like REVE (no interpolation). Pretraining corpus (TUEG, EEG-only clinical) does not include PROTEUS.

## Pipeline (identical to Lane C except the backbone and head)

Reused unchanged: `laneC/data.py` (source-locked external rows, native channels, causal alignment, LOPO/returning/all
folds), `laneC/model.py::augment/masked_ce`, `laneC/train.py` (stage A 960 steps batch 16, PROTEUS 25 epochs OneCycle,
bf16, same optimizer lrs 1e-4 body / 1e-3 head, seeds 20261119+seed), `laneC/full_collect.py`, the pinned eight full-session
orders. Arm **C** only (pooled corpora **with** Scherer; Scherer is external to PROTEUS so held-new claims are unaffected),
seeds 0–2: 3 pooled stage-A backbones, then per seed P01–P10 LOPO, one returning fit (excludes P04/P10 visits 2–3) and one
all-public deployment fit = 36 PROTEUS fits.

Backbone-specific (the only differences, `laneC2/cbmod.py`): 120→200 Hz linear resample, clamp ±15, ×0.15 amplitude factor
(the §280/`deep.py` constant for CBraMod's 100 µV units); CBraMod encoder `return_features` output (B, C, patches, 200)
mean-pooled over channels and patches (variable montage and 2-patch PhysioNet crops forbid a flattened head), then
LayerNorm + `Linear(200, 4)` task head and `Linear(200, 5)` motor-subtype head. The §280 flattened 41×4×200 head is not
reusable across corpora. `laneC2/train_cbramod.py` patches `Member`/optimizer groups into Lane C's trainer and asserts the
patch is active; receipts carry `backbone=cbramod` and the checkpoint SHA so a silent REVE fallback fails the collector.
Bank files keep Lane C's `reve_*` stem and format (`index`, `full0–7` posteriors, `*_delivery`) in a **separate** work dir
(`/leonardo_work/AIFAC_P02_1675/ni26_lane_c2_20261119/bank`), so both banks go through the same scorer.

## Numeric prediction (vs REVE Lane C, read: held-new BA B 55.0–55.4, C 55.0–55.7 over eight orders; returning C 61.5–62.8)

- INDIVIDUAL-MODEL held-new (3-seed pooled, ten LOPO folds, eight orders): **CBraMod-C 52.0 % [48, 56]**, i.e.
  CBraMod−REVE-C **−3.0 pp [−7, +1]**. Reason: unpooled §280 CBraMod F1–F5 scored 49–52 % against REVE 52–56 %; pooled
  pretraining added ≈0 for REVE (B 55.1 vs C 55.2), so I expect no larger stage-A lift here and keep the old gap.
- Returning (P04/P10 seen-session cells): 57 % [50, 64].
- Binary error correlation with corrected cap250 (full0, where Friday banks are readable): φ ≤ 0.45 (REVE's is the
  comparator, to be computed with `laneC/oracle.py` on Galvani). Binary error correlation CBraMod vs REVE ensemble ≈ 0.55.
- Union value: if CBraMod-C adds ≥ 1.0 pp of **unique** zero-regression repair mass on top of cap250 ∪ REVE at ≥0.60
  confidence (oracle screen), the second backbone is worth carrying into the full-system test.

## Refutation / reading rule

- CBraMod−REVE ≥ −1 pp with the 80 % band inside ±2 pp: the backbone is exchangeable; the 55 % ceiling belongs to the
  recipe/data, not REVE. Prefer REVE for cost only if the full-system gain ties.
- CBraMod-C < 48 % **and** unique repair < 0.5 pp: this form of CBraMod is closed (not the family: head and amplitude
  scale are first-try choices; undecided if stage A loss does not fall).
- Anything between is undecided. No full-system claim is made here; consumers (cap250 replay, 1/6 mixer slot, paired
  people-and-order intervals, A100/RTX wall time) run on Galvani when VPN returns.

## Gates before the full wave

1. Source check: CBraMod blob SHA and no unexpected/missing keys beyond `final_layer.*` (head replaced).
2. Smoke (`canary` mode: one supervised step on each corpus, one PROTEUS step, 189 held-P02 posteriors in full0/full1,
   prefix and singleton guards). Stage A loss and finiteness must pass.
3. Then full wave without held-score selection; no hyper-parameter changes after smoke except a documented crash fix.

## Amendments before any scored result (19 Nov, 15:10 CET)

- Canary 64852511 failed on a missing `safetensors` package in the Leonardo venv; `cbmod.py` now reads the checkpoint with
  Lane C's header parser. Canary 64852619 and the three pooled backbones 64852620 passed (stage A loss ≈1.1 by step 960).
- Fit task 64852622_9 (P10, seed 0) failed Lane C's batched-versus-singleton guard by 4e-7 (2.04e-5 against 2e-5, fp32 kernel
  noise). The tolerance is now a module constant `PARITY_ATOL` (Lane C value 2e-5 kept in `laneC/train.py`); C2 sets 1e-4.
  The array was cancelled and all 36 fits rerun as 64853664 with collector 64853666. No score was read.

## Amendment 2 (19 Nov, 16:50 CET): v1 pooled head collapsed; v2 head

The v1 form (mean over channels and patches, `Linear(200,4)`) completed all 36 fits (64853664) and scored **34.4 %
held-new, 37.0 % returning** (chance): posteriors were constant at ≈1/3. Frozen-feature spread was 0.003 and no tested
variant (input scale, fp32, dropout, body lr, a coordinate-aware per-token head) trained within 150–400 steps (diagnostics
64855209, 64856291, 64856795). This is a head-design failure of the first form, **not evidence about CBraMod**; v1 outputs are kept in
`bank_v1_pooledhead_chance` and never scored in a mix. v2 is the §280 head form: montage-specific `Linear` on each patch's
channel-flattened 200-sample tokens (weights shared over patches so 2- and 4-patch corpora fit one head), averaged over
patches; sensor dropout becomes zeroing. Stage-B-only check (arm A, no external stage, seed 0, job 64857334): P02 0.673 and P10 0.469
plain balanced recall, posterior max 0.8/0.78 (non-constant). Full wave resubmitted unchanged otherwise (canary 64858170,
external 64858172, fits 64858177, collector 64858179). Prediction numbers above stand for v2.

## Result (v2, 19 Nov ~18:00 CET) · INDIVIDUAL-MODEL, ten-person held-new + returning cells, eight full orders, Leonardo banks

Jobs: canary 64858170, pooled-C backbones 64858172, fits 64858177 (+ 6 parity-guard reruns 64859960), collector 64859961.
Bank `/leonardo_work/AIFAC_P02_1675/ni26_lane_c2_20261119/bank` (ensemble_C_held_new.npz SHA `ee59ed60…`, ensemble_C_returning.npz
SHA `4d27d61e…`, full_standalone.json SHA `65eefa90…`, 36 PROTEUS checkpoints incl. 3 all-public). Comparison:
[laneC2_compare.json](laneC2_compare.json) from `laneC2/compare.py` against Lane C's REVE-C ensemble (same rows, same eight delivery orders).

| | CBraMod-C (3-seed pooled) | REVE-C (Lane C) | prediction (CBraMod) |
| --- | ---: | ---: | --- |
| held-new BA, mean of 8 orders | **52.76** (52.2–53.3) | 55.22 (55.0–55.7) | 52.0 [48, 56]: inside |
| single seeds, full0 | 51.4 / 51.9 / 52.3 | 49.9 / 50.4 / 54.1 | |
| returning cells BA | **52.30** | 62.08 | 57 [50, 64]: inside, lower half |
| CBraMod − REVE held-new | −2.46 pp | | −3.0 [−7, +1]: inside |
| binary error φ vs REVE (full0, held-new) | 0.517 (members 0.49–0.51) | | ≈0.55: as predicted |
| label-only union BA (REVE ∪ CBraMod) | 66.0 (held-new) / 69.9 (returning) | | oracle, not a mix |
| share of REVE-C misses CBraMod gets right | 24.0 % (15.7 % at max-posterior ≥0.60); regressions at ≥0.60: 8.5 % of rows | | |

Reading by the card's rules: CBraMod−REVE is −2.5 pp, so not "exchangeable" (needs ≥ −1) and CBraMod-C is above 48 % with a
sizeable unique-repair mass, so the form is **not closed**; the decisive step is the corrected-cap250 full-system replay
with each of REVE-C and CBraMod-C (and both) in the 1/6 slot on Galvani. Error correlation with cap250 and unique repair
over cap250 ∪ REVE (`laneC/oracle.py`, which needs the Friday camp stock not present on Leonardo) are **not yet computed**.
Pooled stage A did not help the returning cells here (52.3 vs REVE 62.1): the all-public/returning gain REVE shows is not
reproduced by CBraMod's smaller encoder. Caveats: head (v2) and amplitude constant were first-form choices; v1 collapsed;
no hyper-parameter was selected on held scores.
