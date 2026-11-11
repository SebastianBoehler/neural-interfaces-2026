# Audit recipe: cand_m7r4 (sealed arm)

Top-3 entries are audited by re-running training from committed config and rescoring on the sealed
set; the rescore must land within ±2σ of the submitted score. Our leaderboard entry is our best
sealed upload, so every uploaded arm needs this config. This document is the one for
`runs/sealed/cand_m7r4`: R19e with the frozen `submission.py`, plus the external-head teacher of four
architectures (docs/archive/OVERNIGHT_2NOV.md, "Packaging the candidate"). It shares stages 1–3, 5 and 7, the
corpora, licences and environment with docs/AUDIT_NXIF.md; only the teacher's trunks and export differ.

## One command

From the root of a clone at the audited commit, with the official submodule at
`a66bc23d1a0e686b895c093a642b45c74970758b`, on the Tübingen ML Cloud (Slurm, account `gehler`):

```bash
bash scripts/audit_m7r4.sh runs/sealed/audit_m7r4
```

`NI26_REF=<staged package>` adds a full-path label comparison with it; `NI26_V100=<partition>` replaces
the V100 partition (measured effect under "Validation"). Each stage is a Slurm job chained by `afterok`; a stage is skipped when its
output exists, so the script can be re-run after a failure (or after `audit_nxif.sh`, whose stages
1–3 and SincShallowNet seeds 0–4 it reuses).

| Stage | What | Output |
|---|---|---|
| 1–3 | environment, official data, PROTEUS windows, 7 MOABB corpora (as docs/AUDIT_NXIF.md) | `.venv`, `data/` |
| 4 trunks | SincShallowNet, FBCNet, FBLightConvNet, ShallowFBCSPNet × seeds 0–11, recipe m7r, each seed on its original GPU type (below) | `runs/pretrain/{sincshallow,fbcnet,fblight,shallow}_ea-m7r-s{0..11}.pt` |
| 5 base | R19e members + session EM with the frozen code (`build_sealed.sbatch scripts/sealed_r19e.json`) | `<OUT>_base_r19e_g` |
| 6 teacher | head log-probs (seed mean per architecture) → balanced LR to 3 classes, C 0.01, TorchScript member, weight 4: `export_external_teacher.py BASE OUT --families sincshallow,fbcnet,fblight,shallow --seeds 0-11 --C 0.01 --weight 4` | `<OUT>` + `<OUT>_check.json` |
| 7 checks | official A100 replay, worker-image replica, label agreement (`compare_packages.py`) | `<OUT>_replay*`, `<OUT>_compare.json` |

Pretraining recipe (`experiments/pretrain.py ARCH ea m7r <7 corpora> --seed S --rot 0.05`): as in
docs/AUDIT_NXIF.md (30 epochs, AdamW lr 1e-3, batch 256 drawn equally from the 7 corpora, per-session
Euclidean alignment, rotation ε 0.05, one head per corpus). The stored metadata of all 48 originals
match (epochs 30, lr 1e-3, batch 256, rot 0.05, no other augmentation).

GPU types of the originals (sacct, 29 Oct – 1 Nov), pinned per seed in the script because EEGNeX
trunks moved about one seed apart between A100 and RTX 2080 Ti:

| Family | V100 | RTX 2080 Ti | A100 | Pretraining per seed |
|---|---|---|---|---|
| SincShallowNet | 3–4 | — | 0–2, 5–11 | 759–920 s |
| FBCNet | 0–5 | — | 6–11 | 408–538 s |
| FBLightConvNet | 0–2 | 3–5 | 6–11 | 342–430 s |
| ShallowFBCSPNet | — | 5 | 0–4, 6–11 | 317–391 s |

SincShallowNet seeds 0–4 are also R19e's member init. The script trains them as `audit_nxif.sh` does
(A100 or RTX 2080 Ti), so both recipes share one set; rebuilt on RTX 2080 Ti they were 0.045–0.050
from the originals (0.13–0.16 between seeds; docs/AUDIT_NXIF.md).

## Reference values

- Checksums: `scripts/sealed_m7r4_inputs.sha256` (48 trunks, base and staged package);
  `scripts/sealed_r19e_inputs.sha256` (PROTEUS windows, 7 corpora, export chunking).
- Staged package: `submission.py` `25d51a6e…` (frozen code, worker-proven), `teacher.pt` `c21a03dd…`,
  `weights.npz` `95c74d1a…`, ZIP `060850ee…` (13.7 MB).
- Export checks (3036670, 2 Nov): harness agreement 1.0; traced = eager (0.0; batch 1 7e-7; CUDA
  1.5e-6); 47 of 2,655 labels differ from R19e-g; teacher member weight 0.5.
- Replays: official A100 (3036671) 148 s and worker-image replica (3036672) 120 s, files unchanged,
  same scores.
- Real worker: this package has not run on Codabench. A teacher of the same TorchScript form with
  SincShallowNet and FBCNet heads did (1026185, 1 Nov); FBLightConvNet and ShallowFBCSPNet layers are
  covered by the worker-image replica only.
- Dev evidence (not audit targets): sel Δ +2.33 over R19e on seeds 0–11 (dla 77.41, ds 89.05), +2.11
  on seeds 12–23.

**Tolerance.** GPU training is not bit-deterministic, so rebuilt files will not hash equal. Averaging
12 seeds per architecture damps trunk noise in the teacher. Expected, as for cand_nxif: label
agreement ≥ 99 % on the public windows and a dev score within seed noise (±0.7 sel).

## Compute

- 48 teacher trunks: about 6.7 GPU-hours of pretraining (times above), 5 of them shared with R19e.
- R19e base 16 min on one A100; teacher export 44–47 min; label comparison 43 min; replays 2–3 min.
- Inference: 120–148 s for the 2,655 public windows; scaled linearly, about 20 min for the sealed set
  (~21k windows), inside the 60 min limit.

## Validation

Galvani audit workspace `ni26_audit_nxif` (the fresh workspace of the cand_nxif validation), updated
on 4 Nov to `99bdfa7` plus this recipe. Stages 1–3 and SincShallowNet seeds 0–4 were reused from the
cand_nxif audit (RTX 2080 Ti rebuilds); everything else was rebuilt by
`NI26_REF=<staged cand_m7r4> bash scripts/audit_m7r4.sh` (jobs 3047939–3047952, 00:40–04:45 CET,
mostly queueing).

- **Trunks vs originals** (relative weight distance; two different seeds in brackets):
  - FBCNet: 0.000 on all 12 seeds (V100 and A100, as the originals) [0.55–0.57].
  - FBLightConvNet: 0.011–0.016 [1.17–1.28].
  - SincShallowNet: 0.004–0.006 for seeds 5–11 (A100); 0.045–0.050 for seeds 0–4, RTX 2080 Ti
    rebuilds of A100 / V100 originals [0.13–0.16].
  - ShallowFBCSPNet: 0.03–0.15 on the originals' GPU types [1.25–1.28]. Its training is not
    reproducible within a GPU type, but the rebuilds stay about a tenth of a seed apart.
- **GPU type** (side check 3047953–3047956, seed 0 of the two families with V100 originals): FBCNet
  trained on A100 or RTX 2080 Ti lands 0.042 from the V100 original (0.000 on V100), FBLightConvNet
  0.013 on any of the three. Without V100 (`NI26_V100=a100-galvani`) six FBCNet trunks therefore move
  by under a tenth of a seed distance.
- **Base R19e** (3047948, 16 min, fit 694 s): frozen `submission.py`, harness agreement 1.0.
- **Teacher export** (3047949, 47 min): harness agreement 1.0; traced = eager (0.0; batch 1 7.2e-7;
  CUDA 1.4e-6); member weight 0.5; 45 labels differ from the base (staged: 47). New `teacher.pt`
  `b7a5f088…`, `weights.npz` `55f43259…`.
- **Replays**: official A100 (3047950) 156 s and worker-image replica (3047951) 123 s, files unchanged,
  same scores.
- **Label agreement with the staged cand_m7r4** through the full predict path (3047952): **754 / 755**
  official-test windows (in-sample 0.9711 vs 0.9698) and **2,650 / 2,655** windows overall (99.81 %;
  0.9748 vs 0.9736).

**Verdict:** the recipe rebuilds cand_m7r4 within the expected tolerance (≥ 99 % agreement). In-sample
scores only check that the predictions agree, not how well the package generalises.
