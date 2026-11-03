# Audit recipe: cand_nxif (lead sealed package)

Top-3 entries are audited by re-running training from committed config and rescoring on the sealed
set; the rescore must land within ±2σ of the submitted score. This document is that config for
`runs/sealed/cand_nxif`: R19e with the frozen `submission.py`, plus the EEGNeX + IFNet external-head
teacher (docs/OVERNIGHT_2NOV.md, "Wave 5 result").

## One command

From the root of a clone at the audited commit, with the official submodule at
`a66bc23d1a0e686b895c093a642b45c74970758b`, on the Tübingen ML Cloud (Slurm, account `gehler`):

```bash
bash scripts/audit_nxif.sh runs/sealed/audit_nxif
```

`NI26_REF=<staged package>` adds a full-path label comparison with it. Each stage is a Slurm job
chained by `afterok`; a stage is skipped when its output exists, so the script can be re-run after a
failure.

| Stage | What | Script | Output |
|---|---|---|---|
| 1 prepare | env (`requirements/cloud.txt`, torch 2.14.1+cu126), official PROTEUS data, data audit | `scripts/cloud/prepare.sbatch` | `.venv`, `data/`, `runs/cloud/PREPARED` |
| 2 windows | PROTEUS windows (export_moabb needs them) | `scripts/export_windows.py` | `data/proteus_windows.npz` |
| 3 corpora | 7 MOABB MI corpora; chunking fixed (see checksums) | `scripts/cloud/export_moabb.sbatch` | `data/<corpus>/windows_*.npz` |
| 4 trunks | SincShallowNet × seeds 0–4 (R19e member init); EEGNeX (RTX 2080 Ti), IFNet × seeds 0–11 (teacher); one head per corpus | `scripts/cloud/pretrain.sbatch ARCH ea m7r <7 corpora> "" "" "--rot 0.05"` | `runs/pretrain/sincshallow_ea-m7r-s{0..4}.pt`, `{eegnex,ifnet}_ea-m7r-s{0..11}.pt` |
| 5 base | R19e members + session EM with the frozen code | `scripts/cloud/build_sealed.sbatch scripts/sealed_r19e.json` | `<OUT>_base_r19e_g` |
| 6 teacher | head log-probs (seed mean) → balanced LR to 3 classes, C 0.01, TorchScript member, weight 5 | `scripts/export_external_teacher.py BASE OUT --families eegnex,ifnet --seeds 0-11 --C 0.01 --weight 5` | `<OUT>` + `<OUT>_check.json` |
| 7 checks | official A100 replay, worker-image replica, label agreement | `replay.sbatch`, `worker_replay.sbatch`, `scripts/compare_packages.py` | `<OUT>_replay*`, `<OUT>_compare.json` |

Pretraining recipe (`experiments/pretrain.py`): 30 epochs, AdamW lr 1e-3, weight decay 1e-3, one-cycle
schedule, batch 256 drawn equally from the 7 corpora, per-session Euclidean alignment, rotation
augmentation ε 0.05, inputs at the 41 PROTEUS channels and 120 Hz (spherical-spline interpolation of
missing channels). SincShallowNet seeds 0–4 initialise R19e's `sincshallow_ea_ptm7r_rot05` members;
EEGNeX and IFNet seeds 0–11 feed the teacher, which averages head log-probs over seeds per architecture.

## Reference values

- Checksums: `scripts/sealed_nxif_inputs.sha256` (24 trunks, base and staged package);
  `scripts/sealed_r19e_inputs.sha256` (PROTEUS windows, 7 corpora, export chunking, 5 SincShallowNet trunks).
- Staged package: `submission.py` `25d51a6e…` (frozen code, worker-proven), `teacher.pt` `98f7df45…`,
  `weights.npz` `00ea3dcb…`, ZIP `0a562e62…` (9.0 MB).
- Export checks: harness agreement 1.0; traced = eager (0.0; batch 1 4.8e-7; CUDA 2.1e-6); 40 of 2,655
  labels differ from R19e-g; teacher member weight 0.625.
- Replays: official A100 89 s and worker-image replica 70 s, files unchanged, same scores.
- Dev evidence (not audit targets): sel Δ +2.91 over R19e on seeds 0–11; the 12-seed sets
  s0–11 and s12–23 agree (+2.91 / +2.83), dla 78.0.

**Tolerance.** GPU training is not bit-deterministic, so rebuilt files will not hash equal. EEGNeX
trunks also depend on the GPU type (A100 vs RTX 2080 Ti: about one seed apart), so the recipe trains
them on RTX 2080 Ti like the originals. The R19e fresh-clone audit (30 Oct, docs/SEALED_PLAN.md) matched the staged
package on 2,654 of 2,655 labels. Its rebuilt trunks were 0.05 (relative) from the originals, against
0.13–0.16 between two seeds. Averaging 12 seeds per architecture further damps trunk noise in the
teacher. Expected for cand_nxif: label agreement ≥ 99 % on the public windows and a dev score
within seed noise (±0.7 sel). Met on 3 Nov: see "Validation" below.

## External data and licences (declare in the method description)

| Corpus | Participants | Licence |
|---|---:|---|
| PROTEUS (NEMAR nm000290 v1.0.0, official) | 10 | CC BY 4.0 |
| PhysionetMI (EEGMMIDB) | 109 | ODC-By 1.0 |
| Cho2017 (GigaDB 100295) | 52 | CC0 |
| Lee2019_MI (OpenBMI, GigaDB 100542) | 54 | CC0 |
| Dreyer2023 (Zenodo 8089820) | 87 | CC BY 4.0 |
| Schirrmeister2017 (High-Gamma) | 14 | CC BY 4.0 |
| Shin2017A / Shin2017B (TU Berlin hBCI) | 29 | GNU GPL (MOABB `accept=True`, accepted 28 Oct) |

Licences were checked on 28 Oct (docs/OPTIMIZATION_R10.md); re-verify before the method description.
No pretrained third-party weights; no sealed or clinical data.

## Compute

- Final weights:
  - 24 teacher trunks: EEGNeX about 26 min and IFNet about 7 min per seed on one GPU (seed 0: 1,575 s / 433 s), about 6.6 GPU-hours in total;
  - 5 SincShallowNet trunks (R19e member init): 13–23 min each, about 1.5 GPU-hours;
  - R19e base: 16 min on one A100;
  - teacher export 44 min, label comparison 38 min, replays 2 and 8 min.
- Inference: 65–100 s for the 2,655 public windows (A100 replay and worker-image replica). The sealed set (~21k windows) stays well inside the 60 min limit.
- Development compute on Galvani (`sacct`, 1 Oct – 3 Nov, 1,393 jobs, all in this project's
  workspaces): about 300 GPU-hours (A100 139, RTX 2080 Ti 142, other 22) and 1,300 CPU-core-hours.
  Local runs on a Mac are not metered.

## Pinned environment

torch 2.14.1+cu126, braindecode 1.8.1, MNE 1.13.2, MOABB 1.7.2, numpy 2.5.4, scikit-learn 1.9.1,
neuroai 0.4.0, exca 0.5.29 (`requirements/cloud.txt`; `runs/cloud/requirements.lock.txt` per prepare).
Worker image `lionelyneuro/neural-compet:v3`.

## Validation

Fresh Galvani workspace (3 Nov): an export of `bc46134` plus this recipe, `.venv` and official data
rebuilt by stage 1; the 7 corpora and PROTEUS windows linked in and verified against
`scripts/sealed_r19e_inputs.sha256` (66 files OK).

- **First run** (jobs 3041464–3041471): prepare passed (4 min 32 s), and all 24 EEGNeX / IFNet trunks
  trained (EEGNeX 24–37 min, IFNet 6–15 min per seed). The R19e base stopped: the recipe did not train
  the SincShallowNet trunks that R19e's members start from. Fixed in the script: stage 4 now trains them,
  and the base waits for them. A fresh clone without corpora also stopped the corpora check under
  `pipefail`; fixed.
- **Rebuilt teacher trunks vs originals** (relative weight distance; two different seeds are 0.48–0.78
  apart for EEGNeX and 0.86–0.88 for IFNet). All originals trained on RTX 2080 Ti.
  - IFNet: 0.000 on all 12 seeds (10 rebuilt on A100, 2 on RTX 2080 Ti).
  - EEGNeX: 0.001–0.017 for the 10 seeds rebuilt on RTX 2080 Ti, but 0.446 and 0.476 for seeds 10–11
    rebuilt on A100.
  - Not TF32: EEGNeX seed 10 retrained on A100 with cuDNN TF32 off (job 3043439) is still 0.445 from
    the original, and 0.056 from the A100 TF32-on rebuild. EEGNeX training is reproducible within
    a GPU type and lands about one seed apart across A100 and RTX 2080 Ti.
  - Recipe change: the EEGNeX trunks now train on `2080-galvani` like the originals. The audit
    workspace retrained seeds 10–11 there (job 3043869); the A100 copies are kept in
    `runs/pretrain_a100/` for a hardware-sensitivity check.
- **Relaunch** (jobs 3043395–3043400, 3043869): all stages completed.
  - Trunks: SincShallowNet seeds 0–4 (RTX 2080 Ti, 16–23 min each) are 0.045–0.050 from the originals
    (0.13–0.16 between seeds, as in the 30 Oct audit). EEGNeX seeds 10–11 on RTX 2080 Ti: 0.011 and 0.018.
  - Base R19e (3043396, 16 min, fit 699 s): `submission.py` `25d51a6e…` (frozen), harness agreement 1.0,
    A100 replay files unchanged, in-sample 0.9895.
  - Teacher export (3043397, 44 min): harness agreement 1.0; traced = eager (0.0; batch 1 4.8e-7;
    CUDA 2.0e-6); member weight 0.625; 38 labels differ from the base (staged: 40).
    New `teacher.pt` `154a8286…`, `weights.npz` `eec51eb3…`.
  - Official A100 replay (3043398) 100 s, worker-image replica (3043399) 65 s: files unchanged, same
    scores as each other and as the staged package.
  - Label agreement with the staged cand_nxif through the full predict path (3043400): **755 / 755**
    official-test windows (identical in-sample score 0.9646) and **2,652 / 2,655** windows overall
    (99.89 %; in-sample 0.9766 vs 0.9762).

**Verdict:** the recipe rebuilds cand_nxif from a fresh workspace within the expected tolerance
(≥ 99 % agreement). In-sample scores only check that the predictions agree, not how well the
package generalises. Not covered: an all-A100 rebuild, where EEGNeX lands about a seed away. The
seed-half replication (+2.91 / +2.83) suggests the dev gain holds there, but label agreement
would be lower.
