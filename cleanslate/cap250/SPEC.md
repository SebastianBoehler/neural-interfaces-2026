# Fold-trained cap250 package contract

The archive has 39 files. Its v5.1b base has 65 indexed nets: 14 active ordinary fits, 20
zero-weight ordinary fits retained in the archive, one task teacher and 30 SessionBN traces
from 24 fits. Eye and Rest add six traces. The remaining source fits are in NPZ arrays.
The scope includes the 20 zero-weight fits because the requested rebuild includes every
archived component. All frozen model-selection settings remain fixed across folds.

## Family inventory and ownership

Each row is a family job per listed seed. The GPU time column is for **all seeds in one
fold**, in hours. Values are planning ranges, except zero for CPU families. Ranges include
a training/export allowance; queue time and external pretraining are separate.

| Family | Seeds | Recipe and inputs | PROTEUS dependence | GPU h/fold |
| --- | --- | --- | --- | ---: |
| `shallow_ea` | 0–4 | R19e `build_sealed.py` / `sealed_r19e.json`, group 0, waveform + full-source-session EA | Fit, EA and scale | .167–.333 |
| `shallow_raw` | 0–4 | R19e group 1, waveform + source RMS | Fit and scale | .167–.333 |
| `fbc_ea_fixed` | 0–4 | R19e group 2, FBC fixed FIR with EA and rotation .05, zero mix weight | Fit, EA and scale | .083–.167 |
| `fbc_raw_fixed` | 0–4 | R19e group 3, FBC fixed FIR raw, zero mix weight | Fit and scale | .083–.167 |
| `fbc_ea_fir` | 0–1 | R19e group 4, FBC learned FIR with EA | Fit, EA and scale | .033–.067 |
| `fbc_raw_fir` | 0–1 | R19e group 5, FBC learned FIR raw | Fit and scale | .033–.067 |
| `sinc_ea` | 0–4 | R19e group 6, SincShallow, external m7r init + source EA, rotation .05, zero mix weight | Fit, EA and scale | .083–.250 |
| `fbl_ea` | 0–4 | R19e group 7, FBLight + EA, rotation .05, zero mix weight | Fit, EA and scale | .083–.167 |
| `sbn_fbc_ea` | 0–5 | `export_sbn.py`, `fb_sbn_rot05`; one source fit, two latent-prior exports at 2 and 5 | Fit, EA and scale | .050–.100 |
| `sbn_fbc_raw` | 0–5 | `export_sbn.py`, `fbcnet_raw_sbn`, export latent prior 20 | Fit and scale | .050–.100 |
| `sbn_sinc` | 0–5 | `export_sbn.py`, `sincshallow_ea_ptm7r_rot05_sbn`, external m7r init, export latent prior 20 | Fit, EA and scale | .100–.200 |
| `sbn_fbl` | 0–5 | `export_sbn.py`, `fblight_ea_rot05_sbn`, export latent prior 5 | Fit, EA and scale | .050–.100 |
| `teacher` | 0 | `export_external_teacher.py`: 12 EEGNeX and 12 IFNet m7r trunks, source causal EA and 38→3 balanced LR C=.01 | EA and logistic fit; trunks external | .250–.750 |
| `eye` | 0–2 | `train_v6_members.py` / `ocunet.py` nomix, 7 ocular traces, 60 epochs | Fit and within-source-visit normalization | .025–.075 |
| `rest` | 0–2 | `train_v6_members.py` / `rest4.py` sh/r1, task windows plus allowed Graz pre-cue windows, latent prior 2 | Fit and both task/rest source visits | .025–.075 |
| **`geometry`** | 0 | `export_ts.py`, `export_tsem.py`, route-v3 fit, `fit_tpri.py`, and R19e event MAD. Source waveform, labels and visits; real tangent reference/head, PC256, task-contrast gallery, PS1, personal TSEM prior | All fitted arrays | 0 |
| `hilbert` | 0 | `export_ts2.py`, Hilbert tangent reference/scaler/head using source windows | All fitted arrays | 0 |
| `identity` | 0 | `add_bio_gate.py`, `bio_gate.fold_params`, `add_own_stage.py`, `ownspace.py`; external spectral calibrator plus source EEG/labels | Spectral galleries and three personal classifiers; base calibrator external | 0 |
| **`count`** | 0 | `fit_cf.py` / `studies/14nov/rundetect`; source waveforms, task+run IDs, nested source cross-fitting and design histogram | Projection, within-run scale, detector thresholds and hypothesis prior | 0 |
| **`hard`** | 0–1 | Optional F16e/F32e hard arm, native FBC waveform student, 40 epochs and CPU actor export | Entire fit; hard loss has no PhysioNet auxiliary | .167–.667 |
| `cbramod` | 0–2 | Optional Lane C2 CBraMod-C, external stage A, 25-epoch source stage B, `cb_export.py` | Stage B weights and visit alignment; stage A external | .150–.300 |

I own the bold families (**geometry, count, hard**), shared code, assembler, graph and package
evaluation. Coordinator assigns the other rows. Ordinary and SessionBN builders should coordinate
shared recipe code without writing each other's family files.

Stock ~1.28–2.95 fit GPU-hours per fold or ~15.4–35.4 for 12 folds. Hardcb ~1.60–3.92
per fold or ~19.2–47.0 for 12. Four orders add an estimated ~0.5–1.2 GPU-hours per variant.
Share stock fits across variants with the same roster and data. Then fit both for the hardcb
fit budget, plus both streams. CPU geometry, queues, input staging and validation determine
calendar time as much as the GPU budget. See the pipeline card for timing sources and scope.

## Fixed job interface

```text
python cleanslate/cap250/families/<family>.py train --people P01,P02,... --seed N --out DIR
    [--first-session-only P04] [--data /absolute/path/proteus_windows.npz]
python cleanslate/cap250/assemble.py --people P01,P02,... --parts DIR --out PKG
    [--first-session-only P04] [--data /absolute/path/proteus_windows.npz]
python cleanslate/cap250/train.py --people P01,P02,... --out PKG
    [--first-session-only P04] [--data /absolute/path/proteus_windows.npz]
```

- Use comma-separated unique IDs. A returning fold includes the person in `--people` and
  passes that person in `--first-session-only`. Derive permitted rows before any fitted
  preprocessing, validation or external/public join. Keep the official channel order and
  original row order. Roster-derived priors, galleries and cached features follow the same
  session restriction. For sources with alternate waveforms, join on the source session.
- `DIR` is an empty writable directory, scoped to one fit. The graph uses
  `PARTS/<family>/s<seed>/`. A direct invocation must be independent of every other fit,
  and it must exit nonzero on missing assets, wrong scope or export failures.
- `--data` defaults to `CS_DATA` when set, else repo `data/proteus_windows.npz`. Repository
  assets resolve under the repo root unless explicitly mounted as `CAP250_ASSETS` (same
  relative paths, e.g. `runs/pretrain`). Paths are absolute inside workers. Use the public
  source fit selection helper `cleanslate/cap250/source.py` once available. Until then, use
  `cleanslate/harness/data.py:train_rows` or equivalent with the same row order.
- GPU families use CUDA inside Slurm. CPU families can use a CPU Slurm job. Each fresh job runs
  the original full training recipe by default; a source-only reduced smoke may use
  `CAP250_SMOKE=1`. Reduced artifacts may exercise assembly and must never enter parity or
  held scores. A family needs an authenticated receipt when assembled, so stage direct fits via
  `worker.py` when assembling them outside the graph. The graph receipt records the source row digest,
  variant, git revision, scheduler ID, runtime and emitted file digests.
- Every NPZ uses NumPy arrays with no object dtype. Except named exceptions below, fitted
  members write `weights.npz` with **the actual keys used by cap250**, already prefixed
  `net<I>_` for indexed nets. Do not inherit any PROTEUS-dependent array or trace from a
  historical package. Publish only the artifact filenames below or diagnostics under
  `diagnostics/`. Do not publish checkpoint caches as package artifacts.

| Families | Exact artifact in DIR |
| --- | --- |
| Ordinary nets in the first eight rows | `weights.npz`, index offsets 0, 5, 10, 15, 20, 22, 24, 29 respectively, plus seed. Use the matching `build_sealed.py` array exporter and fold-fitted `align,k,G,scale,shrink`. |
| `teacher` | `weights.npz` with `net34_{align,k,G,scale,shrink,jit_file}` and `teacher.pt`. Teacher TorchScript includes the refit map. |
| `sbn_fbc_ea` | `weights.npz`, indices 35+seed and 41+seed, with `net<I>_{align,k,G,scale,shrink,sjit_file}`; traces `sbn_k2_sN.pt`, `sbn_k5_sN.pt`. File-name k is latent prior. Input alignment k follows the original exported base input, as for all SBN rows. |
| `sbn_fbc_raw` | Same keys at 47+seed, trace `sbn_fbcnet_raw_k20_sN.pt`. |
| `sbn_sinc` | Same keys at 53+seed, trace `sbn_sincshallow_ea_ptm7r_rot05_k20_sN.pt`. |
| `sbn_fbl` | Same keys at 59+seed, trace `sbn_fblight_ea_rot05_k5_sN.pt`. |
| `eye` | `eye_sN.pt` with (B,7,480) input and 3 logits. |
| `rest` | `rest_sN.pt` and `rest_sN.npz` with `align,k,G,scale,shrink`; these five arrays must agree across seeds. |
| `geometry` | `weights.npz` with `ts_*`, `tsem_*`, `route_*`, `em_ev_g`; `tpri.npz` with `tpri,names,sessions`. `route_names` and prior rows use sorted roster. |
| `hilbert` | `weights.npz` with all `ts2_*` keys. |
| `identity` | `weights.npz` with all `bio_*`, `own_*` keys; per-person axis matches sorted roster. |
| `count` | `cf_params.npz` with `cf_*` keys exactly as `fit_cf.py` emits. |
| `hard` | `hard_s1c_sN.pt` in the CPU TorchScript format from `incremental_model.export_actor`; normalized official-order probabilities and `reset()`. |
| `cbramod` | `cb_sN.pt` in `submissions/final_stack/cb_export.py:export_blob` half-storage format. |

External m7r provenance is the seven-corpus recipe in `scripts/audit_nxif.sh` and the
[trunk audit](../../docs/AUDIT_NXIF.md). The external spectral calibrator uses the six corpora
named in `experiments/bio_id.py`. CBraMod-C stage A uses Lane C2 external datasets, including
Scherer; it is eligible for PROTEUS folds. Any independent Scherer experiment needs arm B.
Record hashes and verify corpus scope before reuse. The sealed `teacher.pt` contains its
PROTEUS head and is ineligible as an external initializer.

## Assembly and scheduling

- Use `lean/tools/build6.py` with the authenticated stock switches in
  [CAP250_REBUILD.md](../../friday/CAP250_REBUILD.md). Recreate the v5.1b data file from
  family outputs and fixed scalars. Load the official Solver before publishing a fresh PKG.
- `CAP250_HARD=0|1`, `CAP250_CB=0|1`, `CAP250_TTT=stock|off|tttlite5|tttsparse20|tttlight|tttheavy`.
  Stock defaults to `0/0/stock`; hardcb is `1/1/stock`. Hard alone replaces 5/6 of the outer
  raw slot; CB alone replaces 1/6; both use 0.8 hard + 0.2 CB at 5/6. Use the existing native
  package source fragments. Never edit frozen submission files.
- `geometry` consumes `CAP250_ROUTE_OOF`, the graph-written JSON manifest of per-person
  nested B2 posterior files. The outer fit cannot reuse historical B2 predictions because
  their training rosters can include the outer held person. Each nested scope fits the
  active B2 member set, two cores and PCA. Shared jobs require identical source rosters and
  session restrictions. B2 basis jobs use `worker.py --base-only`.
- `graph.py` emits one Slurm job per family×seed, then package assembly `afterok` on every
  required fit, official harness streams `afterok` on their package, and score `afterok` on
  all requested streams. Keep each fold DAG on one cluster. Stage and compare results across
  sites without cross-cluster Slurm dependencies. The score consumes the harness `jobs.tsv`
  and `pred_<mode>_<person>.npz` schema. Run `gather.py` on the returned shard directories
  to validate and score complete four-order results. The standard harness entrypoint uses `train.py`
  inside an existing Slurm job, which can fit sequentially as an explicit entry mode. Use
  the graph for parallel training and avoid waiting in an allocated GPU job.
- First smoke reduced jobs and package loading, then fresh all-ten stock training plus
  public decision equality with the sealed ZIP in recording order and the first requested
  full-shuffle order. Exact archive reassembly can isolate an
  assembler mismatch. Fresh neural training has documented GPU nondeterminism; compare
  and report all decisions and scores, including mismatches. An archive-only equality
  result cannot stand in for a fresh-train comparison.

## Leonardo launch

From the Git checkout, run `bash cleanslate/cap250/stage_leonardo.sh <revision>`
to stage committed code and the pinned official track under
`$WORK/ni26/cap250/releases/<revision>`. Place source data, external trunk checkpoints,
the identity model and the pinned hard recipe under `$WORK/ni26/cap250/assets`.
Keep the rest export accessible from `RELEASE/data` for the family trainer and
set `CAP250_CB_CHECKPOINT` for CBraMod. The compute nodes read these files locally.

```bash
module load python/3.11.7
root="$WORK/ni26/cap250"
release="$root/releases/$revision"
assets="$root/assets"
export CAP250_CB_CHECKPOINT="$assets/runs/cb_stage_a/cbramod.safetensors"
python -B "$release/cleanslate/cap250/graph.py" --release "$release" \
  --assets "$assets" --revision "$revision" --run "$root/runs/$tag" \
  --variants stock,hardcb --orders 71000,71001,71002,71003 --submit
```

The default Leonardo target uses AIFAC_P02_1636, `boost_usr_prod` A100s and the
existing `$WORK/ni26/.venv312` (CUDA 12.6 and Benchopt 1.10.0). Set `--account AIFAC_P02_1675` for that allocation.
The graph submits all prerequisites, writes `jobs.jsonl`, then releases the submitted
Slurm jobs. Pass the all-ten acceptance collector's same-cluster Slurm ID as
`--afterok` when pre-queuing production. Use `--site galvani` with the Galvani image
and asset paths for overflow.
Use the graph to run the normal parallel package builds. `train.py` also accepts the
same roster directly in a compute allocation with the full staged release and assets.
