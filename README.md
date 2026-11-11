# Neural Interfaces 2026 — Track 2, BCI decoding

Three-class EEG decoding (motor imagery, mental subtraction, word generation) on PROTEUS, for the EEG/EMG Foundation
Challenge 2026. Private repository: [SebastianBoehler/neural-interfaces-2026](https://github.com/SebastianBoehler/neural-interfaces-2026).

**Start with [docs/STATE.md](docs/STATE.md).** It holds the current score, the hard bounds, the system, the open
levers and the decisions waiting. Everything under `docs/archive/` is history and may contradict it.

## Current documents

| File | What it holds |
|---|---|
| [docs/STATE.md](docs/STATE.md) | Entry point: standing, hard bounds, system, how results are measured, open levers |
| [docs/LEDGER.md](docs/LEDGER.md) | Index of what is adopted, open and closed; new runs are pre-registered here |
| [docs/PREDICTIONS_11NOV.md](docs/PREDICTIONS_11NOV.md) | Full record of 11 Nov: 33 pre-registered sections with predictions, results and verdicts |
| [docs/SEALED_UPLOADS.md](docs/SEALED_UPLOADS.md) | Upload log, packages, the 10 Nov rescoring |
| [docs/REVIEW_5NOV.md](docs/REVIEW_5NOV.md) | Directions closed 5–10 Nov, with the method class each was tested on |
| [docs/AUDIT_SBN.md](docs/AUDIT_SBN.md), [AUDIT_NXIF.md](docs/AUDIT_NXIF.md) | Rebuild recipes for the uploaded packages (members, teacher, fast runtime) |
| [docs/COMPETITION.md](docs/COMPETITION.md), [OFFICIAL_CONTRACT.md](OFFICIAL_CONTRACT.md) | Rules, dates, quotas; the official evaluation contract |
| [docs/CLOUD_RUNS.md](docs/CLOUD_RUNS.md), [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) | Galvani job and worker-replica procedures |

## Layout

- `2026-competition/`: official source, a pinned submodule (`a66bc23`). Never edited.
- `submissions/ensemble_v4/submission.py`: the code of the best upload (routing v4.1). `ensemble_fast/` is its base;
  `ensemble/` reproduces the three earlier uploads. All three are SHA-pinned: never edited in place.
- `replay/`: the dev chain replay. `python replay/check.py` reproduces the six reference scores in about a minute.
- `studies/11nov/`: one-off study scripts of 11 Nov (fine-tuning, identity routing, new feature cores, diagnostics).
- `experiments/`, `scripts/`: training, feature and export code that the packages and their audit recipes use.
  Unreferenced code was removed on 11 Nov (`docs/archive/README.md` lists it; `git show 05d2335:<path>` recovers it).
- `evidence/`: compact tracked results. `data/`, `runs/`, `.venv/` are local only and gitignored.

## Setup

```bash
git clone --recurse-submodules https://github.com/SebastianBoehler/neural-interfaces-2026.git
cd neural-interfaces-2026
bash scripts/setup.sh
```

Requires Git and [uv](https://docs.astral.sh/uv/). Source `scripts/environment.sh` from Bash, not zsh.

## Rules of work

- Local Python jobs run only through `scripts/run_capped.sh`, one at a time. Heavy work goes to Galvani.
- Every run is pre-registered in the ledger (mechanism, numeric prediction, refutation rule) before it is scored.
- Uploads need the user's explicit go-ahead and a live check of phase and quota.
- Datasets, environments, credentials and checkpoints are never committed.
