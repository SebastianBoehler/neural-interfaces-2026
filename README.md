# Neural Interfaces 2026 — Track 2, BCI decoding

Three-class EEG decoding (motor imagery, mental subtraction, word generation) on PROTEUS, for the EEG/EMG Foundation
Challenge 2026. Private repository: [SebastianBoehler/neural-interfaces-2026](https://github.com/SebastianBoehler/neural-interfaces-2026).

**Start with [docs/STATE_11NOV.md](docs/STATE_11NOV.md).** It holds the current score, packages, running jobs and open
decisions. Everything under `docs/archive/` is history and may contradict it.

## Current documents

| File | What it holds |
|---|---|
| [docs/STATE_11NOV.md](docs/STATE_11NOV.md) | Entry point: standing, pipeline, running jobs, decisions, open directions |
| [docs/PREDICTIONS_11NOV.md](docs/PREDICTIONS_11NOV.md) | Pre-registered ledger: every run since 11 Nov with prediction, result and verdict |
| [docs/SEALED_UPLOADS.md](docs/SEALED_UPLOADS.md) | Upload log, staged packages, the 10 Nov rescoring |
| [docs/REVIEW_5NOV.md](docs/REVIEW_5NOV.md) | Every direction closed since 5 Nov, with the method class it was tested on |
| [docs/AUDIT_SBN.md](docs/AUDIT_SBN.md), [AUDIT_NXIF.md](docs/AUDIT_NXIF.md), [AUDIT_M7R4.md](docs/AUDIT_M7R4.md) | Rebuild recipes for uploaded and staged packages |
| [docs/COMPETITION.md](docs/COMPETITION.md) | Rules, dates, quotas, warm-up history |
| [OFFICIAL_CONTRACT.md](OFFICIAL_CONTRACT.md), [BASELINE_RESULT.md](BASELINE_RESULT.md) | The official evaluation contract and the reproduced linear baseline (37.75 %) |
| [docs/CLOUD_RUNS.md](docs/CLOUD_RUNS.md), [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) | Galvani job and worker-replica procedures |

## Layout

- `2026-competition/`: official source, a pinned submodule (`a66bc23`). Never edited.
- `submissions/ensemble/submission.py`: lean code that reproduces the three uploaded packages byte for byte.
- `submissions/ensemble_fast/submission.py`: the same outputs at about half the runtime; the base for future uploads.
- `experiments/`, `scripts/`: training, feature, replay and export code (audit recipes depend on it).
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
