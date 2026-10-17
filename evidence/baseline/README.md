# Baseline evidence snapshot

These files preserve the completed 14 October 2026 CPU baseline run.
They contain metadata and metrics, not EEG recordings or model weights.

- `source_revision.txt`: exact official source revision.
- `environment.json`: core installed package versions.
- `data_audit.json`: channel metadata, participant split and window counts.
- `validation.json`: replay checks, metrics and original archive checksum.

Paths in `validation.json` refer to the original local `runs/baseline/` outputs.
Those generated outputs are deliberately ignored by Git and are not included
in a fresh clone. Reproduce them with `bash scripts/run_baseline.sh`.
Dependency versions are tracked in `requirements/baseline-macos-arm64.txt`.
