#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$PROTEUS_ROOT/2026-competition"
mkdir -p "$PROTEUS_ROOT/runs/baseline"
python -m pip freeze > "$PROTEUS_ROOT/runs/baseline/requirements.lock.txt"
git rev-parse HEAD > "$PROTEUS_ROOT/runs/baseline/source_revision.txt"
benchopt prepare tracks/bci_decoding --config tracks/bci_decoding/training.yml \
  2>&1 | tee "$PROTEUS_ROOT/runs/baseline/prepare.log"
python "$PROTEUS_ROOT/scripts/audit_proteus.py" \
  2>&1 | tee "$PROTEUS_ROOT/runs/baseline/data_audit.log"
benchopt run tracks/bci_decoding --config tracks/bci_decoding/training.yml \
  -r 1 --no-cache --no-plot --no-html --no-display --output baseline \
  2>&1 | tee "$PROTEUS_ROOT/runs/baseline/train.log"
python "$PROTEUS_ROOT/scripts/validate_baseline.py"
