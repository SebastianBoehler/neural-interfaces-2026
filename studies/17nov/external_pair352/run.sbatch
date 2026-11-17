#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --job-name=ni26-pair352
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
TASK_ROOT=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
TASK_CODE=/mnt/lustre/work/gehler/gfh527/ni26_pair352_20261117_v2
TASK_OUT="$TASK_ROOT/runs/diag/external_pair352_v2"
cd "$TASK_ROOT"
.venv/bin/python - "$TASK_CODE" <<'PY'
import hashlib,json,sys
from pathlib import Path
code=Path(sys.argv[1]); result=json.loads((code/'smoke_result.json').read_text())
assert result['projected_wall_s']<2400 and result['maxrss_kib']<6*1024**2
assert result['causal_17prefix_exact'] and result['training_twin_exact'] and result['all_8_source_orders_pinned']
for name,digest in result['source_hashes'].items():
    assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256(Path('experiments/features.py').read_bytes()).hexdigest()=='360f335d5a9224aad67af9725907cda07ce799fb574a62ae44bb7eb6cbe3dabd'
PY
.venv/bin/python "$TASK_CODE/producer.py" --root "$TASK_ROOT" --out "$TASK_OUT" --smoke "$TASK_CODE/smoke_result.json"
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('experiments/features.py').read_bytes()).hexdigest()=='360f335d5a9224aad67af9725907cda07ce799fb574a62ae44bb7eb6cbe3dabd'
PY
.venv/bin/python "$TASK_CODE/collect.py" --root "$TASK_ROOT" --out "$TASK_OUT"
