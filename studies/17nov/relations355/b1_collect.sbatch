#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-rel355-B1-whole
set -euo pipefail
cd "${R355_B1:?}"
sha256sum -c instrument.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
ws=Path(os.environ['R355_B1']); p=json.loads((ws/'b1_guard.json').read_text())
assert not p['labels_opened'] and p['execution_mode']=='B1' and len(p['folds'])==40
assert all(x['eight_actual_orders_bitwise'] and x['unchanged_state'] and x['reset_copy_prefix_interleaving_exact'] for x in p['folds'])
PY
export R355_WS="$R355_B1"
bash "$R355_WS/code/collect.sbatch"
