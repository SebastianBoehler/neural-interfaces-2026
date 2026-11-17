#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-rel355-whole-guarded
set -euo pipefail
cd "${R355_GUARD:?}"
sha256sum -c source.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
p=json.loads((Path(os.environ['R355_WS'])/'order_guard.json').read_text())
assert p['labels_opened'] is False and p['eight_orders_actual_B1'] and len(p['folds'])==40
assert all(x['eight_actual_orders_bitwise'] and x['unchanged_state'] and x['b1_b64_max']<=1e-6 for x in p['folds'])
PY
bash "$R355_WS/code/collect.sbatch"
