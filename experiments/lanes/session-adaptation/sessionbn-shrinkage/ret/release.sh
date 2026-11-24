#!/bin/bash
# Stage the Lane I returning release on Galvani: scripts, hash files, raw-trace-patched fold packages (k20/k10/k5). Then submit via
# `ssh mlcloud-galvani "cd $dest && sbatch build.sbatch"` (package build is a CPU job, never the login node).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(dirname "$(dirname "$here")")
base=/mnt/lustre/work/gehler/gfh527; dest=$base/ni26_laneI_returning_20261119
inner=$base/ni26_friday_returning_native_inner_v2_20261119; bundle=$base/ni26_js321_20261117_returning_bundle2
ssh mlcloud-galvani "mkdir -p $dest/pkg $dest/logs"
rsync -a "$here"/patch_k.py "$here"/bn_worker.py "$here"/bn_worker.sbatch "$here"/collect_bn.py "$here"/collect_bn.sbatch "$repo"/laneA/lane_inputs.py mlcloud-galvani:$dest/
ssh mlcloud-galvani "cd $dest && cat > build.sbatch <<EOF
#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/build-%j.out
set -euo pipefail
cd $dest
for f in P04 P10; do rm -rf pkg/\\\${f}_k*; $base/neural-interfaces-2026/.venv/bin/python -W ignore patch_k.py $bundle/\\\${f}_s23/base pkg \\\$f; done
(cd $inner && sha256sum worker.py inputs.py assets.py trace_inputs.py) > parents.sha256
sha256sum lane_inputs.py bn_worker.py bn_worker.sbatch patch_k.py collect_bn.py > source.sha256
EOF"
