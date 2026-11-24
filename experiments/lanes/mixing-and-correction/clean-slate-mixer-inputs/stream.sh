#!/bin/bash
# Revisit streams on Galvani (run on the login node: only sbatch). REV = release dir holding revisit/ and out/.
# usage: stream.sh tc KEY ARRAY [VARIANTS]   -> Team C C1 packages, key e.g. 'within|0|all_loso', out/tc_<tag>_P*.npz
#        stream.sh tb KEYS ARRAY             -> Team B R1 packages in the worker image, out/tb_P*.npz
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
REV=${REV:-$base/ni26_revisit_20261121}; TC=$base/ni26_teamC_20261120; TB=$base/ni26_csB_20261120/r1
mode=${1:?}; key=${2:?}; arr=${3:?}; var=${4:-A1,A2}
tag=$(echo "$key" | cut -d'|' -f1,2 | tr '|' '_')
g="--account=gehler -p ${PART:-a100-galvani,a100-preemptable-galvani,2080-galvani,2080-preemptable-galvani} --gres=gpu:1 -c 8 --mem=40G --exclude=${EXCL:-galvani-cn122,galvani-cn116} -t ${T:-01:30:00}"
mkdir -p "$REV/out" "$REV/logs"
P='P=$(printf "P%02d" $((SLURM_ARRAY_TASK_ID+1)))'
case $mode in
tc)
  sbatch --parsable $g --array="$arr" -J rev-tc-$tag -o "$REV/logs/tc-$tag-%A_%a.out" --wrap "set -e; $P; \
    export PATH=$base/neural-interfaces-2026/.venv/bin:\$PATH OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1; cd $TC; \
    python -u $REV/revisit/stream_tc.py --root $TC --pkg work/C1/pkg_\$P --held \$P --key '$key' --variants $var --out $REV/out/tc_${tag}_\$P.npz" ;;
tb)
  sbatch --parsable $g --array="$arr" -J rev-tb -o "$REV/logs/tb-%A_%a.out" --wrap "set -e; $P; \
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=4,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/out,PYTHONPATH=/cs/harness:/cs \
    --bind $TB/cleanslate:/cs:ro,$TB/run:/work:ro,$REV/out:/out:rw,$REV/revisit:/rev:ro,$TC/runs/shuffle_perms.npz:/perms/shuffle_perms.npz:ro,$base/neural-interfaces-2026/2026-competition:/official:ro,$base/neural-interfaces-2026/data/proteus_windows.npz:/data/proteus_windows.npz:ro \
    --pwd /out $base/neural-interfaces-2026/runs/cloud/ni26_worker.sif \
    python -B /rev/stream_tb.py --pkg /work/new_\$P/pkg --held \$P --keys '$key' --out /out/tb_\$P.npz" ;;
esac
