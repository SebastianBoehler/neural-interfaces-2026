#!/bin/bash
# Teacher line (ledger s145 D): install MOABB 1.7.2 into .venv on a serial node (internet), then download and window the
# seven m7 corpora with scripts/export_moabb.py, one serial job per subject chunk (8 GB each: the serial QOS caps a
# user's memory). Raw files: data/moabb_raw ($MNE_DATA); windows: data/<corpus>/windows_<lo>-<hi>.npz.
# Usage (Leonardo login node, $WORK/ni26): scripts/cloud/leo_corpora.sh
set -euo pipefail
mkdir -p runs/diag/corpora data/moabb_raw
I=$(sbatch --parsable --account=AIFAC_P02_1675 -p lrd_all_serial --cpus-per-task=2 --mem=8G --time=00:30:00 -J ni26-moabb -o runs/diag/corpora/install-%j.out --wrap "module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; .venv/bin/python -m pip install -q moabb==1.7.2 && .venv/bin/python -c 'import moabb; print(moabb.__version__)'; rc=\$?; echo EXIT \$rc; exit \$rc")
echo "install $I"
for spec in PhysionetMI:1-55 PhysionetMI:56-109 Cho2017:1-26 Cho2017:27-52 Lee2019_MI:1-14 Lee2019_MI:15-28 Lee2019_MI:29-42 Lee2019_MI:43-54 \
            Dreyer2023:1-44 Dreyer2023:45-87 Schirrmeister2017:1-7 Schirrmeister2017:8-14 Shin2017A:1-29 Shin2017B:1-29; do
  n=${spec%%:*}; s=${spec#*:}
  sbatch --parsable --dependency=afterok:$I --account=AIFAC_P02_1675 -p lrd_all_serial --cpus-per-task=2 --mem=8G --time=04:00:00 -J ni26-corp-$n -o runs/diag/corpora/$n-$s-%j.out \
    --wrap "module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; export MNE_DATA=\$PWD/data/moabb_raw PYTHONWARNINGS=ignore; .venv/bin/python -I scripts/export_moabb.py $n --subjects $s; rc=\$?; echo EXIT \$rc; exit \$rc" > /dev/null
done
squeue -u $USER -h -o "%j %T %r" | grep corp | sort | uniq -c
