#!/bin/bash
# 13 Nov 02:25 UTC wave (ledger s143): fresh OCU and REST seeds, then the fresh-everything package replay (PKG3).
set -euo pipefail
W=scripts/cloud; mkdir -p runs/diag/dbg
O=${OCU_JOB:-$($W/leo_array.sh ocu17 AIFAC_P02_1675 01:30:00 "python -I studies/12nov/ocunet.py" "19,21,23,25,27,29,31,33,35")}  # OCU_JOB: reuse an array already submitted
E=$(sbatch --parsable --account=AIFAC_P02_1675 -p lrd_all_serial --cpus-per-task=4 --mem=16G --time=02:00:00 -J ni26-export45 -o runs/diag/export45-%j.out --wrap "unset PYTHONPATH; . .venv312/bin/activate; export BENCHOPT_DATA_HOME=\$PWD/data PYTHONWARNINGS=ignore; python -I scripts/export_windows.py -4.5 && test -s data/proteus_windows_start-4.5.npz; rc=\$?; echo EXIT \$rc; exit \$rc")
S=$(sbatch --parsable --dependency=afterok:$E --account=AIFAC_P02_1675 -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=00:25:00 -J ni26d-rest4smoke -o runs/diag/dbg/rest4smoke-%j.out --wrap "module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate; export NI26_WS=\$PWD OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore; python -I studies/11nov/rest4/rest4.py smoke; rc=\$?; echo EXIT \$rc; exit \$rc")
R=$(DEP="--dependency=afterok:$S" $W/leo_array.sh rest4f AIFAC_P02_1675 02:00:00 "python -I studies/11nov/rest4/rest4.py" "21,22,27,28,33,34,39,40,45,46,51,52")
CP="rsync -a --ignore-existing runs/diag/ocunet/probs/ runs/ocunet/probs/ && rsync -a --ignore-existing runs/diag/rest4/probs/ runs/rest4/probs/ &&"
P=$(DEP="--dependency=afterany:$O:$R" $W/leo_cpu.sh pkg_fresh2 AIFAC_P02_1636 04:00:00 "$CP FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8 FT_SEEDS=6-11 OCU_SEEDS=9,10,11,12,13,14,15,16,17 REST_SEEDS=3,4,5,6,7,8 python -I studies/13nov/pkg.py > runs/night13/pkg_fresh2.txt 2>&1 || { echo SCORER FAILED >> runs/night13/pkg_fresh2.txt; exit 1; }")
echo "ocu $O | export $E -> rest4 smoke $S -> rest4 $R | pkg_fresh2 $P"
