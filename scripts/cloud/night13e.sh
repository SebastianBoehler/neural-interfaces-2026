#!/bin/bash
# 13 Nov, TTT12 (ledger s142): fine-tuning on labels independent of the tuned nets. The smoke runs seed 0 (ttt6 smoke is fixed to it); the seed >= 6 trunk path was verified live in ttt7f (sinc seed 6).
set -euo pipefail
W=scripts/cloud; R="python -I studies/13nov/ttt_reduce_score.py"
SM=$(sbatch --parsable --account=AIFAC_P02_1675 -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=00:25:00 -J ni26d-ttt12smoke -o runs/diag/dbg/ttt12smoke-%j.out --wrap "module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate; export NI26_WS=\$PWD OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore; LAB=indep python -I studies/13nov/ttt9.py smoke && LAB=agree python -I studies/13nov/ttt9.py smoke && LAB=truelow python -I studies/13nov/ttt9.py smoke")
echo smoke $SM
for LAB in indep agree truelow; do
  A=$(DEP="--dependency=afterok:$SM" $W/leo_array.sh ttt12$LAB AIFAC_P02_1675 01:00:00 "LAB=$LAB python -I studies/13nov/ttt9.py" "0-47")
  for S in 0-5 6-11; do
    DEP="--dependency=afterany:$A" $W/leo_cpu.sh ttt12_${LAB}_s$S AIFAC_P02_1636 03:00:00 "PFX=ttt9$LAB VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=$S $R runs/diag/ttt/probs9_$LAB > runs/night13/ttt12_${LAB}_s$S.txt 2>&1 || { echo SCORER FAILED >> runs/night13/ttt12_${LAB}_s$S.txt; exit 1; }" >/dev/null
  done; echo "$LAB array $A"
done
