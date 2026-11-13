#!/bin/bash
# Night of 13 Nov (ledger s135-137): arrays released at once, scorings held by afterany, tables in runs/night13/.
# Run on the Leonardo login node from $WORK/ni26 (sbatch only). Usage: scripts/cloud/night13.sh TTT8_JOB OCU9_JOB
set -euo pipefail
TTT8=$1; OCU9=$2; mkdir -p runs/night13; W=scripts/cloud
sc() {  # NAME DEPS CMD -> a CPU scoring job whose table goes to runs/night13/NAME.txt; a failing scorer fails the job
  DEP="--dependency=afterany:$2" $W/leo_cpu.sh "$1" AIFAC_P02_1636 03:00:00 "$3 > runs/night13/$1.txt 2>&1 || { echo SCORER FAILED >> runs/night13/$1.txt; exit 1; }"; }
SM=$(sbatch --parsable --account=AIFAC_P02_1636 -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=00:20:00 -J ni26d-ttt9smoke -o runs/diag/dbg/ttt9smoke-%j.out --wrap "module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate; export NI26_WS=\$PWD OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore; LAB=true python -I studies/13nov/ttt9.py smoke && LAB=stack python -I studies/13nov/ttt9.py smoke; echo EXIT \$?")
echo "ttt9 smoke $SM"
T9T=$(DEP="--dependency=afterok:$SM" $W/leo_array.sh ttt9t AIFAC_P02_1636 01:00:00 "LAB=true python -I studies/13nov/ttt9.py" "0-23")
T9S=$(DEP="--dependency=afterok:$SM" $W/leo_array.sh ttt9s AIFAC_P02_1636 01:00:00 "LAB=stack python -I studies/13nov/ttt9.py" "0-23")
T8F=$($W/leo_array.sh ttt8f AIFAC_P02_1675 01:15:00 "python -I studies/13nov/ttt8.py" "24-71")
echo "arrays: ttt9 true $T9T, ttt9 stack $T9S, ttt8 fresh $T8F"
V="cum10,cum20,cum10e5,cum5e5,cum10lr,cum10c2"; R="python -I studies/13nov/ttt_reduce_score.py"
sc ttt8_s0-5 $TTT8 "PFX=ttt8 VARS=$V FORMS=all6,delta1,delta2 SEEDS=0-5 $R runs/diag/ttt/probs8"
sc ttt8_s6-11 $T8F "PFX=ttt8 VARS=$V FORMS=all6,delta1,delta2 SEEDS=6-11 $R runs/diag/ttt/probs8"
sc ttt8_s12-17 $T8F "PFX=ttt8 VARS=$V FORMS=all6,delta1,delta2 SEEDS=12-17 $R runs/diag/ttt/probs8"
sc ttt9_true $T9T "PFX=ttt9true VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=0-5 $R runs/diag/ttt/probs9_true"
sc ttt9_stack $T9S "PFX=ttt9stack VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=0-5 $R runs/diag/ttt/probs9_stack"
CP="rsync -a --ignore-existing runs/diag/ocunet/probs/ runs/ocunet/probs/ &&"
sc stack_ocu9 $OCU9 "$CP FT_DIR=runs/diag/ttt/probs6 FT_PFX=ttt6 FT_SEEDS=0-5 OCU_SEEDS=0,1,2,3,4,5,6,7,8 STACK_TAG=_ocu9 SUBSETS=FT,OCU,FT+OCU,FT+REST+OCU+MIX python -I studies/13nov/stack.py"
sc stackf $T8F:$OCU9 "$CP FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8 FT_SEEDS=6-11 OCU_SEEDS=3,4,5,6,7,8 STACK_TAG=_fresh SUBSETS=FT,OCU,FT+OCU,FT+REST+OCU+MIX python -I studies/13nov/stack.py"
squeue -u $USER -h -o "%j %T %r" | sort | uniq -c
