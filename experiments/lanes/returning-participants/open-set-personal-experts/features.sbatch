#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=boost_usr_prod
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=logs/features-%j.out
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/leonardo_work/AIFAC_P02_1636/ni26
net=/leonardo_work/AIFAC_P02_1675/ni26_laneB10_20261120
py=$root/.venv312/bin/python
module load python/3.11.7
for corpus in pub sch net; do
    case "$corpus" in
        pub) path=$root/data/proteus_windows.npz ;;
        sch) path=$root/data/scherer2015/windows.npz ;;
        net) path=$net/ext/p0.npz ;;
    esac
    "$py" laneRP/features.py "$corpus" "$path" features --smoke
done
"$py" laneRP/features.py pub "$root/data/proteus_windows.npz" features &
pubpid=$!
"$py" laneRP/features.py sch "$root/data/scherer2015/windows.npz" features &
schpid=$!
seq 0 18 | xargs -P 5 -I@ "$py" laneRP/features.py net "$net/ext/p@.npz" features
wait "$pubpid"
wait "$schpid"
