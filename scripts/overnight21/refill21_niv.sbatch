#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/refill-niv-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
W=/mnt/lustre/work/gehler/gfh527
G=$W/neural-interfaces-2026
run() {
  apptainer exec --cleanenv --env PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1 \
    --bind "$W,$G/data/scherer2015/windows.npz:/external.npz:ro" \
    --pwd "$PWD" "$G/runs/cloud/ni26_worker.sif" python -B "$@"
}
case "${1:?}" in
  setup)
    C=$W/ni26_niv_finalC_20261121
    T=$W/ni26_final_confirm_20261121
    mkdir -p out zips
    cp "$C"/{scherer_rows.tsv.gz,cohort_a.npz,inputs.sha256} .
    cp -r "$C"/{niv,studies,replay} .
    grep -q 'full:0,full:s71024' score.sbatch
    ln -s "$(readlink -f "$C/zips/hss.zip")" zips/hss.zip
    ln -s "$(readlink -f "$C/zips/hss_n25.zip")" zips/hss_n25.zip
    ln -s "$T/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip" zips/hss_n25_fc.zip
    sed -i 's/SEEDS = tuple(71200 + k/SEEDS = tuple(71032 + k/' niv/orders.py
    run -c 'import sys; sys.path.insert(0, "niv"); import orders; assert orders.SEEDS == tuple(range(71032, 71040))'
    sed -i '/ niv\/orders.py$/d; / zips\/hss_n15.zip$/d' inputs.sha256
    sha256sum niv/orders.py >> inputs.sha256
    echo '7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07  zips/hss_n25_fc.zip' >> inputs.sha256
    sha256sum --quiet -c inputs.sha256
    run refill21_niv.py enrol
    sha256sum zips/enrolled/*.zip > enrolled.sha256
    ;;
  smoke)
    run refill21_niv.py smoke
    ;;
  collect)
    run niv/collect.py out out/report_hss_n25_fc_vs_hss.json --a hss_n25_fc --b hss --data /external.npz
    run refill21_niv.py collect
    ;;
  *) exit 2 ;;
esac
