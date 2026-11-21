#!/bin/bash
# Run once on Galvani from the new release F directory after transferring the two refill21_niv files.
# Login node performs file staging and sbatch only. Each accepted job is logged immediately.
set -euo pipefail
W=/mnt/lustre/work/gehler/gfh527
F=$W/ni26_niv_finalF_20261121
T=$W/ni26_final_confirm_20261121
C=$W/ni26_niv_finalC_20261121
cd "$F"
mkdir -p logs
set -o noclobber
exec 3>jobs.tsv

submit() {
  local id
  id=$(sbatch --parsable "$@")
  printf '%s\t%s\n' "$id" "$*" >&3
  printf '%s %s\n' "$id" "$*" >&2
  printf '%s\n' "$id"
}

sed 's/arms="full:0"; person=5/arms="full:0,full:s71024"; person=5/' "$C/score.sbatch" > score.sbatch
cp "$C/enrol_run.sbatch" .
test -s "$W/ni26_niv_finalE_20261121/out/hss_n25/p5_c0.npz"

setup=$(submit -J ni26-f-setup refill21_niv.sbatch setup)
declare -a smokes=() arrays=()
for tag in hss hss_n25 hss_n25_fc; do
  smokes+=("$(submit -J "ni26-f-smoke-$tag" --time=15 --dependency="afterok:$setup" score.sbatch "$tag.zip" "$tag" smoke)")
done
gate=$(submit -J ni26-f-gate --dependency="afterok:$(IFS=:; echo "${smokes[*]}")" refill21_niv.sbatch smoke)
for tag in hss hss_n25 hss_n25_fc; do
  arrays+=("$(submit -J "ni26-f-new-$tag" --time=40 --array=0-35%3 --dependency="afterok:$gate" score.sbatch "$tag.zip" "$tag")")
  arrays+=("$(submit -J "ni26-f-ret-$tag" --time=45 --array=0-8%1 --dependency="afterok:$gate" enrol_run.sbatch "$tag" ret)")
done
submit -J ni26-f-collect --dependency="afterok:$(IFS=:; echo "${arrays[*]}")" refill21_niv.sbatch collect >/dev/null

cd "$T"
arms=(
  v66_cp1_cap250=9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed
  final_stack_hard_safe_st_team_b_nine25_tbgraph_fc=7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07
  final_stack_hard_safe_st_team_b_nine15_tbgraph_fc=6e6bbbb841db4ff1ea5e83e4bad75a090ab0a30455fa5ca570063b5ba828d1d4
)
for node in 102 116 121 205 206 218 221 225 226 229; do
  case "$node" in
    102) gpu=(--partition=2080-preemptable-galvani --gres=gpu:rtx2080ti:1) ;;
    1*) gpu=(--partition=2080-galvani --gres=gpu:rtx2080ti:1) ;;
    2*) gpu=(--partition=a100-galvani --gres=gpu:a100:1) ;;
  esac
  prev=
  for start in now 2026-11-21T14:25 2026-11-21T16:35; do
    dependency=()
    if [[ -n "$prev" ]]; then dependency=(--dependency="afterany:$prev"); fi
    prev=$(submit -J "ni26-f-e2e-cn$node" --nodelist="galvani-cn$node" "${gpu[@]}" \
      --begin="$start" "${dependency[@]}" -o 'logs/%x-%j.log' fc_confirm_gpu.sbatch e2e "${arms[@]}")
  done
done
