#!/usr/bin/env bash
# Run from $CAP250_RUN/src on Leonardo. This script only stages directories and submits Slurm jobs.
set -euo pipefail
W=/leonardo_work/AIFAC_P02_1636/ni26
REL=$W/cap250/releases/15a639718601e959015fd533fa9106476614aa8c
SRC=$W/cap250/runs/production-20261120-32a94ab7
RUN=$W/cap250/runs/redteam-21nov
CODE=$RUN/src/scripts/redteam
SAFE=$W/cap250/runs/overnight-21nov-pkgcmp/hardcb_safe
export CAP250_RELEASE=$REL CAP250_RUN=$RUN CAP250_ASSETS=$W/cap250/assets CAP250_DATA=$W/data/proteus_windows.npz
export CAP250_PYTHON=$W/.venv312/bin/python CAP250_SITE=leonardo CAP250_REV=15a63971 CAP250_SMOKE=0
export CAP250_CB_STAGEA=/leonardo_work/AIFAC_P02_1636/ni26_aux_20261120/runs/cb_stage_a
export CAP250_CB_CHECKPOINT=$W/cap250/assets/runs/cb_stage_a/cbramod.safetensors
SB=(sbatch --parsable --account=AIFAC_P02_1636 --partition=boost_usr_prod --gres=gpu:1 --chdir="$RUN" --output="$RUN/logs/%x-%j.out")
orders=71000,71001,71002,71003
people=(P04 P10)
wrapper="bash $REL/cleanslate/cap250/slurm.sh"
mkdir -p "$RUN/logs" "$RUN/smoke"

submit() {
  local name=$1 dep=$2 minutes=$3 command=$4 job
  local opts=()
  [[ -z $dep ]] || opts=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
  job=$("${SB[@]}" "${opts[@]}" --job-name="$name" --cpus-per-task=4 --mem=16G --time="$minutes" --wrap="set -e; $command")
  printf '%s\t%s\t%s\n' "$job" "$name" "$dep" >> "$RUN/jobs.tsv"
  printf '%s %s\n' "$job" "$name" >&2
  printf '%s\n' "$job"
}

stream() {
  local name=$1 dep=$2 pkg=$3 person=$4 sessions=$5 order=$6 out=$7 arm=$8 then=${9:-}
  local program=cleanslate/cap250/stream_fold.py extra=""
  if [[ -n $arm ]]; then
    program=$CODE/trace_stream.py
    extra="--arm $arm --device cuda:0"
  fi
  mkdir -p "$(dirname "$out")"
  submit "$name" "$dep" 45 "$wrapper $program --pkg $pkg --held $person --sessions $sessions --orders $order --data $CAP250_DATA --out $out $extra $then"
}

case ${1:?first or followup} in
  first)
    build_h1=$(submit rt-h1-build "" 20 "bash $CODE/prepare.sh h1 $SRC $RUN")
    r0="$RUN/smoke/r0-P04.npz"
    smoke_h1=$(stream rt-h1-r0z "$build_h1" "$RUN/R0z/ret_P04/pkg" P04 2,3 71000 "$r0" "" \
      "&& $wrapper $CODE/check_labels.py $r0 $SRC/hardcb/pred_ret_P04.npz")
    for dose in g6:route_slot_g=6 g12:route_slot_g=12 beta02:own_beta=0.2 beta04:own_beta=0.4 kappa40:v6_tpri_kappa=40; do
      arm=${dose%%:*}
      build_dose=$(submit "rt-h5-$arm-build" "" 20 "bash $CODE/prepare.sh dose $SRC $RUN $dose")
      p01="$RUN/smoke/$arm-P01.npz"
      smoke_dose=$(stream "rt-h5-$arm-p01" "$build_dose" "$RUN/$arm/new_P01/pkg" P01 all 71000 "$p01" "$arm" \
        "&& $wrapper $CODE/check_labels.py $p01 $SRC/hardcb/pred_new_P01.npz")
      for who in "${people[@]}"; do
        stream "rt-h5-$arm-$who" "$smoke_dose" "$RUN/$arm/ret_$who/pkg" "$who" 2,3 "$orders" "$RUN/$arm/pred_ret_$who.npz" "$arm" >/dev/null
      done
    done
    for arm in R1a R5 R2 R4; do
      for who in "${people[@]}"; do
        stream "rt-h1-$arm-$who" "$smoke_h1" "$RUN/$arm/ret_$who/pkg" "$who" 2,3 "$orders" "$RUN/$arm/pred_ret_$who.npz" "$arm" >/dev/null
      done
    done
    ;;
  followup)
    smoke_h1=$(awk '$2 == "rt-h1-r0z" { print $1 }' "$RUN/jobs.tsv")
    for who in "${people[@]}"; do
      out="$RUN/R0z/pred_ret_$who.npz"
      stream "rt-h1-r0z-$who" "${smoke_h1:?}" "$RUN/R0z/ret_$who/pkg" "$who" 2,3 "$orders" "$out" "" \
        "&& $wrapper $CODE/check_labels.py $out $SRC/hardcb/pred_ret_$who.npz" >/dev/null
    done
    build_h2=$(submit rt-h2-build "" 20 "bash $CODE/prepare.sh leak $SRC $RUN")
    for who in "${people[@]}"; do
      stream "rt-h2-leak-$who" "$build_h2" "$RUN/leak_all/ret_$who/pkg" "$who" 2,3 "$orders" "$RUN/leak_all/pred_ret_$who.npz" "" >/dev/null
    done
    stream rt-h3-insample-P04 "" "$SRC/stock/ret_P04/pkg" P04 1 71000 "$RUN/insample_P04.npz" ""
    for who in "${people[@]}"; do
      out="$RUN/hardcb_safe/pred_ret_$who.npz"
      stream "rt-ref-safe-$who" "" "$SAFE/ret_$who/pkg" "$who" 2,3 "$orders" "$out" "" \
        "&& $wrapper $CODE/check_labels.py $out $SRC/hardcb/pred_ret_$who.npz" >/dev/null
    done
    ;;
  *) exit 2 ;;
esac
