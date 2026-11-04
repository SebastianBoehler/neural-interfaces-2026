#!/usr/bin/env bash
# Audit rebuild of the sealed arm cand_m7r4 (docs/AUDIT_M7R4.md) on the Tuebingen ML Cloud.
# Run from the root of a clone at the audited commit, with the official submodule at a66bc23:
#   bash scripts/audit_m7r4.sh [OUT_DIR]                  default runs/sealed/audit_m7r4
#   NI26_REF=<staged package dir> bash scripts/audit_m7r4.sh   also compares labels with it
# Stages 1-3, 5 and 7 are those of scripts/audit_nxif.sh; only the teacher's trunks (4) and export (6) differ.
# Every stage is a Slurm job chained by afterok dependencies and is skipped when its output exists:
#   1 prepare   environment (requirements/cloud.txt, torch 2.14.1+cu126), official PROTEUS data, data audit
#   2 windows   data/proteus_windows.npz (scripts/export_windows.py; export_moabb.py needs it)
#   3 corpora   7 MOABB motor-imagery corpora, chunked as recorded in scripts/sealed_r19e_inputs.sha256
#   4 trunks    recipe m7r (rot 0.05, 30 epochs, one head per corpus), seeds 0-11 of SincShallowNet, FBCNet,
#               FBLightConvNet and ShallowFBCSPNet, each on the GPU type that trained the original;
#               SincShallowNet 0-4 are also the R19e member init (as in audit_nxif.sh)
#   5 base      R19e with the frozen submission.py (build_sealed.py scripts/sealed_r19e.json, A100)
#   6 teacher   export_external_teacher.py: the four families' heads, seeds 0-11, C 0.01, weight 4
#   7 checks    official A100 replay, worker-image replica, full-path label agreement with NI26_REF
# GPU training is not bit-deterministic: expect a few windows to differ, not identical files.
set -euo pipefail
OUT=${1:-runs/sealed/audit_m7r4}
BASE=${OUT}_base_r19e_g
X=${NI26_EXCLUDE:-galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122}
G="--account=gehler --exclude=$X --gres=gpu:1"; GA="$G -p a100-galvani,2080-galvani"
E='export PATH=$PWD/.venv/bin:$PATH BENCHOPT_DATA_HOME=$PWD/data'
CORPORA=physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b
dep() { if [ -n "$1" ]; then echo "--dependency=afterok$1"; fi; }
sub() { sbatch --parsable "$@"; }
mkdir -p runs/cloud runs/sealed

# NI26_PREP_OPTS overrides prepare's partition when cpu-galvani is full, e.g. "-p 2080-galvani --gres=gpu:1"
P=""; [ -f runs/cloud/PREPARED ] || P=":$(sub --exclude="$X" ${NI26_PREP_OPTS:-} scripts/cloud/prepare.sbatch)"
W="$P"; [ -f data/proteus_windows.npz ] || W=":$(sub $GA $(dep "$P") --time=01:00:00 --mem=32G \
  -o runs/cloud/audit-windows-%j.out --wrap "$E; python scripts/export_windows.py")"

C="$W"  # dataset, n subjects, chunk, folder, expected files
while read -r ds n k dir files; do
  have=$( (ls data/"$dir"/windows_*.npz 2>/dev/null || true) | wc -l)
  [ "$have" -eq "$files" ] || C="$C:$(sub --exclude="$X" --array=0-$(( (n - 1) / k )) $(dep "$W") \
    -o runs/cloud/audit-$dir-%A_%a.out scripts/cloud/export_moabb.sbatch "$ds" "$n" "$k")"
done <<'EOF'
PhysionetMI 109 10 physionetmi 11
Cho2017 52 6 cho2017 9
Lee2019_MI 54 3 lee2019_mi 18
Dreyer2023 87 10 dreyer2023 9
Schirrmeister2017 14 2 schirrmeister2017 7
Shin2017A 29 5 shin2017a 6
Shin2017B 29 5 shin2017b 6
EOF

pt() {  # arch, then "seeds:partition" groups: one array per group unless all its checkpoints exist
  local a=$1 g r s; shift
  for g in "$@"; do
    r=${g%%:*}
    for s in $(seq "${r%-*}" "${r#*-}"); do
      [ -f "runs/pretrain/${a}_ea-m7r-s$s.pt" ] || { echo -n ":$(sub $G -p "${g#*:}" --array="$r" \
        --time=05:00:00 $(dep "$C") -o runs/cloud/audit-$a-%A_%a.out \
        scripts/cloud/pretrain.sbatch "$a" ea m7r $CORPORA "" "" "--rot 0.05")"; break; }
    done
  done
}
S="$(pt sincshallow 0-4:a100-galvani,2080-galvani)"  # init of the R19e member sincshallow_ea_ptm7r_rot05
# The originals' GPU types (sacct, 29 Oct - 1 Nov). EEGNeX moved about one seed apart between A100 and
# RTX 2080 Ti (docs/AUDIT_NXIF.md), so every seed trains where its original did. NI26_V100 replaces the
# V100 partition where it is unavailable (docs/AUDIT_M7R4.md gives the measured effect).
PV=${NI26_V100:-v100-galvani}
T="$(pt sincshallow 5-11:a100-galvani)$(pt fbcnet 0-5:$PV 6-11:a100-galvani)"
T="$T$(pt fblight 0-2:$PV 3-5:2080-galvani 6-11:a100-galvani)"
T="$T$(pt shallow 0-4:a100-galvani 5:2080-galvani 6-11:a100-galvani)"

CFG=scripts/sealed_r19e.json  # its reference package exists only in our workspace; drop it on a fresh clone
if [ ! -d runs/sealed/ensemble_all_r19e_em_r15 ]; then
  CFG=runs/sealed/sealed_r19e_noref.json
  python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c.pop('reference'); json.dump(c, open(sys.argv[2], 'w'), indent=1)" \
    scripts/sealed_r19e.json "$CFG"
fi
B=""; [ -f "$BASE/weights.npz" ] || B=":$(sub --exclude="$X" $(dep "$C$S") \
  --export=ALL,NI26_SOURCE_REV="${NI26_SOURCE_REV:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}" \
  scripts/cloud/build_sealed.sbatch "$CFG" "$BASE")"

K=":$(sub $GA --cpus-per-task=8 --mem=48G --time=03:00:00 $(dep "$C$S$T$B") -o "${OUT}_export_%j.log" \
  --wrap "$E; python scripts/export_external_teacher.py $BASE $OUT --families sincshallow,fbcnet,fblight,shallow --seeds 0-11 --C 0.01 --weight 4")"
R=$(sub --exclude="$X" $(dep "$K") scripts/cloud/replay.sbatch "$OUT")
V=$(sub --exclude="$X" $(dep "$K") scripts/cloud/worker_replay.sbatch "$OUT")
echo "prepare${P:-:done} windows${W#"$P"} corpora${C#"$W"} trunks$S$T base${B:-:done} teacher$K replay:$R worker:$V"
if [ -n "${NI26_REF:-}" ]; then
  echo "compare:$(sub $GA --cpus-per-task=8 --mem=48G --time=02:00:00 $(dep "$K") -o "${OUT}_compare_%j.log" \
    --wrap "$E; python scripts/compare_packages.py $OUT $NI26_REF ${OUT}_compare.json")"
fi
