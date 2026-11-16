#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" NI26_OUT="$PWD/outputs" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
export PYTHONWARNINGS=ignore
unset PYTHONPATH
. .venv/bin/activate
sha256sum -c source.sha256 >/dev/null
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 data/proteus_windows.npz | sha256sum -c -
printf '%s  %s\n' 6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f runs/shuffle_perms.npz | sha256sum -c -
python -c 'import torch; assert torch.cuda.is_available()'
if [[ ${1:-} == smoke ]]; then
    for arm in off matched split; do
        python studies/16nov/lane3/run.py "$arm" 0 P01 --smoke
    done
else
    i=${SLURM_ARRAY_TASK_ID:?}
    (( i >= 0 && i < 60 )) || exit 2
    arms=(off matched split)
    arm=${arms[$((i / 20))]}
    seed=$(((i % 20) / 10))
    printf -v held 'P%02d' "$((i % 10 + 1))"
    python studies/16nov/lane3/run.py "$arm" "$seed" "$held"
fi
