#!/bin/bash
# Run on Galvani from the verified release after queuing its source smoke.
set -euo pipefail
source_smoke=${1:?source smoke job ID}
[[ $source_smoke =~ ^[0-9]+$ ]]
test ! -e jobs.txt
sha256sum --quiet -c actor.sha256
sha256sum --quiet -c score.sha256
held_smoke=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$source_smoke held.sbatch smoke)
source=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$held_smoke --array=0-19%6 fit.sbatch matrix)
held=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=aftercorr:$source --array=0-19%6 held.sbatch matrix)
score=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$held held.sbatch collect)
printf 'source_smoke=%s\nheld_smoke=%s\nsource=%s\nheld=%s\nscore=%s\n' \
    "$source_smoke" "$held_smoke" "$source" "$held" "$score" | tee jobs.txt
