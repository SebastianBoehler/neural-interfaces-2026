"""Release the CPU experiment only after its own smoke and stop pending duplicate production."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import sha

GPU_ARRAY = '3115206'
GPU_SCORE = '3115207'


def run(*args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def main():
    root = Path(os.environ['EXPERT_ROOT']).resolve(strict=True)
    marker = json.loads((root / 'runs/expert306_smoke/passed.json').read_text())
    lock = root / 'runs/input306/inputs.json'; smoke = root / 'runs/expert306_smoke/P01.npz'
    if sha(lock) != marker['input_lock_sha256'] or sha(smoke) != marker['smoke_sha256']:
        raise ValueError('CPU smoke provenance mismatch')
    source = os.environ['EXPERT_MANIFEST_SHA']
    if sha(root / 'expert.sha256') != source: raise ValueError('CPU source mismatch')
    active = len(run('squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i').splitlines())
    if active > 272: raise ValueError('submission reserve already exceeded')
    intent = root / 'runs/expert306_smoke/route.intent'
    with intent.open('x') as f:
        f.write(f'cpu_smoke_sha256 {sha(smoke)}\nactive_before {active}\n'); f.flush(); os.fsync(f.fileno())
    expected = {f'{GPU_ARRAY}_{i}' for i in range(10)}
    def verify_pending():
        rows = run('squeue', '-h', '-r', '-j', GPU_ARRAY, '-o', '%i|%T').splitlines()
        records = [x.split('|') for x in rows]
        if {x[0] for x in records} != expected or any(x[1] != 'PENDING' for x in records):
            raise ValueError('GPU duplicate is not wholly pending; CPU release stopped')
        if run('squeue', '-h', '-j', GPU_SCORE, '-o', '%T') != 'PENDING':
            raise ValueError('GPU duplicate scorer is not pending')
    verify_pending()
    # Hold pending work so a dependency transition cannot start it while checked.
    run('scontrol', 'hold', GPU_ARRAY)
    try:
        verify_pending()
    except Exception:
        run('scontrol', 'release', GPU_ARRAY)
        raise
    run('scancel', *sorted(expected)); run('scancel', GPU_SCORE)
    with intent.open('a') as f: f.write(f'cancelled_pending_gpu_array {GPU_ARRAY}\ncancelled_pending_gpu_score {GPU_SCORE}\n')
    production = run('sbatch', '--parsable', '--array=0-9%2', f'--export=ALL,EXPERT_MANIFEST_SHA={source}', root / 'run.sbatch')
    if not production.isdigit(): raise ValueError('unexpected CPU producer job id')
    with intent.open('a') as f: f.write(f'cpu_production {production}\n')
    scorer = run('sbatch', '--parsable', f'--dependency=afterok:{production}',
                 f'--export=ALL,EXPERT_MANIFEST_SHA={source},EXPERT_JOB_ID={production}', root / 'score.sbatch')
    if not scorer.isdigit(): raise ValueError('unexpected scorer job id')
    with intent.open('a') as f: f.write(f'cpu_scorer {scorer}\n')
    print(f'CPU production={production} scorer={scorer}; only duplicate pending GPU production cancelled', flush=True)


if __name__ == '__main__': main()
