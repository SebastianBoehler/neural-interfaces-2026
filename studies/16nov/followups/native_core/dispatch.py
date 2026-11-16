"""Release the complete §301 DAG only after concrete parity and source checks."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

PRODUCER = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116')
CHAIN = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_chain_v3_20261116')
PARITY = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_parity_v3_20261116/runs/s301_parity/report.json')
REF = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_20261116/runs/s301_reference')
ALLREF = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all')
CORE = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so')
TRAIN = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def submit(cwd, *args):
    array = next((x.split('=', 1)[1] for x in args if x.startswith('--array=')), None)
    span = array.split('%', 1)[0] if array else '0'
    added = int(span.split('-')[1]) - int(span.split('-')[0]) + 1 if '-' in span else 1
    active = subprocess.check_output(['squeue', '-u', 'gfh527', '-r', '-h'], text=True).splitlines()
    if len(active) + added > 272:
        raise ValueError(f'capacity changed before submit: active={len(active)}, requested={added}')
    result = subprocess.run(['sbatch', '--parsable', *args], cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()
    if not result.isdigit():
        raise ValueError(f'unexpected Slurm job id {result!r}')
    return result


def guard():
    for root, manifest in ((PRODUCER, 'source.sha256'), (PRODUCER, 'producer.sha256'),
                           (CHAIN, 's295-source.sha256'), (CHAIN, 'chain.sha256'),
                           (TRAIN, 'trainref.sha256')):
        subprocess.run(['sha256sum', '-c', manifest], cwd=root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['sha256sum', '-c', 'runs/s301_trainref/library.sha256'], cwd=TRAIN, check=True, stdout=subprocess.DEVNULL)
    gpu_log = (TRAIN / 'logs/trainref-gpu-3114263.out').read_text()
    if 'native whole-session reference/feature parity passed; no score' not in gpu_log or 'NVIDIA' not in gpu_log:
        raise ValueError('training-reference CPU/GPU proof is absent')
    report = json.loads(PARITY.read_text())
    if report.get('held_accuracy_computed') is not False or report.get('library_sha256') != sha(CORE):
        raise ValueError('full native parity report does not match the library')
    records = report.get('records', {})
    for person in ('P01', 'P10'):
        for order in ('within0', 'full0'):
            for device in (0, 1):
                item = records[f'{person}_{order}.npz|device{device}']
                if (item['feature_rel_l2'] > 1e-6 or item['feature_max_abs'] > 1e-4 or
                        item['probability_max_abs'] > 1e-5 or item['non_tied_label_changes'] != 0):
                    raise ValueError('registered full feature/head parity did not pass')
    manifest = json.loads((ALLREF / 'manifest.json').read_text())
    names = {f'P{i:02d}.npz' for i in range(1, 11)}
    if set(manifest['files']) != names or manifest['held_accuracy_computed'] is not False:
        raise ValueError('ten-fold reference heads incomplete')
    for name, digest in manifest['files'].items():
        if sha(ALLREF / name) != digest:
            raise ValueError(f'changed reference head {name}')
    for person in ('P01', 'P10'):
        with np.load(ALLREF / f'{person}.npz', allow_pickle=False) as a, np.load(REF / f'{person}_within0.npz', allow_pickle=False) as b:
            for key in ('fit_rows', 'mean', 'scale', 'coef', 'intercept'):
                if not np.allclose(a[key], b[key], atol=1e-10, rtol=1e-10):
                    raise ValueError(f'{person}: full/reference fold differs: {key}')
    active = subprocess.check_output(['squeue', '-u', 'gfh527', '-r', '-h'], text=True).splitlines()
    running = subprocess.check_output(['squeue', '-u', 'gfh527', '-r', '-h', '-t', 'RUNNING,COMPLETING'], text=True).splitlines()
    if len(active) + 49 > 272 or len(running) > 30:
        raise ValueError(f'insufficient capacity: active={len(active)}, running={len(running)}')


def main():
    intent = PRODUCER / 'runs/s301_dispatch.jsonl'
    if intent.exists():
        raise FileExistsError(intent)
    guard()
    source_sha = sha(PRODUCER / 'producer.sha256')
    path = 'studies/16nov/followups/native_core/'
    with intent.open('x') as f:
        def record(kind, job):
            f.write(json.dumps({'kind': kind, 'job': job}, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
            print(f'{kind}: {job}', flush=True)
        prep = submit(PRODUCER, '--hold', '--array=0', path + 'prepare.sbatch'); record('prepare_smoke_held', prep)
        pc = submit(PRODUCER, '--gres=none', f'--dependency=afterok:{prep}', path + 'check_smoke.sbatch', 'prepare'); record('prepare_check', pc)
        rest = submit(PRODUCER, f'--dependency=afterok:{pc}', '--array=1-9%2', path + 'prepare.sbatch'); record('prepare_rest', rest)
        fit = submit(PRODUCER, f'--dependency=afterok:{rest}', '--array=0', path + 'fit.sbatch'); record('fit_smoke', fit)
        fc = submit(PRODUCER, '--gres=none', f'--dependency=afterok:{fit}', path + 'check_smoke.sbatch', 'fit'); record('fit_check', fc)
        fitrest = submit(PRODUCER, f'--dependency=afterok:{fc}', '--array=1-9%2', path + 'fit.sbatch'); record('fit_rest', fitrest)
        assemble = submit(PRODUCER, f'--dependency=afterok:{fitrest}', path + 'assemble.sbatch'); record('assemble', assemble)
        stage = submit(CHAIN, '--gres=none', f'--export=ALL,S301_PRODUCER_SHA={source_sha}', f'--dependency=afterok:{assemble}', path + 'stage.sbatch'); record('stage', stage)
        zero = submit(CHAIN, '--gres=none', f'--dependency=afterok:{stage}', '--array=0-7%2', path + 'member.sbatch', 'zero'); record('zero', zero)
        treatment = submit(CHAIN, '--gres=none', f'--dependency=afterok:{zero}', '--array=0-15%2', path + 'member.sbatch', 'treatment'); record('member', treatment)
        score = submit(CHAIN, '--gres=none', f'--dependency=afterok:{treatment}', path + 'score.sbatch'); record('score', score)
        running = subprocess.check_output(['squeue', '-u', 'gfh527', '-r', '-h', '-t', 'RUNNING,COMPLETING'], text=True).splitlines()
        if len(running) > 30:
            raise ValueError(f'preparation remains held; running={len(running)} exceeds 30')
        subprocess.run(['scontrol', 'release', prep], check=True)
        record('prepare_smoke_released', prep)


if __name__ == '__main__':
    main()
