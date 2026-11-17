"""Measure completed worker pairs with the same GPU and CPU allocation."""
import json
from pathlib import Path
import subprocess
from time import monotonic, sleep

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
REPO = ROOT/'neural-interfaces-2026'
QUALIFIED = ROOT/'ni26_native366_exact_ea_v1_20261117'
VISIBLE_GPU = ['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,compute_mode',
               '--format=csv,noheader,nounits']


def gpu():
    row, = subprocess.check_output(VISIBLE_GPU, text=True).strip().splitlines()
    uuid, used, total, mode = (entry.strip() for entry in row.split(','))
    return uuid, int(used), int(total), mode


def run_group(arm, mode, queue):
    work = Path.cwd()
    package = QUALIFIED/('stress_' + arm)
    env = 'OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1'
    binds = (f'{work}:/work:rw,{package}:/package:ro,{REPO}/2026-competition:/official:ro,'
             f'{REPO}/data/proteus_windows.npz:/input.npz:ro')
    common = ['apptainer', 'exec', '--nv', '--cleanenv', '--containall', '--no-home',
              '--no-mount', 'hostfs', '--env', env, '--bind', binds, '--pwd', '/work',
              str(REPO/'runs/cloud/ni26_worker.sif'), 'python', 'replay96.py']
    reports = []
    gpu_id, start_mem, total, compute_mode = gpu()
    peak = start_mem
    start = monotonic()
    for batch in queue:
        processes = {}
        try:
            for slot in batch:
                name = f'{arm}_{mode}_{slot}'
                with Path(name + '.log').open('x') as output:
                    processes[name] = subprocess.Popen([*common, name + '.json'],
                                                       stdout=output, stderr=subprocess.STDOUT)
            while any(p.poll() is None for p in processes.values()):
                peak = max(peak, gpu()[1])
                sleep(1)
            for name, process in processes.items():
                if process.returncode:
                    raise RuntimeError(f'{name} failed with status {process.returncode}')
                reports.append(json.loads(Path(name + '.json').read_text()))
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
    return dict(seconds=monotonic()-start, gpu=gpu_id, compute_mode=compute_mode,
                vram_mib_total=total, vram_mib_start=start_mem, vram_mib_peak=peak,
                workers=reports)


def main():
    # Reverse the order of the schedule modes in the second implementation.
    suites = [('reference', 'serial'), ('reference', 'overlap'),
              ('native', 'overlap'), ('native', 'serial')]
    expected = json.loads((QUALIFIED/'reference_stress.json').read_text())
    expected = expected['streams']['first']['forced-unknown-378']['labels'][:96]
    groups = {}
    for arm, mode in suites:
        schedule = [[0], [1]] if mode == 'serial' else [[0, 1]]
        result = run_group(arm, mode, schedule)
        groups[f'{arm}_{mode}'] = result
        print(f'{arm} {mode}: {result["seconds"]:.2f}s, {result["vram_mib_peak"]} MiB', flush=True)
    for key, group in groups.items():
        group['prefix_exact'] = all(w['labels'] == expected for w in group['workers'])
    counts = [w['activity'] for group in groups.values() for w in group['workers']]
    exact = all(g['prefix_exact'] for g in groups.values()) and all(a == counts[0] for a in counts)
    if len({g['gpu'] for g in groups.values()}) != 1:
        raise RuntimeError('groups used different GPUs')
    data = dict(exact=exact, groups=groups)
    Path('result.json').write_text(json.dumps(data, indent=2) + '\n')
    if not exact:
        raise RuntimeError('parallel replay changed labels or work')
    print('EXACT_CONCURRENT_WORKER_REPLAY', flush=True)


if __name__ == '__main__':
    main()
