"""Queue the next source-locked §304 wave only after validating the prior wave."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
CONTROL = Path(__file__).resolve().parent
os.environ['NI26_ROOT'] = str(ROOT)
sys.path[:0] = [str(CONTROL), str(ROOT / 'experiments')]
import protocol
import validate as V

WAVES = ('0-19,60-79,120-139,180-199', '20-39,80-99,140-159,200-219',
         '40-59,100-119,160-179,220-239')
CHAIN = Path('/mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_20261116/studies/16nov/followups/coral_repair/chain/stage.sbatch')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def submit(*args):
    result = command('sbatch', '--parsable', *map(str, args))
    if not result.isdigit():
        raise ValueError(f'unexpected sbatch result {result!r}')
    return result


def exclusive(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())


def completed_wave(stage, previous):
    expected = {f'{previous}_{i}' for i in range(240) if (i % 60) // 10 in range(2 * (stage - 1), 2 * stage)}
    rows = command('sacct', '-n', '-X', '-P', '-j', previous, '-o', 'JobID,State,ExitCode').splitlines()
    records = [row.split('|') for row in rows]
    if ({r[0] for r in records} != expected or
            any(len(r) != 3 or r[1:] != ['COMPLETED', '0:0'] for r in records)):
        raise ValueError(f'prior wave {previous} is not the complete successful 80-task array')


def main(stage, previous):
    if stage not in (1, 2) or not previous.isdigit():
        raise ValueError('invalid continuation stage')
    completed_wave(stage, previous)
    subprocess.run(['sha256sum', '-c', str(ROOT / 'coral304.sha256')], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    d, folds, perms = V.inputs(ROOT, protocol)
    expected = {f'{a}_s{s}_{h}.npz' for a in V.ARMS for s in range(stage * 2) for h in folds}
    actual = {p.name for p in (ROOT / 'runs/coral304').glob('*.npz')}
    if actual != expected:
        raise ValueError(f'prior matrix mismatch missing={len(expected-actual)} extra={len(actual-expected)}')
    hashes = {}
    for arm in V.ARMS:
        for seed in range(stage * 2):
            for held in sorted(folds):
                name = f'{arm}_s{seed}_{held}.npz'
                hashes[name] = V.fold(ROOT, d, folds, perms, arm, seed, held)[0]
    if stage == 2 and not CHAIN.is_file():
        raise FileNotFoundError(f'chain staging entry missing: {CHAIN}')
    active = len(command('squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i').splitlines())
    # One array plus next callback, or array plus assembler plus chain stage.
    needed = 81 if stage == 1 else 82
    if active + needed > 272:
        if active >= 272:
            raise RuntimeError(f'no capacity to schedule a retry: active={active}')
        journal = ROOT / 'runs/coral304_intents'
        journal.mkdir(parents=True, exist_ok=True)
        retry = submit(f'--chdir={ROOT}', '--begin=now+10minutes', CONTROL / 'continue.sbatch', stage, previous)
        exclusive(journal / f'wave{stage + 1}.retry-{retry}.json',
                  dict(job=retry, previous=previous, active=active, needed=needed))
        print(f'insufficient capacity: active={active}, needed={needed}; retry={retry}', flush=True)
        return
    journal = ROOT / 'runs/coral304_intents'
    journal.mkdir(parents=True, exist_ok=True)
    exclusive(journal / f'wave{stage + 1}.intent.json', dict(previous=previous, active=active,
              source_manifest=V.MANIFEST, prior_hashes=hashes, array=WAVES[stage]))
    producer = ROOT / 'studies/16nov/followups/coral_repair/job.sbatch'
    job = submit(f'--chdir={ROOT}', f'--array={WAVES[stage]}%6', producer)
    exclusive(journal / f'wave{stage + 1}.job.json', dict(job=job, previous=previous))
    if stage == 1:
        follow = submit(f'--chdir={ROOT}', f'--dependency=afterok:{job}', CONTROL / 'continue.sbatch', '2', job)
        exclusive(journal / 'wave3.callback.json', dict(job=follow, dependency=job))
        print(f'wave2={job} wave3_callback={follow}', flush=True)
    else:
        assembler = submit(f'--chdir={ROOT}', f'--dependency=afterok:{job}', CONTROL / 'assemble.sbatch')
        exclusive(journal / 'assemble.job.json', dict(job=assembler, dependency=job))
        stage_job = submit(f'--chdir={CHAIN.parents[5]}', f'--dependency=afterok:{assembler}', CHAIN, ROOT)
        exclusive(journal / 'chain_stage.job.json', dict(job=stage_job, dependency=assembler))
        print(f'wave3={job} assembler={assembler} chain_stage={stage_job}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
