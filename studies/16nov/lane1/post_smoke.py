"""Verify a registered full-stage smoke, then queue that branch's remainder."""
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')
ARMS = {'coral': ('C3A', 'C3S', 'C10A', 'C10S'), 'plain': ('zero_eegn', 'E'), 'dist': ('D',)}


def remaining(branch):
    return {'coral': list(range(1, 32)), 'plain': list(range(1, 8)) + list(range(9, 16)),
            'dist': list(range(17, 24))}[branch] if branch in ARMS else _bad(branch)


def _bad(branch):
    raise ValueError(f'unknown branch {branch}')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_controls(score, guard):
    cache = {}
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
    for order in ORDERS:
        archive = ROOT / f'runs/s281_v66_reference/s242c250c_{order}.npz'
        ctrl = ROOT / f'runs/chainout/s281_v66p_control_{order}.npz'
        from s281_routing import verify_archive_file
        ah = verify_archive_file(archive, archive.parent / 'sha256.remote.txt')
        route = json.loads((ROOT / f'runs/s281_v66_route/{order}.json').read_text())
        score.verify_route(route, order, ah, BASES, sessions)
        with np.load(ctrl) as outputs, np.load(archive) as ref:
            guard.verify_archive(outputs, ref)
        for base in BASES:
            score.load_output(ROOT, 'control', order, base, n, cache)
    return n, cache


def check_smoke(branch, score, guard, n, cache):
    manifest_name = {'coral': 'coral', 'plain': 'plain', 'dist': 'dist'}[branch]
    staged = json.loads((ROOT / f'runs/lane1_inputs/{manifest_name}.json').read_text())
    if staged.get('group') != manifest_name:
        raise ValueError('candidate stage manifest identity mismatch')
    for name, digest in staged['inputs_sha256'].items():
        if name.endswith('.npz') and sha(ROOT / 'runs/diag/ttt/probs18' / name) != digest:
            raise ValueError(f'staged stream changed: {name}')
    smoke_arms = {'coral': ('C3A',), 'plain': ('zero_eegn', 'E'), 'dist': ('zero_eegn', 'D')}[branch]
    with np.load(ROOT / 'runs/newmember/s281_v66p_control_within0_gate.npz') as ctl:
        control_gate = dict(ctl)
    for arm in smoke_arms:
        with np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_within0_gate.npz') as z:
            guard.verify(control_gate, dict(z), zero=arm == 'zero_eegn')
        for base in BASES:
            output = score.load_output(ROOT, arm, 'within0', base, n, cache)
            if arm == 'zero_eegn':
                ctl = score.load_output(ROOT, 'control', 'within0', base, n, cache)
                score.verify_zero(ctl, output)
    if branch == 'coral':
        # The new EEG zero arm is not staged yet. Independently compare the
        # current control with the already-validated §281 zero/control pair.
        old = Path('/mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116/runs/s281_v66_cp')
        for base in BASES:
            current = score.load_output(ROOT, 'control', 'within0', base, n, cache)
            with np.load(old / f'control_within0_{base}.npz') as c, np.load(old / f'zero_coral_within0_{base}.npz') as z:
                score.verify_zero(c['scores'], z['scores'])
                score.verify_zero(current, c['scores'])


def check_absent(branch):
    specs = {'coral': ('C3A', 'C3S', 'C10A', 'C10S'),
             'plain': ('zero_eegn', 'E'), 'dist': ('zero_eegn', 'E', 'D')}
    for i in remaining(branch):
        arm, order = specs[branch][i // 8], ORDERS[i % 8]
        targets = [ROOT / f'runs/chainout/s281_v66p_{arm}_{order}.npz',
                   ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz',
                   ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json']
        targets += [ROOT / f'runs/s281_v66_cp/{arm}_{order}_{b}.{suffix}'
                    for b in BASES for suffix in ('npz', 'json')]
        if any(p.exists() or p.is_symlink() for p in targets):
            raise FileExistsError(f'{arm} {order}: a treatment output already exists')


def main(branch):
    if branch not in ARMS or Path.cwd().resolve() != ROOT:
        raise ValueError('invalid branch or workspace')
    sys.path.insert(0, str(ROOT / 'studies/14nov/newmember'))
    score = importlib.import_module('score_s_v66p'); guard = importlib.import_module('s281_guard')
    if not (ROOT / 'runs/lane1_inputs/control-dry.json').is_file():
        raise ValueError('missing completed control/operator dry check')
    n, cache = verify_controls(score, guard)
    check_smoke(branch, score, guard, n, cache); check_absent(branch)
    ledger = ROOT / 'runs/lane1_inputs/job-ledger.jsonl'
    marker = ROOT / f'runs/lane1_inputs/submitted_{branch}.intent'
    with marker.open('x') as f:
        f.write(f'validated; submitter={os.environ["SLURM_JOB_ID"]}\n')
    indices = {'coral': '1-31%3', 'plain': '1-7,9-15%3', 'dist': '17-23%2'}[branch]
    group = 'coral' if branch == 'coral' else 'eeg'
    cmd = ['sbatch', '--parsable', f'--array={indices}',
           f'--dependency=afterok:{os.environ["SLURM_JOB_ID"]}',
           'studies/16nov/lane1/chain_treatment.sbatch', group]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    if not result.isdecimal():
        raise ValueError(f'unexpected sbatch result: {result}')
    with ledger.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({'branch': branch, 'job': result, 'array': indices,
                            'submitter': os.environ['SLURM_JOB_ID']}) + '\n')
        f.flush(); os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f'{branch} smoke and all controls verified; queued remainder job {result}', flush=True)
    finalize_if_last(ledger)


def finalize_if_last(ledger):
    with ledger.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0); entries = [json.loads(line) for line in f if line.strip()]
        branches = {e['branch']: e['job'] for e in entries if e['branch'] in ARMS}
        if len(branches) < 3:
            print(f'waiting for remaining branches: {sorted(set(ARMS) - set(branches))}', flush=True)
            return
        if any(e['branch'] == 'score' for e in entries):
            print('score was already queued by another branch', flush=True)
            return
        path = ROOT / 'studies/16nov/lane1/lock_streams.py'
        spec = importlib.util.spec_from_file_location('lane1_stream_lock', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.main()
        dependency = ':'.join([branches[b] for b in ('coral', 'plain', 'dist')] + [os.environ['SLURM_JOB_ID']])
        result = subprocess.run(['sbatch', '--parsable', f'--dependency=afterok:{dependency}',
                                 'studies/16nov/lane1/chain_score.sbatch'],
                                check=True, capture_output=True, text=True).stdout.strip()
        if not result.isdecimal():
            raise ValueError(f'unexpected score job id: {result}')
        f.seek(0, os.SEEK_END)
        f.write(json.dumps({'branch': 'score', 'job': result, 'dependencies': dependency}) + '\n')
        f.flush(); os.fsync(f.fileno())
        print(f'locked all 192 candidate streams and queued final score job {result}', flush=True)


def dry_controls():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('invalid workspace')
    sys.path.insert(0, str(ROOT / 'studies/14nov/newmember'))
    score = importlib.import_module('score_s_v66p'); guard = importlib.import_module('s281_guard')
    n, cache = verify_controls(score, guard)
    old = Path('/mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116/runs/s281_v66_cp')
    for base in BASES:
        current = score.load_output(ROOT, 'control', 'within0', base, n, cache)
        with np.load(old / f'control_within0_{base}.npz') as c, np.load(old / f'zero_coral_within0_{base}.npz') as z:
            score.verify_zero(c['scores'], z['scores'])
            score.verify_zero(current, c['scores'])
    expected = {'control': ('none', 'none', False), 'zero_eegn': ('eegn', 'none', True),
                'E': ('eegn', 'none', False), 'D': ('eegn_dist2', 'none', False),
                'C3A': ('fb_coral3', 'none', False), 'C3S': ('none', 'fb_coral3', False),
                'C10A': ('fb_coral10', 'none', False), 'C10S': ('none', 'fb_coral10', False)}
    if score.RECIPES != expected or len(remaining('coral')) != 31 or len(remaining('plain')) != 14 or len(remaining('dist')) != 7:
        raise ValueError('recipe or task mapping mismatch')
    for branch in ARMS:
        check_absent(branch)
    directory = ROOT / 'runs/lane1_inputs'; directory.mkdir(exist_ok=True)
    with (directory / 'control-dry.json').open('x') as f:
        json.dump({'n': n, 'orders': ORDERS, 'bases': BASES, 'recipes': expected}, f, sort_keys=True, indent=2)
    print('all eight archived controls, control CP, routes, recipes and task mappings verified', flush=True)


if __name__ == '__main__':
    dry_controls() if sys.argv[1] == 'dry' else main(sys.argv[1])
