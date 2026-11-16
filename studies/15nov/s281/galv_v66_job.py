"""§281 Galvani task mapping and execution; no job submission.

Use the companion sbatch script from the fixed private workspace. CP tasks run
six independent processes, with one BLAS/OMP thread each; all exits are read.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path('/mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116')
PYTHON = '/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python'
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')
RECIPES = {'control': ('none', 'none', '0'), 'zero_coral': ('fb_coral', 'none', '1'),
           'S1': ('fb_coral', 'none', '0'), 'S2': ('fb_am', 'none', '0'),
           'S3': ('fb_coral,fb_am', 'none', '0'), 'S4': ('none', 'fb_coral', '0'),
           'S5': ('none', 'fb_am', '0')}
PHASE_ARMS = {'control': ('control',), 'treatment': ('zero_coral', 'S1', 'S2', 'S3', 'S4', 'S5'),
              'cp': ('control', 'zero_coral', 'S1', 'S2', 'S3', 'S4', 'S5')}


def mapping(phase, task):
    arms = PHASE_ARMS.get(phase)
    if arms is None or not isinstance(task, int) or not 0 <= task < len(arms) * len(ORDERS):
        raise ValueError(f'invalid {phase} task index: {task}')
    return arms[task // len(ORDERS)], ORDERS[task % len(ORDERS)]


def check_session_lengths(path):
    import numpy as np
    with np.load(path, allow_pickle=True) as z:
        sess = z['sess|all_loso']
    if sess.ndim != 1 or not len(sess):
        raise ValueError('empty or invalid all-LOSO sessions')
    _, lengths = np.unique(sess, return_counts=True)
    if np.max(lengths) >= 250:
        raise ValueError(f'study/package cap boundary differs at 250: max session {int(np.max(lengths))}')
    print(f'all-LOSO sessions {len(lengths)}, lengths {int(np.min(lengths))}–{int(np.max(lengths))} (<250)', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_private(path, root):
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f'missing or non-private file: {path}')
    relative = path.relative_to(root)
    if (path.stat().st_nlink != 1 or
            any(root.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1))):
        raise ValueError(f'private inputs must be copies, not links: {path}')


def check_private_tree(tree, root):
    if not tree.is_dir() or tree.is_symlink() or not tree.resolve().is_relative_to(root):
        raise ValueError(f'missing or non-private tree: {tree}')
    for directory, dirs, files in os.walk(tree, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            if p.is_symlink() or (p.is_file() and p.stat().st_nlink != 1):
                raise ValueError(f'private inputs must be copies, not links: {p}')


def check_outputs(paths, root=WORKSPACE):
    for path in paths:
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError(f'output outside private workspace: {path}')
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'refusing to overwrite: {path}')


def preflight(root=WORKSPACE):
    """Check shared inputs before submission; runners retain stricter own guards."""
    if Path.cwd().resolve() != root or Path(__file__).resolve().parents[3] != root:
        raise ValueError(f'work only in private workspace {root}')
    for tree in ('data', 'runs', 'lean', 'studies', 'replay', 'experiments', 'scripts'):
        check_private_tree(root / tree, root)
    essential = ('data/proteus_windows.npz', 'runs/replay_data/bases.npz', 'runs/shuffle_perms.npz',
                 'runs/rundetect/proj_d6_r1.npz', 'runs/rundetect/decisions.npz',
                 'runs/sealed/final_zips/v66_cp1_cap250.zip', 'runs/s281_leo_inputs/remote.sha256')
    for rel in essential:
        require_private(root / rel, root)
    check_session_lengths(root / 'runs/replay_data/bases.npz')
    if sha256(root / 'runs/shuffle_perms.npz') != '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f':
        raise ValueError('shuffled-delivery order file differs from pinned file')
    if sha256(root / 'runs/sealed/final_zips/v66_cp1_cap250.zip') != '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':
        raise ValueError('v6.6 ZIP hash mismatch')
    manifest = root / 'runs/s281_leo_inputs/remote.sha256'
    if sha256(manifest) != 'd86be876782dd4f4b06cd4f16eb0ff9dcf717e8b9cc38c263cc497ffc01609d1':
        raise ValueError('Leonardo base manifest hash mismatch')
    entries = {}
    for line in manifest.read_text().splitlines():
        h, name = line.split()
        if name in entries:
            raise ValueError(f'duplicate manifest entry: {name}')
        entries[name] = h
    if len(entries) != 96:
        raise ValueError('expected exactly 96 base streams')
    ref_manifest = root / 'runs/s281_v66_reference/sha256.remote.txt'
    require_private(ref_manifest, root)
    ref_entries = {}
    for line in ref_manifest.read_text().splitlines():
        h, name = line.split()
        if name in ref_entries:
            raise ValueError(f'duplicate reference: {name}')
        ref_entries[name] = h
    for o in ORDERS:
        archive = root / f'runs/s281_v66_reference/s242c250c_{o}.npz'
        require_private(archive, root)
        if archive.name not in ref_entries or sha256(archive) != ref_entries[archive.name]:
            raise ValueError(f'archived control mismatch: {archive}')
        stream_dirs = set()
        for fam in ('fb_coral', 'fb_am'):
            for seed in range(6):
                name = f'ttt18{o}_{fam}_base-all_loso-s{seed}.npz'
                # The runner picks one directory per order. Require precisely one
                # copy, otherwise its first-match lookup could hide conflicting data.
                paths = [root / 'runs' / rel / name for rel in ('ttt/probs18', 'diag/ttt/probs18')]
                found = [p for p in paths if p.is_file()]
                if len(found) != 1 or name not in entries:
                    raise ValueError(f'expected one chain base stream: {name}, found {found}')
                require_private(found[0], root)
                stream_dirs.add(found[0].parent)
                if sha256(found[0]) != entries[name]:
                    raise ValueError(f'base stream SHA mismatch: {name}')
        if len(stream_dirs) != 1:
            raise ValueError(f'{o}: both families must be in the same stream directory')
        for rel, pattern in (('ocushuf', 'oc_nomix-all_loso-s*.npz'),
                             ('restshuf', 'rest4_sh_r1-all_loso-s*.npz')):
            dirs = [root / 'runs' / r / rel / o for r in ('', 'diag')]
            chosen = next((d for d in dirs if d.is_dir()), None)
            if chosen is None or not list(chosen.glob(pattern)):
                raise ValueError(f'{o}: missing {rel} streams')
            for f in chosen.glob(pattern):
                require_private(f, root)
        dirs = [root / 'runs' / r / 'ttt/probs18' for r in ('', 'diag')]
        chosen = next((d for d in dirs if list(d.glob(f'ttt18{o}LPC_fb_r15c250*'))), None)
        if chosen is None:
            raise ValueError(f'{o}: missing r15c250 streams')
        for family in ('fb', 'fbraw', 'fbl', 'sinc'):
            for seed in range(6):
                for suffix in ('base', 'r15c250'):
                    require_private(chosen / f'ttt18{o}LPC_{family}_{suffix}-all_loso-s{seed}.npz', root)
    t7dirs = [root / 'runs' / r / 'ttt/probs7' for r in ('', 'diag')]
    t7 = next((d for d in t7dirs if list(d.glob('ttt7_fb_gC5*'))), None)
    if t7 is None:
        raise ValueError('missing recording-order ttt7 streams')
    for family in ('fb', 'fbraw', 'fbl', 'sinc'):
        for seed in range(6):
            require_private(t7 / f'ttt7_{family}_base-all_loso-s{seed}.npz', root)
    for rel in ('runs/chainout', 'runs/newmember', 'runs/s281_v66_cp', 'runs/s281_v66_orch/logs', 'runs/s281_v66_orch/slurm'):
        d = root / rel
        if d.is_symlink() or not d.resolve().is_relative_to(root):
            raise ValueError(f'output directory is not private: {d}')
        d.mkdir(parents=True, exist_ok=True)
    outputs = []
    for o in ORDERS:
        for arm in RECIPES:
            outputs += [root / f'runs/chainout/s281_v66p_{arm}_{o}.npz',
                        root / f'runs/newmember/s281_v66p_{arm}_{o}_gate.npz',
                        root / f'runs/newmember/s281_v66p_{arm}_{o}_provenance.json']
            for base in BASES:
                p = root / f'runs/s281_v66_cp/{arm}_{o}_{base}'
                outputs += [p.with_suffix('.npz'), p.with_suffix('.json')]
    check_outputs(outputs, root)
    for arm in RECIPES:
        for order in ORDERS:
            found = list((root / 'runs/newmember').glob(f's281_v66p_{arm}_shufcombo_{order}*.npz'))
            if found:
                raise FileExistsError(f'refusing to overwrite chain summaries: {found}')
        for order in ORDERS:
            for base in BASES:
                check_outputs((root / f'runs/s281_v66_orch/logs/{arm}_{order}_{base}.log',), root)
    print('preflight OK; this does not submit or score jobs', flush=True)


def clean_env():
    env = {key: os.environ[key] for key in ('HOME', 'PATH', 'TMPDIR', 'LD_LIBRARY_PATH') if key in os.environ}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', NI26_WS=str(WORKSPACE), PYTHONWARNINGS='ignore')
    return env


def run_children(commands, directory, env):
    """Start all commands, then wait for every launched child, even on failure."""
    logs, processes, failed = [], [], []
    try:
        for name, _ in commands:
            logs.append((name, (directory / f'{name}.log').open('x')))
        for (name, command), (_, log) in zip(commands, logs):
            try:
                p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                processes.append((name, p)); print(f'child {name} pid={p.pid} started', flush=True)
            except OSError as exc:
                failed.append((name, str(exc))); print(f'child {name} failed to start: {exc}', flush=True)
    finally:
        for name, process in processes:
            rc = process.wait(); print(f'child {name} exit={rc}', flush=True)
            if rc:
                failed.append((name, rc))
        for _, log in logs:
            log.close()
    if failed:
        raise RuntimeError(f'child failures: {failed}')


def run_task(phase, task):
    if Path.cwd().resolve() != WORKSPACE or Path(__file__).resolve().parents[3] != WORKSPACE:
        raise ValueError(f'work only in private workspace {WORKSPACE}')
    check_session_lengths(WORKSPACE / 'runs/replay_data/bases.npz')
    arm, order = mapping(phase, task); env = clean_env()
    print(f'phase={phase} task={task} arm={arm} order={order}', flush=True)
    if phase in ('control', 'treatment'):
        add, swap, zero = RECIPES[arm]
        env.update(NM_NAME=arm, ADD=add, SWAP=swap, ZERO=zero, NM_SEEDS='0,1,2,3,4,5')
        subprocess.run([PYTHON, '-B', '-I', str(WORKSPACE / 'studies/14nov/newmember/run_s_v66.py'), order],
                       cwd=WORKSPACE, env=env, check=True)
    else:
        directory = WORKSPACE / 'runs/s281_v66_orch/logs'
        for base in BASES:
            p = WORKSPACE / f'runs/s281_v66_cp/{arm}_{order}_{base}'
            check_outputs((p.with_suffix('.npz'), p.with_suffix('.json'), directory / f'{arm}_{order}_{base}.log'))
        cmds = [(f'{arm}_{order}_{b}', [PYTHON, '-B', '-I', str(WORKSPACE / 'studies/14nov/newmember/s281_cp.py'), arm, order, b]) for b in BASES]
        run_children(cmds, directory, env)


def smoke():
    """Run a separate workspace's within0 controls and CP B2_s0; never score."""
    root = WORKSPACE / 'smoke'
    preflight(root)
    print('NOTE: runner emits intermediate pre-personalisation diagnostics; do not use these as treatment evidence. The complete final scorer supplies the treatment verdict.', flush=True)
    env = clean_env(); env['NI26_WS'] = str(root)
    runner = root / 'studies/14nov/newmember/run_s_v66.py'
    for arm in ('control', 'zero_coral', 'S1'):
        add, swap, zero = RECIPES[arm]
        e = dict(env, NM_NAME=arm, ADD=add, SWAP=swap, ZERO=zero, NM_SEEDS='0,1,2,3,4,5')
        print(f'smoke chain {arm} within0', flush=True)
        subprocess.run([PYTHON, '-B', '-I', str(runner), 'within0'], cwd=root, env=e, check=True)
    cp = root / 'studies/14nov/newmember/s281_cp.py'
    commands = [(f'smoke_{arm}_within0_B2_s0', [PYTHON, '-B', '-I', str(cp), arm, 'within0', 'B2_s0'])
                for arm in ('control', 'zero_coral', 'S1')]
    run_children(commands, root / 'runs/s281_v66_orch/logs', env)
    sys.path.insert(0, str(root / 'studies/14nov/newmember'))
    import score_s_v66p as scorer
    cache = {}; n = 2655
    outputs = {arm: scorer.load_output(root, arm, 'within0', 'B2_s0', n, cache)
               for arm in ('control', 'zero_coral', 'S1')}
    scorer.verify_zero(outputs['control'], outputs['zero_coral'])
    print('smoke full-stage zero and provenance OK; no score computed', flush=True)


def main():
    if sys.argv[1:] == ['preflight']:
        preflight()
    elif sys.argv[1:] == ['preflight-smoke']:
        preflight(WORKSPACE / 'smoke')
    elif sys.argv[1:] == ['smoke']:
        smoke()
    elif len(sys.argv) == 3 and sys.argv[1] in PHASE_ARMS and sys.argv[2].isdecimal():
        run_task(sys.argv[1], int(sys.argv[2]))
    else:
        raise SystemExit('usage: galv_v66_job.py preflight | preflight-smoke | smoke | {control,treatment,cp} TASK')


if __name__ == '__main__':
    main()
