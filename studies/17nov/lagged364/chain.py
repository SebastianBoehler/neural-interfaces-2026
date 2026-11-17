"""Private §323 chain with a matched EEG relation admitted as one add-member slot."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'ni26_s323_chain_20261117'
ROOT = HERE / 'chain'
MEMBER = Path('studies/14nov/newmember')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
FAMILIES = {'instant': 'n364instant', 'lag': 'n364lag'}
RECIPES = {'lag_zero': ('n323s0c', 'none', True),
           **{a: (f, 'none', False) for a, f in FAMILIES.items()}}
ORIGINAL_SOURCE = 'baf212bddfc79ed4d8bce9a9600c5c3db56d039d72c08f8fcfc92997dafbd55c'
ORIGINAL_STREAM_LOCK = 'a01b1706d8eb0a9857c19e51b4fe26391c6f4c0464982504845011f4a59f1cd3'
ROWS = 2655
BASE = 'B2_pool'


def sha(path):
    import hashlib
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def name(family, order):
    return f'ttt18{order}_{family}_base-all_loso-ensemble2.npz'


def patch(path, old, new):
    if path.is_symlink():
        raise ValueError('refuse writable source symlink')
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f'original source anchor changed: {path.name}')
    updated = text.replace(old, new)
    compile(updated, str(path), 'exec')
    path.chmod(0o644)
    path.write_text(updated)
    path.chmod(0o444)


def prepare():
    manifest = OLD / 'representation-source.sha256'
    if ROOT.exists() or sha(manifest) != ORIGINAL_SOURCE:
        raise ValueError('private workspace must be new and original chain hash pinned')
    subprocess.run(['sha256sum', '-c', str(manifest)], cwd=OLD, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(['cp', '-a', str(OLD), str(ROOT)], check=True)
    patch(ROOT / MEMBER / 's281_provenance.py', '\n\ndef validate_recipe',
          f'\nRECIPES.update({RECIPES!r})\n\ndef validate_recipe')
    p = ROOT / MEMBER / 'run_s.py'
    patch(p, "'n323s1f', 'n323s1c') for f in ADD + SW)",
          "'n323s1f', 'n323s1c', 'n364instant', 'n364lag') for f in ADD + SW)")
    old = 'assert metadata["seeds"] == [0, 1] and metadata["operation"] == "arithmetic_probability_mean"'
    patch(p, old, 'assert metadata["seeds"] == [0, 1]\n'
          '            operation = ("bounded_residual_after_arithmetic_probability_mean" '
          'if fam in ("n364instant", "n364lag") else "arithmetic_probability_mean")\n'
          '            assert metadata["operation"] == operation')
    sources = {entry.split()[1]: entry.split()[0] for entry in manifest.read_text().splitlines()}
    for p in ('run_s.py', 's281_provenance.py'):
        relative = str(MEMBER / p)
        sources[relative] = sha(ROOT / relative)
    with (HERE / 'runtime.sha256').open('x') as f:
        f.writelines(f'{h}  {n}\n' for n, h in sorted(sources.items()))
    print(f'private chain prepared; runtime SHA256 {sha(HERE / "runtime.sha256")}', flush=True)


def load_scorer():
    sys.path.insert(0, str(ROOT / MEMBER))
    import score_s_v66p
    if Path(score_s_v66p.__file__).resolve() != ROOT / MEMBER / 'score_s_v66p.py':
        raise ValueError('wrong scorer imported')
    return score_s_v66p


def original_streams():
    old = OLD / 'runs/representation_inputs'
    if sha(old / 'lock.json') != ORIGINAL_STREAM_LOCK:
        raise ValueError('original stream lock changed')
    return load_scorer().remote_manifest(old / 'streams.sha256')


def provenance(arm, order, root):
    return json.loads((root / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json').read_text())


def verify_order(scorer, order, sessions):
    from s281_guard import verify, verify_archive
    from s281_routing import verify_archive_file
    archive = OLD / f'runs/s281_v66_reference/s242c250c_{order}.npz'
    route = OLD / f'runs/s281_v66_route/{order}.json'
    digest = verify_archive_file(archive, archive.parent / 'sha256.remote.txt')
    scorer.verify_route(json.loads(route.read_text()), order, digest, scorer.BASES, sessions)
    with np.load(OLD / f'runs/chainout/s281_v66p_control_{order}.npz') as c, np.load(archive) as a:
        verify_archive(c, a)
    ref = f'runs/newmember/s281_v66p_control_{order}_gate.npz'
    for arm, arm_root in (('lag_zero', ROOT), ('S0C', OLD)):
        rel = f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz'
        with np.load(OLD / ref) as c, np.load(arm_root / rel) as a:
            verify(dict(c), dict(a), zero=arm == 'lag_zero')
    with np.load(ROOT / f'runs/newmember/s281_v66p_lag_zero_{order}_gate.npz') as z:
        with np.load(OLD / ref) as c:
            if any(not np.array_equal(c[k], z[k]) for k in c.files):
                raise ValueError(f'{order}: zero gate or upstream score changed')


def validate(orders, arms, streams):
    s = load_scorer()
    with np.load(OLD / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        sessions = len(np.unique(z['sess|all_loso']))
        n = len(z['sess|all_loso'])
    if n != ROWS:
        raise ValueError('public replay row count differs')
    s.verify_package(OLD / 'runs/sealed/final_zips/v66_cp1_cap250.zip',
                     '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed')
    output, cache = {}, {}
    from s281_guard import verify
    for order in orders:
        verify_order(s, order, sessions)
        for arm in ('control', 'S0C', 'lag_zero', *arms):
            source = OLD if arm in ('control', 'S0C') else ROOT
            prov = provenance(arm, order, source)
            s.verify_streams(prov, arm, order, streams)
            output[arm, order] = s.load_output(source, arm, order, BASE, n, cache)
            if arm in FAMILIES:
                with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c:
                    with np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz') as a:
                        verify(dict(c), dict(a), zero=False)
        s.verify_zero(output['control', order], output['lag_zero', order])
    return output


def cell(arm, order, resume_cp=None):
    if arm not in RECIPES or order not in ORDERS:
        raise ValueError('unregistered chain cell')
    add, swap, zero = RECIPES[arm]
    context = dict(os.environ, NI26_WS=str(ROOT), ADD=add, SWAP=swap, ZERO=str(int(zero)),
                   NM_NAME=arm, NM_SEEDS='0,1')
    start = time.monotonic()
    run = ROOT / MEMBER / 'run_s_v66.py'
    cp = ROOT / MEMBER / 's281_cp.py'
    if resume_cp is None:
        subprocess.run([sys.executable, '-B', '-I', str(run), order],
                       env=context, cwd=ROOT, check=True)
    else:
        from repair import verify_resume
        verify_resume(arm, order, resume_cp)
    upstream = time.monotonic() - start
    subprocess.run([sys.executable, '-B', '-I', str(cp), arm, order, BASE], env=context, cwd=ROOT, check=True)
    if zero:
        validate((order,), (), original_streams())
    timing = dict(arm=arm, order=order, upstream_seconds=upstream,
                  chain_seconds=time.monotonic() - start)
    with (HERE / f'time_{arm}_{order}.json').open('x') as f:
        json.dump(timing, f)
    print(timing, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    else:
        cell(*sys.argv[1:])
