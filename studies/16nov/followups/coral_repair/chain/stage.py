"""Validate assembler provenance and privately stage its 192 canonical streams."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
FAMILIES = ('fb304','fb_uc304','fb_cc10','fb_cc30')
PRODUCERS = ('plain','unconditional10','conditional10','conditional30')
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def expected_names():
    return {f'ttt18{o}_{fam}_base-all_loso-s{seed}.npz' for o in ORDERS for fam in FAMILIES for seed in range(6)}


def verify_validation(v, source):
    expected_folds = {f'{arm}_s{seed}_P{person:02d}.npz' for arm in PRODUCERS for seed in range(6) for person in range(1,11)}
    if v.get('counts') != {'folds':240, 'streams':192}: raise ValueError('wrong producer/stream count')
    if set(v.get('producer_hashes', {})) != expected_folds: raise ValueError('wrong producer fold matrix')
    if set(v.get('stream_hashes', {})) != expected_names(): raise ValueError('wrong stream matrix')
    if v.get('producer_manifest_sha256') != sha(source / 'coral304.sha256'): raise ValueError('producer source manifest differs')
    if v.get('data_sha256') != sha(ROOT / 'data/proteus_windows.npz'): raise ValueError('data hash differs')
    if v.get('permutation_sha256') != sha(ROOT / 'runs/shuffle_perms.npz'): raise ValueError('permutation hash differs')
    for name, digest in v['producer_hashes'].items():
        path = source / 'runs/coral304' / name
        if not path.is_file() or path.is_symlink() or sha(path) != digest: raise ValueError(f'producer fold differs: {name}')


def main(source):
    source = Path(source).resolve(strict=True)
    if source == ROOT or (ROOT / '.git').exists(): raise ValueError('expected separate private workspaces')
    streams = source / 'runs/coral304_streams'
    vpath = streams / 'validation.json'; manifest = streams / 'streams.sha256'
    if vpath.is_symlink() or manifest.is_symlink(): raise ValueError('linked assembler manifests forbidden')
    validation = json.loads(vpath.read_text()); verify_validation(validation, source)
    records = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split()
        if name in records or Path(name).name != name: raise ValueError('duplicate or invalid stream name')
        records[name] = digest
    if records != validation['stream_hashes'] or set(records) != expected_names(): raise ValueError('stream manifest differs')
    target = ROOT / 'runs/diag/ttt/probs18'
    out = ROOT / 'runs/coral304_inputs'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'streams.sha256').exists() or (out / 'lock.json').exists(): raise ValueError('staging already attempted')
    if any((target / name).exists() for name in records): raise ValueError('refusing to overwrite candidate streams')
    for name, digest in sorted(records.items()):
        src, dst = streams / name, target / name
        if not src.is_file() or src.is_symlink() or sha(src) != digest: raise ValueError(f'bad source: {name}')
        with np.load(src, allow_pickle=False) as z:
            if not np.array_equal(z['index'], np.arange(2655)): raise ValueError(f'wrong index: {name}')
            p = z['proba']
            if p.shape != (2655,3) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1),1,atol=1e-6):
                raise ValueError(f'invalid probability stream: {name}')
            if 'metadata' not in z: raise ValueError(f'missing stream metadata: {name}')
        with src.open('rb') as f, dst.open('xb') as g: shutil.copyfileobj(f,g)
        if sha(dst) != digest or sha(src) != digest: raise ValueError(f'copy/source changed: {name}')
    lockpath = out / 'streams.sha256'
    with lockpath.open('x') as f:
        for name, digest in sorted(records.items()): f.write(f'{digest}  {name}\n')
    with vpath.open('rb') as f, (out / 'validation.json').open('xb') as g: shutil.copyfileobj(f,g)
    if sha(out / 'validation.json') != sha(vpath): raise ValueError('validation copy differs')
    lock = {'manifest_sha256':sha(lockpath), 'assembler_validation_sha256':sha(vpath),
            'producer_manifest_sha256':validation['producer_manifest_sha256'], 'source':str(source)}
    with (out / 'lock.json').open('x') as f: json.dump(lock,f,sort_keys=True,indent=2)
    print('OK: 240 matched producer folds and 192 private streams verified',flush=True)


if __name__ == '__main__': main(sys.argv[1])
