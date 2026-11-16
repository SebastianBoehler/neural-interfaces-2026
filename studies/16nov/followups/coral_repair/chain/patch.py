"""Adapt the original §292 scorer/runner only inside a private §304 snapshot."""
import json
import re
import sys
from pathlib import Path

FAMILIES = ('fb304', 'fb_uc304', 'fb_cc10', 'fb_cc30')
RECIPES = {'control': ('none','none',False), 'zero304': ('fb304','none',True),
           'PLA': ('fb304','none',False), 'PLS': ('none','fb304',False),
           'UCA': ('fb_uc304','none',False), 'UCS': ('none','fb_uc304',False),
           'CCA10': ('fb_cc10','none',False), 'CCS10': ('none','fb_cc10',False),
           'CCA30': ('fb_cc30','none',False), 'CCS30': ('none','fb_cc30',False)}
CONTRASTS = [('CCA10','UCA'),('CCA30','UCA'),('CCS10','UCS'),('CCS30','UCS'),
             ('UCA','PLA'),('UCS','PLS'),('CCA10','PLA'),('CCA30','PLA'),('CCS10','PLS'),('CCS30','PLS')]


def replace_once(src, old, new):
    if src.count(old) != 1: raise ValueError(f'unexpected source anchor: {old[:90]!r}')
    return src.replace(old, new, 1)


def patch_sources(sources):
    out = dict(sources)
    p = out['s281_provenance.py']
    start, end = p.index('RECIPES = {'), p.index('\n\n\ndef validate_recipe')
    out['s281_provenance.py'] = p[:start] + 'RECIPES = ' + repr(RECIPES) + p[end:]
    out['run_s.py'] = replace_once(out['run_s.py'],
        'all(f in ("eegn", "eegn_dist2", "fb_coral3", "fb_coral10") for f in ADD + SW)',
        'all(f in ("fb304", "fb_uc304", "fb_cc10", "fb_cc30") for f in ADD + SW)')
    cp = out['s281_cp.py']
    cp, count = re.subn(r'    assert arm in \{[^\n]+\}', '    assert arm in {' + ', '.join(repr(x) for x in sorted(RECIPES)) + '}', cp, count=1)
    if count != 1: raise ValueError('unexpected CP arm guard')
    out['s281_cp.py'] = cp
    score = out['score_s_v66p.py']
    score = replace_once(score, "ARMS = ('control', 'zero_eegn', 'E', 'D', 'C3A', 'C3S', 'C10A', 'C10S')",
                         'ARMS = ' + repr(tuple(RECIPES)))
    score = score.replace("'zero_eegn'", "'zero304'")
    score = replace_once(score, "[('D','E')]", repr(CONTRASTS))
    score = replace_once(score, "root / 'runs/lane1_inputs/streams.sha256'", "root / 'runs/coral304_inputs/streams.sha256'")
    score = replace_once(score, "families = ('eegn', 'eegn_dist2', 'fb_coral3', 'fb_coral10')", 'families = ' + repr(FAMILIES))
    verify = """    validation_path = path.parent / 'validation.json'
    if not validation_path.is_file() or sha256(validation_path) != lock.get('assembler_validation_sha256'):
        raise ValueError('assembler validation manifest changed')
    validation = json.loads(validation_path.read_text())
    if validation.get('counts') != {'folds': 240, 'streams': 192} or validation.get('stream_hashes') != records:
        raise ValueError('assembler matrix differs from locked streams')
    if validation.get('producer_manifest_sha256') != lock.get('producer_manifest_sha256'):
        raise ValueError('producer source manifest identity differs')
    if validation.get('data_sha256') != sha256(ROOT / 'data/proteus_windows.npz') or validation.get('permutation_sha256') != sha256(ROOT / 'runs/shuffle_perms.npz'):
        raise ValueError('assembler data or permutations differ')
    source = Path(lock['source'])
    if sha256(source / 'coral304.sha256') != lock['producer_manifest_sha256']:
        raise ValueError('producer source manifest changed')
    folds = validation.get('producer_hashes', {})
    expected_folds = {f'{arm}_s{seed}_P{person:02d}.npz' for arm in ('plain', 'unconditional10', 'conditional10', 'conditional30') for seed in range(6) for person in range(1,11)}
    if set(folds) != expected_folds:
        raise ValueError('producer fold matrix differs')
    for name, digest in folds.items():
        if sha256(source / 'runs/coral304' / name) != digest:
            raise ValueError(f'producer fold changed: {name}')
"""
    score = replace_once(score, '    return records\n\n\ndef load_output(', verify + '    return records\n\n\ndef load_output(')
    out['score_s_v66p.py'] = score
    return out


def main(root):
    root = Path(root).resolve(strict=True)
    if (root / '.git').exists() or not (root / 'runs/s281_v66_reference').is_dir():
        raise ValueError('expected isolated private snapshot with archived references')
    base = root / 'studies/14nov/newmember'
    names = ('s281_provenance.py','run_s.py','s281_cp.py','score_s_v66p.py')
    paths = {n:base/n for n in names}
    for path in paths.values():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError(f'expected private regular file: {path}')
    changed = patch_sources({n:p.read_text() for n,p in paths.items()})
    for n,p in paths.items(): p.write_text(changed[n])
    print(json.dumps({'workspace':str(root),'recipes':RECIPES},sort_keys=True))


if __name__ == '__main__': main(sys.argv[1])
