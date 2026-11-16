"""Adapt existing §281 scripts inside a private workspace, never this checkout."""
import json
import os
import sys
from pathlib import Path

RECIPES = {
    'control': ('none', 'none', False), 'zero_eegn': ('eegn', 'none', True),
    'E': ('eegn', 'none', False), 'D': ('eegn_dist2', 'none', False),
    'C3A': ('fb_coral3', 'none', False), 'C3S': ('none', 'fb_coral3', False),
    'C10A': ('fb_coral10', 'none', False), 'C10S': ('none', 'fb_coral10', False),
}
REL = Path('studies/14nov/newmember')


def replace_once(src, old, new):
    if src.count(old) != 1:
        raise ValueError(f'expected exactly one source anchor: {old[:100]!r}')
    return src.replace(old, new, 1)


def patch_sources(sources):
    out = dict(sources)
    p = out['s281_provenance.py']
    start, end = p.index('RECIPES = {'), p.index('\n\n\ndef validate_recipe')
    out['s281_provenance.py'] = p[:start] + 'RECIPES = ' + repr(RECIPES) + p[end:]
    out['run_s.py'] = replace_once(out['run_s.py'],
        'all(f in ("fb_coral", "fb_am") for f in ADD + SW)',
        'all(f in ("eegn", "eegn_dist2", "fb_coral3", "fb_coral10") for f in ADD + SW)')
    out['s281_cp.py'] = replace_once(out['s281_cp.py'],
        '{"control", "zero_coral", "S1", "S2", "S3", "S4", "S5"}', repr(set(RECIPES)))
    score = out['score_s_v66p.py']
    score = replace_once(score, "ARMS = ('control', 'zero_coral', 'S1', 'S2', 'S3', 'S4', 'S5')",
                         'ARMS = ' + repr(tuple(RECIPES)))
    score = score.replace("'zero_coral'", "'zero_eegn'")
    score = replace_once(score, "[('S3','S2'), ('S3','S1')]", "[('D','E')]")
    score = replace_once(score, "root / 'runs/s281_leo_inputs/remote.sha256'",
                         "root / 'runs/lane1_inputs/streams.sha256'")
    summary = '''def cell_weighted_summary(delta, orders, cell_counts, bootstrap=4000, seed=0):
    d, w = np.asarray(delta, dtype=float), np.asarray(cell_counts, dtype=float)
    if d.shape != (len(orders), len(w)) or not np.isfinite(d).all() or np.any(w <= 0):
        raise ValueError('invalid cell-weighted inputs')
    groups = [np.array([i for i, o in enumerate(orders) if o.rstrip('0123456789') == s])
              for s in ('within', 'full', 'runs')]
    if any(len(v) == 0 for v in groups) or sum(map(len, groups)) != len(orders):
        raise ValueError('all three order schemes are required')
    avg = np.mean([d[v].mean(axis=0) for v in groups], axis=0)
    rng, samples = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        pp = rng.integers(0, len(w), len(w))
        sample = np.mean([d[rng.choice(v, size=len(v), replace=True)][:, pp].mean(axis=0)
                          for v in groups], axis=0)
        samples.append(np.average(sample, weights=w[pp]))
    return {'mean': float(np.average(avg, weights=w)),
            'interval': np.percentile(samples, [10, 90]).tolist()}


'''
    score = replace_once(score, 'def verify_zero(control, zero):', summary + 'def verify_zero(control, zero):')
    score = replace_once(score, "    ctl = summarize(np.stack([metric['control', o] for o in ORDERS]), ORDERS, PEOPLE)",
        "    cell_counts = np.array([len(np.unique(cells[people == p])) for p in PEOPLE])\n"
        "    ctl = summarize(np.stack([metric['control', o] for o in ORDERS]), ORDERS, PEOPLE)\n"
        "    ctl_cell = cell_weighted_summary(np.stack([metric['control', o] for o in ORDERS]), ORDERS, cell_counts)\n"
        "    print(f'control cell-equal: {ctl_cell}')")
    score = replace_once(score, "        s = summarize(d, ORDERS, PEOPLE)",
        "        s = summarize(d, ORDERS, PEOPLE)\n"
        "        cell_s = cell_weighted_summary(d, ORDERS, cell_counts)\n"
        "        print(f'{arm} - {reference} cell-equal: {cell_s}')")
    start, end = score.index('def remote_manifest(path):'), score.index('\n\ndef load_output(')
    fn = '''def remote_manifest(path):
    lock = json.loads((path.parent / 'lock.json').read_text())
    if sha256(path) != lock.get('manifest_sha256'):
        raise ValueError('lane1 stream manifest hash mismatch')
    records = {}
    for line in path.read_text().splitlines():
        digest, name = line.split()
        if name in records:
            raise ValueError(f'duplicate stream {name}')
        records[name] = digest
    families = ('eegn', 'eegn_dist2', 'fb_coral3', 'fb_coral10')
    expected = {f'ttt18{o}_{f}_base-all_loso-s{s}.npz'
                for o in ORDERS for f in families for s in range(6)}
    if set(records) != expected:
        raise ValueError('lane1 stream manifest does not cover exact 192 inputs')
    return records
'''
    out['score_s_v66p.py'] = score[:start] + fn + score[end:]
    return out


def main(root):
    root = Path(root).resolve(strict=True)
    if (root / '.git').exists() or root == Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026'):
        raise ValueError('refusing to patch a checkout or the main Galvani project')
    if not (root / 'runs/s281_v66_reference').is_dir():
        raise ValueError('private workspace lacks the independent §257 references')
    paths = {n: root / REL / n for n in ('s281_provenance.py', 'run_s.py', 's281_cp.py', 'score_s_v66p.py')}
    for path in paths.values():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError(f'expected private regular copy: {path}')
        if any(p.is_symlink() for p in (path.parent, *path.parents) if p != root.parent):
            raise ValueError(f'linked parent: {path}')
    patched = patch_sources({k: p.read_text() for k, p in paths.items()})
    for name, path in paths.items():
        path.write_text(patched[name])
    print(json.dumps({'workspace': str(root), 'patched': list(paths), 'recipes': RECIPES}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: patch_private.py PRIVATE_WORKSPACE')
    main(sys.argv[1])
