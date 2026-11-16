"""Score §281 v6.6 personalisation outputs, after all required guards.

Usage: python -I studies/14nov/newmember/score_s_v66p.py
Requires control, zero_coral, S1-S5 for all eight orders and six bases,
plus the per-order §201 identity/contrast routing traces. The zero arm must
match control at the full-stage score level on every row. Scores mix
probabilities and log-scores; only row-wise argmax is meaningful.

This scores new people under the fold-specific routing emulation; it is not
packaged-gate validation, a returning-person score, or a routed-tail check.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from s281_provenance import RECIPES  # noqa: E402

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')
ARMS = ('control', 'zero_coral', 'S1', 'S2', 'S3', 'S4', 'S5')
PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_package(path, expected):
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f'v6.6 package hash mismatch: {path}')


def person_accuracy(scores, y, cell, person, people):
    scores, y, cell, person = map(np.asarray, (scores, y, cell, person))
    if scores.shape != (len(y), 3) or cell.shape != y.shape or person.shape != y.shape:
        raise ValueError('score, label, cell and person shapes do not match')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    pred = scores.argmax(axis=1)
    by_person = {p: [] for p in people}
    for c in np.unique(cell):
        m = cell == c
        owners = np.unique(person[m])
        if len(owners) != 1 or owners[0] not in by_person:
            raise ValueError(f'cell {c}: invalid person ownership')
        by_person[owners[0]].append(100 * np.mean([np.mean(pred[m][y[m] == k] == k) for k in np.unique(y[m])]))
    if any(not values for values in by_person.values()):
        raise ValueError('missing person or empty cells')
    return np.array([np.mean(by_person[p]) for p in people])


def summarize(delta, orders, people, bootstrap=4000, seed=0):
    d = np.asarray(delta, dtype=float)
    if d.shape != (len(orders), len(people)) or not np.isfinite(d).all():
        raise ValueError('invalid order/person deltas')
    groups = {s: np.array([i for i, o in enumerate(orders) if o.rstrip('0123456789') == s])
              for s in ('within', 'full', 'runs')}
    if any(len(v) == 0 for v in groups.values()) or sum(map(len, groups.values())) != len(orders):
        raise ValueError('all three schemes are required')
    avg = np.mean([d[v].mean(axis=0) for v in groups.values()], axis=0)
    rng = np.random.default_rng(seed); samples = []
    for _ in range(bootstrap):
        pp = rng.integers(0, len(people), len(people))
        samples.append(np.mean([d[rng.choice(v, size=len(v), replace=True)][:, pp].mean() for v in groups.values()]))
    return {'mean': float(avg.mean()), 'interval': np.percentile(samples, [10, 90]).tolist(),
            'people': {p: float(v) for p, v in zip(people, avg)},
            'people_up': int((avg > 0).sum()),
            'schemes': {s: float(d[v].mean()) for s, v in groups.items()}}


def verify_zero(control, zero):
    if not np.array_equal(control, zero):
        raise ValueError('zero-weight arm differs from control after full personalisation')


def verify_route(route, order, archive_sha, bases, sessions):
    if route.get('order') != order or route.get('archive_sha256') != archive_sha:
        raise ValueError(f'{order}: routing trace order or archive mismatch')
    if set(route.get('bases', {})) != set(bases):
        raise ValueError(f'{order}: missing or extra routing bases')
    for b in bases:
        v = route['bases'][b]
        if (v.get('sessions') != sessions or v.get('routed_sessions') != 0 or
                v.get('contrast_windows') != 0 or v.get('bio_windows') != 0 or v.get('routed') != []):
            raise ValueError(f'{order} {b}: routing not verified as zero')


def resolve_input(name, root):
    path = Path(name)
    if not path.is_absolute():
        raise ValueError(f'input path is not absolute: {name}')
    if path.is_file():
        return path
    if root is not None:
        anchors = {'runs', 'data', 'studies', 'replay', 'experiments', 'scripts', 'lean', 'evidence', 'submissions'}
        hits = [i for i, part in enumerate(path.parts) if part in anchors]
        if hits:
            candidate = root.joinpath(*path.parts[hits[0]:])
            if candidate.is_file():
                return candidate
        if path.name.startswith('ttt18') and path.name.endswith('.npz'):
            candidate = root / 'runs/s281_leo_inputs' / path.name
            if candidate.is_file():
                return candidate
    raise ValueError(f'missing tracked input: {name}')


def verify_inputs(files, cache, root=None):
    if not isinstance(files, dict) or not files:
        raise ValueError('input manifest has no hashes')
    for name, expected in files.items():
        path = resolve_input(name, root)
        key = str(path.resolve())
        if key not in cache:
            cache[key] = sha256(path)
        if cache[key] != expected:
            raise ValueError(f'tracked input changed: {name}')


def verify_manifest(manifest, arm, order, cache, root=None):
    recipe = manifest.get('recipe', {})
    expected = RECIPES.get(arm)
    actual = (recipe.get('add'), recipe.get('swap'), recipe.get('zero'))
    if (expected is None or recipe.get('name') != arm or actual != expected or
            recipe.get('seeds') != '0,1,2,3,4,5'):
        raise ValueError(f'{arm}: incorrect recipe provenance')
    if (manifest.get('order') != order or manifest.get('ft_variant') != 'r15c250' or
            manifest.get('ft_tag') != f'ttt18{order}LPC'):
        raise ValueError(f'{arm} {order}: incorrect order or fine-tuning provenance')
    verify_inputs(manifest.get('inputs_sha256'), cache, root)


def verify_streams(manifest, arm, order, remote):
    add, swap, _ = RECIPES[arm]
    families = [] if add == swap == 'none' else (add if add != 'none' else swap).split(',')
    inputs = manifest['inputs_sha256']
    for family in families:
        for seed in range(6):
            name = f'ttt18{order}_{family}_base-all_loso-s{seed}.npz'
            found = [v for k, v in inputs.items() if Path(k).name == name]
            if len(found) != 1 or name not in remote or found[0] != remote[name]:
                raise ValueError(f'{arm} {order}: stream missing, duplicated or differs from Leonardo: {name}')


def remote_manifest(path):
    if sha256(path) != 'd86be876782dd4f4b06cd4f16eb0ff9dcf717e8b9cc38c263cc497ffc01609d1':
        raise ValueError('Leonardo stream manifest hash mismatch')
    remote = {}
    for line in path.read_text().splitlines():
        h, name = line.split()
        if name in remote:
            raise ValueError(f'duplicate Leonardo stream {name}')
        remote[name] = h
    if len(remote) != 96:
        raise ValueError('expected 96 Leonardo streams')
    return remote


def load_output(root, arm, order, base, n, cache):
    cp = root / f'runs/s281_v66_cp/{arm}_{order}_{base}.npz'
    sidecar = cp.with_suffix('.json')
    chain = root / f'runs/chainout/s281_v66p_{arm}_{order}.npz'
    prov = root / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
    stage = root / 'lean/src66/cpers.py'
    # Files must all exist; no fallback to older s281_v66 outputs.
    if not all(p.is_file() for p in (cp, sidecar, chain, prov, stage)):
        raise ValueError(f'{arm} {order} {base}: missing output or provenance')
    m = json.loads(prov.read_text())
    verify_manifest(m, arm, order, cache, root)
    side = json.loads(sidecar.read_text())
    if (side.get('arm'), side.get('order'), side.get('base')) != (arm, order, base):
        raise ValueError(f'{cp}: input manifest identity differs')
    files = side.get('inputs_sha256')
    verify_inputs(files, cache, root)
    required = (chain, root / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz', stage,
                root / 'studies/14nov/newmember/s281_cp.py',
                *(root / f'studies/14nov/rundetect/{name}.py' for name in ('package', 'jointcount', 'common', 'detect')),
                root / 'data/proteus_windows.npz', root / 'runs/shuffle_perms.npz')
    def recorded(p):
        if not p.is_file():
            return False
        key = str(p.resolve())
        if key not in cache:
            cache[key] = sha256(p)
        found = [digest for name, digest in files.items() if Path(name).name == p.name]
        return len(found) == 1 and found[0] == cache[key]
    missing = [str(p) for p in required if not recorded(p)]
    if missing:
        raise ValueError(f'{cp}: missing required input hashes: {missing}')
    with np.load(cp, allow_pickle=False) as z:
        for key, value in (('arm', arm), ('order', order), ('base', base),
                           ('source_sha256', sha256(chain)), ('stage_sha256', sha256(stage)),
                           ('provenance_sha256', sha256(prov)), ('input_manifest_sha256', sha256(sidecar))):
            if str(z[key].item()) != value:
                raise ValueError(f'{cp}: {key} differs')
        if bool(z['routing_checked'].item()) or bool(z['scores_are_probabilities'].item()):
            raise ValueError(f'{cp}: unexpected routing/score flags')
        scores = z['scores'].copy()
    if scores.shape != (n, 3) or not np.isfinite(scores).all():
        raise ValueError(f'{cp}: invalid score array')
    return scores


def collect(root, n, session_count):
    cache, outputs = {}, {}
    verify_package(root / 'runs/sealed/final_zips/v66_cp1_cap250.zip',
                   '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed')
    remote = remote_manifest(root / 'runs/s281_leo_inputs/remote.sha256')
    from s281_routing import verify_archive_file
    for order in ORDERS:
        archive = root / f'runs/s281_v66_reference/s242c250c_{order}.npz'
        route = root / f'runs/s281_v66_route/{order}.json'
        if not archive.is_file() or not route.is_file():
            raise ValueError(f'{order}: missing routing trace or archive')
        archive_sha = verify_archive_file(archive, archive.parent / 'sha256.remote.txt')
        verify_route(json.loads(route.read_text()), order, archive_sha, BASES, session_count)
        for arm in ARMS:
            prov = root / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
            if not prov.is_file():
                raise ValueError(f'{arm} {order}: missing chain provenance')
            verify_streams(json.loads(prov.read_text()), arm, order, remote)
        for base in BASES:
            control = load_output(root, 'control', order, base, n, cache)
            zero = load_output(root, 'zero_coral', order, base, n, cache)
            verify_zero(control, zero)
            outputs['control', order, base] = control
        for arm in ARMS[2:]:
            for base in BASES:
                outputs[arm, order, base] = load_output(root, arm, order, base, n, cache)
    return outputs


def main():
    if len(sys.argv) != 1:
        raise SystemExit('This scorer takes no arguments and requires the complete preregistered matrix.')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y = z['y|all_loso']; cells = z['cell|all_loso']; people = z['person|all_loso']
        session_count = len(np.unique(z['sess|all_loso']))
    outputs = collect(ROOT, len(y), session_count)
    metric = {(a, o): np.mean([person_accuracy(outputs[a, o, b], y, cells, people, PEOPLE) for b in BASES], axis=0)
              for a in ARMS if a != 'zero_coral' for o in ORDERS}
    print('§281 new-person cell-balanced accuracy; six bases, scheme-equal, 80% people/order bootstrap')
    print('Routing uses §201 fold-specific emulation; packaged gate and routed returning-person tail are not validated.')
    print('P04/P10 are held-out new people, not returning-session scores.')
    ctl = summarize(np.stack([metric['control', o] for o in ORDERS]), ORDERS, PEOPLE)
    print(f'control: {ctl["mean"]:.3f} {ctl["interval"]} P04 {ctl["people"]["P04"]:.3f} P10 {ctl["people"]["P10"]:.3f}')
    for arm, reference in [(a, 'control') for a in ARMS[2:]] + [('S3','S2'), ('S3','S1')]:
        d = np.stack([metric[arm, o] - metric[reference, o] for o in ORDERS])
        s = summarize(d, ORDERS, PEOPLE)
        print(f'{arm} - {reference}: {s["mean"]:+.3f} {s["interval"]} up {s["people_up"]}/10 '
              f'P04 {s["people"]["P04"]:+.3f} P10 {s["people"]["P10"]:+.3f} schemes {s["schemes"]}')


if __name__ == '__main__':
    main()
