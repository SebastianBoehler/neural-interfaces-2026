"""Patch private copies of the existing §292 chain/scorer; no new evaluator."""
import importlib.util
import json
import sys
from pathlib import Path

from config import GROUPS, recipes


STREAM = '''def stream(fam, bn):
    # Fixed two-source-seed ensemble; identical member in every incumbent CP base.
    assert bn in BS and NM_SEEDS == [0, 1]
    if fam not in STREAM:
        path = f"{NM_DIR}/{TAG}_{fam}_base-{p0}-ensemble2.npz"
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["index"], np.arange(N))
            import json
            metadata = json.loads(str(z["metadata"].item()))
            assert metadata["seeds"] == [0, 1] and metadata["operation"] == "arithmetic_probability_mean"
            assert metadata["family"] == fam and metadata["order"] == O
            STREAM[fam] = norm(z["proba"])
    return STREAM[fam]
'''

REMOTE = '''def remote_manifest(path):
    lock = json.loads((path.parent / 'lock.json').read_text())
    if sha256(path) != lock['manifest_sha256']:
        raise ValueError('representation stream manifest changed')
    verify_inputs(lock['producer_inputs_sha256'], {}, ROOT)
    remote = {}
    for line in path.read_text().splitlines():
        digest, name = line.split()
        if name in remote:
            raise ValueError('duplicate representation stream')
        remote[name] = digest
    expected = {f'ttt18{o}_{f}_base-all_loso-ensemble2.npz' for o in ORDERS for f in FAMILIES315}
    if set(remote) != expected:
        raise ValueError('incomplete ensemble member matrix')
    return remote
'''
VERIFY = '''def verify_streams(manifest, arm, order, remote):
    add, swap, _ = RECIPES[arm]
    families = [] if add == swap == 'none' else (add if add != 'none' else swap).split(',')
    for family in families:
        name = f'ttt18{order}_{family}_base-all_loso-ensemble2.npz'
        found = [v for k, v in manifest['inputs_sha256'].items() if Path(k).name == name]
        if len(found) != 1 or remote.get(name) != found[0]:
            raise ValueError(f'{arm} {order}: ensemble input provenance differs')
'''


def main(group, root):
    root = Path(root).resolve(strict=True)
    if (root / '.git').exists():
        raise ValueError('private workspace required')
    spec = importlib.util.spec_from_file_location('lane292_patch', root / 'studies/16nov/lane1/patch_private.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.RECIPES = recipes(group)
    folder = root / 'studies/14nov/newmember'
    names = ('s281_provenance.py', 'run_s.py', 's281_cp.py', 'score_s_v66p.py')
    source = {name: (folder / name).read_text() for name in names}
    out = module.patch_sources(source)
    replace = module.replace_once
    out = {k: v.replace('zero_eegn', 'zero_repr') for k, v in out.items()}
    out['s281_provenance.py'] = out['s281_provenance.py'].replace("'0,1,2,3,4,5'", "'0,1'")
    run = out['run_s.py']
    run = replace(run, 'all(f in ("eegn", "eegn_dist2", "fb_coral3", "fb_coral10") for f in ADD + SW)',
                  f'all(f in {GROUPS[group]["families"]!r} for f in ADD + SW)')
    start, end = run.index('def stream(fam, bn):'), run.index('\n\n_apply = S.apply')
    run = run[:start] + STREAM + run[end:]
    start, end = run.index('NM_DIR ='), run.index('\nTAG =')
    run = run[:start] + 'NM_DIR = REPO + "/runs/representation_streams" if FAMS_ else None' + run[end:]
    out['run_s.py'] = run
    score = out['score_s_v66p.py']
    score = score.replace("'0,1,2,3,4,5'", "'0,1'")
    score = replace(score, "[('D','E')]", repr(GROUPS[group]['contrasts']))
    score = replace(score, "root / 'runs/lane1_inputs/streams.sha256'", "root / 'runs/representation_inputs/streams.sha256'")
    start, end = score.index('def verify_streams('), score.index('\n\ndef remote_manifest(')
    score = score[:start] + VERIFY + score[end:]
    start, end = score.index('def remote_manifest('), score.index('\n\ndef load_output(')
    score = score[:start] + REMOTE + score[end:]
    score = replace(score, 'def remote_manifest(path):', f'FAMILIES315 = {GROUPS[group]["families"]!r}\n\ndef remote_manifest(path):')
    out['score_s_v66p.py'] = score
    for name, value in out.items():
        compile(value, str(folder / name), 'exec')
        (folder / name).write_text(value)
    # Runtime wrapper forwards explicit seed identity to the patched recipe guard.
    wrapper = folder / 'run_s_v66.py'
    text = wrapper.read_text().replace('"0,1,2,3,4,5"', '"0,1"')
    wrapper.write_text(text)
    (root / 'representation_config.json').write_text(json.dumps(dict(group=group, recipes=recipes(group), **GROUPS[group]), indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
