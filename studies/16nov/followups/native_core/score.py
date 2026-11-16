"""Use the existing guarded shuffled-chain scorer for §301 native members."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('NI26_WS', HERE.parents[3])).resolve()
sys.path[:0] = [str(HERE), str(ROOT / 'studies/16nov/followups/geometry')]
from member_chain import ARMS, ORDERS  # noqa: E402
from chain import sha  # noqa: E402


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected one scorer anchor: {old[:80]}')
    return source.replace(old, new, 1)


def patch_score(source):
    source = replace_once(source, 'from chain import ARMS, ORDERS, sha  # noqa: E402',
                          'from member_chain import ARMS, ORDERS\nfrom chain import sha')
    source = source.replace("'geo_zero'", "'Rzero'")
    old_pairs = "pairs = [(a, 'control') for a in ARMS[1:]] + [('geo_d50', 'geo_s50'), ('geo_d25', 'geo_d50'), ('geo_d100', 'geo_d50')]"
    source = replace_once(source, old_pairs,
                          "pairs = [('Nfixed', 'control'), ('Ntrain', 'control'), ('Ntrain', 'Nfixed')]")
    source = replace_once(source, "'evidence/next_wave_16nov/followups_geometry_results.json'",
                          "'evidence/next_wave_16nov/native_core_s301_results.json'")
    source = replace_once(source, "'outer reconstructed-core posterior delta, fixed PC128/shadow/FT; new-person routing emulation, not packaged-core or returning-tail validation'",
                          "'native core posterior delta anchored to historical rebuilt core; fixed PC128/shadow/FT; new-person routing emulation only'")
    return source


def verify_native(lock, manifest_path):
    root = ROOT
    names = {f'{a}_{o}.npz' for a in ARMS for o in ORDERS}
    if set(lock.get('files', {})) != names:
        raise ValueError('native delta manifest is incomplete')
    for arm in ARMS:
        for order in ORDERS:
            name = f'{arm}_{order}.npz'
            delta = root / 'runs/s301_native/deltas' / name
            if sha(delta) != lock['files'][name]:
                raise ValueError(f'{name}: changed delta')
            prov = root / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
            entries = json.loads(prov.read_text())['inputs_sha256']
            match = [v for k, v in entries.items() if Path(k).name == name]
            manifests = [v for k, v in entries.items() if Path(k).resolve() == manifest_path]
            if match != [lock['files'][name]] or manifests != [sha(manifest_path)]:
                raise ValueError(f'{name}: chain lacks locked input provenance')


def main():
    path = ROOT / 'studies/16nov/followups/geometry/score.py'
    context = {'__name__': 's301_existing_score', '__file__': str(path)}
    exec(compile(patch_score(path.read_text()), str(path), 'exec'), context)
    context['verify_geometry'] = verify_native
    context['main']()


if __name__ == '__main__':
    main()
