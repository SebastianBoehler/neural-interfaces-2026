"""Verify §304 zero/add/swap full-chain smoke without calculating metrics."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = ROOT / 'studies/14nov/newmember'
sys.path.insert(0, str(HERE))
from s281_guard import verify
from s281_routing import verify_archive_file
spec = importlib.util.spec_from_file_location('score304', HERE / 'score_s_v66p.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
    n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
order = 'within0'
archive = ROOT / f'runs/s281_v66_reference/s242c250c_{order}.npz'
route = ROOT / f'runs/s281_v66_route/{order}.json'
archive_sha = verify_archive_file(archive, archive.parent / 'sha256.remote.txt')
s.verify_route(json.loads(route.read_text()), order, archive_sha, s.BASES, sessions)
s.verify_package(ROOT / 'runs/sealed/final_zips/v66_cp1_cap250.zip',
                 '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed')
remote = s.remote_manifest(ROOT / 'runs/coral304_inputs/streams.sha256')
with np.load(ROOT / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as z: control = dict(z)
for arm in ('zero304','PLA','CCS10'):
    prov = ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
    s.verify_streams(json.loads(prov.read_text()), arm, order, remote)
    with np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz') as z:
        verify(control, dict(z), zero=arm == 'zero304')
    for base in s.BASES:
        value = s.load_output(ROOT, arm, order, base, n, {})
        if arm == 'zero304':
            ref = s.load_output(ROOT, 'control', order, base, n, {})
            s.verify_zero(ref, value)
print('OK: §304 zero/add/swap gate, stream lock, all six CP, archive route and package', flush=True)
