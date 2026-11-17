"""Verify zero and both joint mechanisms using the original §304 guards."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'studies/14nov/newmember'
sys.path[:0]=[str(HERE),str(ROOT/'studies/16nov/followups/coral_repair/chain/recovery')]
from s281_guard import verify
from s281_routing import verify_archive_file
from control_source import install
spec=importlib.util.spec_from_file_location('score318',HERE/'score_s_v66p.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
install(s,ROOT)
with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
    n=len(z['y|all_loso']); sessions=len(np.unique(z['sess|all_loso']))
order='within0'; archive=ROOT/f'runs/s281_v66_reference/s242c250c_{order}.npz'
s.verify_route(json.loads((ROOT/f'runs/s281_v66_route/{order}.json').read_text()),order,
               verify_archive_file(archive,archive.parent/'sha256.remote.txt'),s.BASES,sessions)
s.verify_package(ROOT/'runs/sealed/final_zips/v66_cp1_cap250.zip',
                 '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed')
remote=s.remote_manifest(ROOT/'runs/coral304_inputs/streams.sha256')
with np.load(ROOT/f'runs/newmember/s281_v66p_control_{order}_gate.npz') as z: control=dict(z)
for arm in ('zero318','JA','JS','FBL2'):
    prov=ROOT/f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
    s.verify_streams(json.loads(prov.read_text()),arm,order,remote)
    with np.load(ROOT/f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz') as z:
        verify(control,dict(z),zero=arm=='zero318')
    for base in s.BASES:
        result=s.load_output(ROOT,arm,order,base,n,{})
        if arm=='zero318':s.verify_zero(s.load_output(ROOT,'control',order,base,n,{}),result)
print('OK: §318 zero/add/swap/FBL gate, streams, six CP, route and package',flush=True)
