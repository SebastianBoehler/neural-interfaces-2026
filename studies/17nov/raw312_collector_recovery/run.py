"""Run the unchanged §312 collector with its intended sibling imports."""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2')
source=(ROOT/'collect.py').read_text()
old="sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'studies/14nov/newmember')]"
new=old+"\nsys.path.insert(0, str(ROOT))"
anchor='from run import ARMS, ORDERS'
guard='''sys.path.insert(0, str(ROOT))
from run import ARMS, ORDERS
_expected_modules = {'run': ROOT/'run.py', 'data': ROOT/'data.py',
                     'protocol': ROOT/'experiments/protocol.py',
                     'score_s_v66p': ROOT/'studies/14nov/newmember/score_s_v66p.py'}
for _name, _path in _expected_modules.items():
    if Path(sys.modules[_name].__file__).resolve() != _path.resolve():
        raise ValueError(f'wrong imported module: {_name}')'''
if source.count(old)!=1 or source.count(anchor)!=1:raise ValueError('collector import anchor changed')
if os.environ.get('NI26_WS')!=str(ROOT):raise ValueError('wrong producer workspace')
check=sys.argv[1:] == ['--check-imports']
if sys.argv[1:] and not check:raise ValueError('unexpected wrapper arguments')
namespace={'__name__':'collector_import_check' if check else '__main__','__file__':str(ROOT/'collect.py')}
exec(compile(source.replace(old,new,1).replace(anchor,guard,1),str(ROOT/'collect.py'),'exec'),namespace)
if check:
    if namespace['ARMS']!=('S','F','T','C') or len(namespace['ORDERS'])!=8:raise ValueError('constants changed')
    path=ROOT/'output/fb_S_s0_P01.npz';sha=namespace['sha']
    with np.load(path,allow_pickle=False) as z: meta=json.loads(str(z['metadata']))
    if (meta['held'],meta['architecture'],meta['arm'],meta['seed'])!=('P01','fb','S',0):raise ValueError('metadata differs')
    if len(meta['fit_people'])!=9 or len(meta['inner_fit_people'])!=8:raise ValueError('source split differs')
    if meta['source_manifest']!=sha(ROOT/'source.sha256') or sha(path.with_suffix('.pt'))!=meta['checkpoint_sha256']:
        raise ValueError('source/checkpoint mismatch')
    for raw, expected in meta['raw_hashes'].items():
        if sha(raw)!=expected:raise ValueError('raw source mismatch')
    print('OK: actual run/data/protocol/scorer imports, constants and source/checkpoint/raw metadata; no labels opened')
