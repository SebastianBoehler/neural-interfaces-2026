"""Lock the six independently restored source-only fits before any target scoring."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

WORK = Path('/mnt/lustre/work/gehler/gfh527')
PRIMARY = WORK/'ni26_joint382_v1_20261118'
sys.path.insert(0,str(PRIMARY))
from restore import joint, prior, init
sys.path.insert(0,str(WORK/'ni26_conflict388_v1_20261118'))
from actor388 import ARCH, REFERENCES, ROWS, SEEDS
ARMS = ('T','S','C','blank')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    records = {}
    for seed in SEEDS:
        runs = []
        for arm in ('T','S','C'):
            key = f'{arm}_{seed}'
            r = json.loads((Path('fits')/(key+'.json')).read_text())
            assert (r['arm'], r['seed'], r['steps']) == (arm,seed,120)
            assert r['source']==sha('train.py') and r['rows']==ROWS and 'H100' in r['device']
            assert r['bundle']=='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
            assert r['weights']==sha(Path('fits')/(key+'.pt'))
            records[key] = r
            runs.append(r)
        for key in ('before','fixed','episodes'):
            assert all(r[key]==runs[0][key] for r in runs)
        for key in ('gradient','after'):
            assert len({json.dumps(r[key],sort_keys=True) for r in runs})==3
    return records


def models(native):
    assert json.loads(Path('fits.json').read_text())==verify()
    nets = {}
    for seed in SEEDS:
        for arm in ARMS[:-1]:
            m = init(native,'J')
            m.load_state_dict(torch.load(f'fits/{arm}_{seed}.pt',map_location='cpu',weights_only=True))
            m.spatial_conv[1].mode = 'real'
            nets[seed,arm] = m.eval()
        blank = copy.deepcopy(nets[seed,'T'])
        blank.spatial_conv[1].mode = 'blank'
        nets[seed,'blank'] = blank
    return nets


if __name__=='__main__':
    joint.check_manifest(PRIMARY,'67829e9fe023605f5426d1bb035f9b51f00e8fc026c9b6824fdae3035b92c5a9')
    Path('fits.json').write_text(json.dumps(verify(),indent=2))
