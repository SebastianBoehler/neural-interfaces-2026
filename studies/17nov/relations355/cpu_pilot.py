"""Source-only B1 CPU timing; no query inference or label member access."""
import copy
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

old = Path(os.environ['R355_WS'])
out = Path(os.environ['R355_CPU'])
root = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
sys.path.insert(0,str(old/'code'))
from model import ARMS, Relations
from run import sha

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
with np.load(root/'data/proteus_windows.npz') as z:
    people = z['subject'].copy()
    rows = np.flatnonzero(people != 'Dreyer2026Proteus/P01')[:128]
    x = torch.from_numpy(z['X'][rows].copy())
assert len(x) == 128 and not any(people[rows] == 'Dreyer2026Proteus/P01')
proof = dict(scope='SOURCE ONLY CPU B1 timing, no held forward/labels/accuracy',
    torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),
    interop_threads=torch.get_num_interop_threads(),dtype='float32',rows=rows.tolist(),models={})
with torch.no_grad():
    for family, arms in ARMS.items():
        fitted = json.loads((old/'output'/family/'proof.json').read_text())['folds'][0]
        assert fitted['held'] == 'Dreyer2026Proteus/P01'
        for arm in arms:
            checkpoint = old/'output'/family/f'P01_{arm}.pt'
            assert sha(checkpoint) == fitted['arms'][arm]['checkpoint_sha256']
            net = Relations(arm)
            net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)); net.eval()
            state = {k:v.clone() for k,v in net.state_dict().items()}
            t = time.monotonic(); cold = net(x[:1]).softmax(1); cold_s = time.monotonic()-t
            t = time.monotonic()
            values = torch.cat([net(x[i:i+1]).softmax(1) for i in range(128)])
            elapsed = time.monotonic()-t
            assert torch.equal(cold,values[:1]) and torch.isfinite(values).all()
            twin = copy.deepcopy(net)
            assert torch.equal(values[:17],torch.cat([twin(x[i:i+1]).softmax(1) for i in range(17)]))
            assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
            proof['models'][arm] = dict(seconds_128=elapsed,cold_seconds=cold_s,
                checkpoint_sha256=sha(checkpoint),source_repeat_copy_exact=True,
                seconds_per_window=elapsed/128)
# Four independent one-thread arm partitions, each all10models x8orders + guard prefixes.
# Slowest arm gives conservative wall; 2x margin plus120s import/load/startup headroom.
proof['projected_wall_seconds_4process'] = max(v['seconds_per_window'] for v in proof['models'].values()) * (2655*8+10*3*17)*2+120
proof['maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
proof['viable'] = proof['projected_wall_seconds_4process'] <= 1080 and proof['maxrss_kib'] <= 2500*1024
(out/'cpu_pilot.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
