"""Four fixed CPU partitions; assemble only after every exact B1 proof is complete."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ws, old = map(Path,(os.environ['R355_CPU'],os.environ['R355_WS']))
pilot = json.loads((ws/'cpu_pilot.json').read_text())
assert pilot['viable'] and pilot['projected_wall_seconds_4process'] <= 1080
assert pilot['threads'] == pilot['interop_threads'] == 1
cores = sorted(os.sched_getaffinity(0))
assert len(cores) >= 4
arms = {'lag':('instant','lag6'),'envelope':('offset','aligned')}
processes = []
for core, arm in zip(cores,sum((list(v) for v in arms.values()),[])):
    env = dict(os.environ,R355_ARM=arm,R355_CORE=str(core))
    log = (ws/f'{arm}.log').open('w')
    proc = subprocess.Popen([sys.executable,str(ws/'instrument/cpu_banks.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
    processes.append((arm,proc,log))
errors = [(arm,p.wait()) for arm,p,_ in processes]
for _,_,log in processes:
    log.close()
assert all(code == 0 for _,code in errors), errors


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


records, backends = [], []
for family, members in arms.items():
    original_path = old/'output'/family/'proof.json'
    original = json.loads(original_path.read_text())
    updated = copy.deepcopy(original)
    updated.update(execution_mode='CPU_B1',fit_proof_sha256=sha(original_path))
    target = ws/'output'/family
    target.mkdir(parents=True,exist_ok=False)
    for arm in members:
        part = ws/'partitions'/arm
        proof = json.loads((part/'proof.json').read_text())
        assert not proof['labels_opened'] and len(proof['folds']) == 10
        assert proof['device']=='cpu' and proof['dtype']=='float32'
        assert proof['threads']==proof['interop_threads']==1
        assert proof['torch']==pilot['torch']
        backends.append(tuple(proof[k] for k in ('device','dtype','torch','threads','interop_threads','mkldnn')))
        for rec, fitted in zip(proof['folds'],updated['folds']):
            assert rec['held'] == fitted['held'] and rec['arm'] == arm
            assert rec['eight_actual_orders_bitwise'] and rec['reset_copy_prefix_interleaving_exact'] and rec['unchanged_state']
            name = f'{rec["held"].split("/")[-1]}_{arm}'
            assert sha(part/f'{name}.npz') == rec['bank_sha256']
            assert rec['checkpoint_sha256'] == fitted['arms'][arm]['checkpoint_sha256']
            shutil.copyfile(part/f'{name}.npz',target/f'{name}.npz')
            (target/f'{name}.pt').symlink_to(old/'output'/family/f'{name}.pt')
            fitted['arms'][arm].update(bank_sha256=rec['bank_sha256'],new_b1_guards=rec)
            records.append(rec)
    (target/'proof.json').write_text(json.dumps(updated,indent=2)+'\n')
assert len(set(backends)) == 1 and len(records) == 40
(ws/'b1_guard.json').write_text(json.dumps(dict(execution_mode='CPU_B1',labels_opened=False,
    device='cpu',four_processes=True,backend=backends[0],folds=records),indent=2)+'\n')
print('ALL40_CPU_B1_MODELS_EIGHT_ACTUAL_ORDERS_COMPLETE_NO_GOLD',flush=True)
