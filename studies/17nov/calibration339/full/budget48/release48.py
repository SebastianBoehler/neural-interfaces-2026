"""Prospectively authorized resource envelope only; immutable scientific consumer."""
import inspect
import json
import os
import sys
from pathlib import Path
CHAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_chain_v1_20261117')
sys.path.insert(0,str(CHAIN/'studies/17nov/representation_chain'))
import release339 as old
from config import sha
if sha(CHAIN/'representation-source.sha256')!='84205fc97487176b6efa31e2e7b5b006dd8164f8e3e8d3d670d00e4d4dc3053c':raise ValueError('consumeridentitychanged')
text=inspect.getsource(old.main)
if text.count('if projection>32:')!=1:raise ValueError('budgetanchorchanged')
env=dict(vars(old));exec(compile(text.replace('if projection>32:','if projection>48:'),'339_authorized48CPUh_envelope','exec'),env)
env['main']()
p=CHAIN/'released.json';r=json.loads(p.read_text());r.update(actual_CPUcorehour_envelope=48,budget_source_sha256=sha(Path(os.environ['BUDGET_ROOT'])/'source.sha256'));p.write_text(json.dumps(r,indent=2))
