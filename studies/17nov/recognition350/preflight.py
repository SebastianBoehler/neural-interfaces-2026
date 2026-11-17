"""Clean worker import and one actual source window per generated mode; no labels."""
import json
from pathlib import Path
import numpy as np
import torch
from returning_pair_guard import ROOT,zero
from record import load_model
from package_guard import frame,instrument
from returning_assets import sha

torch.set_num_threads(1)
with np.load(ROOT/'neural-interfaces-2026/data/proteus_windows.npz',allow_pickle=True) as z:
    channels=z['ch_names'].copy();raw=z['X'][np.flatnonzero((z['subject']=='Dreyer2026Proteus/P04')&(z['session'].astype(str)=='1'))[0]].copy()
p=json.loads(Path('build.json').read_text())
for entry in p['artifacts']:
    path=Path('bundles')/entry['fold']/str(entry['mode'])
    if any(sha(path/k)!=v for k,v in entry['files'].items()):raise ValueError('private generated source changed')
    model=instrument(load_model(path,channels,'policy350_preflight_'+entry['fold']+str(entry['mode']),'cpu'),False)
    zero(model);frame(model,raw);model.reset_state();zero(model)
print('SIX_PRIVATE_MODES_REAL_SOURCE_ONE_WINDOW_LOAD_RESET_OK',flush=True)
