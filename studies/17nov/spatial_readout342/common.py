"""Pinned source-only encoder and explicit token layouts for §342."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly

OLD = Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_A100source_v1_20261117')
DATA = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/proteus_windows.npz')
LABEL = Path('/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116/source_labels.py')
WEIGHT_SHA = 'a28e97f7bd73bd7ff55f3b5d680f48807c335cf3134d1281228a061056aaf52b'
MODULE_SHA = 'ff99c31afa9dceb1923f796e349ea9e132d633b8816c251a6e50be00e3306c8d'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    if Path(value.__file__).resolve() != path.resolve():
        raise ValueError('module ownership differs')
    return value


def load():
    if sha(DATA) != DATA_SHA:
        raise ValueError('official tensor changed')
    with np.load(DATA, allow_pickle=False) as z:
        d = {k: z[k] for k in ('X', 'subject', 'session', 'ch_names')}
    people = np.array([str(s).split('/')[-1] for s in d['subject']])
    source = np.flatnonzero(np.isin(people, ['P04', 'P05', 'P06', 'P07', 'P08', 'P10']))
    query = np.flatnonzero(people == 'P09')
    if set(people[source]) != {'P04', 'P05', 'P06', 'P07', 'P08', 'P10'}:
        raise ValueError('SOURCE6 mismatch')
    return d, people, source, query


def labels(rows, people):
    if not set(people[rows]) <= {'P04', 'P05', 'P06', 'P07', 'P08', 'P10'}:
        raise ValueError('source-only labels enforced')
    if sha(LABEL) != '35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc':
        raise ValueError('selected-label API changed')
    return module('ni26_s342_labels', LABEL).read_training_labels(DATA, rows, len(people))


def encoder(prior):
    assets = OLD / 'assets'
    if sha(assets / 'atom-large.safetensors') != WEIGHT_SHA or sha(assets / 'labram.py') != MODULE_SHA:
        raise ValueError('cached default encoder changed')
    sys.path.insert(0, str(assets / 'runtime'))
    upstream = module('ni26_s342_upstream', assets / 'labram.py')
    torch.manual_seed(329)
    model = upstream.labram_large_patch200_200(num_classes=0, init_values=.1, use_mean_pooling=False).eval()
    del model.logit_scale
    if prior == 'pretrained':
        own = model.state_dict()
        allowed = json.loads((OLD / 'compatibility.json').read_text())['removed_pretraining_keys']
        mapped = {}
        for name, value in load_file(str(assets / 'atom-large.safetensors')).items():
            key = name[8:] if name.startswith('student.') else name
            if key in own:
                mapped[key] = value
            elif name not in allowed:
                raise ValueError('unexpected checkpoint key')
        if set(mapped) != set(own) or len(mapped) != 425:
            raise ValueError('strict 425-key load failed')
        model.load_state_dict(mapped, strict=True)
    elif prior != 'random':
        raise ValueError('unknown prior')
    return model, upstream


def inputs(raw):
    x = resample_poly(raw.astype(np.float64), 5, 3, axis=-1).astype(np.float32) / 100
    if x.shape[1:] != (41, 800) or not np.isfinite(x).all():
        raise ValueError('input preprocessing differs')
    return torch.from_numpy(x.reshape(-1, 41, 4, 200))


def layouts(patch):
    if patch.ndim != 3 or patch.shape[1:] != (164, 400):
        raise ValueError('patch shape differs')
    grid = patch.reshape(-1, 41, 4, 400)
    return dict(pooled400=patch.mean(1), temporal1600=grid.mean(1).reshape(-1, 1600),
                spatial16400=grid.mean(2).reshape(-1, 16400), full65600=grid.reshape(-1, 65600))


def state_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def gpu():
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('explicit A100 required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
