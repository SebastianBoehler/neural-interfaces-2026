"""Pinned per-window native covariance cache; no labels until explicitly selected."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

BASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
DATA=BASE/'data/proteus_windows.npz'
DATA_SHA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
PRODUCER=Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116')
PRODUCER_SHA='21c1af2e0784329da2f662319ecdac3f6757715cbcece604fa1db46d21b659bf'
READER=Path('/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116/source_labels.py')
READER_SHA='35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc'
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def labels(rows,n):
    if sha(READER)!=READER_SHA:raise ValueError('labelreader changed')
    spec=importlib.util.spec_from_file_location('selected334labels',READER)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    return reader.read_training_labels(DATA,np.asarray(rows),n)


def metadata():
    if sha(DATA)!=DATA_SHA or sha(PRODUCER/'producer.sha256')!=PRODUCER_SHA:raise ValueError('334data/produceridentity')
    with np.load(DATA,allow_pickle=False) as z:d={k:z[k] for k in ('subject','session','interface','ch_names')}
    d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
    people=np.array([str(p).split('/')[-1] for p in d['subject']])
    fit=np.flatnonzero(np.isin(people,[f'P{i:02d}' for i in range(4,11)]))
    query=np.flatnonzero(np.isin(people,['P02','P03']))
    with np.load(BASE/'runs/shuffle_perms.npz') as z:
        perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    if any(not np.array_equal(np.sort(p),np.arange(len(people))) for p in perms.values()):raise ValueError('badorders')
    return d,people,fit,query,perms


def official():
    sys.path[:0]=[str(BASE/'experiments'),str(BASE/'2026-competition')]
    spec=importlib.util.spec_from_file_location('protocol334',BASE/'experiments/protocol.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    source=Path(result.score.__globals__['balanced_accuracy'].__code__.co_filename).resolve()
    expected=(BASE/'2026-competition/tracks/bci_decoding/benchmark_utils/metrics.py').resolve()
    module=sys.modules[result.score.__globals__['balanced_accuracy'].__module__]
    if Path(module.__file__).resolve()!=expected:raise ValueError('metricmoduleidentity')
    if source!=expected or sha(source)!='fcc739400806a74259f248618501376a4fe53f2b5a40ccdc20db6a9e2369787d':raise ValueError('officialmetricpath/hash')
    if sha(BASE/'experiments/protocol.py')!='390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c':raise ValueError('protocolhash')
    return result
