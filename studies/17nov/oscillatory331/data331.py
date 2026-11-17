"""Selected source labels, immutable official metric and source-only scaling."""
import hashlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

BASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
SOURCE='528284dc7c07c835534ca94e6637c00df44a38755b62313d5cfbfac25a265ef4'
DATA=BASE/'data/proteus_windows.npz'
DATA_SHA='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
LABEL=Path('/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116/source_labels.py')
LABEL_SHA='35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check():
    if sha(BASE/'source.sha256')!=SOURCE or sha(DATA)!=DATA_SHA or sha(LABEL)!=LABEL_SHA:raise ValueError('331dependencyidentity')
    for line in (BASE/'source.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha(BASE/name)!=expected:raise ValueError('331source dependency changed')


def labels(rows,n):
    if sha(LABEL)!=LABEL_SHA:raise ValueError('selected-labelreader changed')
    spec=importlib.util.spec_from_file_location('source331labels',LABEL)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.read_training_labels(DATA,rows,n)


def load():
    check()
    with np.load(DATA,allow_pickle=False) as z:
        d={k:z[k] for k in ('X','subject','session','interface','ch_names')}
    d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
    people=np.array([str(p).split('/')[-1] for p in d['subject']])
    fit=np.flatnonzero(np.isin(people,[f'P{i:02d}' for i in range(4,11)]))
    query=np.flatnonzero(np.isin(people,['P02','P03']))
    if set(people[fit])!={f'P{i:02d}' for i in range(4,11)} or set(people[query])!={'P02','P03'}:raise ValueError('source7/queryscope')
    return d,fit,query,people


def calibrate(d,fit):
    from model import Decoder
    rms=float(np.sqrt(np.mean(d['X'][fit].astype(np.float64)**2)))
    if not np.isfinite(rms) or rms<=0:raise ValueError('sourceRMSinvalid')
    model=Decoder('local',torch.zeros(287),torch.ones(287));blocks=[]
    with torch.no_grad():
        for r in np.array_split(fit,max(1,(len(fit)+63)//64)):
            blocks.append(model.moments(torch.from_numpy((d['X'][r]/rms).astype(np.float32))).numpy())
    moment=np.concatenate(blocks).astype(np.float64)
    mean=moment.mean(0).astype(np.float32);sd=moment.std(0).clip(1e-6).astype(np.float32)
    meta=dict(fit_rows_sha256=digest(fit),fit_input_sha256=digest(d['X'][fit]),rms=rms,
              moment_mean_sha256=digest(mean),moment_sd_sha256=digest(sd),excluded=['P01','P02','P03'])
    return rms,torch.from_numpy(mean),torch.from_numpy(sd),meta


def official():
    sys.path.insert(0,str(BASE/'experiments'))
    import protocol
    if Path(protocol.__file__).resolve()!=(BASE/'experiments/protocol.py').resolve():raise ValueError('wrong official protocol module')
    return protocol
