"""Run the pinned §431 route with only actual source labels available."""
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys

import numpy as np
from source_labels import read_training_labels

BASE=Path('/mnt/lustre/work/gehler/gfh527')
ROOT=BASE/'ni26_aux431_20261118'
BANK=BASE/'ni26_friday_ea_returning_fresh_v2_20261118'
DATA=ROOT/'data/proteus_windows.npz'
HERE=Path(__file__).resolve().parent
PROTO,ORDER,STACK='seen_session','full0','B2_pool'
SESSIONS={f'Dreyer2026Proteus/{p}|{d}' for p in ('P04','P10') for d in ('2','3')}
FIRSTS=dict(zip(sorted(SESSIONS),(2153,2383,1238,1326)))
LEGACY_AGES=dict(zip(sorted(SESSIONS),(10,10,28,10)))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def inputs():
    with np.load(DATA,allow_pickle=False) as z:
        people,days=(z[k] for k in ('subject','session'))
    stream=np.char.add(np.char.add(people,'|'),days)
    test=np.flatnonzero(np.isin(stream,list(SESSIONS)))
    expected={}
    for p,n in (('P04',2278),('P10',2270)):
        selected=(people==f'Dreyer2026Proteus/{p}')&(days!='1')
        expected[f'{p}_s23']=np.flatnonzero(~selected)
        assert len(expected[f'{p}_s23'])==n
    assert len(test)==762
    return people,days,stream,test,expected


def blocked(key):
    return key.split('|')[0].lower() in {'y','y_tr','y_te','labels','label','target','targets'}


class SourceArchive(Mapping):
    def __init__(self,raw,path,source):
        self.raw,self.path,self.source=raw,path,source
        self.files=[k for k in raw.files if not blocked(k) or k=='y_tr' and source is not None]

    def __getitem__(self,key):
        if blocked(key):
            if key!='y_tr' or self.source is None:
                raise RuntimeError(f'test label denied at {self.path}:{key}')
            assert np.array_equal(np.sort(self.raw['idx_tr']),self.source)
        return self.raw[key]

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self,key):
        return key in self.files

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.raw.close()

    def close(self):
        self.raw.close()


@contextmanager
def without_test_labels(splits):
    load=np.load
    opened={}
    def read(file,*args,**kwargs):
        path=Path(file).absolute()
        if any(x in path.parts for x in ('labels','chainout','retv6','aux431')) or path.name=='shufpkg.npz':
            raise RuntimeError(f'cached downstream prediction denied: {path}')
        if path not in opened:
            opened[path]=sha(path)
        raw=load(file,*args,**kwargs)
        assert hasattr(raw,'files'),path
        match=re.fullmatch(r'seen_session-(P(?:04|10)_s23)\.npz',path.name)
        source=splits[match[1]] if match and path.parent.name=='feats_exact' else None
        return SourceArchive(raw,path,source)
    np.load=read
    try:
        yield opened
    finally:
        np.load=load


def replace(src,before,after):
    assert src.count(before)==1,before
    return src.replace(before,after)


def module(name,path,src,**symbols):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    m.__dict__.update(symbols)
    sys.modules[name]=m
    exec(compile(src,str(path),'exec'),m.__dict__)
    return m


class OwnTargets:
    def __init__(self,splits,n):
        self.splits,self.n=splits,n
        self.fold=None

    def __getitem__(self,rows):
        assert self.fold in self.splits and np.isin(rows,self.splits[self.fold]).all()
        return read_training_labels(DATA,np.asarray(rows),self.n)


def remove_reference(head):
    return replace(head,'; REF = np.load(REPO + "/runs/shufpkg.npz")','')


def construct(people,days,splits):
    env=dict(BASES=STACK,ORDERS=ORDER,ARMS='base,ER,T',PROTO=PROTO,RT_PROTO=PROTO,
             GATE_V51B='1',FT_SFX='QF',FT_FORM='s15',RT_MEMBERS='1',SCHEMES='full',PERMS='1')
    assert not set(('LABEL_RT','TAU','G3','PSC','KAPPA','BETA','NI26_REPLAY_DATA','SAVE_OUT'))&os.environ.keys()
    os.environ.update(env)
    sys.path[:0]=[str(ROOT/p) for p in ('studies/14nov/pview','studies/13nov','replay','experiments')]

    loader=ROOT/'replay/rlib.py'
    src=replace(loader.read_text(),'; Y = f["y"]','')
    src=replace(src,'Y[i], i','np.full(len(i), -1, dtype=int), i')
    module('rlib',loader,src)

    shuffle=ROOT/'studies/13nov/shuffle.py'
    src=replace(shuffle.read_text(),'len(A.L.Z[f"y|{p}"])','len(A.L.Z[f"idx|{p}"])')
    source_line=next(line for line in src.splitlines() if line.startswith('DICTS = {'))
    src=replace(src,source_line,'DICTS = (A.L.Z, A.pl.B, SC.B)')
    module('shuffle',shuffle,src)

    own=ROOT/'studies/12nov/ownspace.py'
    targets=OwnTargets(splits,len(people))
    src=replace(own.read_text(),'z["X"].astype(np.float64), z["y"], z["interface"]',
                'z["X"].astype(np.float64), INPUT_TARGETS, z["interface"]')
    module('ownspace',own,src,INPUT_TARGETS=targets)

    public=np.flatnonzero(np.isin(people,('Dreyer2026Proteus/P04','Dreyer2026Proteus/P10'))&(days=='1'))
    assert len(public)==381
    only_public=np.full(len(people),-1,dtype=int)
    only_public[public]=read_training_labels(DATA,public,len(people))
    pview=ROOT/'studies/14nov/pview/data.py'
    src=replace(pview.read_text(),'len(W["y"])','len(W["subject"])')
    src=replace(src,'Y = Z["y|all_loso"].astype(int); assert (Y == W["y"]).all()','Y = PUBLIC_TARGETS')
    module('data',pview,src,PUBLIC_TARGETS=only_public)

    path=ROOT/'studies/14nov/pview/chain.py'
    src=replace(path.read_text(),'exec(compile(HEAD, F6, "exec"), R6)',
                                'exec(compile(remove_reference(HEAD), F6, "exec"), R6)')
    ctx=module('returning_pview',path,src,remove_reference=remove_reference)
    original=ctx.K.members
    def members(proto,session,person):
        assert proto==PROTO and targets.fold is None
        targets.fold=ctx.K.NM[proto,session][2]
        try:
            return original(proto,session,person)
        finally:
            targets.fold=None
    ctx.K.members=members
    return ctx
