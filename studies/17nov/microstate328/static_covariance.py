"""Unchanged spatial-token occupancy versus matched angular covariance (§328 adjunct)."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
ORIGINAL=Path('/mnt/lustre/work/gehler/gfh527/ni26_microstate328_v2_20261117')
ORIGINAL_SHA='4dade504b4011b6697c3a2a78d073b342b05cce70581ec24da173e950b1c3527'
DATA=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/proteus_windows.npz')
LABELS=Path('/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116/source_labels.py')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    start=time.monotonic()
    if sha(ORIGINAL/'source.sha256')!=ORIGINAL_SHA:raise ValueError('originalsource differs')
    for folder in (ROOT,ORIGINAL):
        for line in (folder/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha(folder/name)!=expected:raise ValueError('immutable source differs')
    previous=json.loads((ORIGINAL/'result.json').read_text())
    if sha(DATA)!=previous['guard']['data_sha256'] or sha(LABELS)!=previous['label_reader_sha256']:raise ValueError('data/labelreader changed')
    rep=module('ni26_micro328_rep',ORIGINAL/'representation.py');reader=module('ni26_micro328_labels',LABELS)
    with np.load(ORIGINAL/'artifacts.npz') as z:
        fit,query,centers,random,scale=[z[k] for k in ('source_rows','query_rows','centers','random','scale')]
    rows=np.r_[fit,query];n=len(fit)
    with np.load(DATA,allow_pickle=False) as z:
        raw=z['X'][rows];people=np.array([str(s).split('/')[-1] for s in z['subject']]);total=len(people)
        stream=np.char.add(np.char.add(z['subject'][query],'|'),z['session'][query])
    if set(people[fit])!={f'P{i:02d}' for i in range(4,11)} or np.any(people[rows]=='P01'):raise ValueError('sourcequery scope differs')
    bank=rep.features(raw,centers,random,float(scale));topo,_=rep.topographies(raw)
    matrix=np.einsum('ntc,ntd->ncd',topo,topo)/480
    if not np.allclose(np.trace(matrix,axis1=1,axis2=2),1,atol=1e-8) or np.linalg.eigvalsh(matrix).min() < -1e-10:raise ValueError('angularmoment invariant failed')
    i,j=np.triu_indices(41);cov=matrix[:,i,j]*np.where(i==j,1.,np.sqrt(2.))
    if cov.shape!=(len(raw),861) or not np.isfinite(cov).all():raise ValueError('angularcov width/finite failed')
    random_occ=rep.summarize(rep.tokenize(raw,random))[:,:16]
    rawblock=bank['raw287'];occ=bank['occupancy']
    features=dict(raw287=rawblock,raw_occupancy=bank['raw_occupancy'],raw_random_occupancy=np.c_[rawblock,random_occ],
        raw_cov=np.c_[rawblock,cov],raw_cov_occupancy=np.c_[rawblock,cov,occ],raw_cov_random_occupancy=np.c_[rawblock,cov,random_occ])
    for name in ('raw287','raw_occupancy'):
        digest=hashlib.sha256(np.ascontiguousarray(features[name]).tobytes()).hexdigest()
        if digest!=previous['results'][name]['feature_sha256']:raise ValueError('originalfeature identity failed')
    for name,value in features.items():
        if not np.array_equal(value[:,:287],rawblock) or not np.isfinite(value).all():raise ValueError('rawblock identity/finite failed')
    if not np.array_equal(np.einsum('ntc,ntd->ncd',rep.topographies(raw[:1])[0],rep.topographies(raw[:1])[0])[0]/480,matrix[0]):raise ValueError('rowindependence failed')
    yfit=reader.read_selected(DATA,'y',fit,total);models={}
    for name,value in features.items():
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000,class_weight='balanced'));m.fit(value[:n],yfit)
        if int(m[-1].n_iter_.max())>=2000:raise ValueError('fixedhead convergence failed')
        models[name]=m
    yquery=reader.read_selected(DATA,'y',query,total);interface=reader.read_selected(DATA,'interface',query,total)
    sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'2026-competition')];import protocol
    result={}
    for name,m in models.items():
        pred=m.predict(features[name][n:]);scores={}
        for person in ('P02','P03'):
            mask=people[query]==person;scores[person]=protocol.score(yquery[mask],pred[mask],dict(stream=stream,interface=interface),mask)
            if name in ('raw287','raw_occupancy') and scores[person]['balanced_accuracy_cell_mean']!=previous['results'][name]['people'][person]['balanced_accuracy_cell_mean']:
                raise ValueError('original controlscore identity failed')
        result[name]=dict(people=scores,mean_person_cell_ba=float(np.mean([s['balanced_accuracy_cell_mean'] for s in scores.values()])),dimensions=features[name].shape[1])
    report=dict(source_only=True,P01_labels_opened=False,previously_exposed_queries=['P02','P03'],results=result,
        source_sha256=sha(ROOT/'source.sha256'),original_source_sha256=ORIGINAL_SHA,codebook_sha256=previous['guard']['codebook_sha256'],seconds=time.monotonic()-start,
        limits='same16centroids/split/sourceheads;smallsourcegeometrystatisticcomparison;noouter/current-systemgain')
    with (ROOT/'static_result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
