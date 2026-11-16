"""Train one source-only §314 fold and emit stateless probability streams."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT=Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'experiments')]
import protocol
from features import extract,TF_DIM,DYNAMIC_DIM

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def rms_fit(X,rows):
 if not len(rows):raise ValueError('empty normalization fit')
 sums=np.zeros(41,np.float64)
 for start in range(0,len(rows),16):
  x=X[rows[start:start+16]].astype(np.float64);sums+=np.square(x).sum(axis=(0,2))
 value=np.sqrt(sums/(len(rows)*480))
 if not np.isfinite(value).all() or np.any(value<1e-8):raise ValueError('invalid training-only RMS')
 return value.astype(np.float32)


def features(X,rows,rms,seed):
 tf=np.empty((len(rows),TF_DIM),np.float32);dynamic=np.empty((len(rows),DYNAMIC_DIM),np.float32)
 for start in range(0,len(rows),16):
  i=rows[start:start+16];tf[start:start+len(i)],dynamic[start:start+len(i)]=extract(X[i],rms,seed)
 return tf,dynamic


def fit_model(x,y,c):
 if set(np.unique(y))!={0,1,2}:raise ValueError('training split lacks a class')
 scaler=StandardScaler().fit(x)
 model=LogisticRegression(C=c,solver='lbfgs',max_iter=1000,tol=1e-4).fit(scaler.transform(x),y)
 if model.n_iter_.max()>=1000 or not np.array_equal(model.classes_,[0,1,2]):raise ValueError('logistic optimization failed')
 return scaler,model


def main(task,smoke=False):
 if not 0<=task<20:raise ValueError('expected 20 fold/seed tasks')
 seed,index=task//10,task%10
 d=protocol.load();X=d['X'];folds=list(protocol.folds(d,'all_loso'));held,tr,te=folds[index]
 if len(folds)!=10 or X.shape[1:]!=(41,480):raise ValueError('unexpected dataset/folds')
 people=sorted(set(d['subject'][tr]));hold=set(d['subject'][te])
 if len(people)!=9 or len(hold)!=1 or set(people)&hold:raise ValueError('held-person leakage')
 validation=people[seed];inner=tr & (d['subject']!=validation);val=tr & (d['subject']==validation)
 ir,vr,train,test=map(np.flatnonzero,(inner,val,tr,te))
 if set(d['subject'][ir])&set(d['subject'][vr]) or np.any(te[ir]) or not len(vr):raise ValueError('validation overlap')
 output=ROOT/f'runs/s314/folds/{held}_s{seed}{"_smoke" if smoke else ""}.npz'
 if output.exists():raise FileExistsError(output)
 started=time.monotonic();inner_rms=rms_fit(X,ir)
 # The inner transform receives source rows only; held rows are not needed for selection.
 source=np.r_[ir,vr];itf,idyn=features(X,source,inner_rms,seed);n=len(ir)
 if np.count_nonzero(np.ptp(idyn[:n],axis=0)>1e-6)<100:raise ValueError('temporal feature bank inactive')
 # Real source time-reversal check preserves the corresponding per-bin TF statistics.
 a,b=extract(X[ir[:2]],inner_rms,seed);ar,br=extract(X[ir[:2],:,::-1].copy(),inner_rms,seed)
 if np.max(np.abs(b.reshape(-1,16,12,4,4,2)-br.reshape(-1,16,12,4,4,2)[:,:,:,:,::-1,:]))<1e-4:
  raise ValueError('temporal bank has no time-order sensitivity')
 np.testing.assert_allclose(a[:,:164].reshape(-1,41,4),ar[:,:164].reshape(-1,41,4)[:,:,::-1],atol=2e-5)
 np.testing.assert_allclose(a[:,164:].reshape(-1,17,41,4),ar[:,164:].reshape(-1,17,41,4)[:,:,:,::-1],atol=2e-5,rtol=1e-5)
 inner_data={'TF':itf,'TFD':np.concatenate((itf,idyn),axis=1)};chosen={};val_losses={}
 for arm,feat in inner_data.items():
  losses=[]
  for c in (.01,.1,1.):
   scaler,model=fit_model(feat[:n],d['y'][ir],c)
   losses.append(float(log_loss(d['y'][vr],model.predict_proba(scaler.transform(feat[n:])),labels=[0,1,2])))
  if not np.isfinite(losses).all():raise ValueError('nonfinite source validation loss')
  chosen[arm]=(.01,.1,1.)[int(np.argmin(losses))];val_losses[arm]=losses
 del inner_data,itf,idyn
 final_rms=rms_fit(X,train);all_rows=np.r_[train,test];tf,dyn=features(X,all_rows,final_rms,seed);n=len(train)
 combined=np.concatenate((tf,dyn),axis=1)
 if not np.array_equal(combined[:,:TF_DIM],tf):raise ValueError('TF features changed in temporal arm')
 results={'model|rms':final_rms};model_meta={};trained={}
 for arm,feat in (('TF',tf),('TFD',combined)):
  scaler,model=fit_model(feat[:n],d['y'][train],chosen[arm])
  if arm=='TFD' and np.max(np.abs(model.coef_[:,TF_DIM:]))<=1e-10:raise ValueError('temporal features ignored by classifier')
  p=model.predict_proba(scaler.transform(feat[n:])).astype(np.float64)
  if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-10):raise ValueError('invalid posterior')
  results[arm]=p;trained[arm]=(scaler,model,feat[n:])
  for name,value in (('coef',model.coef_),('intercept',model.intercept_),('scaler_mean',scaler.mean_),('scaler_scale',scaler.scale_)):
   results[f'model|{arm}|{name}']=value
  model_meta[arm]={'coef_sha256':digest(model.coef_),'intercept_sha256':digest(model.intercept_),
                   'scaler_mean_sha256':digest(scaler.mean_),'scaler_scale_sha256':digest(scaler.scale_),
                   'iterations':int(model.n_iter_.max()),'features':feat.shape[1]}
 with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
  for order in ORDERS:
   perm=z[f'{order[:-1]}|{order[-1]}|all_loso'].astype(np.int64)
   if not np.array_equal(np.sort(perm),np.arange(len(X))):raise ValueError('invalid pinned permutation')
   rows=perm[te[perm]];loc=np.searchsorted(test,rows)
   if not np.array_equal(test[loc],rows):raise ValueError('held order index mismatch')
   results[f'{order}|index']=rows
   for arm in ('TF','TFD'):
    results[f'{order}|{arm}']=results[arm][loc]
    scaler,model,test_features=trained[arm]
    replay=model.predict_proba(scaler.transform(test_features[loc]))
    if not np.allclose(replay,results[f'{order}|{arm}'],atol=1e-12,rtol=0):raise ValueError('stateless shuffled replay mismatch')
 meta={'held':held,'seed':seed,'source_people':people,'validation_person':validation,'train_rows_sha256':digest(train),
       'validation_rows_sha256':digest(vr),'inner_fit_rows_sha256':digest(ir),'test_rows_sha256':digest(test),
       'inner_rms_sha256':digest(inner_rms),'final_rms_sha256':digest(final_rms),'chosen_C':chosen,'source_val_logloss':val_losses,
       'model':model_meta,'data_sha256':sha(ROOT/'data/proteus_windows.npz'),'permutations_sha256':sha(ROOT/'runs/shuffle_perms.npz'),
       'source_sha256':sha(ROOT/'s314-source.sha256'),'seconds':time.monotonic()-started,'held_accuracy_computed':False,'smoke':smoke}
 results['metadata']=json.dumps(meta,sort_keys=True)
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('xb') as f:np.savez(f,**results)
 print(f'{held}/s{seed}: source-only supervised TF/TFD trained and 8 orders emitted; {meta["seconds"]:.1f}s; no held score',flush=True)


if __name__=='__main__':main(int(sys.argv[1]),len(sys.argv)==3 and sys.argv[2]=='smoke')
