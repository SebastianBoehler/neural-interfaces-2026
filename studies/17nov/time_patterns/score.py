"""Score only the complete validated whole-decoder matrix (§314)."""
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from audit import ROOT,validate
from produce import ORDERS,sha
sys.path[:0]=[str(ROOT/'studies/14nov/newmember'),str(ROOT/'experiments')]
import protocol
from score_s_v66p import BASES,PEOPLE,person_accuracy,summarize


def main():
 d,result,folds=validate();n=len(d['subject'])
 path=ROOT/'studies/16nov/lane2/score.py';spec=importlib.util.spec_from_file_location('s314_reference',path)
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 helper.ORDERS=ORDERS;helper.REF=ROOT/'runs/lane4_reference';reference=helper.check_reference(n)
 # Held labels are opened only after every fold, stream, model and reference is validated.
 with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as z:y=z['y'].copy()
 person=np.array([str(x).split('/')[-1] for x in d['subject']]);cell=np.char.add(np.char.add(d['stream'],'|'),d['interface'])
 if any(set(y[cell==c])!={0,1,2} for c in np.unique(cell)):raise ValueError('cell lacks a required class')
 pp={};official={};mask=np.ones(n,bool)
 for arm in ('TF','TFD','reference'):
  for order in ORDERS:
   preds=([reference[order,b] for b in BASES] if arm=='reference' else [result[arm,s,order] for s in (0,1)])
   pp[arm,order]=np.mean([person_accuracy(p,y,cell,person,PEOPLE) for p in preds],axis=0)
   official[arm,order]=float(np.mean([100*protocol.score(y,p.argmax(1),d,mask)['balanced_accuracy_cell_mean'] for p in preds]))
 def contrast(a,b):return summarize(np.stack([pp[a,o]-pp[b,o] for o in ORDERS]),ORDERS,PEOPLE)
 report={'official_equal_cell_by_order':official,
         'official_equal_cell_scheme_equal':{a:float(np.mean([np.mean([official[a,o] for o in ORDERS if o.startswith(s)]) for s in ('within','full','runs')])) for a in official},
         'absolute_people_order':{a:summarize(np.stack([pp[a,o] for o in ORDERS]),ORDERS,PEOPLE) for a in official},
         'people_order_intervals':{'TFD_minus_TF':contrast('TFD','TF'),'TF_minus_reference':contrast('TF','reference'),'TFD_minus_reference':contrast('TFD','reference')},
         'fold_sha256':folds,'source_sha256':sha(ROOT/'s314-source.sha256'),
         'seed_aggregation':'argmax each independently trained seed, score, then average',
         'interpretation':'whole static decoder; external reference is new-person routed-chain emulation, not a matched baseline or package proof'}
 target=ROOT/'evidence/next_wave_16nov/s314_results.json';target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
 print('validated complete §314 matrix and saved',target,flush=True)


if __name__=='__main__':main()
