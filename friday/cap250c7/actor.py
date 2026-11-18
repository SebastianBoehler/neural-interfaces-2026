"""Run the exact cap250 solver and arbitrate from its two live outputs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import replay
from benchmark_utils.streaming import fresh_model
from data import permutation

from arbitrate import log_scores,variants
from features import batch,CausalFeatures,select


def stream(template,coefficients,eeg,rows):
    model=fresh_model(template)
    cp_step=model.cp.step
    observed=[]
    def capture(*args):
        answer=cp_step(*args)
        observed.append(answer.copy())
        return answer
    model.cp.step=capture
    state=CausalFeatures()
    stock,old,cap,features=[],[],[],[]
    predictions={arm:[] for arm in coefficients}
    for t,r in enumerate(rows):
        label=model.predict(eeg[int(r):int(r)+1].cuda())
        assert label.shape==(1,) and label.device.type=='cpu' and label.dtype==torch.long
        assert len(observed)==1 and len(model.v5hist)==t+1
        left=log_scores(np.asarray(model.v5hist[-1])[None],np.array([True]))[0]
        right=log_scores(np.asarray(observed.pop())[None],np.array([t<90]))[0]
        x,a,b=state.step(left,right)
        assert b==int(label[0])
        stock.append(b)
        old.append(left)
        cap.append(right)
        features.append(x)
        for arm,parameters in coefficients.items():
            predictions[arm].append(int(select(x,a,b,parameters)))
    activity=replay.activity(model)
    assert not model.dead and not model.cp.dead and not model.cf.dead
    assert activity['errors']==0 and activity['recognized']==-1 and activity['windows']==len(rows)
    model.cp.step=cp_step
    del model
    old=np.asarray(old)
    cap=np.asarray(cap)
    x,a,b=batch(old,cap)
    np.testing.assert_allclose(x,features,atol=1e-12,rtol=0)
    assert np.array_equal(b,stock)
    for arm,parameters in coefficients.items():
        assert np.array_equal(select(x,a,b,parameters),predictions[arm])
        assert np.array_equal(select(x,b,b,parameters),b)
    fixed=variants(old,cap)['causal']
    return dict(stock=stock,inner=a.tolist(),causal=fixed.tolist(),**predictions),activity,x


def run(index,smoke):
    assert torch.__version__=='2.14.1+cu130' and 'A100' in torch.cuda.get_device_name()
    with np.load('/external.npz',allow_pickle=False) as z:
        x=torch.from_numpy(z['X'])
        people,days,channels=(z[k] for k in ('subject','session','ch_names'))
    assert x.shape==(2130,41,480) and x.dtype==torch.float32
    roster=sorted(set(people))
    source=Path('coefficients.json')
    lock=Path('coefficients.sha256').read_text().split()[0]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==lock
    record=json.loads(source.read_text())
    assert set(record['source_people'])=={f'P{i:02d}' for i in range(1,11)}
    assert len(roster)==9 and all(p.startswith('Scherer2015/') for p in roster)
    assert record['source']==hashlib.sha256(Path('source.sha256').read_bytes()).hexdigest()
    person=roster[index]
    sessions=np.char.add(np.char.add(people,'|'),days)
    cap=json.loads(Path(f'/prior/cap_{index}.json').read_text())
    old=json.loads(Path(f'/prior/old_{index}.json').read_text())
    template=replay.load(Path('/package'),channels.tolist())
    output={}
    for order in (0,1):
        perm=permutation(sessions,None,'full',order)
        for day in sorted(set(days[people==person])):
            key=f'{person}|{day}|{order}'
            rows=perm[(people==person)&(days==day)]
            assert rows.tolist()==cap[key]['rows']==old[key]['rows']
            predictions,activity,features=stream(template,record['coefficients'],x,rows)
            assert predictions['stock']==cap[key]['labels'] and activity==cap[key]['activity']
            mismatch=int(np.count_nonzero(np.asarray(predictions['inner'])!=old[key]['labels']))
            output[key]=dict(rows=rows.tolist(),predictions=predictions,activity=activity,
                             standalone_old_label_differences=mismatch)
            if smoke:
                prefix,_,step=stream(template,record['coefficients'],x,rows[:17])
                assert all(v[:17]==prefix[a] for a,v in predictions.items())
                np.testing.assert_allclose(features[:17],step,atol=2e-4,rtol=1e-5)
                break
        if smoke:
            break
    assert 'neural-interfaces-2026/.venv/' not in Path('/proc/self/maps').read_text()
    dest=Path('smoke.json' if smoke else f'held_{index}.json')
    dest.write_text(json.dumps(dict(coefficient_sha=lock,source=record['source'],sessions=output)))
    Path(str(dest)+'.sha256').write_text(hashlib.sha256(dest.read_bytes()).hexdigest())
    print(person,'sessions',len(output),'inner vs standalone differences',
          sum(r['standalone_old_label_differences'] for r in output.values()),flush=True)


if __name__=='__main__':
    run(int(sys.argv[1]),sys.argv[2:]==['smoke'])
