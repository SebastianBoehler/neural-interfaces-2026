"""Seal separate held-session member banks after verifying all independent fits."""
import argparse
import json
from pathlib import Path

import numpy as np
from probe import ORDERS,digest,fingerprint,read_inputs,write_json
from returning_inputs import FAMILIES,LAYOUT,PEOPLE,archived,rows

HERE=Path(__file__).resolve().parent
SHALLOW_SMOKE_SOURCE='0ba595347da89d770b83419341f687c20cfeb7a1837a3caa83a7745d7a2b1057'


def load(name,held,seed,source,test,delivery):
    tag=f'{name}_{held}_seed{seed}'
    record=HERE/f'{tag}_receipt.json'
    result=HERE/f'{tag}_orders.npz'
    weights=HERE/f'{tag}_fresh.pt'
    info=json.loads(record.read_text())
    assert info['family']==name and info['fold']==held and info['seed']==seed
    assert info['held_labels_opened'] is False and info['historical_reconstruction'] is False
    assert info['shared_by_comparators'] and info['deterministic']
    assert info['duplicate_fit_bit_exact']==((held,seed)==('P10',0))
    assert info['fit_rows_sha']==fingerprint(source) and info['held_rows_sha']==fingerprint(test)
    expected_source=(SHALLOW_SMOKE_SOURCE if (name,held,seed)==('shallow_ea','P10',0)
                     else digest(HERE/'source.sha256'))
    assert info['source_manifest']==expected_source
    assert info['audit_sha']==digest(HERE/'inputs_audit.json')
    assert digest(weights)==info['checkpoint_sha'] and digest(result)==info['predictions_sha']
    assert set(info['checks'])==set(LAYOUT) and max(info['delta_from_recording'].values())>1e-4
    assert bool(info['spectral'])==(name=='fbcnet_ea_tf')
    for key,checks in info['checks'].items():
        assert sorted(checks)==[f'Dreyer2026Proteus/{held}|{day}' for day in ('2','3')]
        assert all(v['first_direct_max_abs']==0 and v['prefix_exact'] and v['reset_exact']
                   and v['batch_max_abs']<=2e-5 for v in checks.values())
    with np.load(result,allow_pickle=False) as z:
        assert np.array_equal(z['index'],test) and tuple(z['orders'])==LAYOUT
        assert np.array_equal(z['delivery'],np.stack([test,*[a[np.isin(a,test)] for a in delivery.values()]]))
        prediction=z['proba']
        assert prediction.shape==(len(LAYOUT),len(test),3) and np.isfinite(prediction).all()
        assert np.allclose(prediction.sum(2),1,rtol=0,atol=1e-12)
    return prediction,info,digest(record)


def main(root):
    data,_=read_inputs(root)
    folds,index,delivery=rows(root,data)
    legacy=archived(root,index)
    audit=json.loads((HERE/'inputs_audit.json').read_text())
    assert fingerprint(index)==audit['global_rows_sha']
    assert {k:fingerprint(v) for k,v in delivery.items()}==audit['orders']
    all_info={}
    for name,(slot,nseeds,*_) in FAMILIES.items():
        state={}
        values=np.full((nseeds,len(LAYOUT),len(index),3),np.nan)
        for seed in range(nseeds):
            for p,(source,test) in folds.items():
                probs,info,receipt=load(name,p,seed,source,test,delivery)
                where=np.searchsorted(index,test)
                assert np.array_equal(index[where],test) and np.isnan(values[seed,:,where]).all()
                values[seed][:,where,:]=probs
                state[f'{p}_seed{seed}']=dict(receipt=receipt,state=info['state'],
                        max_batch_abs=max(check['batch_max_abs']
                                          for item in info['checks'].values() for check in item.values()),
                        spectral=info['spectral'],gpu=info['gpu'],node=info['node'],
                        source_manifest=info['source_manifest'])
        assert np.isfinite(values).all()
        proba=values[:,1:]
        assert proba.shape==(nseeds,8,762,3)
        recorded=values[:,0]
        bank=HERE/f'{name}_seen_session_orders.npz'
        with bank.open('xb') as handle:
            np.savez_compressed(handle,index=index,seeds=np.arange(nseeds),orders=np.array(ORDERS),
                               delivery=np.stack(list(delivery.values())),proba=proba,recording=recorded)
        old=legacy[name]
        bank_info=dict(slot=slot,bank=str(bank),sha256=digest(bank),shape=list(proba.shape),
                       source_fits=len(state),recording_max_abs_vs_archive=float(np.abs(recorded-old).max()),
                       recording_argmax_changes_vs_archive=int(np.count_nonzero(recorded.argmax(2)!=old.argmax(2))),
                       max_causal_delta=float(np.abs(proba-recorded[:,None]).max()),fits=state)
        all_info[name]=bank_info
        print(name,bank_info['shape'],bank_info['sha256'],
              'recording difference',bank_info['recording_max_abs_vs_archive'],flush=True)
    result=dict(historical_reconstruction=False,held_labels_opened=False,shared_by_comparators=True,
                source=digest(HERE/'source.sha256'),inputs_audit=digest(HERE/'inputs_audit.json'),
                global_rows_sha=fingerprint(index),full0_first={k:list(v[3]) for k,v in PEOPLE.items()},
                families=all_info)
    write_json(HERE/'returning_result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
