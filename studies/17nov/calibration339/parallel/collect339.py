"""All independent producer guards/hashes before source-held calibration labels."""
import json
import os
from pathlib import Path
import numpy as np
from shared339 import src,inputs,OUTER,CAL,VIEWS,BASE_SHA,sha,digest,ensemble,apply,weights,fit


def main():
    root=Path(os.environ['CAL_ROOT']);d,rows,cal,people,perms=inputs();records={};banks={};transfer={}
    keys={v+'|'+o for v in VIEWS for o in perms}
    for seed in (0,1):
        folder=root/f'seed{seed}';m=json.loads((folder/'record.json').read_text())
        if m['seed']!=seed or m['source_sha256']!=sha(root/'source.sha256') or m['scientific_base']!=BASE_SHA:raise ValueError('source/seedidentity')
        if m['outer']!=OUTER or m['cal']!=CAL or m['calibration_labels_opened'] or m['outer_labels_opened']:raise ValueError('labelscope')
        if m['fit_people']!=[f'P{i:02d}' for i in range(3,11)] or m['fit_rows_sha256']!=digest(rows) or m['cal_rows_sha256']!=digest(cal):raise ValueError('sourceexclusion')
        if m['checkpoint_sha256']!=sha(folder/'model.pt') or m['logits_sha256']!=sha(folder/'logits.npz'):raise ValueError('immutableoutputs')
        if m['training']['steps']!=1480 or m['training']['max_gradient']<=0 or m['cost']['projection_seconds']>480:raise ValueError('fixedrecipe/cost')
        if 'A100' not in m['device'] or m['kernel']['threads']!=6:raise ValueError('device/kernel')
        old=src.verify(OUTER,seed)
        with np.load(src.PROD/'output'/f'{OUTER}_s{seed}'/'result.npz') as z:original=json.loads(str(z['metadata']))
        if m['initial_param_sha256']!=original['initial_sha256']['P0'] or m['original_source9_checkpoint']!=original['checkpoints']['P0']:raise ValueError('originalseed/checkpoint')
        with np.load(folder/'logits.npz') as z:
            if set(z.files)!={'index'}|{kind+'|'+k for kind in ('held8','insample9') for k in keys} or not np.array_equal(z['index'],cal):raise ValueError('completematrix')
            banks[seed]={k:z['held8|'+k].copy() for k in keys};transfer[seed]={k:z['insample9|'+k].copy() for k in keys}
        if any(v.shape!=(len(cal),3) or not np.isfinite(v).all() for b in (banks[seed],transfer[seed]) for v in b.values()):raise ValueError('nonfinitelogitbank')
        records[str(seed)]=m
    if records['0']['kernel']!=records['1']['kernel'] or records['0']['scale']!=records['1']['scale']:raise ValueError('unregisteredbackend/scalechange')
    # Exact complete1480step SOURCE8 banks/checkpoints/originalsource9 guards now passed for BOTHseeds.
    yc=src.source_labels.read_training_labels(src.data323.DATA,cal,len(d['X']))
    cells=np.char.add(np.char.add(d['stream'][cal],'|'),d['interface'][cal]);w=weights(yc,cells);results={}
    import protocol
    if Path(protocol.__file__).resolve()!=(src.PROD/'experiments/protocol.py').resolve():raise ValueError('wrongowningprotocol')
    for view in VIEWS:
        source={o:ensemble([banks[s][view+'|'+o] for s in (0,1)]) for o in perms}
        for o,lp in source.items():
            if not np.allclose(lp,apply(lp,1.),atol=1e-12,rtol=0):raise ValueError('identitytemperature')
            own=protocol.score(yc,lp.argmax(1),d,people==CAL)['balanced_accuracy_cell_mean']
            if not np.isclose(own,np.dot(w,lp.argmax(1)==yc),atol=1e-12):raise ValueError('originalcell/classweightparity')
        answer=fit(source,yc,w,tuple(perms));changed=0.
        for o in perms:
            lp=ensemble([transfer[s][view+'|'+o] for s in (0,1)]);calibrated=apply(lp,answer['T'])
            if not np.array_equal(lp.argmax(1),calibrated.argmax(1)):raise ValueError('temperaturetransferargmax')
            changed=max(changed,float(abs(np.exp(lp)-np.exp(calibrated)).max()))
        answer['source9_numerical_transfer_max_probability_change']=changed
        answer['active_fit_and_transfer']=answer['max_probability_change']>1e-7 and changed>1e-7;results[view]=answer
    report=dict(scope='ONEsourceheldcalibrationprototype;NOoutermetrics/chain/fullmatrix',source_sha256=sha(root/'source.sha256'),
        scientific_base=BASE_SHA,outer=OUTER,cal=CAL,records=records,temperatures=results,
        outer_labels_opened=False,calibration_labels_opened_after_complete_banks=True,
        seed_aggregation='mean2seedprobabilities BEFOREtemperature;notmeanseedmetrics',
        limitation='samecal fittedNLLoptimization only;source8to9generalization unknown;historicalqueryexposure')
    (root/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
