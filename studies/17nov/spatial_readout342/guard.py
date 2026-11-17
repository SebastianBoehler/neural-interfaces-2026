"""Source-only CPU entry guard and one A100 compatibility/cost smoke."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from common import OLD, encoder, gpu, inputs, labels, layouts, load, sha, state_sha
from head import Head, objective, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('device', choices=['cpu', 'gpu'])
    mode = parser.parse_args().device
    start = time.monotonic(); torch.set_num_threads(2)
    d, people, source, query = load()
    rows = np.flatnonzero(people == 'P04')[:3]
    x = inputs(d['X'][rows]); records = {}; banks = {}
    if mode == 'gpu':
        if not json.loads(Path('cpu_guard.json').read_text())['passed']:
            raise ValueError('CPU entry guard required')
        gpu()
    for prior in ('pretrained', 'random'):
        model, upstream = encoder(prior)
        chans = upstream.get_input_chans(d['ch_names'].astype(str).tolist())
        if chans != json.loads((OLD/'compatibility.json').read_text())['channel_indices']:
            raise ValueError('channel mapping differs')
        before = state_sha(model)
        if mode == 'gpu':
            with torch.no_grad():
                same_weight_cpu = model.forward_features(x, input_chans=chans, return_patch_tokens=True).numpy()
            if prior == 'random':
                torch.save(model.state_dict(), 'random_initial.pt')
            model.cuda()
            if state_sha(model) != before:
                raise ValueError('device transfer changed frozen parameters')
        xt = x.clone().to('cuda' if mode == 'gpu' else 'cpu').requires_grad_(True)
        output = model.forward_features(xt, input_chans=chans, return_patch_tokens=True)
        output.square().mean().backward()
        if not torch.isfinite(xt.grad).all() or not torch.any(xt.grad[..., -1]):
            raise ValueError('last-sample gradient inactive')
        with torch.no_grad():
            a = model.forward_features(xt[:1], input_chans=chans, return_patch_tokens=True)
            model.forward_features(xt[1:2], input_chans=chans, return_patch_tokens=True)
            b = model.forward_features(xt[:1], input_chans=chans, return_patch_tokens=True)
            if not torch.equal(a, b):
                raise ValueError('interleaved statelessness failed')
            replica = copy.deepcopy(model)
            if not torch.equal(a, replica.forward_features(xt[:1], input_chans=chans, return_patch_tokens=True)):
                raise ValueError('deepcopy failed')
            del replica
        if state_sha(model) != before:
            raise ValueError('encoder state mutated')
        patch = output.detach().cpu().numpy(); banks[prior] = patch
        values = layouts(patch)
        # Token layout identities audit channel/time axes, independent of learned representation.
        for channel in (0, 20, 40):
            np.testing.assert_array_equal(values['full65600'].reshape(-1,41,4,400)[:, channel], patch[:,channel*4:channel*4+4])
        if mode == 'gpu':
            cpu = same_weight_cpu
            banks[prior+'_cpu'] = same_weight_cpu
            if not np.allclose(cpu, patch, atol=2e-3, rtol=1e-3):
                raise ValueError('CPU/A100 feature parity failed')
        records[prior] = dict(state_sha256=before, layout_dimensions={k:v.shape[1] for k,v in values.items()})
        del model, output, xt
    # A small real-source oracle establishes C-aware n/L2/bias semantics before full widths.
    y = labels(source, people)
    minirows = np.concatenate([np.flatnonzero(y == c)[:16] for c in range(3)])
    raw = d['X'][source[minirows], :3].mean(-1).astype(np.float64)
    z = np.ascontiguousarray(StandardScaler().fit_transform(raw)); target=np.ascontiguousarray(y[minirows],dtype=np.int64)
    oracle = {}
    for c in (.001, .01, .1):
        result = minimize(lambda t: objective(z,target,t.reshape(3,4),c)[0], np.zeros(12),
                          jac=lambda t: objective(z,target,t.reshape(3,4),c)[1].ravel(),
                          method='L-BFGS-B', options=dict(gtol=1e-10,ftol=1e-15,maxiter=2000))
        theta=result.x.reshape(3,4); loss, grad=objective(z,target,theta,c)
        oracle[str(c)]=dict(loss=loss,max_gradient=float(np.abs(grad).max()),success=bool(result.success))
        if np.abs(grad).max()>1e-7:
            raise ValueError('independent source CPU oracle failed')
        if mode == 'gpu':
            fitted, info=Head().fit(z,target,c)
            if abs(info['loss']-loss)>1e-8 or np.max(np.abs(predict(z,fitted)-predict(z,theta)))>1e-5:
                raise ValueError('GPU C-aware oracle failed')
            oracle[str(c)]=info
    if mode == 'gpu':
        # Actual source48 × full65600 cost smoke. No source-held or query accuracy.
        model, upstream=encoder('pretrained'); model.cuda()
        with torch.no_grad():
            parts=[model.forward_features(inputs(d['X'][source[minirows[i:i+8]]]).cuda(),
                   input_chans=chans,return_patch_tokens=True).cpu().numpy() for i in range(0,len(minirows),8)]
        full=np.concatenate(parts).reshape(len(minirows),65600)
        standardized=np.ascontiguousarray(StandardScaler().fit_transform(full).astype(np.float64)/np.sqrt(164))
        _, cost=Head().fit(standardized,target,.01)
        records['full_width_cost']=cost
        np.savez('same_weight_cpu_gpu_probe.npz',rows=rows,**banks)
    else:
        np.savez('cpu_probe.npz',rows=rows,**banks)
    result=dict(passed=True,mode=mode,labels_scope='SOURCE6 only',P09_labels_opened=False,
                P01_P02_P03_inference=False,records=records,oracle=oracle,
                source_manifest_sha256=sha('source.sha256'),seconds=time.monotonic()-start)
    Path(mode+'_guard.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
