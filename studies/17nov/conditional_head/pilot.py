"""Fit and score the pre-registered seven-person conditional-logit source pilot."""
import ctypes
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import FIT, QUERY, ORDERS, OLD, sha, Native, pointer

DIMS = 44239


from loss import objective, penalty


def native_fit(x, y):
    lib = ctypes.CDLL(os.environ['S309_LIBRARY'])
    pd = ctypes.POINTER(ctypes.c_double); pi = ctypes.POINTER(ctypes.c_int64)
    lib.nc_head_fit_device.argtypes = [pd, pi, ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
                                      ctypes.c_int, pd, pd, pd, pi]
    lib.nc_head_error.restype = ctypes.c_char_p
    theta = np.empty((3, DIMS+1), np.float64); loss = np.empty(1, np.float64)
    grad = np.empty(1, np.float64); calls = np.empty(1, np.int64)
    status = lib.nc_head_fit_device(pointer(x), pointer(y), len(y), DIMS, .3, 1,
                                     pointer(theta), pointer(loss), pointer(grad), pointer(calls))
    if status: raise RuntimeError(lib.nc_head_error().decode())
    return theta, float(loss[0]), float(grad[0]), int(calls[0])


def run():
    start = time.monotonic(); root = Path('runs/s326'); info = json.loads((root/'prepare.json').read_text())
    if info['fit_people'] != list(FIT) or info['queries'] != list(QUERY) or info['held_accuracy_computed']:
        raise ValueError('wrong or scored preparation')
    if sha(root/'train.npz') != info['train_sha256'] or len(info['streams_sha256']) != 16:
        raise ValueError('train input or stream matrix changed')
    for name, digest in info['streams_sha256'].items():
        if sha(root/name) != digest: raise ValueError(f'changed query stream: {name}')
    if not torch.cuda.is_available(): raise RuntimeError('A100 CUDA required; no fallback')
    with np.load(root/'train.npz', allow_pickle=False) as z:
        x = np.ascontiguousarray(z['x']); y = np.ascontiguousarray(z['y'])
        delta = np.ascontiguousarray(z['D']); mean = z['mean'].copy(); scale = z['scale'].copy()
    if x.shape[1] != DIMS or delta.shape != (21, DIMS) or set(y) != {0,1,2}: raise ValueError('invalid features')
    native = Native(); zero, native_loss, native_grad, native_calls = native_fit(x, y)
    loss_cpu, grad_cpu = native.objective(x, y, zero)
    if abs(native_loss-loss_cpu)>1e-9 or abs(native_grad-np.abs(grad_cpu).max())>1e-8 or native_grad>1e-5:
        raise ValueError('native zero objective/gradient mismatch')
    device = torch.device('cuda:0'); xd = torch.as_tensor(x, device=device); yd = torch.as_tensor(y, device=device)
    dd = torch.as_tensor(delta, device=device); w0 = torch.as_tensor(zero, device=device).clone().requires_grad_(True)
    q = float(penalty(w0, dd).item())
    if not np.isfinite(q) or q <= 1e-20: raise ValueError('inactive baseline conditional offsets')
    val = penalty(w0, dd); ga = torch.autograd.grad(val, w0)[0]
    centered = w0[:, :-1] - w0[:, :-1].mean(0, keepdim=True)
    analytic = 2 * (dd @ centered.T).T @ dd / (dd.shape[0] * 3)
    analytic -= analytic.mean(0, keepdim=True)
    if not torch.allclose(ga[:, :-1], analytic, atol=1e-10, rtol=1e-8) or torch.count_nonzero(ga[:, -1]):
        raise ValueError('penalty analytic/autograd mismatch')
    direction = ga / torch.linalg.vector_norm(ga); eps = 1e-5
    finite = float(((penalty(w0+eps*direction, dd)-penalty(w0-eps*direction, dd))/(2*eps)).item())
    exact = float((ga*direction).sum().item())
    if not np.isclose(finite, exact, atol=1e-8, rtol=1e-5) or exact <= 0:
        raise ValueError('penalty finite difference or activity failure')
    shifted = w0.clone(); shifted[:, :-1] += torch.arange(DIMS, device=device, dtype=torch.float64) * 1e-10
    if not torch.allclose(penalty(shifted, dd), val, atol=1e-12, rtol=1e-10): raise ValueError('softmax gauge failure')
    zero_d = penalty(w0, torch.zeros_like(dd))
    zero_d_grad = torch.autograd.grad(zero_d, w0)[0]
    if zero_d.item() != 0 or torch.count_nonzero(zero_d_grad): raise ValueError('zero-D failure')
    torch_loss = objective(xd, yd, w0, dd, 0, q); torch_grad = torch.autograd.grad(torch_loss, w0)[0]
    if abs(torch_loss.item()-native_loss)>1e-9 or np.max(np.abs(torch_grad.cpu().numpy()-grad_cpu))>1e-8:
        raise ValueError('native and modified objective mismatch at λ=0')
    models = {'zero': zero}; diagnostics = {'zero':{'loss':native_loss, 'gradmax':native_grad, 'calls':native_calls}}
    for lam, name in ((.01,'moderate'),(.1,'strong')):
        theta = torch.zeros_like(w0, requires_grad=True)
        opt = torch.optim.LBFGS([theta], lr=1., max_iter=2000, max_eval=2500, history_size=10,
                                tolerance_grad=1e-8, tolerance_change=1e-12, line_search_fn='strong_wolfe')
        calls = [0]
        def closure():
            opt.zero_grad(); value = objective(xd,yd,theta,dd,lam,q); value.backward(); calls[0]+=1
            if not torch.isfinite(value) or not torch.isfinite(theta.grad).all(): raise ValueError('nonfinite optimization')
            return value
        opt.step(closure); final = closure(); maxgrad = float(theta.grad.abs().max().item())
        if maxgrad > 1e-5: raise ValueError(f'{name} did not converge: {maxgrad}')
        models[name] = theta.detach().cpu().numpy()
        diagnostics[name] = {'loss':float(final.item()), 'gradmax':maxgrad, 'calls':calls[0],
                             'penalty':float(penalty(theta, dd).item()/q), 'weight_delta':float(torch.linalg.vector_norm(theta-w0).item())}
    # Save guard evidence and all predictions before opening any query labels for scoring.
    output = {}; changed = {'moderate':0, 'strong':0}
    for person in QUERY:
        for order in ORDERS:
            with np.load(root/f'{person}_{order}.npz', allow_pickle=False) as z:
                rows = z['rows'].copy(); features = (z['features']-mean)/scale
            for name, theta in models.items():
                logits = features @ theta[:,:-1].T + theta[:,-1]
                if not np.isfinite(logits).all(): raise ValueError('invalid query logits')
                output[f'{person}|{order}|{name}'] = np.argmax(logits, axis=1)
                if name != 'zero': changed[name] += int(np.count_nonzero(output[f'{person}|{order}|{name}'] != output[f'{person}|{order}|zero']))
            output[f'{person}|rows'] = rows
    if any(n <= 0 for n in changed.values()): raise ValueError(f'no changed predictions: {changed}')
    with (root/'pilot-predictions.npz').open('xb') as f: np.savez(f, **output)
    meta = {'fit_people':FIT, 'queries':QUERY, 'q':q, 'penalty_gradient_norm':exact, 'finite_difference':finite,
            'changed_labels':changed, 'diagnostics':diagnostics, 'predictions_sha256':sha(root/'pilot-predictions.npz'),
            'prepare_sha256':sha(root/'prepare.json'), 'device':torch.cuda.get_device_name(),
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(), 'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'elapsed_s':time.monotonic()-start, 'query_labels_read_for_accuracy':False}
    with (root/'pilot-guards.json').open('x') as f: json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == '__main__': run()
