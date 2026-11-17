"""Finite-window actual FIR bank rank/conditioning, not learned-feature information proof."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import convolution_matrix,eigh
from inputs import source,restored,sha,digest

root=Path(os.environ['INFO_ROOT']);started=time.monotonic();torch.set_num_threads(6)
x,rows,people=source();net,scale=restored();coeff=[p.detach().numpy().astype(np.float64) for p in net.spectral_filtering.fir_list]
operator=np.concatenate([convolution_matrix(c,480,mode='same') for c in coeff])
wave=(x[0,0].astype(np.float64)/scale).astype(np.float32)
with torch.no_grad():actual=net.spectral_filtering(torch.from_numpy((x[:1].astype(np.float64)/scale).astype(np.float32))).numpy()[0,:,0].ravel()
predicted=operator@wave
if not np.allclose(predicted,actual,atol=1e-5,rtol=1e-5):raise ValueError('finiteoperator/actualsourceFFTguard')
eigen=eigh(operator.T@operator,eigvals_only=True)
if eigen[0]<-1e-10*eigen[-1]:raise ValueError('Gram spectrum materiallynegative')
if eigen[0]<=0:raise ValueError('singular/nonpositive spectrum; report ratherthanclip')
singular=np.sqrt(eigen);rank=int(np.sum(singular>1e-12*singular[-1]))
report=dict(source_sha256=sha(root/'source.sha256'),shape=list(operator.shape),rank_at_relative_sigma_1e_minus12=rank,singular_min=float(singular[0]),
    singular_max=float(singular[-1]),condition=float(singular[-1]/singular[0]),singular_values=singular.tolist(),
    actual_source_fft_max_abs=float(np.abs(predicted-actual).max()),coefficient_hashes=[digest(c) for c in coeff],source_rows=rows.tolist(),
    labels_opened=False,optimizer_run=False,P09_X_used=False,P02P03_inference=False,seconds=time.monotonic()-started,
    scope='linearperchannel480→9×480FIRoutputs;NOT1152learnedsummaryorclassifieridentifiability')
(root/'operator.json').write_text(json.dumps(report));print(json.dumps({k:v for k,v in report.items() if k!='singular_values'}),flush=True)
