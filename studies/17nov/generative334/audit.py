"""Post-pilot source-only latent activity audit; no refit or accuracy."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.special import softmax
from model import transform
from inference import Session,banks,density_stream
from feature_io import load
from data import sha

root=Path(os.environ['GEN_ROOT'])
pilot=Path('/mnt/lustre/work/gehler/gfh527/ni26_gen334_v7_20261117')
with np.load(pilot/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files if k!='eta'};eta=float(z['eta'])
bank,record,manifest=load('source7')
ix=np.flatnonzero(bank['people']=='P04')[:17]
z=transform(model,bank['features'][ix]);density=banks(model,z,eta)
p,state=density_stream(model,density['mixture'],eta,'mixture')
t,_=density_stream(model,density['translation'],eta,'translation')
s=Session(model,eta,'mixture');a=z@s.q;r=z-a@s.q.T
residual=-.5*eta*(s.mean2[None]-2*(r@s.mean.reshape(-1,z.shape[1]).T).reshape(len(z),-1,3))/(z.shape[1]-2)
anchor=-.25*eta*((a[:,None,None]-s.ba[None,:,None]-s.anchor[None,None])**2).sum(-1)
posterior=np.exp(state.evidence).reshape(len(model['geometry']),len(model['offset']),6)
marginals={}
for axis,name in enumerate(('geometry','translation','permutation')):
    v=posterior.sum(axis=tuple(i for i in range(3) if i!=axis))
    marginals[name]=dict(probability=v.tolist(),entropy=float(-(v*np.log(np.maximum(v,1e-300))).sum()),uniform_l1=float(np.abs(v-1/len(v)).sum()))
# Removing the full complement changes only residual density, keeping semantic density intact.
neutral=np.broadcast_to(anchor[:,:,None],density['mixture'].shape)
pneutral,_=density_stream(model,neutral,eta,'mixture')
report=dict(scope='P04 first17sourcewindows; savedmodel; nofit/nolabels/noaccuracy',source_rows=bank['rows'][ix].tolist(),
    feature_manifest_sha256=manifest,model_sha256=sha(pilot/'model.npz'),eta=eta,
    global_geometry_norm=float(np.linalg.norm(model['global_geometry'])),person_geometry_norm=float(np.linalg.norm(model['geometry'][1:])),
    residual_class_range_max=float(np.ptp(residual,axis=-1).max()),anchor_class_range_max=float(np.ptp(anchor,axis=-1).max()),
    mixture_translation_probability_max=float(np.abs(p-t).max()),complement_ablation_probability_max=float(np.abs(p-pneutral).max()),
    latent_marginals=marginals)
(root/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
