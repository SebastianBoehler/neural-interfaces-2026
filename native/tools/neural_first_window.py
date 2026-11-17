"""Isolate native/host-Python/SIF-golden first SOURCE window; no fit or accuracy."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.jit._pickle import restore_type_tag


def main():
    p=argparse.ArgumentParser()
    for key in ('package','fixture','native','project','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(args.project/'2026-competition/tracks/bci_decoding'))
    spec=importlib.util.spec_from_file_location('frozen_first_source_window',args.package/'submission.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with np.load(args.package/'weights.npz',allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    fixture=torch.load(args.fixture,weights_only=True,map_location='cpu')
    with torch.serialization.safe_globals([restore_type_tag]):native=torch.load(args.native,weights_only=True,map_location='cpu')
    x=fixture['known_x'][0].numpy();count=int(z['n_nets']);weights=z['member_weights'].astype(np.float64);weights/=weights.sum()
    bank=module.NetBank([module.with_prefix(z,f'net{i}_') for i in range(count)],weights[1:1+count]!=0,'cpu',args.package)
    actual=bank(x);ids=np.array([i for i,_ in actual]);host=np.stack([v for _,v in actual]);gold=fixture['known_probs'][0].numpy();cpp=native['probs'].numpy()
    if not np.array_equal(ids,fixture['known_ids'][0].numpy()) or not np.array_equal(ids,native['ids'].numpy()):raise ValueError('actual source member ID/order differs')
    maximum=lambda a,b:float(np.max(np.abs(a-b)))
    records=[]
    for row,i in enumerate(ids):
        net=module.with_prefix(z,f'net{i}_');name=str(net.get('sjit_file',net.get('jit_file','shallow' if 'time_w' in net else 'fbc')))
        record=dict(member=int(i),model=name,align=bool(net['align']),
            cpp_vs_SIF=maximum(cpp[row],gold[row]),host_vs_SIF=maximum(host[row],gold[row]),cpp_vs_host=maximum(cpp[row],host[row]),
            cpp_input_vs_host=maximum(native[f'input_{i}'].numpy(),bank.last_inputs[i]))
        records.append(record);print(json.dumps(record),flush=True)
    inputs=[]
    for j,i in enumerate(fixture['input_ids'].tolist()):inputs.append(dict(member=i,host_vs_SIF=maximum(bank.last_inputs[i],fixture['known_input'][0,j].numpy()),cpp_vs_SIF=maximum(native[f'input_{i}'].numpy(),fixture['known_input'][0,j].numpy())))
    eye=module.EyeNet(z,'cpu',args.package).step(x);rest=module.RestMember(z,'cpu',args.package).step(x)
    result=dict(torch_version=torch.__version__,mkldnn_enabled=torch.backends.mkldnn.enabled,matmul_precision=torch.get_float32_matmul_precision(),members=records,inputs=inputs,
        eye_cpp_vs_host=maximum(native['eye'].numpy(),eye),eye_host_vs_SIF=maximum(eye,fixture['known_eye'][0].numpy()),
        rest_cpp_vs_host=maximum(native['rest'].numpy(),rest),rest_host_vs_SIF=maximum(rest,fixture['known_rest'][0].numpy()),
        source_windows=1,task_labels_loaded=False,optimizer_steps=0,accuracy_computed=False,tolerance_changed=False)
    args.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
