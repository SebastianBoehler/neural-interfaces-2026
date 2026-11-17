"""No-training complete checkpoint→query→named-official-metric replay of §330."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent
FIT=Path('/mnt/lustre/work/gehler/gfh527/ni26_finetune330_v3_A100_20261117')
SOURCE=Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_A100source_v1_20261117')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for value in iter(lambda:f.read(1<<20),b''):h.update(value)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def main():
    torch.set_num_threads(6)
    report=json.loads((FIT/'result.json').read_text())
    for folder in (ROOT,FIT,SOURCE):
        for line in (folder/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha(folder/name)!=expected:raise ValueError('immutable sourcechanged')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('sameA100route required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
    sys.path[:0]=[str(SOURCE/'assets/runtime'),'/mnt/lustre/work/gehler/gfh527/ni26_visual324_v2_20261117']
    import shared
    fitmodule=module('ni26_330_replay_definition',FIT/'train.py');upstream=module('ni26_330_replay_upstream',SOURCE/'assets/labram.py')
    d,fit,query,people=shared.load_data()
    with np.load(SOURCE/'source_banks.npz') as z:
        if not np.array_equal(z['source_rows'],fit) or not np.array_equal(z['query_rows'],query):raise ValueError('sourcequerybankidentityfailed')
        raw=torch.from_numpy(z['raw287'][len(fit):].astype(np.float32))
    x=torch.from_numpy((resample_poly(d['X'][query].astype(np.float64),5,3,axis=-1).astype(np.float32)/100).reshape(-1,41,4,200))
    guard=json.loads((SOURCE/'compatibility.json').read_text());probabilities={}
    for arm,expected in report['checkpoint_sha256'].items():
        path=FIT/(arm+'.pt')
        if sha(path)!=expected:raise ValueError('fittedcheckpointchanged')
        state=torch.load(path,map_location='cpu',weights_only=True)
        enc=upstream.labram_large_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False);del enc.logit_scale
        model=fitmodule.Decoder(enc,guard['channel_indices'],state['mu'].numpy(),state['sd'].numpy());model.load_state_dict(state,strict=True);model.cuda().eval()
        values=[]
        with torch.no_grad():
            for start in range(0,len(query),16):values.append(torch.softmax(model(x[start:start+16].cuda(),raw[start:start+16].cuda()),1).cpu().numpy())
        probabilities[arm]=np.concatenate(values);del model,state
    np.savez(ROOT/'query_probabilities.npz',index=query,**probabilities)
    sys.path.insert(0,str(ROOT/'2026-competition'));protocol=module('ni26_330_replay_official_protocol',ROOT/'experiments/protocol.py')
    actual=Path(sys.modules[protocol.balanced_accuracy.__module__].__file__)
    if sha(actual)!=sha(ROOT/'2026-competition/benchmark_utils/metrics.py'):raise ValueError('actualofficialmetricchanged')
    labels=shared.labels(query,len(d['X']));reader=module('ni26_330_replay_labels',shared.LABELS);interface=reader.read_selected(shared.DATA,'interface',query,len(d['X']))
    scores={}
    for arm,p in probabilities.items():
        scores[arm]={}
        for person in ('P02','P03'):
            mask=people[query]==person;value=protocol.score(labels[mask],p[mask].argmax(1),dict(stream=d['stream'][query],interface=interface),mask)
            if value!=report['scores'][arm][person]:raise ValueError('completeofficialscorereplaydiffers:'+arm+'/'+person)
            scores[arm][person]=value
    result=dict(passed=True,no_training=True,all4checkpoint_hashes_verified=True,query_probability_sha256=sha(ROOT/'query_probabilities.npz'),
        original_result_sha256=sha(FIT/'result.json'),producer_source_sha256=report['source_sha256'],source_sha256=sha(ROOT/'source.sha256'),
        explicit_protocol_path=str(protocol.__file__),actual_metric_path=str(actual),actual_metric_sha256=sha(actual),scores=scores,
        limits='exactcompletecheckpoint/metricreplay onA100;notouter/workerverification')
    with (ROOT/'replay.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
