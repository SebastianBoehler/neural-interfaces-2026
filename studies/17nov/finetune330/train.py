"""Complete source-only Atom-L full-FT versus frozen/initialization factorial (§330)."""
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_2080_v1_20261117')
SOURCE_SHA='057627db1f85aab13e0cab308f0a21d4e7c6645202b86fe2202160f4e4e44833'
EPOCHS=12


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for value in iter(lambda:f.read(1<<20),b''):h.update(value)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Decoder(nn.Module):
    def __init__(self,encoder,chans,mu,sd):
        super().__init__();self.encoder=encoder;self.chans=chans
        self.register_buffer('mu',torch.as_tensor(mu,dtype=torch.float32));self.register_buffer('sd',torch.as_tensor(sd,dtype=torch.float32))
        torch.manual_seed(330);self.head=nn.Linear(687,3);nn.init.normal_(self.head.weight,std=.001);nn.init.zeros_(self.head.bias)

    def features(self,x,raw):
        value=self.encoder.forward_features(x,input_chans=self.chans,return_patch_tokens=True).mean(1)
        return (torch.cat([value,raw],1)-self.mu)/self.sd

    def forward(self,x,raw):return self.head(self.features(x,raw))


def parity(net,x,raw):
    net.eval();cpu=copy.deepcopy(net).cpu()
    with torch.no_grad():
        a=torch.softmax(cpu(x[:3].cpu(),raw[:3].cpu()),1).numpy()
        b=torch.softmax(net(x[:3].cuda(),raw[:3].cuda()),1).cpu().numpy()
        first=net(x[:1].cuda(),raw[:1].cuda());net(x[1:2].cuda(),raw[1:2].cuda())
        if not torch.equal(first,net(x[:1].cuda(),raw[:1].cuda())):raise ValueError('statelessinterleavingfailed')
    error=float(np.max(np.abs(a-b)));margin=np.sort(a,axis=1)[:,-1]-np.sort(a,axis=1)[:,-2]
    changes=int(np.sum((a.argmax(1)!=b.argmax(1))&(margin>1e-6)))
    if error>1e-3 or changes:raise ValueError('CPU/GPUactualfittedreadoutparityfailed')
    del cpu;return dict(probability_max_abs=error,nontied_argmax_changes=changes)


def main():
    started=time.monotonic();torch.set_num_threads(6)
    for folder in (ROOT,SOURCE):
        for line in (folder/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha(folder/name)!=expected:raise ValueError('sourcechanged')
    if sha(SOURCE/'source.sha256')!=SOURCE_SHA:raise ValueError('329sourceidentitydiffers')
    guard=json.loads((SOURCE/'compatibility.json').read_text());readout=json.loads((SOURCE/'fitted_readout_guard.json').read_text())
    if not guard['passed'] or guard['labels_opened'] or len(readout)!=9 or any(v['probability_max_abs']>1e-3 or v['nontied_argmax_changes'] for v in readout.values()):raise ValueError('329numericalguardsrequired')
    if guard['peak_gpu_bytes']>8*1024**3 or guard['comparisons']['pretrained']['B1_median_ms']>=100:raise ValueError('329resourceguardfailed')
    if not torch.cuda.is_available() or '2080 Ti' not in torch.cuda.get_device_name(0):raise ValueError('explicit2080routeonly')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.enable_cudnn_sdp(False)
    sys.path[:0]=[str(SOURCE/'assets/runtime'),'/mnt/lustre/work/gehler/gfh527/ni26_visual324_v2_20261117']
    import shared
    if sha(SOURCE/'assets/atom-large.safetensors')!=guard['weight_sha256'] or sha(SOURCE/'assets/labram.py')!=guard['module_sha256']:raise ValueError('passedassetschanged')
    upstream=module('ni26_atom330_upstream',SOURCE/'assets/labram.py')
    from safetensors.torch import load_file
    weights=load_file(str(SOURCE/'assets/atom-large.safetensors'))
    d,fit,query,people=shared.load_data();rows=np.r_[fit,query];n=len(fit);y=shared.labels(fit,len(d['X']))
    release=json.loads((ROOT/'release.json').read_text())
    bank_sha=sha(SOURCE/'source_banks.npz')
    if bank_sha!=release['source_banks_sha256']:raise ValueError('releasedfeaturebankchanged')
    with np.load(SOURCE/'source_banks.npz') as b:
        if not np.array_equal(b['source_rows'],fit) or not np.array_equal(b['query_rows'],query):raise ValueError('sourcebankrowsdiffer')
        raw=b['raw287'].astype(np.float32);initial={p:b[p+'_pooled_raw'].astype(np.float32) for p in ('pretrained','random')}
    x=resample_poly(d['X'][rows].astype(np.float64),5,3,axis=-1).astype(np.float32).reshape(-1,41,4,200)/100
    xt=torch.from_numpy(x);rt=torch.from_numpy(raw);yt=torch.from_numpy(y).long();classweight=torch.tensor(n/(3*np.bincount(y,minlength=3)),device='cuda',dtype=torch.float32)
    lossfn=nn.CrossEntropyLoss(weight=classweight);results={};smokes={};training={};checkpoints={}
    nb=math.ceil(n/16);steps=EPOCHS*nb
    def make(prior):
        torch.manual_seed(329);enc=upstream.labram_large_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False);del enc.logit_scale
        if prior=='pretrained':
            own=enc.state_dict();mapped={}
            for name,value in weights.items():
                k=name[8:] if name.startswith('student.') else name
                if k in own:mapped[k]=value
                elif name not in guard['removed_pretraining_keys']:raise ValueError('unexpectedpretrainingkey')
            if set(mapped)!=set(own):raise ValueError('encoderloadincomplete')
            enc.load_state_dict(mapped,strict=True)
        mu=initial[prior][:n].mean(0);sd=initial[prior][:n].std(0);sd=np.maximum(sd,1e-6)
        return Decoder(enc,guard['channel_indices'],mu,sd).cuda()
    for prior in ('pretrained','random'):
        for adaptive in (False,True):
            name=prior+('_FT' if adaptive else '_frozen');net=make(prior);before=shared.state_sha(net.encoder)
            for p in net.encoder.parameters():p.requires_grad_(adaptive)
            params=[dict(params=net.head.parameters(),lr=.001)]
            if adaptive:params.append(dict(params=net.encoder.parameters(),lr=.0001))
            opt=torch.optim.AdamW(params,weight_decay=.01)
            cache=torch.as_tensor((initial[prior][:n]-net.mu.detach().cpu().numpy())/net.sd.detach().cpu().numpy(),device='cuda')
            batches=[np.random.default_rng(330+ep).permutation(n) for ep in range(EPOCHS)]
            batch=np.r_[[np.flatnonzero(y==c)[0] for c in range(3)],batches[0][:13]]
            initial_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};costs=[]
            for _ in range(3):
                net.train();opt.zero_grad();xb=xt[batch].cuda().requires_grad_(True)
                torch.cuda.synchronize();t=time.perf_counter()
                logits=net(xb,rt[batch].cuda()) if adaptive else net.head(cache[batch])
                loss=lossfn(logits,yt[batch].cuda());loss.backward()
                if adaptive:
                    gradient=sum(float(p.grad.square().sum()) for p in net.encoder.parameters() if p.grad is not None)
                    if gradient<=0 or not np.isfinite(gradient) or xb.grad is None or not torch.any(xb.grad[..., -1]!=0):raise ValueError('encoder/inputgradientinactive')
                opt.step();torch.cuda.synchronize();costs.append(time.perf_counter()-t)
                if not torch.isfinite(loss):raise ValueError('smokeupdate nonfinite')
            if adaptive and shared.state_sha(net.encoder)==before:raise ValueError('encoderupdateinactive')
            smoke_parity=parity(net,xt[:3],rt[:3]);peak=torch.cuda.max_memory_allocated()
            projected=float(max(costs[1:])*steps*4+180)
            if peak>8*1024**3 or projected>24*60:raise ValueError('330measuredcost/memorybudgetfailed')
            smokes[name]=dict(cost_seconds=costs,conservative_fourarm_seconds=projected,peak_gpu_bytes=peak,parity=smoke_parity)
            net.load_state_dict(initial_state,strict=True)
            if shared.state_sha(net.encoder)!=before:raise ValueError('smokeinitialrestorefailed')
            opt=torch.optim.AdamW(params,weight_decay=.01);schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.01+.99*.5*(1+math.cos(math.pi*step/steps)))
            losses=[];lastgrad=0.
            for ep in range(EPOCHS):
                net.train();loss_sum=0.
                for start in range(0,n,16):
                    batch=batches[ep][start:start+16];opt.zero_grad()
                    logits=net(xt[batch].cuda(),rt[batch].cuda()) if adaptive else net.head(cache[batch])
                    loss=lossfn(logits,yt[batch].cuda())
                    if not torch.isfinite(loss):raise ValueError('sourcefitlossnonfinite')
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):raise ValueError('nonfinitegradient')
                    lastgrad=max(float(p.grad.abs().max()) for p in net.parameters() if p.grad is not None)
                    opt.step();schedule.step();loss_sum+=float(loss)*len(batch)
                losses.append(loss_sum/n);print(name,ep+1,losses[-1],flush=True)
            parity_result=parity(net,xt[:3],rt[:3]);after=shared.state_sha(net.encoder)
            if adaptive==(after==before):raise ValueError('encoderfrozen/FTidentityfailed')
            net.eval();probs=[];pfit=[]
            with torch.no_grad():
                for start in range(0,len(rows),16):
                    p=torch.softmax(net(xt[start:start+16].cuda(),rt[start:start+16].cuda()),1).cpu().numpy()
                    probs.append(p)
            p=np.concatenate(probs)
            if not np.isfinite(p).all():raise ValueError('inferenceprobabilitynonfinite')
            pred=p[:n].argmax(1)
            training[name]=dict(epochloss=losses,final_max_gradient=lastgrad,fit_accuracy=float(np.mean(pred==y)),
                fit_class_recalls=[float(np.mean(pred[y==c]==c)) for c in range(3)],initial_encoder_sha256=before,final_encoder_sha256=after,parity=parity_result,
                source_scaler_sha256=hashlib.sha256(np.r_[net.mu.detach().cpu().numpy(),net.sd.detach().cpu().numpy()].tobytes()).hexdigest())
            path=ROOT/(name+'.pt');torch.save(net.state_dict(),path);checkpoints[name]=sha(path);results[name]=p[n:]
            del net,opt,cache,initial_state
    # All four final fits and actual fitted CPU/GPU guards must pass before querylabels.
    yquery=shared.labels(query,len(d['X']));reader=module('ni26_atom330_labels',shared.LABELS)
    interface=reader.read_selected(shared.DATA,'interface',query,len(d['X']))
    sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'2026-competition')];import protocol
    scores={}
    for name,p in results.items():
        scores[name]={}
        for person in ('P02','P03'):
            mask=people[query]==person;scores[name][person]=protocol.score(yquery[mask],p[mask].argmax(1),dict(stream=d['stream'][query],interface=interface),mask)
    report=dict(source_only=True,P01_labels_opened=False,fit_people=sorted(set(people[fit])),source_rows_sha256=shared.digest(fit),source_banks_sha256=bank_sha,previously_exposed_queries=['P02','P03'],device=torch.cuda.get_device_name(0),
        attention_backend='SDPA MATH only',TF32=False,epochs=EPOCHS,scores=scores,source_training=training,smoke=smokes,
        checkpoint_sha256=checkpoints,source_sha256=sha(ROOT/'source.sha256'),prior_source_sha256=SOURCE_SHA,
        weight_sha256=guard['weight_sha256'],seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limits='fixed12epochs may underconverge;twoexposedpeople;noouter/current-system/A100workerresult')
    if sha(SOURCE/'source_banks.npz')!=bank_sha:raise ValueError('featurebankchangedduringfit')
    with (ROOT/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
