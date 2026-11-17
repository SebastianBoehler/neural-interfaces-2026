"""SOURCE-only initialization/batch/first-update oracle; requires separate fit registration."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from convert_package import save_tensors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    dependency=Path('/mnt/lustre/work/gehler/gfh527/ni26_direct346_v1_20261117')
    sys.path.insert(0,str(dependency))
    from common346 import source,native,sha,digest,state_sha,CONTROL
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data,fit,_,people,_=source();model=native(data);record={};fields={}
    with (CONTROL/'record.json').open() as stream:reference=json.load(stream)
    def capture(self,x,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None):
        if state_sha(self.net)!=reference['trace']['initial_state_sha256']:raise ValueError('original rawA initializer changed')
        if self.scale!=reference['trace']['scale'] or weighted or head_only or avg is not None or self.aug or self.rot or self.align:raise ValueError('source recipe changed')
        self._prepare()
        if sha(inspect.getfile(type(self.net)))!='604e690ad1743a8dc9eeb87c0c3065d28619cbd4dbda8c9790e60924cafbe81a':raise ValueError('original FBC architecture owner changed')
        initial={key:value.detach().clone() for key,value in self.net.state_dict().items()}
        plan=[batch for _ in range(40) for batch in self._batches(rows,rng)]
        if len(plan)!=1240 or len(x)!=1898:raise ValueError('exact1898source/40×31 plan changed')
        padded=np.full((1240,64),-1,np.int64);sizes=np.asarray([len(batch) for batch in plan],np.int64)
        for i,batch in enumerate(plan):padded[i,:len(batch)]=batch
        from deep import param_groups
        opt=torch.optim.AdamW(param_groups(self.net,.001,1),lr=.001,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=1240)
        schedule=[]
        for _ in plan:
            schedule.append([opt.param_groups[0]['lr'],opt.param_groups[0]['betas'][0]])
            opt.step();scheduler.step()  # no gradients; this records the original scheduler only
        if any(not torch.equal(initial[k],v) for k,v in self.net.state_dict().items()):raise ValueError('schedule fixture mutated initializer')
        opt=torch.optim.AdamW(param_groups(self.net,.001,1),lr=.001,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=1240)
        self.net.train();idx=plan[0];opt.zero_grad();started=time.perf_counter();logits=self.net(x[idx]);loss=loss_fn(logits,y[idx]);forward_seconds=time.perf_counter()-started
        started=time.perf_counter();loss.backward();backward_seconds=time.perf_counter()-started
        gradients={key:value.grad.detach().clone() for key,value in self.net.named_parameters() if value.grad is not None}
        if not gradients or any(not torch.isfinite(v).all() for v in gradients.values()) or max(float(v.abs().max()) for v in gradients.values())<=0:raise ValueError('original first source gradient inactive/nonfinite')
        started=time.perf_counter();opt.step();optimizer_seconds=time.perf_counter()-started
        updated=self.net.state_dict();body='spatial_conv.0.parametrizations.weight.original';head='final_layer.parametrizations.weight.original'
        body_change=float((updated[body]-initial[body]).abs().max());head_change=float((updated[head]-initial[head]).abs().max())
        if body_change<=0 or head_change<=0 or float(self.net.final_layer.weight.norm(dim=1).max())>.500001:raise ValueError('original first body/head update or constraint inactive')
        fields.update(source_only=np.int64(1),epochs=np.int64(40),initialization_seed=np.int64(self.seed),source_people=np.array(sorted(set(people[fit]))),excluded_people=np.array(['P02','P03','P09']),
            source_X=x.numpy(),source_y=y.numpy(),source_session=np.asarray(self._row_sess,np.int64),batch_indices=padded,batch_sizes=sizes,
            expected_schedule=np.asarray(schedule,np.float64),first_logits=logits.detach().numpy(),first_loss=loss.detach().numpy())
        fields.update({'initial_'+key:value.numpy() for key,value in initial.items()})
        fields.update({'first_gradient_'+key:value.numpy() for key,value in gradients.items()})
        fields.update({'first_state_'+key:value.detach().numpy() for key,value in self.net.state_dict().items()})
        record.update(original_initializer_sha256=reference['trace']['initial_state_sha256'],native_fixture_initial_state_sha256=state_sha_from(initial),
            rows_sha256=digest(self._order),source_X_sha256=digest(x.numpy()),source_y_sha256=digest(y.numpy()),scale=self.scale,
            batch_plan_sha256=digest(padded),first_loss=float(loss),optimizer_steps=1,query_labels_opened=False,query_forward=False,
            first_forward_seconds=forward_seconds,first_backward_seconds=backward_seconds,first_optimizer_seconds=optimizer_seconds,
            first_body_change_max=body_change,first_head_change_max=head_change)
    model._train=types.MethodType(capture,model);model.fit(fit)
    args.out.mkdir(exist_ok=False);save_tensors(args.out/'training_fixture.pt',fields)
    record.update(fixture_sha256=sha(args.out/'training_fixture.pt'),producer_sha256=sha(Path(__file__)),source_people=sorted(set(people[fit])))
    (args.out/'proof.json').write_text(json.dumps(record,indent=2))


def state_sha_from(state):
    import hashlib
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):digest.update(key.encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


if __name__=='__main__':main()
