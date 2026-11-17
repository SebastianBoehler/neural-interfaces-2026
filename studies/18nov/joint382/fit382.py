"""Train Galvani's native A6 reference under a pinned fold-excluded source schedule."""
import json
from pathlib import Path
import sys
import torch
from kernel import train
from restore import prior,joint,init,source_arrays


def main(arm,seed,smoke):
    torch.set_num_threads(2)
    assert 'A100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    data=source_arrays(d,rows,runtime)
    net=init(runtime.net,arm)
    probe=torch.from_numpy(data['x'][:12])
    with torch.no_grad():
        cpu=net(probe)
        net.cuda()
        gpu=net(probe.cuda()).cpu()
        error=(gpu-cpu).abs().max().item()
        assert error<5e-3 and torch.equal(gpu.argmax(1),cpu.argmax(1)),(arm,error)
    rec=train(net,*(torch.from_numpy(data[k]).cuda() for k in ('x','y','weights')),
              data['streams'],arm,seed,3 if smoke else 120)
    name=('smoke_' if smoke else '')+arm+'_'+str(seed)
    rec.update(arm=arm,source_sha=joint.sha(Path('source.sha256')),scope=joint.FIT_PEOPLE,
               rows=joint.digest(rows),held_labels_opened=False,probe_max_error=error)
    if not smoke:
        path=Path(name+'.pt')
        torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},path)
        rec['checkpoint_sha']=joint.sha(path)
    Path(name+'.json').write_text(json.dumps(rec,indent=2))
    print(name,rec['seconds'],rec['groups'],flush=True)


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
