"""Measure the scale change implied by the trained memory on unlabeled EEG."""
import json
from pathlib import Path

import numpy as np
import torch

from train import BASE, DATA, ORDERS, PERM, Decoder, digest, sha


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    with np.load(DATA) as z:
        x = torch.from_numpy(z['X']).float()
        person = np.array([s.split('/')[-1] for s in z['subject']])
        stream = np.char.add(np.char.add(person,'|'),z['session'])
    with np.load(PERM) as z:
        orders = {o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    result = {}
    for held in sorted(set(person)):
        src = np.flatnonzero(person != held)
        target = np.flatnonzero(person == held)
        donor = src[stream[src] == sorted(set(stream[src]))[0]]
        proof = json.loads(Path(held+'.json').read_text())
        weight = Path(held+'_memory.pt')
        assert sha(weight) == proof['checkpoints']['memory'] and digest(src) == proof['train_rows_sha256']
        net = Decoder().eval()
        net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
        scale = x[src].square().mean().sqrt()
        mu, second, code = net.unpack(net.encode(x[target]/scale))
        donor_code = net.unpack(net.encode(x[donor]/scale))[2]
        var = (net.second-net.mean.square()).clamp_min(1e-6)
        blank = net.read(torch.zeros(1,49)).chunk(2,-1)[1].tanh()
        cells = []
        for o,p in orders.items():
            sessions = []
            for session in set(stream[target]):
                idx = np.searchsorted(target,p[np.isin(p,target[stream[target] == session])])
                age = torch.arange(1,len(idx),dtype=torch.float32)[:,None]
                avg = (20*net.mean+mu[idx[:-1]].cumsum(0))/(20+age)
                energy = (20*net.second+second[idx[:-1]].cumsum(0))/(20+age)
                desired = .5*(var.log()-(energy-avg.square()).clamp_min(1e-6).log())
                losses = [desired[31:].square().mean().item()]
                for history in (code[idx[:-1]],donor_code[torch.arange(len(idx)-1)%len(donor)]):
                    state = torch.cat((history.cumsum(0)/age,age/(age+20)),-1)
                    actual = net.read(state).chunk(2,-1)[1].tanh()-blank
                    losses.append((actual-desired)[31:].square().mean().item())
                sessions.append(losses)
            cells.append(np.mean(sessions,0))
        cells = np.stack(cells)
        result[held] = np.mean([cells[:4].mean(0),cells[4:6].mean(0),cells[6:].mean(0)],0).tolist()
    blank, real, donor = np.mean(list(result.values()),0)
    Path('gain379.json').write_text(json.dumps(dict(columns=['blank','real','donor'],mse=result,
        reduction_vs_blank=float(1-real/blank), reduction_vs_donor=float(1-real/donor)),indent=2))


if __name__ == '__main__':
    main()
