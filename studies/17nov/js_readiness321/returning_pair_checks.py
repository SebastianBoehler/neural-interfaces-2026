"""Shared three-arm source-package inference guards, without labels or a metric."""
import hashlib

import numpy as np
import torch

from package_guard import frame,reference_mix,same_gate


class Frames:
    """Match independent processes' RNG streams while interleaving the three real models."""
    def __init__(self,keys):
        torch.manual_seed(3211701);torch.cuda.manual_seed_all(3211701)
        self.rng={k:(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()) for k in keys}

    def __call__(self,models,raw):
        data={};hashes=[]
        for k,m in models.items():
            cpu,cuda=self.rng[k];torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
            data[k]=frame(m,raw);self.rng[k]=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
            hashes.append(tuple(hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in self.rng[k]))
        if len(set(hashes))!=1:raise ValueError('matched independent-process RNG streams diverged across source arms')
        return data


def compare(cur,data,t,weights,reference,raw_view=None):
    a,pa,ta=data['base'];b,pb,tb=data['off'];c,pc,tc=data['js']
    if a!=b or not np.array_equal(pa,pb):raise ValueError('source off/base final identity failed')
    for field in ('core1','mix','u','shadow','out_v5'):
        if not np.array_equal(ta[field],tb[field]) or not np.array_equal(ta[field],tc[field]):raise ValueError('source common inner changed '+field)
    for field in ('contrast','identity'):
        if not same_gate(ta[field],tb[field]) or not same_gate(ta[field],tc[field]):raise ValueError('source common gate changed')
    if any(not np.array_equal(cur['base'].ft.lab,m.ft.lab) or not np.array_equal(cur['base'].ft.wt,m.ft.wt) for m in cur.values()):raise ValueError('source FT label/weight chain differs')
    for slot in cur['base'].ft.X:
        for m in cur.values():
            if len(m.ft.X[slot])!=len(cur['base'].ft.X[slot]) or (m.ft.X[slot] and not np.array_equal(m.ft.X[slot][-1],cur['base'].ft.X[slot][-1])):raise ValueError('source inner FT inputs differ')
    if cur['base'].ft.gens is not None:
        for slot,g in cur['base'].ft.gens.items():
            if any(not torch.equal(g.get_state(),m.ft.gens[slot].get_state()) for m in cur.values()):raise ValueError('source inner FT generator state differs')
    expected,share=reference_mix(cur['js'],tc,weights,reference);error=float(np.abs(expected-tc['v6']['mix6']).max())
    if error>1e-8 or expected.argmax()!=tc['v6']['mix6'].argmax():raise ValueError('actual source JS owning recipe differs')
    m=cur['js']
    if m.js_step!=t+1 or m.js_done!=t+1 or m.js_calls!=t+1 or m.js_bank.groups[0][0].n:raise ValueError('once/current/readonly source JS input failed')
    if not np.array_equal(m.nets.last_inputs[35],cur['base'].nets.last_inputs[35]):raise ValueError('current source net35 input changed')
    if any(m.log6['err'] or m.cp.dead or m.cp.log['err'] or m.dead or m.ft.over for m in cur.values()):raise ValueError('source package/FT/CP runtime or budget failed')
    view_error=0. if raw_view is None else float(np.abs(ta['u']-raw_view).max())
    if view_error>1e-8:raise ValueError('actual raw source view differs from complete native bank')
    return error,float(np.abs(pc-pa).max()),view_error
