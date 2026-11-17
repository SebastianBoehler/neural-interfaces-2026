"""Independent deterministic CPU original→both same-weight execution variants through actual215FT/CP."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from phase_guard import VARIANTS,load,check_pair,reset_pair
from guard_shared import OLD,DATA
from phase_state import neural_state
from reference import sha


def main():
    start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
    with np.load(DATA,allow_pickle=False) as z:x=z['X'][:215].copy();channels=z['ch_names']
    results={}
    for variant,folder in VARIANTS.items():
        old=load(OLD,'cpu215_old_'+variant,channels,'cpu',True);new=load(folder/'prepared_jsall','cpu215_new_'+variant,channels,'cpu',True)
        maximum,saved,initial,final=check_pair(old,new,x,'cpu')
        if old.ft.refits!=20 or new.ft.refits!=20 or old.cp.refits!=10 or new.cp.refits!=10 or old.cp.fused!=125 or new.cp.fused!=125:
            raise ValueError('CPU215 FT/CP activity failed')
        generators=[{s:hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest() for s,g in model.ft.gens.items()} for model in (old,new)]
        if generators[0]!=generators[1]:raise ValueError('CPU augmentation generator mismatch')
        reset=reset_pair(old,new,x,saved,'cpu')
        results[variant]=dict(max_output_error=maximum,initial=initial,final=final,reset=reset,generators=generators,
                             ft_refits=[old.ft.refits,new.ft.refits],cp_refits=[old.cp.refits,new.cp.refits],cp_fused=[old.cp.fused,new.cp.fused],
                             complete_sha256=sha(folder/'complete.json'))
    r=dict(status='deterministic_CPU215_semantics_passed',variants=results,source_sha256=sha(Path('source.sha256')),
           deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads(),
           seconds=time.monotonic()-start,accuracy_computed=False,scope='CPU same-weight semantics;not GPU production/timing qualification')
    Path('cpu_semantic.json').write_text(json.dumps(r,indent=2));print(r['status'],flush=True)


if __name__=='__main__':main()
