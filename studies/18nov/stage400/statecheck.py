"""Validate archive multipliers against the package's literal previous-30-window state."""
import json
import numpy as np
from capture import HERE,ORIG
from cross import TRACE
from score_s_v66p import BASES,ORDERS


def main():
    worst=0.
    with np.load(ORIG/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        streams=z['sess|all_loso']
    for o in ORDERS:
        scheme=o.rstrip('0123456789')
        with np.load(ORIG/'runs/shuffle_perms.npz') as z:
            perm=z[f'{scheme}|{o[len(scheme):]}|all_loso']
        with np.load(TRACE/f'RB_{o}.npz') as z:
            for b in BASES:
                p=z[b+'|mix'];q=z[b+'|rebal']
                for s in np.unique(streams):
                    rows=perm[streams[perm]==s]
                    for i,r in enumerate(rows):
                        past=rows[max(i-30,0):i]
                        marginal=(5/3+p[past].sum(0))/(5+len(past))
                        out=p[r]/marginal**1.15
                        out/=out.sum()
                        worst=max(worst,float(abs(out-q[r]).max()))
    assert worst<1e-10
    (HERE/'state_guard.json').write_text(json.dumps(dict(previous_windows=30,prior_count=5,power=1.15,max_error=worst)))


if __name__=='__main__':
    main()
