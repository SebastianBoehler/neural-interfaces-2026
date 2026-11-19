"""Use the actual F-C20 raw-family substitution without changing protected state."""
from contextlib import contextmanager

import numpy as np

from locked32 import ARMS,PUPILS
from outer import MASS,full_parent,replace,within_parent


@contextmanager
def arms(order,posteriors,active):
    owner,load=(full_parent,full_parent.full0) if order=="full0" else (within_parent,within_parent.rebuilt)
    native=owner.candidate
    state,observed={},{}
    assert MASS==5/6 and tuple(posteriors)==ARMS[1:]

    def variants(ctx,archive,patches,incumbent):
        assert patches and not state
        stock,expected=native(ctx,archive,patches,incumbent)
        state["stock"]=stock
        for arm,p in posteriors.items():
            score,deltas,proof=replace(native,ctx,archive,patches,incumbent,p,active)
            assert deltas==expected
            state[arm],observed[arm]=score,proof
        np.testing.assert_array_equal(state["identity"],stock)
        np.testing.assert_array_equal(state["hard"],state["detached"])
        assert observed["hard"]==observed["detached"]
        zero=observed["identity"]
        assert zero["slot_movement"]==0
        for k in PUPILS:
            assert observed[k]["slot_movement"]>.01
            assert all(observed[k][s]!=zero[s] for s in ("mix","rebalance","em","tsem"))
            assert all(observed[k][s]==zero[s] for s in ("shadow","gate","refit_states"))
        assert observed["joint"]["tsem"]!=observed["hard"]["tsem"]
        return stock,expected

    owner.candidate=variants
    try:
        with load() as ctx:
            assert tuple(state)==ARMS and len(observed["joint"]["refit_states"])==4
            if order=="full0":
                ref,zero,parent,warm=ctx["matched"]["stock"],ctx["zero"],ctx["proof"],ctx["warm"]
            else:
                ref,zero,parent=ctx["reference"],ctx["archive_zero"],ctx["parent"]
                warm=sum(ctx["ft_work"].values())
            assert warm==9600
            np.testing.assert_array_equal(state["stock"],ref["raw"])
            yield dict(raw=state,proof=observed,reference=ref,zero=zero,parent=parent,warm=warm,
                       **{k:ctx[k] for k in ("count","final","windows","delivery")})
    finally:
        owner.candidate=native
