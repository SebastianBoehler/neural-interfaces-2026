"""Use the signed raw-slot replacement above the true adapted-weight operation."""
from contextlib import contextmanager

import numpy as np

from bridge_owner import PROTO, STACK, stamp
from outer import full_parent, replace, within_parent
from parents22 import ARMS


@contextmanager
def doubling(ctx, archive):
    """Keep the F-C20 raw-slot guard and double the independent live FBL member."""
    original_mix = ctx.A.mix
    slot = ctx.SLOT['fbl']
    frozen = archive[slot] + np.mean([ctx.FR17['fbl',s]-ctx.FR6['fbl',s] for s in range(6)], axis=0)
    warm = ctx.CU17['fbl',0] - ctx.FT_BASE['fbl',0]
    order = ctx.Pperm
    actual = ctx.norm(frozen+warm)[order]
    warm_rows = int(np.count_nonzero(np.any(actual != ctx.norm(frozen)[order],axis=1)))
    assert warm_rows == 2375 and np.abs(warm).max() > .001
    proof = {}

    def outer_mix(bn, proto, families, extra):
        assert (bn,proto)==(STACK,PROTO) and not proof
        assert len(dict(extra))==len(extra) and list(extra).count(('sc_fbl',1.))==1
        live = ctx.A.members(bn,proto)['sc_fbl']
        np.testing.assert_array_equal(live,actual)
        before,share = original_mix(bn,proto,families,extra)
        assert share==1/16
        updated = [(name,w*2 if name=='sc_fbl' else w) for name,w in extra]
        changed,share = original_mix(bn,proto,families,updated)
        np.testing.assert_allclose(changed,(16*before+live)/17,atol=2e-14,rtol=0)
        assert share==1/17 and np.abs(changed-before).max() > .001
        proof.update(warm_rows=warm_rows, adapted=stamp(live), share=share, mix=stamp(changed))
        return changed,share

    ctx.A.mix = outer_mix
    try:
        yield proof
    finally:
        ctx.A.mix = original_mix


@contextmanager
def run_arms(order, pupils):
    owner, load = (full_parent,full_parent.full0) if order=='full0' else (within_parent,within_parent.rebuilt)
    base = owner.candidate
    all_raw, traces = {}, {}

    def visit(ctx, archive, patches, incumbent):
        assert patches and not all_raw
        stock, deltas = base(ctx,archive,patches,incumbent)
        all_raw['stock'] = stock
        active = np.ones(len(stock),dtype=bool)
        for name,values in (('identity',None),('multi',pupils['multi'])):
            raw, used, trace = replace(base,ctx,archive,patches,incumbent,values,active)
            assert used==deltas
            all_raw[name],traces[name]=raw,trace
        np.testing.assert_array_equal(all_raw['identity'],stock)
        for name,values in (('FBL2',None),('joint',pupils['multi']),('single_joint',pupils['single'])):
            with doubling(ctx,archive) as boost:
                raw, used, trace = replace(base,ctx,archive,patches,incumbent,values,active)
                assert used==deltas and boost.get('mix')==trace['mix']
            all_raw[name],traces[name]=raw,dict(trace,boost=boost)
        assert traces['joint']['slot_movement']==traces['multi']['slot_movement'] > .01
        assert all(traces[arm]['boost']['adapted']==traces['FBL2']['boost']['adapted']
                   for arm in ('joint','single_joint'))
        assert all(traces[arm][key]!=traces['FBL2'][key]
                   for arm in ('joint','single_joint') for key in ('mix','rebalance','em','tsem'))
        return stock,deltas

    owner.candidate=visit
    try:
        with load() as ctx:
            assert set(all_raw)==set(ARMS)
            if order=='full0':
                control,zero,parent = ctx['matched']['stock'],ctx['zero'],ctx['proof']
            else:
                control,zero,parent = ctx['reference'],ctx['archive_zero'],ctx['parent']
            np.testing.assert_array_equal(all_raw['stock'],control['raw'])
            yield dict(raw=all_raw,reference=control,zero=zero,parent=parent,traces=traces,
                       **{name:ctx[name] for name in ('count','final','windows','delivery')})
    finally:
        owner.candidate=base
