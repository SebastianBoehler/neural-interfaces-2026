"""Re-run the locked §318 mixer contrasts at the authenticated corrected hook."""

from contextlib import contextmanager

import numpy as np

import inputs as full_parent
import upstream as within_parent
from bridge_owner import PROTO, STACK, stamp

TREATMENTS = dict(identity=(False,1.),SS=(True,1.),FBL2=(False,2.),JS=(True,2.))
NAME = "nm_fb_cc30"


def trial(original, ctx, archive, patches, incumbent, cc, swap, factor, zero):
    old_mix, rebalance, em, tsem=ctx.A.mix,ctx.gm.M.rebal,ctx.gm.em_g,ctx.gm.tsem_g
    p=ctx.Pperm
    cc=cc[p]
    inner={k:stamp(incumbent[k]) for k in ("rebal","gate","shadow")}
    warm={key:stamp(value) for key,value in ctx.CU17.items()}
    reads={key:stamp(value) for key,value in ctx.FT_BASE.items()}
    assert set(warm)==set(reads)=={(f,0) for f in ("fb","fbl","fbraw","sinc")}
    fb_light="fbl"
    slot=ctx.SLOT[fb_light]
    frozen=archive[slot]+np.mean([ctx.FR17[fb_light,s]-ctx.FR6[fb_light,s] for s in range(6)],axis=0)
    warm_delta=ctx.CU17[fb_light,0]-ctx.FT_BASE[fb_light,0]
    expect=ctx.norm(frozen+warm_delta)[p]
    fresh=ctx.norm(frozen)[p]
    fbl_change=float(np.abs(expect-fresh).max())
    assert fbl_change>1e-3 and np.isfinite(cc).all()
    seen={}

    def substitute(bn,proto,fams,extra):
        assert (bn,proto)==(STACK,PROTO) and not seen and NAME not in ctx.A.EXTRA
        members=ctx.A.members(bn,proto)
        weights=dict(extra)
        assert len(weights)==len(extra) and weights["sc_fb"]==weights["sc_fbl"]==weights["sc_rest"]==1.
        np.testing.assert_array_equal(members["sc_fbl"],expect)
        before,share=old_mix(bn,proto,fams,extra)
        assert share==1/16
        next_extra=list(extra)
        if swap:
            next_extra=[(k,v) for k,v in next_extra if k!="sc_fb"]
            next_extra.append((NAME,1.))
        elif zero:
            next_extra.append((NAME,0.))
        next_extra=[(k,factor*v if k=="sc_fbl" else v) for k,v in next_extra]
        expected=weights.copy()
        if swap:
            expected.pop("sc_fb")
            expected[NAME]=1.
        elif zero:
            expected[NAME]=0.
        expected["sc_fbl"]=factor
        assert dict(next_extra)==expected
        if swap or zero:
            native=ctx.A.EXTRA["sc_fb"]
            ctx.A.EXTRA[NAME]=dict(native,**{PROTO:cc})
        try:
            if swap or zero:
                np.testing.assert_array_equal(ctx.A.members(bn,proto)[NAME],cc)
            got=old_mix(bn,proto,fams,next_extra)
        finally:
            if swap or zero:
                del ctx.A.EXTRA[NAME]
        d=(cc-members["sc_fb"]) if swap else 0.
        want=(16*before+d+(factor-1)*expect)/(15+factor)
        np.testing.assert_allclose(got[0],want,atol=2e-14,rtol=0)
        assert got[1]==1/(15+factor)
        if zero:
            np.testing.assert_array_equal(got[0],before)
        difference=float(np.abs(got[0]-before).max())
        assert difference==0 if zero else difference>1e-3
        seen.update(mix=stamp(got[0]), max_mix=difference,share=got[1],
                    fbl_post_warm_sha=stamp(expect),fbl_warm_effect=fbl_change,
                    fbl_warm_rows=int(np.any(expect!=fresh,1).sum()),
                    factor=factor,fb_warm_removed=swap)
        return got

    def hook(name, earlier, gate=False):
        def observe(x,proto,*args):
            assert name not in seen and proto==PROTO
            if gate:
                np.testing.assert_array_equal(args[0],incumbent["gate"])
            got=earlier(x,proto,*args)
            seen[name]=stamp(got)
            return got
        return observe

    ctx.A.mix=substitute
    ctx.gm.M.rebal=hook("rebal",rebalance)
    ctx.gm.em_g=hook("em",em,True)
    ctx.gm.tsem_g=hook("tsem",tsem,True)
    try:
        out,delta=original(ctx,archive,patches,incumbent)
    finally:
        ctx.A.mix,ctx.gm.M.rebal,ctx.gm.em_g,ctx.gm.tsem_g=old_mix,rebalance,em,tsem
    assert all(k in seen for k in ("mix","rebal","em","tsem"))
    assert inner=={k:stamp(incumbent[k]) for k in inner} and ctx.GB[STACK,"P0r"] is incumbent["rebal"]
    assert warm=={key:stamp(value) for key,value in ctx.CU17.items()}
    assert reads=={key:stamp(value) for key,value in ctx.FT_BASE.items()}
    return out,delta,seen


@contextmanager
def variants(order,cc):
    owner,rebuild=(full_parent,full_parent.full0) if order=="full0" else (within_parent,within_parent.rebuilt)
    original=owner.candidate
    results,receipts={}, {}

    def capture(ctx,archive,patches,incumbent):
        assert patches and "sc_fb" in dict(incumbent["items"])
        assert dict(incumbent["items"])["sc_fb"]==1.
        stock,deltas=original(ctx,archive,patches,incumbent)
        results["stock"]=stock
        for arm,(swap,weight) in TREATMENTS.items():
            out,proof,stage=trial(original,ctx,archive,patches,incumbent,cc,swap,weight,arm=="identity")
            assert proof==deltas
            results[arm],receipts[arm]=out,stage
        np.testing.assert_array_equal(results["identity"],stock)
        for arm in ("SS","FBL2","JS"):
            assert all(receipts[arm][k]!=receipts["identity"][k] for k in ("rebal","em","tsem"))
        assert not any(np.array_equal(results["JS"],results[k]) for k in ("SS","FBL2"))
        return stock,deltas

    owner.candidate=capture
    try:
        with rebuild() as ctx:
            if order=="full0":
                stock,zero,parent=ctx["matched"]["stock"],ctx["zero"],ctx["proof"]
            else:
                stock,zero,parent=ctx["reference"],ctx["archive_zero"],ctx["parent"]
            np.testing.assert_array_equal(stock["raw"],results["stock"])
            yield dict(raw=results,reference=stock,zero=zero,parent=parent,stages=receipts,
                       **{k:ctx[k] for k in ("count","final","windows","delivery")})
    finally:
        owner.candidate=original
