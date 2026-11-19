"""Read verified within0 trajectories and independently rebuild their full0 counterparts."""

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from bridge_mix import archived_zero, install
from bridge_orders import pinned_control
from bridge_owner import PROTO, STACK, corrected, guarded_load, owner, pin, sha
from complete_mix import candidate
from complete_sinc_pair import pair_refits, verify_incumbent
from complete_tail import load_count, load_personalization
from complete_zero import FEAT, read
from member_inputs import ORDERS
from upstream import CAMP, ZERO as WITHIN_ZERO, archive_receipt, frozen_path
from within0_complete import input_delivery, three_aux
from within0_pair_inputs import authenticate


WITHIN = CAMP / "fc18_necessity_v2"
FULL = CAMP / "full0_common_terminal_v1"
MATRIX = CAMP / "full0_common_matched_fit_v1"
WITHIN_PROOF = "b0b86ac866e9cecb8e5bff88dc5fb28602e504d2e41ef9420ca783b22ba7524c"
WITHIN_BANK = "cfd6e64970b2134b73fcfb715200ef1d0ae8647870fdd3ab16179710a3cbfd1e"
FULL_PROOF = "3e96c32dd86984df589e4414e7e039410b9df8e11521b92bce5670d27961510b"
FULL_BANK = "d99fab293f89a9059af7ab35488a22aa3a88f151887339ffaa652efb415ce6a8"
FULL_SRC = "74527664de7163c98b4f633c5f926fdec471b15cc656baec7c5810b58b9d335a"
FULL_MATRIX = "62f4362728056faef2b1937efa7e450a21319908cb2e8cb43a19f4cfa16bbebb"


def meta(path):
    return json.loads(path.read_text())


def selected_arrays(bank, delivery):
    with np.load(bank, allow_pickle=False) as source:
        np.testing.assert_array_equal(source["delivery"], delivery)
        np.testing.assert_array_equal(source["index"], np.arange(2655))
        mapped = {arm: {k: source[f"{arm}|{k}"] for k in ("raw", "filtered", "terminal")} for arm in ("stock", "no_ft")}
    return mapped


@contextmanager
def within0():
    anchor, parent = archive_receipt()
    pin(WITHIN/"complete.json", WITHIN_PROOF)
    path = WITHIN/"complete.npz"
    pin(path, WITHIN_BANK)
    c18 = meta(WITHIN/"complete.json")
    assert c18["stock_bitwise"] and c18["off_zero_exact"] and c18["rows"] == 2655 and not c18["held_labels_opened"]
    assert c18["historical_zero"] == WITHIN_ZERO
    assert c18["output"]["sha256"] == WITHIN_BANK and c18["arms"]["no_ft"]["ft_steps"] == 0
    _, _, provenance, pins = pinned_control("within0")
    final = load_personalization()
    with guarded_load(pins, provenance):
        ctx = owner("within0")
        p = input_delivery()
        np.testing.assert_array_equal(ctx.Pperm, p)
        matched = selected_arrays(path, p)
        with np.load(parent, allow_pickle=False) as source:
            for k in ("raw", "filtered", "terminal"):
                np.testing.assert_array_equal(matched["stock"][k], source[f"corrected|cap250_{k}"])
        yield dict(matched=matched, count=load_count(ctx), final=final, delivery=p, order="within0",
                   windows=read(FEAT,"X").astype(np.float64), proof=WITHIN_PROOF, zero=WITHIN_ZERO,
                   warm=9600, frozen_zero=True)


@contextmanager
def full0():
    from full0_common_case import BRIDGE, BRIDGE_SOURCE, PINS, ZERO, ZERO_SHA, ZERO_SOURCE, qualified_zero

    order = "full0"
    pin(FULL/"source.sha256", FULL_SRC)
    pin(FULL/"matched_full0.json", FULL_PROOF)
    target = FULL/"matched_full0.npz"
    pin(target, FULL_BANK)
    anchor = meta(FULL/"matched_full0.json")
    qualified_zero(ZERO_SHA, order, ZERO, ZERO_SOURCE)
    assert anchor["complete"] and anchor["matches_upstream"] and not anchor["held_labels_opened"]
    assert anchor["order"] == order and anchor["archive_zero"] == ZERO_SHA and anchor["matrix_source"] == FULL_MATRIX
    assert anchor["bridge"] == sha(BRIDGE) and anchor["output"]["sha256"] == FULL_BANK
    source = authenticate(pin, anchor["bridge"], order=order, bridge=BRIDGE, source_sha=BRIDGE_SOURCE)
    raw, gate, provenance, expected = pinned_control(order)
    for f, digest in PINS.items():
        pin(f,digest)
    final = load_personalization()
    with guarded_load(expected, provenance):
        ctx = owner(order)
        p = ctx.Pperm
        np.testing.assert_array_equal(p,input_delivery(order))
        original = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        archived_zero(ctx,install(ctx,{}),raw,gate)
        frozen = Path(source["bank"]["path"])
        pin(frozen,source["bank"]["bank_sha256"])
        with np.load(frozen,allow_pickle=False) as f:
            j = ORDERS.index(order)
            np.testing.assert_array_equal(f["delivery"][j],p)
            ctx.FR17["sinc",0] = f["proba"][j]
        with np.load(source["fbl_bank"]["path"],allow_pickle=False) as f:
            np.testing.assert_array_equal(f["delivery"],p)
            ctx.FR17["fbl",0] = f["proba"]
        with np.load(source["bridge"]["path"],allow_pickle=False) as f:
            bridge = dict(f)
        np.testing.assert_array_equal(bridge["delivery"],p)
        sessions = bridge["session"]
        first = np.concatenate([p[sessions[p]==s][:20] for s in dict.fromkeys(sessions[p])])
        assert len(first)==280
        matrices, proofs = pair_refits(MATRIX,FULL_MATRIX,ctx,source,anchor["bridge"],first,order)
        assert proofs==anchor["refits"]
        work = {f:sum(x["causal"]["steps"] for x in meta(MATRIX/"members"/f"{f}_s0.json")["visits"].values())
                for f in proofs}
        assert set(work.values())=={2400} and len(work)==4
        ctx.FT_BASE,ctx.CU17 = matrices["base"],matrices["causal"]
        three_aux(ctx,3)
        patches = corrected(p,order)
        tape=install(ctx,patches)
        verify_incumbent(ctx,bridge,"corrected",tape)
        on,delta=candidate(ctx,original,patches,tape)
        assert delta==anchor["active_seed0_deltas"]["corrected"]
        off=frozen_path(ctx,original,patches,tape,"V6r")
        literal=frozen_path(ctx,original,patches,tape,"V6Fr")
        np.testing.assert_array_equal(off,literal)
        np.testing.assert_array_equal(on[first],off[first])
        with np.load(target,allow_pickle=False) as f:
            np.testing.assert_array_equal(f["delivery"],p)
            np.testing.assert_array_equal(f["corrected|P0r_posterior"],tape["shadow"][np.argsort(p)])
            stock={k:f[f"corrected|cap250_{k}"] for k in ("raw","filtered","terminal")}
            np.testing.assert_array_equal(stock["raw"],on)
        yield dict(matched={"stock":stock,"no_ft":{"raw":off}}, count=load_count(ctx), final=final,
                   delivery=p, order=order, windows=read(FEAT,"X").astype(np.float64), proof=FULL_PROOF,
                   zero=ZERO_SHA, warm=sum(work.values()), frozen_zero=True)
