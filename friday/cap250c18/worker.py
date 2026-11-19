"""Seal the FT × CSP necessity factorial without touching held labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import sha
from tails import ARMS, exact_prefix, run_arm
from upstream import FEAT, ORDER, rebuilt


def main(smoke):
    if not smoke:
        source = json.loads(Path("smoke.json").read_text())
        assert source["passed"] and source["scope"] == "smoke" and not source["held_labels_opened"]
        assert source["source"] == sha(Path("source.sha256"))
    with rebuilt() as ctx:
        count = ctx["count"]
        all_sessions = tuple(count.SESS)
        assert len(all_sessions) == 14 and ctx["first_20_exact"] == 280
        delivery = ctx["delivery"]
        with np.load(FEAT,allow_pickle=False) as data:
            people, visits = data["subject"], data["session"]
        who = "Dreyer2026Proteus/P01|1"
        def first_visit(s):
            ids = count.C.order_rows(ORDER,s,count.PROTO)
            return set(people[ids]) == {"Dreyer2026Proteus/P01"} and set(visits[ids]) == {"1"}

        p01 = next(s for s in all_sessions if first_visit(s))
        rows = count.C.order_rows(ORDER,p01,count.PROTO)
        assert who in ctx["visits"] and len(rows) == 190 and set(people[rows]) == {who.split("|")[0]}
        complete = all_sessions if not smoke else (p01,)
        delivered = np.concatenate([count.C.order_rows(ORDER,s,count.PROTO) for s in complete])
        if not smoke:
            np.testing.assert_array_equal(delivered,delivery)
        proofs, banks = {}, {}
        for name,(ft,cp) in ARMS.items():
            stage, item = run_arm(ctx,ft,cp,complete)
            if smoke and cp:
                assert (item["csp_refits"],item["csp_fusions"]) == (9,100)
            if not smoke and cp:
                assert (item["csp_refits"],item["csp_fusions"]) == (126,1395)
            profiles = ctx["visits"][who] if smoke else ctx["ft_work"]
            assert set(profiles.values()) == ({170} if smoke else {2400})
            item.update(ft_steps=sum(profiles.values()) if ft else 0,
                        warm_consumed=ft, csp_constructed=cp)
            if not cp:
                assert item["csp_refits"] == item["csp_fusions"] == item["csp_features"] == 0
            if name == "stock":
                for k in stage:
                    np.testing.assert_array_equal(stage[k][delivered],ctx["reference"][k][delivered])
            verify = [p01]
            for s in all_sessions:
                if s != p01:
                    verify += [s,p01]
                    break
            # Use one independent matched path for the interleaved control visit in a smoke.
            if smoke:
                other, _ = run_arm(ctx,ft,cp,(verify[1],))
            else:
                other = stage
            item["prefix_count"] = [exact_prefix(ctx,ft,cp,other if s==verify[1] else stage,s) for s in verify]
            assert min(item["prefix_count"]) > 0
            proofs[name], banks[name] = item, stage
            print(f"PASS F-C18 {name}",item["count_effect"],item["csp_refits"],item["csp_fusions"],item["ft_steps"],flush=True)
        for a,b in (("stock","no_csp"),("no_ft","both_off")):
            np.testing.assert_array_equal(banks[a]["raw"],banks[b]["raw"])
            np.testing.assert_array_equal(banks[a]["filtered"],banks[b]["filtered"])
        assert not np.array_equal(banks["stock"]["filtered"][delivered],banks["no_ft"]["filtered"][delivered])
        result = dict(
            passed=True, scope="smoke" if smoke else "complete", order=ORDER, held_labels_opened=False,
            source=sha(Path("source.sha256")), historical_zero=ctx["archive_zero"], parent=ctx["parent"],
            independent_count=True, stock_bitwise=True, matched_p0r=True, first_20_exact=ctx["first_20_exact"],
            off_zero_exact=ctx["off_zero_exact"], max_raw_ft_effect=ctx["raw_delta"], rows=len(delivered), arms=proofs)
        if not smoke:
            payload = dict(delivery=delivery,index=np.arange(len(delivery)),p0r=ctx["p0r"])
            for name,stage in banks.items():
                payload.update({f"{name}|{k}":p for k,p in stage.items()})
            result["output"] = write_npz(Path.cwd()/"complete.npz",payload)
        output = Path.cwd()/("smoke.json" if smoke else "complete.json")
        write_receipt(output,**result)
    print("PASS F-C18",result["scope"],sha(output),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","complete"))
    main(p.parse_args().mode=="smoke")
