"""Score four component arms on the authenticated Fable R6 causal tape."""
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

import returning_diet_worker as r6
from diet_decomposition_patch import ARMS, patch
from diet_decomposition_tail import intervene


def smoke(model, folder, data, signals, rows, items, receipt, age):
    prefix = rows[:84]
    half_source = r6.patch_em((folder / "submission.py").read_text(), 0.5)
    half = r6.modified(model, items, "half", route=True)
    for a, b in zip(intervene(model, items, "all"), r6.diet(model, items, "quarter"), strict=True):
        np.testing.assert_array_equal(a, b, err_msg="recomposed quarter")
    with TemporaryDirectory() as temp:
        for arm, settings in ARMS.items():
            root = Path(temp) / arm
            r6.stage(root / r6.ROSTER, folder, patch(half_source, settings))
            live, _ = r6.load(data, root, r6.ROSTER, receipt)

            def check(m, n):
                r6.check(m, n)
                tail = m.routeB.tail
                expected_count = age - 1 if settings[0] else n
                assert (m.log6["rec"], m.cf.t, m.cp.t, len(tail.em.A), len(tail.tsem.U)) == (
                    age - 1, expected_count, n, n, n
                ), f"{arm}: active path counts differ"
                assert tail.em.state is not None and tail.tsem.state is not None, f"{arm}: EM fits inactive"

            seen = r6.tap(live, signals, prefix, r6.fresh, check, r6.DEVICE)
            result = intervene(model, items, arm)
            short = intervene(model, items[:len(prefix)], arm)
            for name, scores, cut in zip(("label", "terminal", "route_out"), result, short, strict=True):
                recorded = np.asarray([getattr(d, name) for d in seen])
                expected = scores[:len(prefix)]
                np.testing.assert_array_equal(expected, cut, err_msg=f"{arm} {name} causal prefix")
                compare = np.testing.assert_array_equal if name == "label" else partial(
                    np.testing.assert_allclose, rtol=0, atol=2e-5
                )
                compare(expected, recorded, err_msg=f"{arm} {name} live parity")
            idx = 1 if arm == "a" else 2
            assert np.max(np.abs(result[idx][age - 1:] - half[idx][age - 1:])) > 2e-5, f"{arm} is inactive"
            if arm == "a":
                np.testing.assert_array_equal(result[2], half[2], err_msg="CountFilter changed RouteB")
            del live
    print("PASS live labels, stage activity, causal prefixes and exact recomposed quarter", flush=True)


if __name__ == "__main__":
    parser = r6.parser()
    parser.add_argument("--smoke-key")
    args = parser.parse_args()
    r6.run(args, arms=ARMS, intervention=intervene, smoke_check=smoke, smoke_key=args.smoke_key)
