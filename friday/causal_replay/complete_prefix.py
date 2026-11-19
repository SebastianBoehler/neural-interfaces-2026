"""Exercise count and CSP prefixes on actual latched sessions under the verified zero."""
import json

import numpy as np

from bridge_owner import HERE, MEANS, guarded_load, owner, pin, sha
from complete_tail import active_prefixes, load_count, load_personalization
from complete_zero import FEAT, HASHES, STAGES, no_route, read

ZERO = MEANS / "terminal_full0_v2/zero.json"
ZERO_SHA = "9b9dc4070b234a6df2c2d806c7ad5f2ddf2dbd231370fae688369a78b046663c"


def main():
    pin(ZERO, ZERO_SHA)
    zero = json.loads(ZERO.read_text())
    control = STAGES / "stages_full0.npz"
    for path in (control, FEAT):
        pin(path, HASHES[path])
    pin(MEANS / "dependents_full0_v2/dependents_full0.json", HASHES[MEANS / "dependents_full0_v2/dependents_full0.json"])
    no_route()
    if not zero["exact_archive_zero"] or not zero["three_aux_exact"] or zero["held_labels_opened"]:
        raise ValueError("historical terminal proof is missing")
    final = load_personalization()
    with guarded_load() as opened:
        ctx = owner()
        count = load_count(ctx)
        latch = count.latch("full0")
        if {str(s): t for s, t in latch.items()} != zero["stages"]["latch"]:
            raise ValueError("historical count detector latch changed")
        with np.load(control, allow_pickle=False) as z:
            replay = {key: z[key] for key in ("raw", "filtered", "terminal")}
        windows = read(FEAT, "X").astype(np.float64)
        evidence = active_prefixes(count, final, **replay, windows=windows, latch=latch)
    receipt = HERE / "active_prefix.json"
    with receipt.open("x") as handle:
        json.dump(dict(**evidence, zero=ZERO_SHA, archive=HASHES[control],
                       source_sha256=sha(HERE / "source.sha256"),
                       held_labels_opened=False, refit_matrix_opened=False,
                       inputs={str(p): digest for p, digest in sorted(opened.items())}),
                  handle, sort_keys=True, indent=2)
        handle.write("\n")
    receipt.chmod(0o444)
    print(f"PASS active count and CSP prefix {sha(receipt)}", flush=True)


if __name__ == "__main__":
    main()
