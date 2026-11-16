"""Read-only verification of the previously scored §281 control, not a scorer."""
import json
import sys
from pathlib import Path

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116')
OUT = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane2/reference_sha256.json')
sys.path.insert(0, str(ROOT / 'studies/14nov/newmember'))
import score_s_v66p as S  # noqa: E402
from s281_routing import verify_archive_file  # noqa: E402
import numpy as np  # noqa: E402

ORDERS = tuple(f'{s}{i}' for s in ('within', 'full', 'runs') for i in (0, 1))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
    S.verify_package(ROOT / 'runs/sealed/final_zips/v66_cp1_cap250.zip',
                     '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed')
    remote = S.remote_manifest(ROOT / 'runs/s281_leo_inputs/remote.sha256')
    cache, hashes = {}, {}
    for order in ORDERS:
        archive = ROOT / f'runs/s281_v66_reference/s242c250c_{order}.npz'
        sha = verify_archive_file(archive, archive.parent / 'sha256.remote.txt')
        route = json.loads((ROOT / f'runs/s281_v66_route/{order}.json').read_text())
        S.verify_route(route, order, sha, S.BASES, sessions)
        for arm in ('control', 'zero_coral'):
            prov = ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
            S.verify_streams(json.loads(prov.read_text()), arm, order, remote)
        for base in S.BASES:
            a = S.load_output(ROOT, 'control', order, base, n, cache)
            b = S.load_output(ROOT, 'zero_coral', order, base, n, cache)
            S.verify_zero(a, b)
            rel = f'runs/s281_v66_cp/control_{order}_{base}.npz'
            hashes[rel] = S.sha256(ROOT / rel)
        print(f'{order}: route, six controls, full-stage zeros and input provenance OK', flush=True)
    with OUT.open('x') as f:
        json.dump({'root': str(ROOT), 'outputs': hashes}, f, indent=2, sort_keys=True)
    print(f'validated {len(hashes)} immutable control files; {OUT}', flush=True)


if __name__ == '__main__':
    main()
