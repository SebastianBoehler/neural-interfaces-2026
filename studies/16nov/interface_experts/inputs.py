"""Lock §301 source, references and unlabelled gate features without scoring."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import sha, DATA_SHA

W = Path('/mnt/lustre/work/gehler/gfh527')
PROD = W / 'ni26_s301_native_producer_v3_20261116'
REF = W / 'ni26_s301_native_reference_all_20261116/runs/s301_reference_all'
STYLE = W / 'ni26_s306_style_20261116/runs/style306/features.npz'
CORE = W / 'ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so'
TRAIN = W / 'ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so'
PARITY = W / 'ni26_s301_native_parity_v3_20261116/runs/s301_parity/report.json'
EXPECTED = {'producer.sha256': '21c1af2e0784329da2f662319ecdac3f6757715cbcece604fa1db46d21b659bf',
            'manifest.json': '71ee9a414844cce01f70b3d355d15c92c9a10fcb01e899c4ebe77d196f51e14e',
            'libnative_core.so': '4bcbb983086f690780c5690b25fcf2f429a88ff3377d108a7f4b12535fa0fb56',
            'libtrainref.so': '1508f45481b8a396d181b99e07cd72206383e51b8674e876cacadc5dbae131de',
            'report.json': '0fecfd2110a48728079934c6e6e59db3b80a1532f5a4af49c08809b026aa4256',
            'features.npz': '71bb7069550aa09c75df37a2be2e59a6607af584a469fed7cd7f3bf4262864e6',
            'metrics.py': 'fcc739400806a74259f248618501376a4fe53f2b5a40ccdc20db6a9e2369787d',
            'proteus_windows.npz': DATA_SHA,
            'shuffle_perms.npz': '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'}


def check_sources():
    for manifest in ('source.sha256', 'producer.sha256'):
        subprocess.run(['sha256sum', '-c', manifest], cwd=PROD, check=True, stdout=subprocess.DEVNULL)


def paths():
    return ([PROD / 'producer.sha256', REF / 'manifest.json', CORE, TRAIN, PARITY, STYLE,
             PROD / 'data/proteus_windows.npz', PROD / 'runs/shuffle_perms.npz',
             PROD / '2026-competition/benchmark_utils/metrics.py'] +
            [PROD / f'runs/s301_native_inputs/P{i:02d}.npz' for i in range(1, 11)] +
            [REF / f'P{i:02d}.npz' for i in range(1, 11)] +
            [PROD / 'runs/coremember.npz', PROD / 'runs/coremember_within2_within3_full1_runs1.npz'])


def main():
    root = Path(os.environ['EXPERT_ROOT']).resolve(strict=True)
    output = root / 'runs/input306/inputs.json'
    if output.exists(): raise FileExistsError(output)
    check_sources(); files = {}
    for path in paths():
        digest = sha(path)
        if path.name in EXPECTED and EXPECTED[path.name] != digest:
            raise ValueError(f'changed known input: {path}')
        files[str(path)] = digest
    with (REF / 'manifest.json').open() as f: reference = json.load(f)
    if any(files[str(REF / p)] != digest for p, digest in reference['files'].items()):
        raise ValueError('reference-head manifest mismatch')
    if set(reference['files']) != {f'P{i:02d}.npz' for i in range(1, 11)}:
        raise ValueError('incomplete source reference matrix')
    check_sources(); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(dict(files=files, held_labels_opened=False), f, sort_keys=True, indent=2); f.write('\n')
    with output.with_suffix('.sha256').open('x') as f: f.write(sha(output) + '  inputs.json\n')
    print('locked', len(files), 'source inputs; sha256', sha(output), flush=True)


if __name__ == '__main__': main()
