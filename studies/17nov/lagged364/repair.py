"""Reuse the failed smoke's proven upstream bytes; admit registered CP arms."""
import ast
import json
import subprocess
from pathlib import Path

from chain import HERE, MEMBER, ROOT, load_scorer, patch, provenance, sha

V1 = HERE.parent / 'ni26_lagged364_v1_20261117'
SMOKE = {
    'runs/chainout/s281_v66p_lag_zero_within0.npz':
        '1783b05b9d7d3c0c23f31936ba8657f3d858a97d94661adce59a65c280ec7d48',
    'runs/newmember/s281_v66p_lag_zero_within0_gate.npz':
        'bb2e972e1b9efbfcd08d89af1fece8827469522463ed7cfc5ae0b2eab8e5e028',
    'runs/newmember/s281_v66p_lag_zero_within0_provenance.json':
        'bfcc0bcbd46d6c2fae773321db75441373b38d62e9e9a5475a745f5ddfd08bac',
}
CONTROL = MEMBER / 's281_cp.py'
PROOF = 'aa91ec14b970c4ea43ba47c2e187c41b866298ea319941344b60b74c603d350b'
RUNTIME = 'a4fe2f5350db8c5a8c3dae1e8e517b415167759e1cbe9e032b6bf8ca65546ac2'
FAILURE = '6a19db41a24d271e60cdf079b21010777e893bd011c7eb2e6684bbb1df67bb36'
OLD_CP = '2737bcfea067b0adb5c92d051b90d43ed0839f2d56f7d94b6374fcde40721807'


def verify_resume(arm, order, flag):
    if (arm, order, flag) != ('lag_zero', 'within0', '--resume-cp'):
        raise ValueError('only the frozen-zero CP step may resume')
    if any(sha(root / path) != h for root in (ROOT, V1 / 'chain') for path, h in SMOKE.items()):
        raise ValueError('original smoke upstream bytes changed')
    scorer = load_scorer()
    scorer.verify_manifest(provenance(arm, order, ROOT), arm, order, {}, ROOT)


def main():
    if (ROOT.exists() or sha(V1 / 'runtime.sha256') != RUNTIME
            or sha(V1 / 'proof.json') != PROOF or sha(V1 / 'slurm-3122524.out') != FAILURE
            or sha(V1 / 'chain' / CONTROL) != OLD_CP):
        raise ValueError('original stage or failed smoke identity differs')
    subprocess.run(['sha256sum', '-c', str(V1 / 'runtime.sha256')], cwd=V1 / 'chain',
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['cp', '-a', str(V1 / 'chain'), str(ROOT)], check=True)
    rule = next(line for line in (ROOT / CONTROL).read_text().splitlines()
                if line.startswith('    assert arm in {'))
    if len(ast.literal_eval(rule.split(' in ', 1)[1])) != 10:
        raise ValueError('old CP guard did not have the original ten names')
    patch(ROOT / CONTROL, rule + '\n', '')
    patch(ROOT / CONTROL, 'from s281_provenance import InputTracker, write_manifest',
          'from s281_provenance import InputTracker, RECIPES, write_manifest\n'
          '    assert arm in RECIPES, f"unregistered CP arm: {arm}"')
    old = V1 / 'runtime.sha256'
    with (HERE / 'runtime.sha256').open('x') as f:
        for line in old.read_text().splitlines():
            h, path = line.split()
            f.write(f'{sha(ROOT / CONTROL) if path == str(CONTROL) else h}  {path}\n')
    proof = json.loads((V1 / 'proof.json').read_text())
    proof['inputs'][str(V1 / 'proof.json')] = PROOF
    for path in ('runtime.sha256', 'chain-source.sha256', 'finish-source.sha256'):
        proof['inputs'][str(HERE / path)] = sha(HERE / path)
    with (HERE / 'proof.json').open('x') as f:
        json.dump(proof, f, sort_keys=True, indent=2)
    verify_resume('lag_zero', 'within0', '--resume-cp')
    print('Frozen-zero upstream, code and 24 staged banks preserved; only CP dispatch corrected.',
          flush=True)


if __name__ == '__main__':
    main()
