"""Pin public official DINOv2 code/weights and retain their actual terms."""
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
REPO = 'https://github.com/facebookresearch/dinov2.git'
WEIGHTS = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(root):
    target = root / 'assets'
    if target.exists():
        raise FileExistsError('asset preparation is one-shot')
    target.mkdir(); code = target / 'dinov2'
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git', 'init', str(code)], check=True, env=env)
    subprocess.run(['git', '-C', str(code), 'remote', 'add', 'origin', REPO], check=True, env=env)
    subprocess.run(['git', '-C', str(code), 'fetch', '--depth=1', 'origin', COMMIT], check=True, env=env)
    subprocess.run(['git', '-C', str(code), 'checkout', '--detach', COMMIT], check=True, env=env)
    actual = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError('official code revision mismatch')
    license_text = (code / 'LICENSE').read_text()
    card = (code / 'MODEL_CARD.md').read_text()
    if 'Apache License' not in license_text or 'Apache License 2.0' not in card or 'ViT-S' not in card:
        raise ValueError('expected official code/model terms not confirmed')
    checkpoint = target / 'dinov2_vits14_pretrain.pth'
    with urllib.request.urlopen(WEIGHTS, timeout=120) as source, checkpoint.open('xb') as output:
        while chunk := source.read(1 << 20):
            output.write(chunk)
    files = [code / 'LICENSE', code / 'MODEL_CARD.md', checkpoint]
    files += list((code / 'dinov2').rglob('*.py'))
    files += [p for p in code.glob('NOTICE*') if p.is_file()]
    hashes = {str(p.relative_to(root)): sha(p) for p in sorted(files)}
    record = dict(repository=REPO, revision=COMMIT, checkpoint_url=WEIGHTS,
                  checkpoint_bytes=checkpoint.stat().st_size, files=hashes,
                  code_license='Apache-2.0', model_license='Apache-2.0 per pinned official MODEL_CARD',
                  access='public HTTPS;no login/acceptance/paid service', labels_opened=False)
    with (target / 'manifest.json').open('x') as output:
        json.dump(record, output, indent=2)
    for path in files + [target / 'manifest.json']:
        path.chmod(0o444)
    return record


def verify(root):
    info = json.loads((root / 'assets/manifest.json').read_text())
    if info['revision'] != COMMIT or info['checkpoint_url'] != WEIGHTS:
        raise ValueError('asset identity differs')
    for path, expected in info['files'].items():
        if sha(root / path) != expected:
            raise ValueError(f'changed visual asset:{path}')
    return info
