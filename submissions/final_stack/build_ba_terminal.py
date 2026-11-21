"""Build a new hard+nine25 package with the signed Team A MANAS A1 terminal term."""
import argparse
from copy import copy
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import torch

from build_hard_team_b import SOLVER, once, verify_parent
from patch_dead import NEW
from patch_selftest import WINDOW, windows

HERE = Path(__file__).resolve().parent
PARENT = "final_stack_hard_safe_st_team_b_nine25_tbgraph.zip"
PARENT_SHA = "6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9"
PRODUCTION_SHA = "4bd61c969408980e00eff3ac501ed53bef593eb0e3be04d0e20a4216ebd0fb87"
MONTAGE = ('C2', 'CP2', 'P2', 'Pz', 'CPz', 'Cz', 'C1', 'CP1', 'P1', 'P3', 'PO7', 'P5', 'TP7', 'CP5', 'CP3',
           'C3', 'PO8', 'P6', 'P4', 'TP8', 'CP6', 'CP4', 'C6', 'C4', 'FC6', 'FC4', 'FC2', 'F6', 'F4', 'F2',
           'AF4', 'AF3', 'F5', 'F3', 'F1', 'Fz', 'FC5', 'FC3', 'FC1', 'FCz', 'C5')
POSTERIOR = "        return logits.argmax(-1)"
FORWARD = '''        wave = torch.from_numpy((self.alignment @ raw)[None]).to(self.device)
        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
            f = self.net.features(wave).float()
'''
NEW_FORWARD = '''        wave = torch.from_numpy((self.alignment @ raw)[None])
        if graph is None:
            f = self._forward(wave.to(self.device))
        else:
            f = graph.run(self._forward, wave, self.device)
'''
FORWARD_METHOD = '''    def _forward(self, wave):
        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
            return self.net.features(wave).float()

'''
DECISION = '                labels.append(int(self.cp.step(x, p, self.cf.step(x, p), self.recB if self.routed else not self.active).argmax()))'
TERMINAL = '''                previous = self.cp.fused
                q = self.cp.step(x, p, self.cf.step(x, p), self.recB if self.routed else not self.active)
                labels.append(int(self.terminal_scores(q, self.cp.fused != previous).argmax()))'''


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def eager_reference(directory, source):
    ns = {}
    source = once(source, POSTERIOR, '        self.last = logits.softmax(-1)[0]\n' + POSTERIOR)
    exec(compile(source, '<signed MANAS A1>', 'exec'), ns)
    torch.set_num_threads(1)
    model = ns['Solver']().load_model(dict(submission_dir=directory, device='cpu', ch_names=MONTAGE,
                                         n_chans=41, n_times=480, sfreq=120))
    results = []
    for raw in windows():
        label = model.predict(torch.from_numpy(raw[None]))
        q = model.last.numpy()
        if not np.isfinite(q).all() or label.item() != q.argmax():
            raise ValueError('invalid signed Team A posterior')
        results.append(q.tolist())
    return results


def partner_source(source):
    text = once(source, POSTERIOR, '        self.last = logits.softmax(-1)[0]\n' + POSTERIOR)
    text = once(text, '        self.age = 0\n', '        self.age = 0\n        self.last = None\n')
    text = once(text, 'folder / "weights.pt"', 'folder / "team_a.pt"')
    text = once(text, '    @torch.inference_mode()\n    def predict(self, x):',
                FORWARD_METHOD + '    @torch.inference_mode()\n    def predict(self, x, graph=None):')
    return once(text, FORWARD, NEW_FORWARD)


def patch(source, partner, license_text, weights_sha, expected, use_graph):
    if source.count(NEW) != 1 or source.count('self._check_team_b(dev)') != 1:
        raise ValueError('signed Team B parent is missing its guard or self-test')
    # The official loader accepts only one Python entry point. Keep MANAS symbols isolated.
    license_header = '\n'.join('# ' + line for line in license_text.splitlines())
    namespace = f'{license_header}\n_TEAM_A = {{}}\nexec(compile({partner!r}, "<MANAS A1>", "exec"), _TEAM_A)\n\n'
    addition = (HERE / 'team_a_terminal.py.inc').read_text()
    for key, value in dict(GRAPH=repr(use_graph), SHA=repr(weights_sha), REFERENCE=repr(expected),
                           WINDOWS=WINDOW.rstrip()).items():
        addition = once(addition, f'__TEAM_A_{key}__', value)
    text = once(source, DECISION, TERMINAL)
    text = once(text, SOLVER, namespace + addition + '\n\n' + SOLVER)
    text = once(text, 'model = TeamBlendStack(z,', 'model = TerminalTeamAStack(z,')
    compile(text, 'submission.py', 'exec')
    return text.encode()


def build(parent, donor_dir, target, *, use_graph):
    verify_parent(parent, PARENT_SHA)
    weights = donor_dir / 'weights.pt'
    weights_sha = digest(weights)
    donor = (donor_dir / 'submission.py').read_text()
    reference = eager_reference(donor_dir, donor)
    with TemporaryDirectory(dir=target.parent) as scratch, ZipFile(parent) as before:
        staged = Path(scratch) / target.name
        source = patch(before.read('submission.py').decode(), partner_source(donor),
                       (donor_dir / 'LICENSE-MANAS.txt').read_text(), weights_sha, reference, use_graph)
        with ZipFile(staged, 'x') as after:
            for entry in before.infolist():
                payload = source if entry.filename == 'submission.py' else before.read(entry)
                after.writestr(copy(entry), payload)
            team_a = ZipInfo('team_a.pt')
            team_a.compress_type = ZIP_DEFLATED
            with weights.open('rb') as inp, after.open(team_a, 'w') as out:
                shutil.copyfileobj(inp, out, 2**20)
        with ZipFile(staged) as after:
            if after.namelist() != before.namelist() + ['team_a.pt'] or after.testzip() is not None:
                raise ValueError('candidate archive inventory or CRC differs')
            if after.read('submission.py') != source:
                raise ValueError('candidate Solver bytes differ')
            with after.open('team_a.pt') as inp:
                if hashlib.file_digest(inp, 'sha256').hexdigest() != weights_sha:
                    raise ValueError('candidate Team A bytes differ')
            for entry in before.namelist():
                if entry != 'submission.py' and before.read(entry) != after.read(entry):
                    raise ValueError(f'candidate changed {entry}')
        os.link(staged, target)
    print(digest(target), target.stat().st_size, target, flush=True)


if __name__ == '__main__':
    import sys

    sys.path.insert(0, str(HERE.parents[1] / '2026-competition'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    donor = parser.add_mutually_exclusive_group(required=True)
    donor.add_argument('--production', type=Path)
    donor.add_argument('--scherer-free', type=Path)
    parser.add_argument('--scherer-free-sha')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    if args.target.exists():
        raise FileExistsError(args.target)
    if args.production:
        verify_parent(args.production, PRODUCTION_SHA)
        with TemporaryDirectory(dir=args.target.parent) as scratch, ZipFile(args.production) as archive:
            archive.extractall(scratch)
            build(args.parent, Path(scratch), args.target, use_graph=not args.eager)
    else:
        import json

        meta = json.loads((args.scherer_free / 'training.json').read_text())
        weights = args.scherer_free / 'weights.pt'
        m = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
        if (digest(weights) != args.scherer_free_sha or meta['weights_sha'] != args.scherer_free_sha
                or tuple(meta['corpora']) != m['corpora'] or meta['inputs'] != m['inputs']
                or any('scherer' in item.lower() for item in (*m['corpora'], *m['inputs']))):
            raise ValueError('Scherer-free checkpoint provenance differs')
        build(args.parent, args.scherer_free, args.target, use_graph=not args.eager)
