"""Independent original raw-FBC initial states; no EEG, fit, forward or labels."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.project / 'experiments'))
    from deep import build
    from session_bn import swap_bn
    from braindecode.models import FBCNet

    owners = [Path(inspect.getfile(fn)) for fn in (build, swap_bn, FBCNet)]
    original_hashes = {str(path): sha(path) for path in owners}
    fields = {'native_format': torch.tensor(1, dtype=torch.int64)}
    torch.set_num_threads(1)
    for seed in (0, 1):
        torch.manual_seed(seed)
        net = build('fbcnet', 41, 480, pretrained=False)
        if swap_bn(net, False, 20.) != 1:
            raise ValueError('original raw FBC must have exactly one BN replacement')
        state = net.state_dict()
        if len(state) != 26:
            raise ValueError(f'original raw FBC state count changed: {len(state)}')
        for key, value in state.items():
            fields[f'seed{seed}_{key}'] = value.detach().cpu().clone()
            if seed == 0 and key.startswith('spectral_filtering.fir_list.'):
                fields['recipe_' + key] = value.detach().cpu().clone()
    if {str(path): sha(path) for path in owners} != original_hashes:
        raise ValueError('original initializer source changed during capture')
    torch.save(fields, args.output)
    proof = dict(source_sha256=original_hashes, torch_version=torch.__version__, seeds=[0, 1],
                 initialization_only=True, eeg_loaded=False, labels_loaded=False,
                 forward_calls=0, optimizer_steps=0, fixture_sha256=sha(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
