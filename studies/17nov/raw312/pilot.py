"""§312 source-only raw recording and encoder-gradient feasibility; no held scoring."""
import hashlib
import json
import os
from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ['NI26_ROOT'])
OUT = Path(os.environ['NI26_OUT'])
DEVICE = os.environ['NI26_DEVICE']
if DEVICE not in ('cpu', 'cuda'):
    raise ValueError('NI26_DEVICE must be cpu or cuda')
RAW = ROOT / 'data/neural_compet/Dreyer2026Proteus/download/nm000290'
BANDS = ((.1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 55))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def target(x):
    f = torch.fft.rfftfreq(480, 1 / 120).to(x.device)
    power = torch.fft.rfft(x).abs().square() / 480 ** 2
    return torch.cat([x.mean(-1)] + [power[:, :, (f >= lo) & (f < hi)].mean(-1)
                     .clamp_min(1e-12).log() for lo, hi in BANDS], 1)


def prototype(arch, x):
    import braindecode.models as bd
    torch.manual_seed(312)
    cls = getattr(bd, arch)
    net = cls(n_chans=41, n_outputs=3, n_times=480, sfreq=120).to(DEVICE)
    if not hasattr(net, 'final_layer'):
        raise ValueError(f'{arch}: no explicit final_layer')
    net.final_layer = nn.Identity()
    net.eval()  # This pilot tests gradient connectivity, not a production BN recipe.
    with torch.no_grad():
        dim = net(x[:2]).flatten(1).shape[1]
    readout = nn.Linear(dim, 287).to(DEVICE)
    parameters = list(net.parameters()) + list(readout.parameters())
    opt = torch.optim.Adam(parameters, lr=1e-4)
    y = target(x)
    mu, sd = y.mean(0), y.std(0).clamp_min(1e-3)
    y = (y - mu) / sd
    masked = x.clone()
    masked[:, :, 120:240] = 0
    def loss():
        return nn.functional.mse_loss(readout(net(masked).flatten(1)), y)
    before = float(loss().detach())
    max_grad = 0.
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        value = loss()
        if not torch.isfinite(value):
            raise ValueError('nonfinite reconstruction loss')
        value.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('missing/nonfinite encoder gradient')
        max_grad = max(max_grad, max(float(g.abs().max()) for g in grads))
        opt.step()
    after = float(loss().detach())
    if max_grad <= 0 or after >= before:
        raise ValueError(f'{arch}: inactive/unfitted pilot {before}->{after}, gradient {max_grad}')
    return dict(architecture=arch, feature_dim=dim, initial_loss=before,
                final_loss=after, encoder_gradient_max=max_grad, steps=10)


def main():
    torch.set_num_threads(6)
    files = sorted(RAW.glob('sub-P02/ses-*/eeg/*task-BaselineCE*_eeg.edf'))
    if not files:
        raise FileNotFoundError('source P02 BaselineCE EDF absent')
    path = files[0]
    with np.load(ROOT / 'data/proteus_windows.npz') as data:
        names = data['ch_names'].astype(str).tolist()
    raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
    if set(names) - set(raw.ch_names):
        raise ValueError(f'missing official channels: {set(names)-set(raw.ch_names)}')
    raw.pick(names).reorder_channels(names)
    if raw.info['sfreq'] <= 150:
        raise ValueError('native Nyquist cannot support official 75Hz filter')
    raw.crop(tmin=0, tmax=min(67.99, raw.times[-1])).load_data()
    rate = float(raw.info['sfreq'])
    raw.notch_filter([50., 60.], verbose='ERROR')
    raw.filter(.1, 75., verbose='ERROR').resample(120., verbose='ERROR')
    values = raw.get_data() * 1e6
    n = min(16, values.shape[1] // 480)
    if n < 4:
        raise ValueError('too few complete real source crops')
    x = values[:, :n * 480].reshape(41, n, 480).transpose(1, 0, 2).astype('float32')
    if not np.isfinite(x).all() or np.max(np.abs(x)) == 0:
        raise ValueError('invalid raw source values')
    # Fixed units conversion preserves amplitude and channel means.
    xt = torch.from_numpy(x / 100.).to(DEVICE)
    results = [prototype(arch, xt) for arch in ('FBCNet', 'ShallowFBCSPNet')]
    report = dict(status='source_only_pilot_passed', device=DEVICE, source_people=['P02'], excluded=['P01'],
                  raw_file=str(path), raw_sha256=sha(path), native_rate=rate,
                  channels=names, crops=n, crop_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                  official_window_sha256=sha(ROOT / 'data/proteus_windows.npz'),
                  filtering='50/60 notch;0.1-75Hz;resample120Hz;first68s',
                  scale='fixed100uV;no centering/RMS/clipping', models=results,
                  limitation='source optimization only; no held labels, task score or official-window parity')
    OUT.mkdir(exist_ok=True)
    destination = OUT / 'pilot.json'
    with destination.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
