"""Project two measured eye signals onto the native, then exported, EEG montage."""
from pathlib import Path
import sys

import mne
import numpy as np

REPO = Path('../neural-interfaces-2026').resolve()
sys.path.insert(0, str(REPO/'scripts'))
from export_external import preprocess, to_proteus

EYES = ['HEOG', 'VEOG']


def read(directory, run, expected):
    path = directory / f'word{run}.vhdr'
    if '=wor2.' in path.read_text():
        path = directory / 'word2_fixed.vhdr'
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose=False)
    assert raw.info['sfreq'] == 1000
    positions = set(mne.channels.make_standard_montage('standard_1005').ch_names)
    scalp = [n for n in raw.ch_names if n in positions]
    times = np.array([o for o, d in zip(raw.annotations.onset, raw.annotations.description)
                      if d in ('Stimulus/S 16', 'Stimulus/S 32')])
    np.testing.assert_array_equal(np.rint(1000 * times), np.rint(1000 * expected))
    starts = np.rint(times * 1000).astype(int)
    assert starts[0] >= 10000 and np.all(np.diff(starts) >= 21000)
    cols = scalp + EYES
    samples = np.stack([raw.get_data(picks=cols, start=t-10000, stop=t-4000)
                        for t in starts]) * 1e6
    means = samples.reshape(20, len(cols), 300, 20).mean(-1)
    means -= means.mean(-1, keepdims=True)
    eye = preprocess(raw.pick(EYES).load_data()).get_data() * 1e6
    crops = np.stack([eye[:, t:t+480] for t in np.rint((times+1)*120).astype(int)])
    return scalp, means[:, :-2], means[:, -2:], crops


def prepare(p, verify):
    with np.load(REPO/'data/shin2018/windows.npz') as z:
        ids = np.flatnonzero((z['subject'] == f'Shin2018/{p}') & (z['win'] == 0))
        runs = z['run'][ids].astype(int)
        onset = z['onset'][ids]
        original, names = z['X'][ids], z['ch_names'].tolist()
    np.testing.assert_array_equal(runs, np.repeat([1, 2, 3], 20))
    directory = REPO/f'data/shin2018/raw/VP{p:03d}'
    recordings = [read(directory, r, onset[runs == r]) for r in (1, 2, 3)]
    if verify:
        sys.path.insert(0, str(REPO/'studies/15nov/shin278'))
        from export_shin import person
        for r, x, scalp, _, _ in person(p, names):
            assert scalp == recordings[r-1][0]
            direct = [names.index(n) for n in names if n in scalp]
            np.testing.assert_array_equal(original[runs == r][:,direct], x[::2][:,direct])
            error = np.abs(original[runs == r] - x[::2]).max()
            print(f'P{p} run {r} original reconstruction max {error} microvolts')
            assert error <= 1e-4
    aligned, quality = [], []
    for i, (scalp, observed, eye, crop) in enumerate(recordings):
        source_names, eeg_s, eye_s, _ = recordings[i-1]
        assert scalp == source_names
        flat = lambda x: x.transpose(0, 2, 1).reshape(-1, x.shape[1])
        coef, _, rank, _ = np.linalg.lstsq(flat(eye_s), flat(eeg_s), rcond=None)
        assert rank == 2 and np.isfinite(coef).all()
        pred = coef.T @ eye
        truth = np.mean(observed**2)
        quality.append([np.mean((observed-v)**2)/truth for v in (pred, pred[..., ::-1])])
        projected, _ = to_proteus(coef.T @ crop, scalp, names)
        direct = [n for n in names if n in scalp]
        assert len(direct) == 16
        np.testing.assert_array_equal(projected[:, [names.index(n) for n in direct]],
                                      (coef.T @ crop)[:, [scalp.index(n) for n in direct]].astype('f4'))
        aligned.append(projected)
    projected = np.concatenate(aligned)
    shifted = projected[..., ::-1]
    assert np.isfinite(projected).all() and projected.std() > 0
    np.testing.assert_allclose(np.linalg.norm(projected.astype('d'),axis=-1),
                               np.linalg.norm(shifted.astype('d'),axis=-1), rtol=1e-12)
    np.savez(f'person{p}.npz', original=original, aligned=original-projected, reversed=original-shifted,
             quality=quality, rms=np.mean(projected.astype('d')**2)**.5, ids=ids, names=names)


if __name__ == '__main__':
    prepare(int(sys.argv[1]), '--verify' in sys.argv)
