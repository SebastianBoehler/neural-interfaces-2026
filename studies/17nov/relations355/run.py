"""Fixed whole-decoder LOSO producers; complete banks before collector label access."""
import argparse
import copy
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from model import ARMS, Relations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA route unavailable; no CPU fallback')


def fit(x, y, arm, epochs):
    torch.manual_seed(355)
    torch.cuda.manual_seed_all(355)
    net = Relations(arm).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=.01)
    rng = np.random.default_rng(355)
    before = net.filters.weight.detach().clone()
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total = 0.
        net.train()
        for start in range(0, len(x), 64):
            ids = order[start:start + 64]
            opt.zero_grad(set_to_none=True)
            logits = net(x[ids])
            loss = torch.nn.functional.cross_entropy(logits, y[ids])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if not torch.isfinite(net.filters.weight.grad).all() or net.filters.weight.grad.norm() == 0:
                raise ValueError('Inactive/nonfinite learned temporal bank')
            opt.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total / len(x))
        print(json.dumps(dict(arm=arm, epoch=epoch + 1, source_loss=losses[-1])), flush=True)
    assert not torch.equal(before, net.filters.weight)
    return net.eval(), losses


@torch.no_grad()
def predict(net, x):
    return torch.cat([net(x[i:i + 64]).softmax(1).cpu() for i in range(0, len(x), 64)]).numpy()


def guards(net, x, arm):
    before = {k: v.clone() for k, v in net.state_dict().items()}
    a = predict(net, x[:17])
    predict(net, x[17:24])
    assert np.array_equal(a, predict(net, x[:17]))
    assert np.array_equal(a, predict(copy.deepcopy(net), x[:17]))
    one = np.concatenate([predict(net, x[i:i + 1]) for i in range(17)])
    assert np.max(np.abs(a - one)) <= 1e-6
    assert np.array_equal(a.argmax(1), one.argmax(1))
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())
    twin = copy.deepcopy(net)
    twin.arm = {'instant': 'lag6', 'lag6': 'instant', 'offset': 'aligned', 'aligned': 'offset'}[arm]
    difference = float((net.representation(x[:4]) - twin.representation(x[:4])).abs().max())
    assert difference > 1e-4, 'Relationship intervention is a silent no-op'
    return dict(reset_interleaving_deepcopy_exact=True, b1_max=float(np.max(np.abs(a-one))),
                changed_relation_max=difference, unchanged_inference_state=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--family', choices=ARMS, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    setup()
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic()
    with np.load(root / 'data/proteus_windows.npz') as z:
        x = torch.from_numpy(z['X'].astype('float32')).cuda()
        y = z['y'].copy()
        people = z['subject'].copy()
        assert x.shape == (2655, 41, 480) and np.isfinite(y).all()
        assert len(z['ch_names']) == len(set(z['ch_names'])) == 41
    roster = sorted(set(people))
    epochs = 2 if args.smoke else 40
    helds = [roster[0]] if args.smoke else roster
    proof = dict(data_sha256=sha(root/'data/proteus_windows.npz'), folds=[], epochs=epochs,
                 source={f.name: sha(f) for f in Path(__file__).parent.glob('*.py')},
                 device=torch.cuda.get_device_name(), torch=torch.__version__, stateless=True,
                 tf32=False, seed=355, labels_scored=False)
    for held in helds:
        train, query = np.flatnonzero(people != held), np.flatnonzero(people == held)
        if args.smoke:
            # All smoke computations use source rows; no query posterior or label is used.
            query = train[:24]
        assert held not in set(people[train]) and len(set(people[train])) == 9
        record = dict(held=held, source_people=sorted(set(people[train])),
                      rows_sha256=hashlib.sha256(train.tobytes()).hexdigest(), arms={})
        for arm in ARMS[args.family]:
            start = time.monotonic()
            net, losses = fit(x[train], torch.as_tensor(y[train], device='cuda'), arm, epochs)
            check = guards(net, x[train[:24]], arm)
            probability = predict(net, x[query])
            assert probability.shape == (len(query), 3) and np.isfinite(probability).all()
            torch.save(net.state_dict(), out / f'{held.split("/")[-1]}_{arm}.pt')
            name = f'{held.split("/")[-1]}_{arm}.npz'
            np.savez(out / name, rows=query, probability=probability)
            record['arms'][arm] = dict(seconds=time.monotonic()-start, losses=losses, guards=check,
                                       bank_sha256=sha(out/name), checkpoint_sha256=sha(out/f'{held.split("/")[-1]}_{arm}.pt'), parameters=sum(v.numel() for v in net.parameters()))
            print(json.dumps(dict(completed=held, arm=arm, seconds=record['arms'][arm]['seconds'])), flush=True)
        proof['folds'].append(record)
        (out/'progress.json').write_text(json.dumps(proof, indent=2))
    proof['seconds'] = time.monotonic()-t
    proof['maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    proof['cuda_peak_bytes'] = torch.cuda.max_memory_allocated()
    if args.smoke:
        # Per-family 10 folds, 40/2 epoch scaling, 1.5 safety factor plus setup.
        proof['projected_family_seconds'] = 10 * 20 * 1.5 * proof['seconds'] + 120
    (out/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
