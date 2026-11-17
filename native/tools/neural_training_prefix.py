"""Capture six actual original SessionNet source updates for native parity."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook


class EndOfPrefix(Exception):
    pass


def capture(fixture, project):
    sys.path.insert(0, str(project / 'experiments'))
    from deep import build
    from session_bn import SessionNet

    x, labels = fixture['source_X'], fixture['source_y']
    if x.shape != (1898, 41, 480) or labels.shape != (1898,):
        raise ValueError('source fixture shape differs')
    seed = int(fixture['initialization_seed'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = np.arange(len(x))
    model = SessionNet({'X': x.numpy()}, arch='fbcnet', align=False,
                       seed=seed, pretrained=False)
    model.dev = 'cpu'
    model.net = build('fbcnet', 41, 480, pretrained=False).train()
    model._row_sess = fixture['source_session'].numpy()
    model._prepare()
    initial = {key.removeprefix('initial_'): value for key, value in fixture.items()
               if key.startswith('initial_')}
    initialized = model.net.state_dict()
    if set(initial) != set(initialized) or any(
        not torch.equal(initialized[key], value) for key, value in initial.items()
    ):
        raise ValueError('original SessionNet initializer differs')

    original_batches = model._batches

    def batches(batch_rows, random):
        plan = original_batches(batch_rows, random)
        if len(plan) != 31:
            raise ValueError('OneCycle epoch batch count differs')
        if random is not rng:
            return plan
        for i, batch in enumerate(plan):
            count = int(fixture['batch_sizes'][i])
            if not np.array_equal(batch, fixture['batch_indices'][i, :count].numpy()):
                raise ValueError(f'original training batch {i} differs')

        def prefix():
            yield from plan[:6]
            raise EndOfPrefix

        return prefix()

    model._batches = batches  # only stop the iterator, after the original sixth update
    criterion = torch.nn.CrossEntropyLoss()
    output = {'native_format': torch.tensor(1), 'prefix_steps': torch.tensor(6)}
    frame = {}
    completed = 0

    def snapshot(name):
        def hook(_module, _args, value):
            frame[name] = value.detach()

        return hook

    def after_step(optimizer, _args, _kwargs):
        nonlocal completed
        completed += 1
        if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
            raise ValueError('original AdamW consumer differs')
        group = optimizer.param_groups[0]
        schedule = torch.tensor([group['lr'], group['betas'][0]], dtype=torch.float64)
        if not torch.allclose(schedule, fixture['expected_schedule'][completed - 1],
                              rtol=0, atol=1e-15):
            raise ValueError('original OneCycle prefix differs')
        if completed not in (1, 2, 6):
            return
        step = dict(frame, schedule=schedule)
        step.update({'state_' + name: value for name, value in model.net.state_dict().items()})
        for name, parameter in model.net.named_parameters():
            if parameter.grad is None:
                continue
            step['gradient_' + name] = parameter.grad
            for key, value in optimizer.state[parameter].items():
                step[key + '_' + name] = value
        if completed == 1:
            for key, value in fixture.items():
                if key.startswith('first_') and not torch.equal(step[key[6:]], value):
                    raise ValueError('original first-update anchor differs: ' + key)
        if any(not torch.isfinite(value).all() for value in step.values()):
            raise ValueError('nonfinite original source prefix')
        output.update({f'{completed}_{key}': value.detach().clone()
                       for key, value in step.items()})

    hooks = [model.net.register_forward_hook(snapshot('logits')),
             criterion.register_forward_hook(snapshot('loss')),
             register_optimizer_step_post_hook(after_step)]
    try:
        model._train(x, labels, None, rows, 40, .001, rng, torch.Generator().manual_seed(seed),
                     criterion, False)
    except EndOfPrefix:
        pass
    finally:
        for hook in hooks:
            hook.remove()
    if completed != 6 or {k for k in output if k.endswith('_logits')} != {
        '1_logits', '2_logits', '6_logits'
    }:
        raise ValueError('original training prefix did not capture steps 1, 2 and 6')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'project', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    fixture = torch.load(args.fixture, mmap=True, map_location='cpu', weights_only=True)
    output = capture(fixture, args.project)
    torch.save(output, args.output)
    print('ORIGINAL_SOURCE_PREFIX_PASS steps=6 captured=1,2,6 '
          f'first_update_exact seconds={time.monotonic() - started:.3f}')


if __name__ == '__main__':
    main()
