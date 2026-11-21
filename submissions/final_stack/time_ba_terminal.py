"""Counterbalanced additive timing against the exact cap250 ZIP."""
from time import perf_counter
from unittest.mock import patch

import torch

import check_hard_team_b as C
from build_ba_terminal import PARENT, PARENT_SHA

CAP250 = 'v66_cp1_cap250.zip'
CAP250_SHA = '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'
SEALED_REF_S = 2554.9166
SEALED_WINDOWS = 20083


def timing(zips, scratch, candidate, folder, data, device, unpack, play):
    cap, cap_folder = unpack(zips, CAP250, CAP250_SHA, scratch)
    parent, parent_folder = unpack(zips, PARENT, PARENT_SHA, scratch)
    models, load_s = {}, {}
    inputs = dict(cap250=(cap, cap_folder), parent=(parent, parent_folder),
                  graph=(candidate, folder), eager=(candidate, folder))
    for name, (module, directory) in inputs.items():
        start = perf_counter()
        if name == 'eager':
            with patch.object(module.TerminalTeamAStack, 'TEAM_A_GRAPH', False):
                model = C.load(module, directory, data, device)
        else:
            model = C.load(module, directory, data, device)
        torch.cuda.synchronize()
        load_s[name] = perf_counter() - start
        models[name] = model
    timings = {}
    for k, (session, rows) in enumerate(list(C.plans(data, '71000').items())[::2]):
        x = torch.from_numpy(data['X'][rows]).to(device)
        arms = list(models) if k % 2 == 0 else list(models)[::-1]
        times = {}
        for name in arms:
            torch.cuda.synchronize()
            start = perf_counter()
            play(models[name], x, use_graph=(name != 'eager'))
            torch.cuda.synchronize()
            times[name] = perf_counter() - start
        print(session, len(rows), times, flush=True)
        if k:  # discard the first complete warm-up session in each arm
            timings[session] = dict(windows=len(rows), seconds=times)
    windows = sum(t['windows'] for t in timings.values())
    computed = {name: 1000 * sum(t['seconds'][name] for t in timings.values()) / windows for name in models}
    stats = {}
    for name, ms in computed.items():
        added = ms - computed['cap250']
        startup = max(0, load_s[name] - load_s['cap250'])
        stats[name] = dict(ms_per_window=ms, add_vs_cap250_ms=added, add_vs_parent_ms=ms - computed['parent'],
                           projected_s=SEALED_REF_S + SEALED_WINDOWS * added / 1000 + startup)
    return dict(windows=windows, sessions=timings, load_s=load_s, arms=stats)
