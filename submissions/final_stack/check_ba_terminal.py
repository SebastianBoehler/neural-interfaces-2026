"""Check the exact MANAS blend through the official Solver on public PROTEUS."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import check_hard_team_b as C
from build_ba_terminal import PARENT, PARENT_SHA, digest
from team_b_speed_bench import check_features

CANDIDATE = 'final_stack_hard_safe_st_team_b_nine25_ba_terminal.zip'
CAP250 = 'v66_cp1_cap250.zip'
CAP250_SHA = '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'
CANDIDATE_MODES = ('graph', 'eager')


def clean(model):
    C.clean(model)
    a = model.team_a
    g = model.team_a_graph
    if (a.age or a.last is not None or a.alignment is not None or np.any(a.covariance_sum)
            or np.any(a.identity_sum) or a.adapter.features or a.person is not None or a.streak is not None
            or model.team_a_p is not None or (g is not None and (g.graph is not None or g.replays))):
        raise RuntimeError('Team A template carries session state')


def play(template, windows, *, capture=False, use_graph=None):
    model = C.fresh_model(template)
    if use_graph is False:
        model.team_a_graph = None
    is_candidate = hasattr(model, 'team_a')
    labels, taps = [], []
    original = model.terminal_scores if is_candidate else None
    end = {}
    move = 0.0

    def observe(q, fused):
        result = original(q, fused)
        end.update(scores=result.copy(), unblended=q.copy(), fused=fused)
        return result

    parent_cp = model.cp.step

    def observe_parent(*args):
        q = parent_cp(*args)
        end['scores'] = q.copy()
        return q

    instrument = (patch.object(model, 'terminal_scores', observe) if is_candidate else
                  patch.object(model.cp, 'step', observe_parent)) if capture else nullcontext()
    with instrument:
        for window in windows:
            end.clear()
            labels.append(C.label(model, window))
            if capture:
                scores = end['scores']
                if scores.argmax() != labels[-1]:
                    raise RuntimeError('Solver label bypassed the terminal scores')
                taps.append(np.r_[model.trace['v6']['mix6'], model.trace['out'], scores])
                if is_candidate and model.team_a_p is not None:
                    q = end['unblended']
                    if end['fused']:
                        q = np.exp(q - q.max())
                    q = q / q.sum()
                    np.testing.assert_array_equal(scores, 0.90 * q + 0.10 * model.team_a_p)
                    move = max(move, float(np.max(np.abs(scores - q))))
    if model.dead or model.log6['err'] or model.log6['v6'] != len(windows) or model.routeB.log['err'] or model.cp.nfail or model.cf.dead:
        raise RuntimeError(f'degraded Solver: {model.log6}')
    if getattr(model, 'team_b', None) is not None:
        if len(model.team_b.windows) != len(windows):
            raise RuntimeError('Team B missed a window')
        if windows.is_cuda:
            check_features(model.team_b, len(windows))
    if is_candidate and model.team_a is not None:
        team = model.team_a
        if team.age != len(windows) or len(team.adapter.features) != len(windows) or model.team_a_p is None:
            raise RuntimeError('Team A missed a window')
        if windows.is_cuda and use_graph is not False and template.TEAM_A_GRAPH:
            if model.team_a_graph.graph is None or model.team_a_graph.replays != len(windows):
                raise RuntimeError('Team A graph did not execute once per window')
    if capture and is_candidate and model.team_a is not None and move <= 1e-4:
        raise RuntimeError('Team A did not move the terminal scores')
    return np.asarray(labels), np.asarray(taps), dict(rec=model.log6['rec'], refits=model.ft.refits, fused=model.cp.fused)


def faults(module, directory, data, device):
    stack = module.TerminalTeamAStack
    pred = stack._team_a_probability
    corrupt = {
        'exception': patch.object(stack, '_team_a_probability', side_effect=RuntimeError('injected MANAS failure')),
        'wrong_posterior': patch.object(stack, '_team_a_probability', lambda self, raw: np.roll(pred(self, raw), 1)),
    }
    if device == 'cuda:0' and stack.TEAM_A_GRAPH:
        corrupt['graph_capture'] = patch.object(module.TeamAFeatureGraph, 'capture', side_effect=RuntimeError('injected capture failure'))
    messages = {}
    for name, fault in corrupt.items():
        with fault:
            try:
                C.load(module, directory, data, device)
            except RuntimeError as exc:
                if 'load-time self-test: Team A' not in str(exc):
                    raise
                messages[name] = str(exc)
            else:
                raise RuntimeError(f'load-time fault was ignored: {name}')
    return messages


def dead_guard(template, windows):
    model = C.fresh_model(template)
    for window in windows[:3]:
        C.label(model, window)
    with patch.object(model, '_team_a_probability', side_effect=RuntimeError('injected live Team A failure')) as call:
        labels = [C.label(model, window) for window in windows[3:]]
    if (not model.dead or model.log6['err'] != 1 or call.call_count != 1 or model.team_a.age != 3
            or model.hard_window is not None or model.team_a_p is not None or not model.trace['dead']):
        raise RuntimeError(f'dead-session guard failed: {model.log6}')
    return dict(windows=3 + len(labels), failures=call.call_count)


def zero(parent, module, folder, data, device, smoke):
    with patch.object(module.TerminalTeamAStack, 'TEAM_A_MIX', 0.0):
        disabled = C.load(module, folder, data, device)
    if disabled.team_a is not None:
        raise RuntimeError('Team A loaded in the zero arm')
    counts = {}
    for session, rows in C.D.plans(data, 71120, C.ZERO_STREAMS).items():
        windows = torch.from_numpy(data['X'][rows[:8] if smoke else rows]).to(device)
        parent_result = play(parent, windows, capture=True)
        zero_result = play(disabled, windows, capture=True)
        for x, y in zip(parent_result[:2], zero_result[:2], strict=True):
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                raise RuntimeError(f'{session}: blend zero differed bitwise')
        if parent_result[2] != zero_result[2]:
            raise RuntimeError(f'{session}: blend-zero counts differ')
        counts[session] = len(windows)
    if not smoke and sum(counts.values()) != 379:
        raise RuntimeError(f'wrong zero coverage: {counts}')
    return counts


def compare(template, data, device, order, part, smoke):
    report = {}
    sessions = list(C.plans(data, order).items())
    subset = sessions[:1] if smoke else sessions[part::2]
    for k, (session, rows) in enumerate(subset):
        x = torch.from_numpy(data['X'][rows[:36] if smoke else rows]).to(device)
        # Read the same published candidate as both arms; only its Team A graph differs.
        paths = CANDIDATE_MODES if k % 2 == 0 else CANDIDATE_MODES[::-1]
        actual = {path: play(template, x, capture=smoke, use_graph=(path == 'graph')) for path in paths}
        labels, scores, meta = actual['graph']
        eager, _, eager_meta = actual['eager']
        changes = np.flatnonzero(labels != eager).tolist()
        if changes or meta != eager_meta:
            raise RuntimeError(f'{order} {session}: eager/graph changed {changes} or stage work')
        if smoke:
            other = torch.from_numpy(data['X'][sessions[1][1][:8]]).to(device)
            play(template, other)
            repeat, prefix, _ = play(template, x[:30], capture=True)
            if labels[:30].tobytes() != repeat.tobytes() or scores[:30].tobytes() != prefix.tobytes():
                raise RuntimeError('Team A prefix or interleaved reset differs')
            np.testing.assert_allclose(scores[:, -3:], actual['eager'][1][:, -3:], rtol=0, atol=2e-5)
        report[session] = dict(windows=len(x), changed=len(changes), **meta)
        print(order, session, report[session], flush=True)
    clean(template)
    return report


def unpack(zips, name, sha, scratch):
    archive = zips / name
    if digest(archive) != sha:
        raise ValueError(f'archive SHA-256 differs: {name}')
    folder = scratch / name.removesuffix('.zip')
    module = C.unpack(archive, folder)
    return module, folder


def run(args):
    C.verify_parent(args.data, C.DATA_SHA)
    data = C.D.load(args.data)
    if data['X'].shape != (2655, 41, 480) or len(set(data['stream'])) != 14:
        raise ValueError('unexpected public test export')
    device = 'cpu' if args.stage == 'zero' else 'cuda:0'
    report = dict(stage=args.stage, sha=args.sha, torch=torch.__version__)
    if device == 'cuda:0':
        report['gpu'] = torch.cuda.get_device_name()
        expect = os.environ['NI26_EXPECT_GPU']
        if expect not in report['gpu']:
            raise RuntimeError(f'expected {expect}, got {report["gpu"]}')
    with TemporaryDirectory() as directory:
        root = Path(directory)
        module, folder = unpack(args.zips, CANDIDATE, args.sha, root)
        if args.stage in ('smoke', 'zero'):
            before, original = unpack(args.zips, PARENT, PARENT_SHA, root)
            parent = C.load(before, original, data, device)
            report['zero'] = zero(parent, module, folder, data, device, args.stage == 'smoke')
        if args.stage not in ('zero', 'timing'):
            template = C.load(module, folder, data, device)
            if template.TEAM_A_MIX != 0.10 or template.TEAM_B_MIX != 0.25 or len(template.team_b.nets) != 9:
                raise RuntimeError('wrong MANAS or Team B dose')
            clean(template)
        if args.stage == 'smoke':
            report['load_faults'] = faults(module, folder, data, device)
            rows = C.plans(data, '71000')[C.ZERO_STREAMS[0]][:8]
            report['dead_guard'] = dead_guard(template, torch.from_numpy(data['X'][rows]).to(device))
            report['orders'] = {o: compare(template, data, device, o, 0, True) for o in ('recording', '71000')}
            clean(template)
        if args.stage == 'terminal':
            report.update(order=args.order, sessions=compare(template, data, device, args.order, args.part, False))
        if args.stage == 'timing':
            from time_ba_terminal import timing

            report.update(timing(args.zips, root, module, folder, data, device, unpack, play))
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(args.stage, 'PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('zero', 'smoke', 'terminal', 'timing'))
    parser.add_argument('zips', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--order', choices=('recording', '71000'), default='71000')
    parser.add_argument('--part', type=int, choices=(0, 1), default=0)
    run(parser.parse_args())
