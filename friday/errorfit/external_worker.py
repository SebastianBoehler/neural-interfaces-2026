"""Run cap250 and locked F5 corrections in the official one-window A100 worker."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from data import permutation
from external_inputs import ARMS, MODELS, ORDERS, digest, verify_models
from temporal import TemporalDecoder
from waveform import center


def load_actors(weights):
    result = {}
    with torch.random.fork_rng(devices=[]):
        for name in MODELS:
            kind, seed = name.split("_")
            ckpt = torch.load(weights / f"{name}.pt", weights_only=True, map_location="cpu")
            assert (ckpt["architecture"], ckpt["objective"], ckpt["seed"]) == ("chronological", kind, int(seed))
            model = TemporalDecoder("chronological")
            model.load_state_dict(ckpt["model"], strict=True)
            result[name] = model.cuda().eval()
    return result


def terminal(raw, age, label):
    assert raw.shape == (3,) and raw.argmax() == label
    if age < 90:
        assert (raw >= 0).all() and np.isclose(raw.sum(), 1, atol=1e-5)
        with np.errstate(divide="ignore"):
            return np.log(raw.astype(np.float64))
    assert (raw <= 0).all() and np.isfinite(raw).all()
    return raw.astype(np.float64)


@torch.no_grad()
def correct(model, eeg, state):
    z = model.encode(model.spectrum(eeg))
    blank = model.static(z)
    write = model.writes(z, blank)
    history, age = state if state is not None else (torch.zeros_like(write), 0)
    genuine = model.read(z, blank, history, blank.new_tensor([age]))
    new_state = history + write, age + 1
    outputs = torch.cat((genuine, blank), dim=0)
    difference = 0.20 * center(outputs).clamp(-3, 3)
    if state is None:
        assert torch.equal(genuine, blank)
        stock_api, check_state = model.step(eeg)
        torch.testing.assert_close(genuine, stock_api, rtol=0, atol=0)
        torch.testing.assert_close(new_state[0], check_state[0], rtol=0, atol=0)
    return difference.cpu().numpy(), new_state


def stream(template, actors, eeg, rows):
    model = fresh_model(template)
    step = model.cp.step
    stock_raw = []

    def capture(*args, **kwargs):
        raw = step(*args, **kwargs)
        stock_raw.append(np.array(raw, copy=True))
        return raw

    model.cp.step = capture
    outputs = {arm: [] for arm in ARMS}
    timers = {arm: [] for arm in ("stock", *MODELS)}
    states = {name: None for name in actors}
    activity = dict.fromkeys(actors, 0.)
    for age, row in enumerate(rows):
        start = perf_counter()
        window = eeg[int(row):int(row)+1].cuda()
        label = model.predict(window)
        assert label.device.type == "cpu" and label.dtype == torch.int64 and label.shape == (1,)
        assert len(stock_raw) == 1
        label = int(label[0])
        timers["stock"].append(perf_counter() - start)
        base = terminal(stock_raw.pop(), age, label)
        outputs["stock"].append(label)
        assert np.argmax(base + np.zeros(3)) == label
        # Rotate the measurement order to balance warm-cache effects across actors.
        order = MODELS[age % len(MODELS):] + MODELS[:age % len(MODELS)]
        for name in order:
            start = perf_counter()
            delta, states[name] = correct(actors[name], window, states[name])
            timers[name].append(perf_counter() - start)
            assert delta.shape == (2, 3) and np.isfinite(delta).all() and states[name][1] == age + 1
            actual, blank = delta
            activity[name] = max(activity[name], float(np.max(np.abs(actual - blank))))
            outputs[f"{name}_genuine"].append(int((base + actual).argmax()))
            outputs[f"{name}_blank"].append(int((base + blank).argmax()))
    counters = replay.activity(model)
    assert counters["recognized"] == -1 and counters["windows"] == len(rows)
    assert not any(counters[k] for k in ("errors", "ft_over"))
    assert not model.dead and not model.cp.dead and not model.cf.dead
    assert all(v > 1e-5 for v in activity.values())
    runtime = {name: dict(seconds=float(sum(values)), warm_p50_ms=float(np.median(values[1:])*1e3),
                          warm_p90_ms=float(np.quantile(values[1:], .9)*1e3))
               for name, values in timers.items()}
    return dict(rows=rows.tolist(), arms=outputs, counters=counters, max_history_effect=activity, runtime=runtime)


def main(person, smoke):
    assert torch.__version__ == "2.14.1+cu130" and "A100" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load("/external.npz", allow_pickle=False) as source:
        eeg = torch.from_numpy(source["X"])
        people, days = source["subject"], source["session"]
        channels = source["ch_names"].tolist()
    assert eeg.shape == (2130, 41, 480) and eeg.dtype == torch.float32 and len(channels) == 41
    who = sorted(set(people))[person]
    visits = sorted(set(days[people == who]))
    assert len(visits) == 2
    archive = Path("/prior") / f"cap_{person}.json"
    expected = json.loads(archive.read_text())
    weights = Path("/weights")
    models = verify_models(weights)
    template = replay.load(Path("/package"), channels)
    actors = load_actors(weights)
    streams = np.char.add(np.char.add(people, "|"), days)
    sessions = {}
    for order in ORDERS[:1] if smoke else ORDERS:
        indices = permutation(streams, None, "full", order)
        for day in visits[:1] if smoke else visits:
            key = f"{who}|{day}|{order}"
            rows = indices[(people == who) & (days == day)]
            control = expected[key]
            assert rows.tolist() == control["rows"], key
            run = stream(template, actors, eeg, rows)
            assert run["arms"]["stock"] == control["labels"], key
            assert run["counters"] == control["activity"], key
            sessions[key] = run
            if smoke:
                prefix = stream(template, actors, eeg, rows[:17])
                assert all(left == run["arms"][name][:17] for name, left in prefix["arms"].items()), key
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    result = dict(archive=digest(archive), source=digest("external_source.sha256"),
                  models=models, sessions=sessions, gpu=torch.cuda.get_device_name())
    suffix = "smoke" if smoke else str(person)
    path = Path(f"external_{suffix}.json")
    path.write_text(json.dumps(result) + "\n")
    print(f"{who}: {len(sessions)} paired sessions; exact stock, zero and adaptation counters OK", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:] == ["smoke"])
