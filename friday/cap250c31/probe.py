"""Read the alternate current window and restore the same authentic hard actor state."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from locked import (
    ACTOR_LOCK, ACTORS, F32_SMOKE, PARENT, PARENT_SEAL, TRIAL, input_waves,
    parent, parent_seal, pin, scope, select, sha, stamp, write,
)

HISTORY = {"hist_n", "hist_s1", "hist_s2", "hist_mass", "hist_w1", "hist_w2", "hist_clipped"}
SMOKE_KEY = "Scherer2015/G|1|0"
ORACLE = Path("/pupil_oracle.npz")


def history(actor):
    out = {name.rsplit(".", 1)[-1]: value for name, value in actor.named_buffers() if ".hist_" in name}
    assert set(out) == HISTORY
    return out


def frozen(actor):
    return {name: value.clone() for name, value in actor.state_dict().items() if name.rsplit(".", 1)[-1] not in HISTORY}


def exact_state(a, b):
    assert set(a) == set(b) and all(torch.equal(a[name], b[name]) for name in a)


def actors():
    trial = PARENT / "v66_cap250_hard_trial.zip"
    pin(trial, TRIAL)
    lock_file = Path("/actors.json")
    pin(lock_file, ACTOR_LOCK)
    lock = json.loads(lock_file.read_text())
    assert lock["f32e_source_only"] and lock["device"] == "cpu" and tuple(lock["actors"]["hard"]) == ACTORS
    result = []
    with ZipFile(trial) as archive:
        for seed, pin_digest in enumerate(ACTORS):
            source = archive.read(f"hard_s1c_s{seed}.pt")
            assert hashlib.sha256(source).hexdigest() == pin_digest
            result.append(torch.jit.load(io.BytesIO(source), map_location="cpu").eval())
    return result


@torch.no_grad()
def session(active, control, nominal, local, seq):
    cpu = torch.from_numpy(nominal)
    alt = torch.from_numpy(local)
    state = [history(a) for a in active]
    independent = [history(a) for a in control]
    protected = [frozen(a) for a in active]
    output = np.empty((len(seq), 2, 2, 3), dtype=np.float64)
    with torch.jit.optimized_execution(False):
        for actor, bank in zip(active + control, state + independent, strict=True):
            assert actor.reset().item() == 0 and not any(torch.count_nonzero(v).item() for v in bank.values())
        for age, row in enumerate(seq):
            x, z = cpu[row:row + 1].contiguous(), alt[row:row + 1].contiguous()
            for seed, (a, b, act, ref) in enumerate(zip(active, control, state, independent, strict=True)):
                assert act["hist_n"].item() == ref["hist_n"].item() == age
                saved = {name: value.clone() for name, value in act.items()}
                try:
                    alternate = a(z)
                    assert act["hist_n"].item() == age + 1
                finally:
                    for name, value in saved.items():
                        act[name].copy_(value)
                exact_state(act, saved)
                observed, oracle = a(x), b(x)
                exact_state(act, ref)
                assert act["hist_n"].item() == age + 1
                assert torch.equal(observed, oracle)
                assert observed.shape == alternate.shape == (1, 3) and observed.dtype == alternate.dtype == torch.float64
                output[age, 0, seed] = observed.numpy()[0]
                output[age, 1, seed] = alternate.numpy()[0]
        for a, original in zip(active, protected, strict=True):
            exact_state(frozen(a), original)
    assert np.isfinite(output).all() and (output >= 0).all()
    np.testing.assert_allclose(output.sum(-1), 1., rtol=0, atol=1e-12)
    assert np.max(np.abs(output[:, 0] - output[:, 1])) > 1e-3
    return output


def source():
    subject, day, interface, names, onset, plans = scope()
    nominal, local = input_waves(subject, day, interface, names, onset)
    return sorted(set(subject)), plans, nominal, local


def run(mode, index):
    assert torch.__version__ == "2.14.1+cu130" and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    population, plans, nominal, local = source()
    seal = parent_seal()
    active = actors()
    control = [copy.deepcopy(a) for a in active]
    if mode == "smoke":
        assert population[index] == "Scherer2015/G"
        rows = plans[0]["Scherer2015/G|1"]
        assert len(rows) == 120
        result = session(active, control, nominal, local, rows)
        pin(ORACLE, F32_SMOKE)
        with np.load(ORACLE, allow_pickle=False) as bank:
            np.testing.assert_array_equal(rows, bank[SMOKE_KEY + "|rows"])
            assert stamp(nominal[rows]) == str(bank[SMOKE_KEY + "|x"])
            np.testing.assert_array_equal(result[:, 0], bank[SMOKE_KEY + "|p"][:, 0])
        session(active, control, nominal, local, plans[0]["Scherer2015/G|2"][:17])
        np.testing.assert_array_equal(result[:17], session(active, control, nominal, local, rows[:17]))
        print("PASS genuine current-only reconstruction probe", write("smoke.json", dict(
            passed=True, source=sha("source.sha256"), parents=PARENT_SEAL, nominal_exact=True, state_exact=True,
            reset_prefix_exact=True, rows=len(rows), probability_peak=float(np.max(np.abs(result[:, 0] - result[:, 1]))),
            held_labels_opened=False)), flush=True)
        return
    s = json.loads(Path("smoke.json").read_text())
    assert s["passed"] and s["parents"] == PARENT_SEAL and not s["held_labels_opened"]
    assert s["source"] == sha("source.sha256") and s["state_exact"] and s["nominal_exact"] and s["reset_prefix_exact"]
    who = population[index]
    streams = parent(index, who, plans, seal)
    banks = {}
    for key, (rows, classes, waveform) in streams.items():
        assert waveform == stamp(nominal[rows])
        p = session(active, control, nominal, local, rows)
        np.testing.assert_array_equal(p[:17], session(active, control, nominal, local, rows[:17]))
        lq = np.log(np.maximum(p.mean(2), 1e-12))
        gap = lq[:, 1] - lq[:, 0]
        choices, present = select(gap, classes)
        banks[key + "|rows"] = rows
        banks[key + "|actors"] = p
        banks[key + "|labels"] = choices
        banks[key + "|eligible"] = present
    path = Path(f"bank_{index}.npz")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **banks)
    path.chmod(0o444)
    receipt = dict(passed=True, held_labels_opened=False, bank=sha(path), source=s["source"], parents=PARENT_SEAL,
                   index=index, who=who, keys=sorted(streams), smoke=sha("smoke.json"), state_exact=True)
    print("PASS current-only caps", who, write(f"bank_{index}.json", receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "produce"))
    parser.add_argument("index", type=int, choices=range(9))
    args = parser.parse_args()
    run(args.mode, args.index)
