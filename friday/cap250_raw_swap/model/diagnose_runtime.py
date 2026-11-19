"""Separate cross-device numeric drift from same-device streaming drift."""
import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from external_inputs import PEOPLE, VISITS, ancestry, cohort, deliveries
from held import frozen_member
from qualify_model import (HERE as MODEL, RUNTIME, authenticated, batch_oracle, ensure_reset, immutable_tensors,
                           origin_report, sha)
from source_targets import raw_logits


HERE = Path(__file__).resolve().parent
APPROVED = "2bcf07e5e2780b0e76e855a853cf1fbe5bdd8140356be868b746b25fb538d034"


def comparison(first, second):
    return dict(max_abs=float(np.abs(first - second).max()),
                decision_changes=int(np.count_nonzero(first.argmax(1) != second.argmax(1))))


@torch.no_grad()
def run(index):
    torch.set_num_threads(1)
    if torch.__version__ != RUNTIME or not torch.cuda.is_available():
        raise ValueError("expected the same CUDA worker runtime")
    manifest = MODEL / "artifact.sha256"
    if sha(manifest) != APPROVED:
        raise ValueError("different immutable raw actor bundle")
    expected = {file: digest for digest, file in (line.split() for line in manifest.read_text().splitlines())}
    origin_report()
    ancestry()
    x, subject, _, _, stream = cohort(waveforms=True)
    name = PEOPLE[index]
    rows = np.flatnonzero(subject == name)
    delivery = deliveries(stream, name)
    results = {}
    for seed in (0, 1):
        path = MODEL / f"raw_s1c_s{seed}.pt"
        if sha(path) != expected[path.name]:
            raise ValueError("raw artifact hash differs")
        ckpt, _, _, _ = authenticated(index, seed)
        meta, cpu = batch_oracle(seed, index, x, delivery, rows)
        ref, norm = frozen_member(ckpt, meta)
        ref.cuda()
        norm.forward = MethodType(lambda self, values: normalize(self, values, []), norm)
        module = torch.jit.load(str(path), map_location="cuda").eval()
        original = immutable_tensors(module)
        sessions = {}
        for (order, visit), ordered in delivery.items():
            frames = torch.from_numpy(x[ordered]).cuda()
            with torch.backends.cudnn.flags(allow_tf32=False):
                batch = torch.softmax(raw_logits(ref, frames / meta["scale"]).double(), dim=1).cpu().numpy()
            opposite = delivery[(order, VISITS[1 - VISITS.index(visit)])]
            cross = torch.from_numpy(x[opposite[:10]]).cuda()
            with torch.jit.optimized_execution(False):
                ensure_reset(module)
                online = torch.cat([module(w.unsqueeze(0)) for w in frames])
                ensure_reset(module)
                prefix = torch.cat([module(w.unsqueeze(0)) for w in frames[:31]])
                if not torch.equal(prefix, online[:31]):
                    raise ValueError("CUDA batch versus stream diagnostic changed its own prefix")
                ensure_reset(module)
                for point in cross:
                    module(point.unsqueeze(0))
                ensure_reset(module)
                if not torch.equal(prefix, torch.cat([module(w.unsqueeze(0)) for w in frames[:31]])):
                    raise ValueError("CUDA interleaving diagnostic did not reset exactly")
            stream_prob = online.cpu().numpy()
            cpu_prob = cpu[(order, visit)]
            if not all(np.isfinite(values).all() for values in (batch, stream_prob, cpu_prob)):
                raise ValueError("non-finite runtime comparison")
            sessions[f"{order}|{visit}"] = dict(rows=len(ordered), gpu_batch_cpu_batch=comparison(batch, cpu_prob),
                gpu_stream_cpu_batch=comparison(stream_prob, cpu_prob), gpu_stream_gpu_batch=comparison(stream_prob, batch))
        after = immutable_tensors(module)
        if set(original) != set(after) or any(not torch.equal(v, after[k]) for k, v in original.items()):
            raise ValueError("runtime diagnostic changed frozen parameters")
        results[str(seed)] = sessions
    report = dict(person=name, source=APPROVED, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  held_gold_opened=False, seeds=results)
    output = HERE / f"runtime-{index}.json"
    with output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    print(f"PASS raw runtime diagnosis {sha(output)}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("person", type=int, choices=range(len(PEOPLE)))
    run(cli.parse_args().person)
