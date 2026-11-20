"""Make a recording-order B2 posterior bank for inner held people, using fresh source fits."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from assemble import base_arrays, load_arrays
from components import assets, part_path
from receipts import read_receipt
from route_inputs import b2_fits
from source import ROOT, add_scope, file_digest, row_digest, selection, source_rows

sys.path.insert(0, str(ROOT))
from lean.src66.ensemble import submit, with_prefix
from lean.src66.nets import NetBank
from lean.src66.stages import Tail
from lean.src66.tangent import TangentCore


def members(parts, people, first, data, channels, root):
    row_hash = row_digest(source_rows(people, first, data))
    data_hash = file_digest(data)
    z = base_arrays(channels)
    for family, seed in b2_fits():
        fit = part_path(parts, family, seed)
        if family == "geometry":
            read_receipt(fit, family, seed, row_hash, data_hash, base_only=True)
        else:
            read_receipt(fit, family, seed, row_hash, data_hash)
        z.update(load_arrays(fit / "weights.npz"))
        for name in assets(family, seed):
            if name.endswith(".pt"):
                shutil.copyfile(fit / name, root / name)
    # B2 predates the early boosts, routing, adaptation, Eye/Rest and CountFilter.
    z.update(em_nmin=np.float64(10), em_boost=np.float64(0), tsem_boost=np.float64(0))
    return z, row_hash, data_hash


def stream(z, root, waveforms, streams):
    if not torch.cuda.is_available():
        raise RuntimeError("nested B2 calibration requires a CUDA Slurm allocation")
    torch.set_num_threads(1)
    w = z["member_weights"] / np.sum(z["member_weights"])
    net_ids = np.flatnonzero(w[1:-1])
    net = NetBank([with_prefix(z, f"net{i}_") for i in net_ids],
                  np.ones(len(net_ids), dtype=bool), "cuda:0", root)
    core1 = TangentCore(with_prefix(z, "ts_"))
    core2 = TangentCore(with_prefix(z, "ts2_"))
    tail = Tail(z)
    view = z["tsem_V"].astype(np.float64)

    def tangent(raw):
        p1, features = core1.step(raw)
        return p1, (features - z["tsem_m"]) @ view, core2.step(raw)[0]

    out = np.empty((len(waveforms), 3))
    for session in dict.fromkeys(streams):
        for component in (net, core1, core2, tail):
            component.reset_state()
        for i in np.flatnonzero(streams == session):
            raw = np.nan_to_num(waveforms[i])
            geometry = submit(tangent, raw)
            posteriors = net(raw)
            p1, u, p2 = geometry.result()
            mixed = w[0] * p1
            for j, p in posteriors:
                mixed += w[1 + net_ids[j]] * p
            out[i] = tail.step(raw, u, mixed + w[-1] * p2)
    return out


def run(people, first, data, parts, query, query_first, out):
    if not os.environ.get("SLURM_JOB_ID") or query in people or out.exists():
        raise ValueError("calibration must run in Slurm with a fresh output and an excluded query person")
    with np.load(data, allow_pickle=False) as archive:
        rows, _, sessions = selection(archive, (query,), (query,) if query_first else ())
        channels = archive["ch_names"].tolist()
        x = archive["X"][rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        root = Path(tmp)
        z, row_hash, data_hash = members(parts, people, first, data, channels, root)
        p = stream(z, root, x, sessions)
        metadata = json.dumps(dict(query=query, query_rows=row_digest(rows), source_rows=row_hash,
                                   data_sha256=data_hash, job=os.environ["SLURM_JOB_ID"]))
        file = root / "bank.npz"
        np.savez(file, index=rows, proba=p, metadata=metadata)
        file.rename(out)
    print(f"PASS nested {query}: {len(rows)} posteriors written to {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_scope(parser)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-first", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.people, args.first_session_only, args.data, args.parts, args.query, args.query_first, args.out)
