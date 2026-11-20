"""Source-scoped replacement for laneC/data.py::public (train half, held = nobody) and stage A staging for CBraMod-C."""

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ASSETS, ROOT, load_source  # noqa: E402



def stage_a_dir():
    return Path(os.environ.get("CAP250_CB_STAGEA", ASSETS / "runs/cb_stage_a")).resolve()


def checkpoint():
    return Path(os.environ.get("CAP250_CB_CHECKPOINT", "")).resolve()


def make_public(lane_data, people, first_session_only, data):
    """Return a function with laneC.data.public's return signature for the permitted source rows.

    Prior, rng(20261119), two aligned views and label order are those of public(held='all'): the training rows are
    every permitted row in the original row order; there is no query."""
    d = load_source(people, first_session_only, data)
    state = {}

    def public(root, held, full=False):
        if held != "all" or not full:
            raise ValueError("source-scoped CBraMod fits train on every permitted row (held = 'all')")
        x, stream, names = d["X"], d["stream"], d["ch_names"].tolist()
        n = len(x)
        if len(names) != 41 or len(set(d["person"])) != len(people):
            raise ValueError("unexpected source montage or roster")
        centered = x.astype(np.float64)
        centered -= centered.mean(-1, keepdims=True)
        prior = np.einsum("nct,ndt->cd", centered, centered, optimize=True) / (n * x.shape[-1])
        rng = np.random.default_rng(20261119)
        trained = [lane_data.align(x, stream, rng.permutation(n), prior) for _ in range(2)]
        state.update(rows=n, prior=prior)
        return trained, d["y"], np.arange(n), np.zeros(0, np.int64), {}, names, prior

    public.state = state
    return public
