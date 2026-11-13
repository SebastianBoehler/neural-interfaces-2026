"""Deliver every test session in another order through experiments/protocol.predict_streams (ledger s176): each
session's chunk of window indices is reordered by a rank (global window index -> delivery position), the model streams it
in that order, and the outputs are put back at their window indices. Works for every model with
predict_proba_stream / predict_stream (core1, SessionBN nets, the rest net).
Usage: import deliver; deliver.install(protocol, rank) ... deliver.uninstall(protocol); rank_from(perms, scheme, seed, proto)"""
import numpy as np
_ORIG = {}


def rank_from(perms, scheme, seed, proto="all_loso"):
    order = perms[f"{scheme}|{seed}|{proto}"]; rk = np.full(int(order.max()) + 10000, -1); rk[order] = np.arange(len(order)); return rk


def install(protocol, rank):
    _ORIG.setdefault("ps", protocol.predict_streams)

    def predict_streams(model, d, mask, proba=False):
        idx = np.flatnonzero(mask); out = np.empty((len(idx), 3) if proba else len(idx), dtype=np.float64 if proba else np.int64)
        fn = model.predict_proba_stream if proba else model.predict_stream; st = d["stream"][idx]; start = 0
        while start < len(idx):
            stop = start
            while stop < len(idx) and st[stop] == st[start]:
                stop += 1
            ix = idx[start:stop]; rk = rank[ix]; assert (rk >= 0).all() and len(set(rk)) == len(ix), "the delivery order does not cover this session"
            o = np.argsort(rk, kind="stable"); res = fn(ix[o]); back = np.empty_like(res); back[o] = res; out[start:stop] = back
            start = stop
        return out
    protocol.predict_streams = predict_streams


def uninstall(protocol):
    if "ps" in _ORIG:
        protocol.predict_streams = _ORIG["ps"]
