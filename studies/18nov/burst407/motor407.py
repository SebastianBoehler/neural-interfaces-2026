"""Run §401's digest-locked actor with only its posterior insertion exchanged."""
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from experiment407 import ARMS, BASE, parent

FIT = BASE/'ni26_burst407_v1_20261118'
ACTOR = parent.ACTOR/'full401.py'


def motor_update(raw, member):
    motor = .9 * raw[:, 0] + .1 * member[:, 0]
    pair = raw[:, 1:] / raw[:, 1:].sum(1, keepdims=True)
    new = np.column_stack((motor, pair * (raw.sum(1) - motor)[:, None]))
    np.testing.assert_allclose(new[:, 1:] / new[:, 1:].sum(1, keepdims=True), pair, rtol=0, atol=1e-12)
    np.testing.assert_allclose(new.sum(1), raw.sum(1), rtol=0, atol=1e-14)
    parent.require(np.max(abs(new[:, 0] - raw[:, 0])) > .01, 'inactive motor insertion')
    return new


def members():
    card = json.loads((FIT/'source.json').read_text())
    parent.require(card['members'] == parent.digest(FIT/'members.npz') and
                   card['source'] == parent.digest(FIT/'source.sha256') and
                   not card['query_labels_opened'], 'source ancestry')
    return card


def main(mode):
    parent.require(parent.digest(ACTOR) == '46f6576d7a165e1086810d6ed89cff051fe35f7a3292a55d4481a25dd65996bb', 'actor')
    code = ACTOR.read_text()
    start = code.index('                mass=raw[target,1:].sum(axis=1)')
    end = code.index('            filtered=package.gate', start)
    code = code[:start]+'                new[target]=motor_update(raw[target],model[a])\n'+code[end:]
    actor = ModuleType('full401')
    exec(compile(code, str(ACTOR), 'exec'), actor.__dict__)
    actor.ARMS, actor.SRC, actor.members, actor.motor_update = ARMS, FIT, members, motor_update
    sys.modules['full401'] = actor
    if mode == 'score':
        import score401
        score401.main()
    else:
        parent.require(mode in ('order','actor-smoke'), 'unknown mode')
        actor.main(mode == 'actor-smoke', int(sys.argv[2]))


if __name__ == '__main__':
    main(sys.argv[1])
