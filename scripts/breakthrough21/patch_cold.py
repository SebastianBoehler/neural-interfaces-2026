"""Derive a source-prior cold-start version of the fused nine25 Team B Solver."""
import ast
from pathlib import Path
from functools import partial
import sys

import numpy as np

from patch_terminal import LEAD_SHA, once, verify_parent
from build_cb_speed import build

PARAM = "        W = whitener(self.cov, n).float()\n"
PRIOR_CALL = """        strength = self._lookup_prior(n)
        if strength:
            scale = self.cov.diagonal().sum() / n
            combined = self.cov + (strength * scale) * self.population_cov
            W = whitener(combined, n + strength).float()
        else:
            W = whitener(self.cov, n).float()
"""
ENCODER_CLASS = "    @torch.no_grad()\n    def posterior(self, window):\n"
ENCODER_METHOD = """    population_cov = None

    def _lookup_prior(self, n):
        if self.population_cov is None or n > 40:
            return 0.0
        return 12.0 * (41 - n) / 40

"""
TEAM_B_TEST = "            self._check_team_b(dev)\n"
TEAM_B_LOAD = '            self.team_b = _TEAM_B["Encoder"](pack, dev, self.TEAM_B_MEMBERS)\n'


def patch(source, prior):
    if (prior.shape != (41, 41) or not np.isfinite(prior).all() or
            not np.array_equal(prior, prior.T) or abs(np.trace(prior) - 1) > 1e-10 or
            np.linalg.eigvalsh(prior)[0] <= 0):
        raise ValueError("invalid population covariance")
    start = source.index("exec(compile(", source.index("_TEAM_B = {")) + len("exec(compile(")
    stop = source.index(", '<Team B encoder>', 'exec'), _TEAM_B)", start)
    encoder = ast.literal_eval(source[start:stop])
    encoder = once(encoder, ENCODER_CLASS, ENCODER_METHOD + ENCODER_CLASS)
    encoder = once(encoder, PARAM, PRIOR_CALL)
    compile(encoder, "<Team B encoder>", "exec")
    text = source[:start] + repr(encoder) + source[stop:]
    anchor = TEAM_B_TEST if TEAM_B_TEST in text else TEAM_B_LOAD
    insert = f"            self.team_b.population_cov = torch.tensor({prior.tolist()!r}, dtype=torch.float64, device=dev)\n"
    out = once(text, anchor, anchor + insert)
    compile(out, "submission.py", "exec")
    return out


def main():
    parent, prior_path, prior_sha, target = sys.argv[1:]
    p = Path(prior_path)
    verify_parent(p, prior_sha)
    prior = np.load(p, allow_pickle=False)
    build(Path(parent), Path(target), expected_sha=LEAD_SHA, patch_source=partial(patch, prior=prior))


if __name__ == "__main__":
    main()
