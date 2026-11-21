"""Patch the signed lead's native covariance aligners for the enrolled reference test."""
import ast


def once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one anchor, found {count}: {old[:120]!r}")
    return source.replace(old, new)


CORE_START = """    def features(self, x):
        x = x - x.mean(0, keepdims=True)
        self.n += 1
        B ="""
CORE_END = """        self.log_sum += eig_apply_batch(C, log_floor)
        ref = (self.k * self.prior + self.log_sum) / (self.k + self.n)
"""
CORE_NEW = """        return C, eig_apply_batch(C, log_floor)

    def features(self, x):
        x = x - x.mean(0, keepdims=True)
        self.n += 1
        C, L = self.round3_cov(x)
        self.log_sum += L
        k, prior = (self.k, self.prior) if self.round3_ref is None else self.round3_ref
        ref = (k * prior + self.log_sum) / (k + self.n)
"""
INPUT_START = """    def __call__(self, x):
        X = x[None].astype(np.float64)
        if not self.align:
            return (X / self.scale).astype(np.float32)[0]
        xc ="""
INPUT_END = """        self.n += 1
        self.cov_sum = self.cov_sum + C[0]
        R = (self.k * self.G + self.cov_sum) / (self.k + self.n)
"""
INPUT_NEW = """        return C[0]

    def __call__(self, x):
        X = x[None].astype(np.float64)
        if not self.align:
            return (X / self.scale).astype(np.float32)[0]
        self.n += 1
        self.cov_sum = self.cov_sum + self.round3_cov(x)
        k, prior = (self.k, self.G) if self.round3_ref is None else self.round3_ref
        R = (k * prior + self.cov_sum) / (k + self.n)
"""
TEAM_START = """        xb = bandpass(x[None], self.mask)[0].double()
        c = xb @ xb.T / T
        self.cov = c if self.cov is None else self.cov + c
        W = whitener(self.cov, n).float()
"""
TEAM_NEW = """        c = self.round3_cov(x)
        self.cov = c if self.cov is None else self.cov + c
        cov, count = self.cov, n
        if self.round3_ref is not None:
            k, prior = self.round3_ref
            cov, count = cov + k * prior, count + k
        W = whitener(cov, count).float()
"""
TEAM_HELPER = """    def round3_cov(self, x):
        xb = bandpass(x[None], self.mask)[0].double()
        return xb @ xb.T / T

    def round3_enrol_cov(self, window):
        return self.round3_cov(detrend(window.to(self.device, torch.float32)))

"""

ACTIVATE = """    def round3_activate(self):
        if not self.round3_n0 or not self.recB or self.round3_started >= 0 or self.dead:
            return
        gates = (self.route.last["contrast"], self.route.last["identity"])
        j, _, ramp = max((g for g in gates if g is not None), key=lambda g: g[2])
        if ramp <= 0:
            raise ValueError("recognition did not provide an enrolled gallery index")
        bank = self.round3_bank[j]
        streams = [s for s, _ in self.nets.groups if s.align]
        if self.rest.stream.align:
            streams.append(self.rest.stream)
        for part, cov in zip((self.core1, self.core2, *streams), bank[:-1], strict=True):
            part.round3_ref = (self.round3_n0, cov)
        cov = torch.as_tensor(bank[-1], device=self.team_b.device, dtype=torch.float64)
        self.team_b.round3_ref = (self.round3_n0, cov)
        self.round3_started = self.log6["n"]
        self.round3_person = j

"""


def patch_encoder(source):
    candidates = [n.args[0].value for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "compile"
                  and isinstance(n.args[0], ast.Constant) and "class Encoder:" in str(n.args[0].value)]
    old, = candidates
    enc = once(old, TEAM_START, TEAM_NEW)
    reset = "        self.windows, self.cov, self.embeddings = [], None, None\n"
    enc = once(enc, reset, reset + "        self.round3_ref = None\n")
    anchor = "    def _embed_batch(self, x):\n"
    enc = once(enc, anchor, TEAM_HELPER + anchor)
    return once(source, repr(old), repr(enc))


def revised(source):
    for reset in ("        self.n, self.log_sum, self.slow_sum = 0, np.zeros_like(self.prior), 0.0\n",
                  "        self.n, self.cov_sum = 0, 0.0\n"):
        source = once(source, reset, reset + "        self.round3_ref = None\n")
    source = once(source, CORE_START, "    def round3_cov(self, x):\n        B =")
    source = once(source, CORE_END, CORE_NEW)
    source = once(source, INPUT_START, "    def round3_cov(self, x):\n        xc =")
    source = once(source, INPUT_END, INPUT_NEW)
    source = patch_encoder(source)
    class_start = "class HardWaveEnsemble(Ensemble6):"
    pre, cls = source.split(class_start)
    init = "    def __init__(self, z, dev, root):\n"
    cls = cls.replace(init, init + "        self.round3_names = tuple(map(str, z[\"route_names\"]))\n"
                      "        self.round3_n0, self.round3_bank = 0, {}\n", 1)
    reset = "        self.hard_window = None\n\n    @torch.no_grad()\n    def hard_probability"
    cls = once(cls, reset, "        self.hard_window = None\n"
               "        self.round3_started, self.round3_person = -1, -1\n\n"
               + ACTIVATE + "    @torch.no_grad()\n    def hard_probability")
    cls = once(cls, "            return result\n        finally:\n",
               "            self.round3_activate()\n            return result\n        finally:\n")
    result = pre + class_start + cls
    compile(result, "submission.py", "exec")
    return result
