"""Routing v3 + gated early boost (GEB, eb2, 7 Nov): streaming reference = route_stream3eb.V3TailEB with the boost gate.
Each tail (shadow and routed) has its own SessionEM, which keeps the running mean m of its INPUT margin (top1 - top2) over
windows 0..t-1; gate g(t) = clip((eb_gate_hi - m) / (eb_gate_hi - eb_gate_lo), 0, 1) (1 at t = 0 or when disabled)
multiplies EB's h(t) in that SessionEM and in the SAME tail's TSEM (eb2/gm.py; the package wires TSEM.gate_src to its own
tail's SessionEM, and Route3.bind copies the pair together). Keys: route_stream3eb's plus eb_gate_lo, eb_gate_hi."""
import numpy as np
import route_stream3 as R3
import route_stream3eb as EB


def h_early(T, t):
    return min(max((2 * T - t) / T, 0.0), 1.0)


class SessEMg(EB.SessEMb):
    def __init__(self, lo, hi, boost, T, *a):
        self.g_lo, self.g_hi = lo, hi
        super().__init__(boost, T, *a)

    def reset_state(self):
        super().reset_state()
        self.msum, self.gates = 0.0, []

    def _gate(self, t):
        if self.g_hi <= self.g_lo or t == 0:
            return 1.0
        return min(max((self.g_hi - self.msum / t) / (self.g_hi - self.g_lo), 0.0), 1.0)

    def step(self, f, p):
        self.gates.append(self._gate(len(self.A)))
        out = super().step(f, p)
        s = np.sort(p)
        self.msum += s[-1] - s[-2]
        return out

    def _adapt(self, A, B, f, p, t):  # = EB.SessEMb._adapt with the gated boost
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (f - m) / sd
        Q = B
        for _ in range(self.iters):
            mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
            C = np.cov((Z - Q @ mu).T)
            C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C)
            Q = B * R3.lik(Z, mu, Ci) ** self.lam
            Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0) if self.n0 else self.lam
        e *= 1 + self.boost * (h_early(self.T, t) * self.gates[t])
        q = p * R3.lik(z[None], mu, Ci)[0] ** e
        return q / q.sum()


class TSEMsg(EB.TSEMsb):
    gate_src = None  # this tail's SessEMg

    def _adapt(self, A, conf, lab, u, q, t):  # = EB.TSEMsb._adapt with the gate of the same tail's SessionEM
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (u - m) / sd
        k = max(2, int(self.rho * t / 3))
        sel = [i[np.argsort(-conf[i])[:k]] for i in (np.flatnonzero(lab == c) for c in range(3))]
        if min(len(i) for i in sel) < 2:
            return q
        mu = np.stack([Z[i].sum(0) / (len(i) + self.m0) for i in sel])
        C = np.cov(np.concatenate([Z[i] - mu[c] for c, i in enumerate(sel)]).T)
        C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
        e = self.lam * t / (t + self.n0) if self.n0 else self.lam
        g = self.gate_src.gates[t] if self.gate_src is not None and t < len(self.gate_src.gates) else 1.0
        e *= 1 + self.boost * (h_early(self.T, t) * g)
        qq = q * R3.lik(z[None], mu, np.linalg.inv(C))[0] ** e
        return qq / qq.sum()


class ChainG(R3.Chain):
    def __init__(self, p):
        g = lambda k: float(p[k])
        T, lo, hi = float(p.get("eb_T", 30.0)), float(p.get("eb_gate_lo", 0.0)), float(p.get("eb_gate_hi", 0.0))
        self.rb = R3.Rebal(g("rb_alpha"), g("rb_k"), int(p["rb_w"]))
        self.em = SessEMg(lo, hi, float(p.get("em_boost", 0.0)), T, g("em_lam"), g("em_n0"), g("em_m0"),
                          g("em_shrink"), int(p["em_iters"]), int(p["em_nmin"]))
        self.ts = TSEMsg(float(p.get("tsem_boost", 0.0)), T, g("tsem_lam"), g("tsem_rho"), g("tsem_shrink"),
                         g("tsem_m0"), g("tsem_n0"), int(p["tsem_nmin"]))
        self.ts.gate_src = self.em


class V3TailGEB(R3.V3Tail):
    def __init__(self, p):
        super().__init__(p)
        self.shadow, self.routed = ChainG(p), ChainG(p)
        self.reset_state()
