"""s288 unit tests (python studies/15nov/tsm288/test_spd.py): SPD outputs, identity at the recentred mean, causality, k propagation,
BatchNorm swap, train chunk = one session."""
import sys
from pathlib import Path
WS = Path(__file__).resolve().parents[3]; sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts"), str(Path(__file__).parent)]
import torch
import spd
from spd import D64, SPDSessionBN, TSMNet, logm, sym_fn

torch.manual_seed(0)


def rand_spd(n, d, scale=1.0):
    A = torch.randn(n, d, d, dtype=D64) * scale; return A @ A.transpose(1, 2) + 0.1 * torch.eye(d, dtype=D64)


def test_recentre_identity():
    bn = SPDSessionBN(8); bn.m = 32; bn.train()
    Dg = torch.diag_embed(torch.rand(32, 8, dtype=D64) + 0.2)  # commuting matrices: the log-Euclidean mean of the output is exactly the identity
    assert torch.allclose(logm(bn(Dg)).mean(0), torch.zeros(8, 8, dtype=D64), atol=1e-8)
    S = rand_spd(32, 8); R = bn(S)  # general matrices: the congruence by M^-1/2 shrinks the mean log a lot (log-Euclidean, not affine-invariant, mean)
    assert logm(R).mean(0).norm() < 0.3 * logm(S).mean(0).norm(), (logm(R).mean(0).norm(), logm(S).mean(0).norm())
    # a session whose windows are all the same matrix M maps to the identity ("identity at the recentred mean")
    M = rand_spd(1, 8)[0]; R = bn(M.expand(32, 8, 8).clone())
    assert torch.allclose(R, torch.eye(8, dtype=D64).expand(32, 8, 8), atol=1e-8)
    assert bn.steps == 3 and float(bn.run_log.abs().max()) > 0  # the prior was updated


def test_output_spd_and_symmetric():
    net = TSMNet(n_chans=6, n_times=64, d_out=4); net.eval()
    x = torch.randn(10, 6, 64); S = net.bimap(net.spdbn(net.spd_features(x)))
    assert torch.allclose(S, S.transpose(1, 2), atol=1e-10); assert torch.linalg.eigvalsh(S).min() > 0
    out = net(x); assert out.shape == (10, 3) and torch.isfinite(out).all()


def test_causal_and_prior():
    bn = SPDSessionBN(8); bn.k = 5.0; bn.eval(); S = rand_spd(20, 8)
    a = bn(S)[:10]; b = bn(torch.cat([S[:10], rand_spd(10, 8, 3.0)]))[:10]
    assert torch.allclose(a, b, atol=1e-9), "window t depends on later windows"
    bn.k = 1e9; assert torch.allclose(bn(S), S, atol=1e-6), "huge prior (identity mean) must leave S unchanged"
    bn.k = 0.0; R = bn(S); assert not torch.allclose(R, S), "k=0 must recentre"


def test_net_integration():
    import session_bn as SB
    d = {"X": None}
    net = spd.TSMSessionNet.__new__(spd.TSMSessionNet); torch.nn.Module.__init__(torch.nn.Module())
    net.net = TSMNet(n_chans=6, n_times=64, d_out=4); net.group = 8; net.causal = False; net.k_train = 20.0; net.dev = "cpu"
    net._prepare(); assert net.n_sbn == 1 and len(net.sbns()) == 1 and len(net.spdbns()) == 1
    net.set_k(7.0); assert net.sbns()[0].k == 7.0 and net.spdbns()[0].k == 7.0 and net.spdbns()[0].m == 8
    net.net.train(); x = torch.randn(16, 6, 64); net.net(x).sum().backward()  # two chunks of 8
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.net.parameters())
    try:
        net.net(torch.randn(12, 6, 64)); raise SystemExit("a batch that is not whole chunks must fail")
    except AssertionError:
        pass


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
    print("ALL OK")
