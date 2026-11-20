

# ---- ttt_base extras (appended to ref/submission.py by cleanslate/ttt_base/train.py; same namespace, C_/T_/K_ defined)
class BD(nn.Module):
    """A braindecode model on aligned (B, 41, 480) windows. `.fc` = its output layer (TTT params='bn+fc' subset)."""

    def __init__(s, name, **kw):
        super().__init__()
        import inspect
        import braindecode.models as M
        cls = getattr(M, name)
        if "sfreq" in inspect.signature(cls).parameters:
            kw = {**kw, "sfreq": 120.0}
        s.net = cls(n_chans=C_, n_outputs=K_, n_times=T_, **kw)

    def forward(s, x):
        return s.net(x)

    @property
    def fc(s):
        return s.net.final_layer


ARCHS["bd"] = BD
