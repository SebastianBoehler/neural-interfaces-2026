class TeamAFeatureGraph(TeamBFeatureGraph):
    """Capture only the frozen MANAS forward; prefix updates remain eager."""

    def run(self, forward, aligned, dev):
        if self.graph is None:
            self.graph, self.host, self.static, self.output = self.capture(forward, aligned, dev)
        self.host.copy_(aligned)
        self.static.copy_(self.host, non_blocking=True)
        self.graph.replay()
        self.replays += 1
        # PrefixAdapter keeps the feature for later arrivals.
        return self.output.clone()


class TerminalTeamAStack(TeamBlendStack):
    TEAM_A_MIX = 0.10
    TEAM_A_GRAPH = __TEAM_A_GRAPH__
    TEAM_A_SHA = __TEAM_A_SHA__
    TEAM_A_REFERENCE = __TEAM_A_REFERENCE__

    def __init__(self, z, dev, root):
        self.team_a = None
        super().__init__(z, dev, root)
        if self.TEAM_A_MIX:
            try:
                path = Path(root) / "team_a.pt"
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != self.TEAM_A_SHA:
                        raise ValueError("Team A weights differ from the signed bank")
                if self.cf is None or self.cp is None or not self.routed:
                    raise ValueError("Team A requires the routed count and CSP stages")
                self.team_a = _TEAM_A["Solver"]().load_model(dict(
                    submission_dir=root, device=dev, ch_names=CH_NAMES, n_chans=41, n_times=480, sfreq=120))
                self._check_team_a()
            except Exception as exc:
                raise RuntimeError(f"load-time self-test: Team A raised {exc!r}") from exc

    def reset_state(self):
        super().reset_state()
        self.team_a_p = None
        self.team_a_graph = None
        if self.team_a is not None:
            self.team_a.reset_state()
            if self.TEAM_A_GRAPH and self.team_a.device.type == "cuda":
                self.team_a_graph = TeamAFeatureGraph()

    def _team_a_probability(self, raw):
        label = self.team_a.predict(torch.from_numpy(raw[None]), self.team_a_graph)
        q = self.team_a.last.cpu().numpy()
        self._check_probability(q, "Team A")
        if label.shape != (1,) or label.dtype != torch.long or int(label.cpu()[0]) != q.argmax():
            raise ValueError("Team A label differs from its posterior")
        return q

    @staticmethod
    def _check_probability(q, name):
        if q.shape != (3,) or not np.isfinite(q).all() or (q < 0).any() or abs(float(q.sum()) - 1) > 1e-6:
            raise ValueError(f"invalid {name} posterior")

    def _check_team_a(self):
__TEAM_A_WINDOWS__
        eager = np.stack([self._team_a_probability(x) for x in _xs])
        atol = 0.02 if self.team_a.device.type == "cuda" and torch.cuda.is_bf16_supported() else 0.001
        if not np.allclose(eager, self.TEAM_A_REFERENCE, rtol=0, atol=atol):
            raise ValueError(f"Team A posterior {eager} differs from its CPU reference")
        self.reset_state()
        if self.team_a_graph is not None:
            graph = np.stack([self._team_a_probability(x) for x in _xs])
            if self.team_a_graph.replays != len(_xs) or not np.allclose(graph, eager, rtol=0, atol=2e-5):
                raise ValueError("Team A graph did not reproduce the eager posterior")
        self.reset_state()

    def step(self, x):
        self.team_a_p = None
        return super().step(x)

    def mix6(self, mix, members, p_ft):
        raw = self.hard_window
        base = super().mix6(mix, members, p_ft)
        if members and self.team_a is not None:
            # Evaluate in step62's existing exception guard. Apply the term after count and CSP.
            self.team_a_p = self._team_a_probability(raw)
        return base

    def terminal_scores(self, q, fused):
        if self.team_a_p is None or self.dead:
            return q
        if fused:
            q = np.exp(q - q.max())
        q = q / q.sum()
        self._check_probability(q, "CSP/count")
        return (1 - self.TEAM_A_MIX) * q + self.TEAM_A_MIX * self.team_a_p
