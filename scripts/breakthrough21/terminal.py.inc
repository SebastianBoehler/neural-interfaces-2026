class BreakthroughStack(TeamBlendStack):
    MIXES = __MIXES__
    SPECTRAL_SHA = __SPECTRAL_SHA__
    CB_FILES = __CB_FILES__
    RECORD = __RECORD__

    def __init__(self, z, dev, root):
        self.spectral = self.terminal_cb = None
        super().__init__(z, dev, root)
        if not self.routed or self.cf is None or self.cp is None:
            raise ValueError("terminal mixture requires the routed count and CSP stages")
        if self.SPECTRAL_SHA:
            path = Path(root) / "spectral.npz"
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != self.SPECTRAL_SHA:
                    raise ValueError("spectral bank SHA-256 differs")
            self.spectral = SpectralMember(path, CH_NAMES)
        if self.CB_FILES:
            blobs = []
            for name, sha in self.CB_FILES:
                blob = (Path(root) / name).read_bytes()
                if hashlib.sha256(blob).hexdigest() != sha:
                    raise ValueError(f"{name}: CBraMod bank SHA-256 differs")
                blobs.append(blob)
            self.terminal_cb = CBMember(blobs, dev)
        self._terminal_experts = None

    def reset_state(self):
        super().reset_state()
        self._terminal_experts = None
        if self.spectral is not None:
            self.spectral.reset()
        if self.terminal_cb is not None:
            self.terminal_cb.reset()

    @staticmethod
    def _verify_p(p):
        if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, rtol=0, atol=1e-6):
            raise ValueError("invalid terminal expert probability")
        return p

    def step(self, x):
        self._terminal_experts = None
        return super().step(x)

    def mix6(self, mix, members, p_ft):
        raw = self.hard_window
        result = super().mix6(mix, members, p_ft)
        if members and (self.spectral is not None or self.terminal_cb is not None):
            self._terminal_experts = (
                None if self.spectral is None else self._verify_p(self.spectral.prob(raw)),
                None if self.terminal_cb is None else self._verify_p(self.terminal_cb.prob(raw)),
            )
        return result

    def terminal_scores(self, q, fused):
        if self.RECORD:
            self.terminal_labels = {"lead": int(q.argmax())}
            self.terminal_native = q.copy()
        if self._terminal_experts is None or self.dead:
            return q
        if fused:
            q = np.exp(q - q.max())
        q = self._verify_p(q / q.sum())
        spectral, cb = self._terminal_experts
        for arm, weights in self.MIXES.items():
            s, c = weights
            result = (1 - (s + c)) * q
            if s:
                result += s * spectral
            if c:
                result += c * cb
            if self.RECORD:
                self.terminal_labels[arm] = int(result.argmax())
        return result
