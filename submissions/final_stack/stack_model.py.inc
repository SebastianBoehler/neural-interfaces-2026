class StackEnsemble(HardWaveEnsemble):
    """cap250 with the outer raw slot (five outer FBCNet members) replaced by `hard` (F-C25 pair, when HARD_FILES is
    set) and/or CBraMod-C: proposal = (1 - CB_MIX) * (hard mean | mean of the five outer members) + CB_MIX * CBraMod,
    applied to each of the five outer members at its original weight (1/90), which is the dev chain's raw-slot rule
    `MASS * (proposal - other)` with MASS 5/6 (CBraMod alone: 1/6 of the slot)."""

    CB_FILES = __CB_FILES__
    CB_MIX = 0.2

    def __init__(self, z, dev, root):
        self.cb_blobs = tuple((Path(root) / name).read_bytes() for name, _ in self.CB_FILES)
        if any(hashlib.sha256(blob).hexdigest() != digest
               for blob, (_, digest) in zip(self.cb_blobs, self.CB_FILES, strict=True)):
            raise ValueError("the signed CBraMod member artifacts differ")
        super().__init__(z, dev, root)
        self.cb = CBMember(self.cb_blobs, dev) if self.cb_blobs else None
        if not self.cb_blobs and not self.HARD_FILES:
            raise ValueError("the stack package has no member")

    def reset_state(self):
        super().reset_state()
        if getattr(self, "cb", None) is not None:
            self.cb.reset()

    def mix6(self, mix, members, p_ft):
        if not members:
            return super().mix6(mix, members, p_ft)
        if self.hard_window is None:
            raise ValueError("stack mixer has no current native input")
        x, self.hard_window = self.hard_window, None
        original = np.stack([members[i] for i in self.hard_slots])
        proposal = self.hard_probability(x).mean(0) if self.HARD_FILES else original.mean(0)
        if self.cb is not None:
            proposal = (1 - self.CB_MIX) * proposal + self.CB_MIX * self.cb.prob(x)
        altered = dict(members)
        for i in self.hard_slots:
            altered[i] = proposal
        delta = (self.hard_weights[:, None] * (proposal - original)).sum(0)
        return Ensemble6.mix6(self, mix + delta, altered, p_ft)
