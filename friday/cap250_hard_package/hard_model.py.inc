import hashlib


class HardWaveEnsemble(Ensemble6):
    """Replace five frozen outer raw members with the pinned CPU hard pair."""

    HARD_FILES = (
        ("hard_s1c_s0.pt", "ec33684850e902479f3bf28e472a9ea590e64fa759b0b03704daf7aff1ea4018"),
        ("hard_s1c_s1.pt", "ed83b6b5b2a2765f82afa4a2fcb3f2c526d24a32e8b789b527268bde67b5401e"),
    )

    def __init__(self, z, dev, root):
        self.hard_blobs = tuple((Path(root) / name).read_bytes() for name, _ in self.HARD_FILES)
        if any(hashlib.sha256(blob).hexdigest() != digest
               for blob, (_, digest) in zip(self.hard_blobs, self.HARD_FILES, strict=True)):
            raise ValueError("the signed hard waveform artifacts differ")
        super().__init__(z, dev, root)
        raw_ids = tuple(self.ft.slots["fbraw"])
        if (self.routed != 1 or self.v6r or not self.r15 or self.ft.cap != 250
                or raw_ids != tuple(range(47,53)) or self.ft.member["fbraw"] != 47):
            raise ValueError("expected the original cap250 raw FBCNet layout")
        self.hard_slots = raw_ids[1:]
        self.hard_weights = self.w[1 + np.asarray(self.hard_slots)]
        if not np.allclose(self.hard_weights, 1 / 90, rtol=0, atol=1e-16):
            raise ValueError("the original outer raw contribution has different weight")

    def reset_state(self):
        super().reset_state()
        self.hard_actors = None
        self.hard_window = None

    @torch.no_grad()
    def hard_probability(self, raw):
        if raw.shape != (41,480) or raw.dtype != np.float32:
            raise ValueError("the CPU actor needs the unchanged native raw waveform")
        x = torch.from_numpy(np.ascontiguousarray(raw[None]))
        with torch.jit.optimized_execution(False):
            if self.hard_actors is None:
                self.hard_actors = tuple(torch.jit.load(io.BytesIO(blob), map_location="cpu").eval()
                                         for blob in self.hard_blobs)
                if any(actor.reset().item() != 0 for actor in self.hard_actors):
                    raise ValueError("source hard actor could not reset")
            outputs = [actor(x) for actor in self.hard_actors]
        if any(p.shape != (1,3) or p.dtype != torch.float64 or p.device.type != "cpu" for p in outputs):
            raise ValueError("unexpected output from the pinned CPU actor")
        q = np.stack([p.numpy()[0] for p in outputs])
        if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(1),1.,rtol=0,atol=1e-12):
            raise ValueError("invalid hard waveform probabilities")
        return q

    def step(self, x):
        if self.hard_window is not None:
            raise ValueError("the owning raw frame was not reset")
        self.hard_window = x
        try:
            result = super().step(x)
            if self.hard_window is not None and not self.dead:  # dead session: v5.1b fallback, keep running
                raise ValueError("the actual outer mixer did not consume the current waveform")
            return result
        finally:
            self.hard_window = None

    def mix6(self, mix, members, p_ft):
        if not members:
            return super().mix6(mix, members, p_ft)
        if self.hard_window is None:
            raise ValueError("hard waveform mixer has no current native input")
        mean = self.hard_probability(self.hard_window).mean(0)
        self.hard_window = None
        original = np.stack([members[i] for i in self.hard_slots])
        altered = dict(members)
        for i in self.hard_slots:
            altered[i] = mean
        delta = (self.hard_weights[:,None] * (mean - original)).sum(0)
        return super().mix6(mix + delta, altered, p_ft)
