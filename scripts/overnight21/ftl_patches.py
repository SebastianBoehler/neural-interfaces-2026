"""Text patches for the hardcb fold-package submission.py copies of lane FT-labels (never the files under submissions/).
GUARD: HardWaveEnsemble.__init__ refuses v6_ft_labels=1 ("expected the original cap250 raw FBCNet layout"); drop only that clause.
TEACH: the label chain of the unrecognised path (tail6r) reads the hardcb mix (hard pair and CBraMod proposal applied, no
fine-tuning, rest and eye view as in the v6 chain) instead of the stock raw mix. The stateful hard actors and the CBraMod
covariance accumulator run once per window: the proposal is cached for the window and reused by the final mix6.
Each (old, new) pair must occur exactly once."""

GUARD = [(
    "if (self.routed != 1 or self.v6r or not self.r15 or self.ft.cap != 250",
    "if (self.routed != 1 or not self.r15 or self.ft.cap != 250  # lane FT-labels: v6r allowed",
)]
TEACH = [
    (
        "                label = self.tail6r.step(x, u[:n_tsem], self.mix6(mix, {}, {}), extra)\n            else:\n                label = out_v5\n            p_ft = self.ft.step(dict(nets)",
        "                label = self.tail6r.step(x, u[:n_tsem], self.label_mix6(mix, dict(nets)), extra)\n            else:\n                label = out_v5\n            p_ft = self.ft.step(dict(nets)",
    ),
    (
        "    def step(self, x):\n        return self.step62(x) if self.routed else self.step61(x)\n",
        "    def label_mix6(self, mix, members):\n        return self.mix6(mix, {}, {})\n\n"
        "    def step(self, x):\n        return self.step62(x) if self.routed else self.step61(x)\n",
    ),
    (
        "        x, self.hard_window = self.hard_window, None\n"
        "        original = np.stack([members[i] for i in self.hard_slots])\n"
        "        proposal = self.hard_probability(x).mean(0) if self.HARD_FILES else original.mean(0)\n"
        "        if self.cb is not None:\n"
        "            proposal = (1 - self.CB_MIX) * proposal + self.CB_MIX * self.cb.prob(x)\n"
        "        altered = dict(members)\n",
        "        x, self.hard_window = self.hard_window, None\n"
        "        original = np.stack([members[i] for i in self.hard_slots])\n"
        "        proposal = self.proposal_for(x, original)\n"
        "        altered = dict(members)\n",
    ),
    (
        "    def mix6(self, mix, members, p_ft):\n        if not members:\n            return super().mix6(mix, members, p_ft)\n"
        "        if self.hard_window is None:\n            raise ValueError(\"stack mixer has no current native input\")\n",
        "    def proposal_for(self, x, original, keep=False):\n"
        "        cached = getattr(self, \"_teach\", None)\n"
        "        if cached is not None and cached[0] is x:\n"
        "            self._teach = None\n"
        "            return cached[1]\n"
        "        proposal = self.hard_probability(x).mean(0) if self.HARD_FILES else original.mean(0)\n"
        "        if self.cb is not None:\n"
        "            proposal = (1 - self.CB_MIX) * proposal + self.CB_MIX * self.cb.prob(x)\n"
        "        if keep:\n"
        "            self._teach = (x, proposal)\n"
        "        return proposal\n\n"
        "    def label_mix6(self, mix, members):  # lane FT-labels teacher: the hardcb mix before fine-tuning\n"
        "        if self.hard_window is None:\n"
        "            raise ValueError(\"stack teacher has no current native input\")\n"
        "        x = self.hard_window  # kept: the final mix6 of this window consumes it\n"
        "        original = np.stack([members[i] for i in self.hard_slots])\n"
        "        proposal = self.proposal_for(x, original, keep=True)\n"
        "        altered = dict(members)\n"
        "        for i in self.hard_slots:\n"
        "            altered[i] = proposal\n"
        "        delta = (self.hard_weights[:, None] * (proposal - original)).sum(0)\n"
        "        return Ensemble6.mix6(self, mix + delta, altered, {})\n\n"
        "    def mix6(self, mix, members, p_ft):\n        if not members:\n            return super().mix6(mix, members, p_ft)\n"
        "        if self.hard_window is None:\n            raise ValueError(\"stack mixer has no current native input\")\n",
    ),
]


def apply(text, patches):
    for old, new in patches:
        assert text.count(old) == 1, old[:80]
        text = text.replace(old, new)
    compile(text, "submission.py", "exec")
    return text
