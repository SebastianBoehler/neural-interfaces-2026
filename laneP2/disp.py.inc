class DispCorrection:
    """Lane P2: bounded Gaussian-density correction of the routed raw posterior from the dispersion of one raw FBCNet member's four
    segment logit contributions (laneP/chain/segcorr.py, streaming form). Causal: the densities of window t use windows < t of the SAME
    session (reset_state), fitted on pseudo-labels = argmax of the uncorrected routed output of each window."""

    W, N0, MIN_EFF, CLIP, PRIOR = 0.10, 20.0, 8, 5.0, 20.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt, self.sm, self.sq = np.zeros(3), np.zeros((3, 6)), np.zeros((3, 6))

    def step(self, p, seg):
        z = seg.sum(0)
        x = np.concatenate([z - z.mean(), seg.std(0)])
        out, active = p, False
        if (self.cnt >= self.MIN_EFF).all():
            tot = self.cnt.sum()
            mu0 = self.sm.sum(0) / tot
            mu = (self.sm + self.PRIOR * mu0) / (self.cnt[:, None] + self.PRIOR)
            within = (self.sq - 2 * mu * self.sm + self.cnt[:, None] * mu ** 2).sum(0)
            var0 = self.sq.sum(0) / tot - mu0 ** 2
            v = np.maximum((within + self.N0 * var0) / (tot + self.N0), 1e-6)
            ll = -0.5 * (((x - mu) ** 2) / v).sum(1)
            ll = np.clip(ll - ll.mean(), -self.CLIP, self.CLIP)
            q = p * np.exp(self.W * ll)
            out, active = q / q.sum(), True
        k = int(np.argmax(p))
        self.cnt[k] += 1; self.sm[k] += x; self.sq[k] += x ** 2
        return out, active


class DispEnsemble(StackEnsemble):
    """StackEnsemble (hard pair + CBraMod raw slot) with the dispersion correction on the routed output, and optionally (FBL2_ON) the
    doubled live FBLight member (friday/cap250_fbl2_package, composed on top of the stack mixer). DISP_ON False = the stack bitwise."""

    DISP_ON = True
    FBL2_ON = False
    SEG_MEMBER = 47
    SEG_FILE = "sbn_fbcnet_raw_k20_s0.pt"

    def __init__(self, z, dev, root):
        super().__init__(z, dev, root)
        if self.FBL2_ON:
            members = tuple(self.ft.slots["fbl"])
            weights = self.w[1 + np.asarray(members)]
            mass = float(weights.sum())
            post_rest = (1 - self.w_rest) * mass
            if (members != tuple(range(59, 65)) or self.ft.member["fbl"] != members[0] or self.w_rest != 1 / 16
                    or abs(post_rest - 1 / 16) > 1e-16 or abs(self.routeB.a - 1 / 16) > 1e-16
                    or not np.allclose(weights, 1 / 90, atol=1e-16, rtol=0)):
                raise ValueError("FBLight doubling requires the frozen cap250 layout")
            self.fbl_members, self.fbl_weights, self.fbl_mass, self.fbl_post_rest = members, weights, mass, post_rest
            self.routeB.a = 1 / 17
        self.seg_net = None
        if self.DISP_ON:
            if self.ft.member["fbraw"] != self.SEG_MEMBER or str(z[f"net{self.SEG_MEMBER}_sjit_file"]) != self.SEG_FILE:
                raise ValueError("the dispersion member is not the cap250 raw FBCNet seed 0")
            found = [n for _, ms in self.nets.groups for i, n in ms if i == self.SEG_MEMBER]
            if len(found) != 1 or not isinstance(found[0], SharedFIRSJit) or found[0].fblight:
                raise ValueError("the dispersion member is not live as a shared-FIR SessionBN FBCNet")
            self.seg_net = found[0]
            self.seg_net.seg_on = True
        self.disp = DispCorrection()
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        log = getattr(self, "disp_log", None)
        if log and log["n"]:
            print(f"[disp] windows {log['n']} active {log['active']} changed {log['changed']} skipped {log['skipped']}", file=sys.stderr, flush=True)
        self.disp_log = dict(n=0, active=0, skipped=0, changed=0)
        if getattr(self, "disp", None) is not None:
            self.disp.reset()

    def mix6(self, mix, members, p_ft):
        native = super().mix6(mix, members, p_ft)
        if not self.FBL2_ON or not members:
            return native
        frozen = np.stack([members[i] for i in self.fbl_members])
        adapted = (self.fbl_weights[:, None] * frozen).sum(0) / self.fbl_mass
        if "fbl" in p_ft:
            shifted = np.clip(adapted + p_ft["fbl"] - frozen[0], 1e-6, None)
            adapted = shifted / shifted.sum()
        return (native + self.fbl_post_rest * adapted) / (1 + self.fbl_post_rest)

    def step(self, x):
        p = super().step(x)
        if self.seg_net is None:
            return p
        log, net = self.disp_log, self.seg_net
        log["n"] += 1
        seg, net.seg = net.seg, None  # one segment export per window: a window whose member forward did not run is skipped, never reused
        if seg is None or not np.isfinite(seg).all() or not np.isfinite(p).all():
            log["skipped"] += 1
            return p
        out, active = self.disp.step(p, seg)
        log["active"] += int(active)
        log["changed"] += int(out.argmax() != p.argmax())
        return out
