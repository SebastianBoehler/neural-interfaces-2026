"""PV2: offline personal models of a known person, fitted on that person's public session(s) only (what a package would
store as weights; nothing is fitted on labels during a session).
  ocular  LR (C 0.05) on SessionEM's 48 columns (GE48), per-session z-scored training rows; at run time each window is
          standardised causally (data.zcausal: windows < t of the session plus the stored session stats as 10
          pseudo-windows) -> 3 class posteriors
  em_pm   SessionEM template prior: class means of the session-z-scored routed-tail view on the public session
          (GE48 + eye-net log posteriors [+ cross-fitted personal ocular log posteriors])
  ts_pm   TSEM template prior: route4's own `pri` (class means of the session-standardised core1 PCs) is used as is
  strength s = clip((BA_personal - BA_eye) / 20, 0, 1), leave-one-run-out inside the public session(s) (statistics review)"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import data as D

C_LR, KZ = 0.05, 10


class Ocular:
    def __init__(self, rows, F=None):
        """F: (N, d) training features by global index (default GE48 as stored, recording-order event rates)."""
        F = D.GE if F is None else F
        self.rows = np.asarray(rows); g = [self.rows[D.SESS[self.rows] == s] for s in dict.fromkeys(D.SESS[self.rows])]
        self.prior = D.sess_stats(F, g); X = D.ztrain(F, g)[self.rows]
        self.lr = LogisticRegression(C=C_LR, max_iter=3000).fit(X, D.Y[self.rows])

    def stream(self, F):
        """F: one session's GE48 rows in delivery order -> (n, 3) posteriors, causal."""
        return self.lr.predict_proba(D.zcausal(F, self.prior, KZ))


def public_rows(q):
    """the person's public session(s) in the dev protocol: P04 and P10 -> session 1 (seen_session trains on it)."""
    return np.flatnonzero((D.PER == q) & (D.SES == "1"))


def loro(q):
    """leave-one-run-out inside the public rows, recording order: personal ocular posteriors (cross-fitted)."""
    rows = public_rows(q); key = np.char.add(D.ITF[rows], D.RUN[rows].astype(str)); out = np.zeros((len(rows), 3))
    for k in dict.fromkeys(key):
        te = key == k; m = Ocular(rows[~te]); out[te] = m.stream(D.GE[rows[te]])
    return rows, out


def strength(q):
    rows, P = loro(q); bp, be = D.ba(rows, P.argmax(1)), D.ba(rows, D.EYE[rows].argmax(1))
    return float(np.clip((bp - be) / 20, 0, 1)), bp, be


def em_prior(q, with_ocular):
    """class means (3, d) of the routed-tail SessionEM view on the public rows, each column z-scored per session."""
    rows, P = loro(q); V = [D.GE[rows], np.log(np.clip(D.EYE[rows], 1e-3, 1))]
    if with_ocular:
        V.append(np.log(np.clip(P, 1e-3, 1)))
    V = np.concatenate(V, 1); V = (V - V.mean(0)) / np.maximum(V.std(0), 1e-3)
    return np.stack([V[D.Y[rows] == k].mean(0) for k in range(3)])


# PV5 (coordinator): matched coordinates. Training and test columns come from one code path: the gaze columns
# (session_em.gaze_features, stateless) and the event rates of runs/geshuf.npz (session_em.EventRates), the public
# session streamed in the SAME delivery scheme and seed as the test session (key "<order>|all_loso").
COLS = {"G48": np.arange(48), "G32": np.arange(32)}


def matched_feats(order):
    """-> (N, 48) gaze + event rates by global index, every session streamed in `order` (rec: recording order)."""
    G = np.load(D.REPO + "/runs/geshuf.npz"); ev = G["rec"] if order == "rec" else G[f"{order}|all_loso"]
    assert ev.shape == (D.N, 16) and np.isfinite(ev).all(); return np.c_[D.GE[:, :32], ev]


def em_prior_masked(q, F, cols, d):
    """class means (3, d) of F[:, cols] on the public rows (z-scored per session) and the mask (d,) of pulled columns;
    columns outside `cols` (eye net, ...) get no prior."""
    rows = public_rows(q); V = F[rows][:, cols]; V = (V - V.mean(0)) / np.maximum(V.std(0), 1e-3)
    pm, mk = np.zeros((3, d)), np.zeros(d); pm[:, cols] = np.stack([V[D.Y[rows] == k].mean(0) for k in range(3)]); mk[cols] = 1
    return pm, mk
