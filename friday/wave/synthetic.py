"""Source-only synthetic EEG styles with a variance-matched diagonal control."""

import numpy as np
import torch


KINDS = ("raw_style_full", "raw_style_diagonal")


def diag(matrix):
    return matrix.diagonal(dim1=-2, dim2=-1)


def style_maps(eeg, sessions, weights, kind):
    identity = torch.eye(eeg.shape[1], dtype=torch.float64, device=eeg.device)
    covariances = []
    for rows in sessions:
        x = eeg[rows].double()
        x = x - x.mean(dim=-1, keepdim=True)
        covariance = (x @ x.transpose(-2, -1)) / x.shape[-1]
        mass = weights[rows].double()
        cov = (covariance * (mass / mass.sum())[:, None, None]).sum(dim=0)
        covariances.append(.75 * cov + .25 * cov.diag().median() * identity)
    covariance = torch.stack(covariances)
    eigenvalues, basis = torch.linalg.eigh(covariance)
    if torch.any(eigenvalues <= 0) or not torch.isfinite(eigenvalues).all():
        raise ValueError("source session style covariance is invalid")
    root = (basis * eigenvalues.sqrt().unsqueeze(-2)) @ basis.transpose(-2, -1)
    inverse = (basis * eigenvalues.rsqrt().unsqueeze(-2)) @ basis.transpose(-2, -1)
    full = .75 * identity + .25 * (root[None] @ inverse[:, None])
    output = full @ covariance[:, None] @ full.transpose(-2, -1)
    diagonal = torch.diag_embed((diag(output) / diag(covariance)[:, None]).sqrt())
    torch.testing.assert_close(diag(diagonal @ covariance[:, None] @ diagonal), diag(output),
                               rtol=1e-10, atol=1e-10)
    if not ((full - torch.diag_embed(diag(full))).abs().max() > .01):
        raise ValueError("source covariance intervention has inactive cross-channel mixing")
    return (full if kind == "raw_style_full" else diagonal).float()


class SourceStyle:
    def __init__(self, source_eeg, sessions, source_people, weights, kind, seed):
        self.maps = style_maps(source_eeg, sessions, weights, kind)
        owners = [str(source_people[rows[0]]) for rows in sessions]
        self.donors = [[j for j, p in enumerate(owners) if p != owner] for owner in owners]
        if any(not candidates for candidates in self.donors):
            raise ValueError("style donor needs a different source person")
        self.rng = np.random.default_rng(seed + 20000)
        self.applied = 0

    def __call__(self, eeg, session):
        donor = self.rng.choice(self.donors[session])
        mean = eeg.mean(dim=-1, keepdim=True)
        transformed = self.maps[session, donor] @ (eeg - mean) + mean
        if self.applied == 0 and (not transformed.isfinite().all()
                                  or (transformed - eeg).abs().max() < .001):
            raise ValueError("synthetic session style made no finite change")
        self.applied += 1
        return transformed
