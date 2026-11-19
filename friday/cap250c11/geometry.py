"""Find supported alternatives for difficult earlier refit targets."""
import numpy as np


def spectra(covariances):
    cov = np.asarray(covariances,dtype=np.float64)
    assert cov.ndim == 4 and cov.shape[1:] == (8,41,41)
    variance = np.diagonal(cov,axis1=-2,axis2=-1)
    assert np.isfinite(variance).all() and (variance>0).all()
    correlation2 = cov**2 / (variance[...,None]*variance[...,None,:])
    rms = np.sqrt(np.maximum((correlation2.sum(axis=-1)-1)/40,0))
    features = np.concatenate((np.log(variance),rms),axis=-1)
    assert np.isfinite(features).all()
    return features


def nearest(view, hard, check=False):
    raw = view.reshape(len(view),-1)
    center = np.median(raw,axis=0)
    scale = np.maximum(1.4826*np.median(np.abs(raw-center),axis=0),.05)
    selected = np.clip((raw[hard]-center)/scale,-5,5)
    gram = selected @ selected.T
    norms = gram.diagonal()
    d2 = np.maximum(norms[:,None]+norms[None,:]-2*gram,0)
    if check:
        direct = np.sum((selected[0]-selected[1:])**2,axis=1)
        np.testing.assert_allclose(d2[0,1:],direct,rtol=1e-10,atol=1e-10)
    np.fill_diagonal(d2,np.inf)
    result = np.argsort(d2,axis=1,kind='stable')[:,:7]
    assert result.shape==(len(hard),7) and not (result==np.arange(len(hard))[:,None]).any()
    return hard[result]


def propose(features, labels, weights, check=False):
    assert len(features)==len(labels)==len(weights) and len(labels)>=20
    assert labels.shape==weights.shape and np.isin(labels,(0,1,2)).all()
    hard = np.sort(np.argsort(weights,kind='stable')[:len(labels)//2])
    prior = (np.bincount(labels[hard],minlength=3)+1)/(len(hard)+3)
    views = (features[:,:4],features[:,4:])
    supporters = []
    verdicts = []
    for view in views:
        n = nearest(view,hard,check)
        votes = np.stack([(labels[n]==c).sum(axis=1) for c in (0,1,2)],axis=1)
        chosen = votes.argmax(axis=1)
        actual_votes = votes[np.arange(len(hard)),chosen]
        conditions = ((chosen!=labels[hard]) & (actual_votes>=4) &
                      (actual_votes-votes[np.arange(len(hard)),labels[hard]]>=2) &
                      ((actual_votes+1)/10>=1.15*prior[chosen]))
        supporters.append(conditions)
        verdicts.append(chosen)
    both = supporters[0] & supporters[1] & (verdicts[0]==verdicts[1])
    targets = labels.copy()
    targets[hard[both]] = verdicts[0][both]
    return targets,hard


def rotate(labels, targets, weights, hard):
    """Move proposals within the same original class and hard-confidence ranks."""
    control = labels.copy()
    for c in (0,1,2):
        rows = hard[labels[hard]==c]
        if len(rows)<2:
            control[rows]=targets[rows]
            continue
        rows=rows[np.argsort(weights[rows],kind='stable')]
        control[rows]=np.roll(targets[rows],len(rows)//2)
        assert np.array_equal(np.bincount(control[rows],minlength=3),
                              np.bincount(targets[rows],minlength=3))
    return control
