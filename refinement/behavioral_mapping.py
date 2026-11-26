import numpy as np

def refine(assigned, nearest, own, best, n, threshold=.85):
    assigned,nearest=np.asarray(assigned),np.asarray(nearest)
    own,best=np.asarray(own),np.asarray(best)
    if not (assigned.shape==nearest.shape==own.shape==best.shape) or assigned.ndim!=1:
        raise ValueError("unaligned row arrays")
    if not len(assigned) or not np.isfinite(own).all() or not np.isfinite(best).all():
        raise ValueError("empty or invalid scores")
    if any(np.any((a<0)|(a>=n)) or a.dtype.kind not in "iu" for a in (assigned,nearest)):
        raise ValueError("invalid representative ids")
    if np.any(best<own-1e-10): raise ValueError("global score below assigned score")
    # Preserve structural assignment on ties, never decrease similarity.
    keep=own>=best-1e-12
    target=np.where(keep,assigned,nearest)
    score=np.where(keep,own,best)
    valid=score>threshold
    return target,score,valid
