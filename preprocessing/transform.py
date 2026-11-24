from dataclasses import dataclass
import numpy as np
from config.settings import ZCA_EPSILON, FIT_SAMPLE_SIZE, RANDOM_SEED

@dataclass
class Transform:
    median: np.ndarray
    scale: np.ndarray
    center: np.ndarray
    whitening: np.ndarray

    def apply(self, x):
        x = np.where(np.isfinite(x), x, self.median)
        return (((x-self.median)/self.scale-self.center) @ self.whitening).astype(np.float32)

def fit_transform(x, epsilon=ZCA_EPSILON):
    if x.ndim != 2 or len(x)<2 or epsilon<=0:
        raise ValueError("Need at least two rows and positive epsilon")
    x = np.where(np.isfinite(x),x,np.nan)
    median = np.nanmedian(x,axis=0)
    if not np.isfinite(median).all():
        raise ValueError("All-missing continuous feature")
    x = np.where(np.isfinite(x),x,median)
    q = np.percentile(x,[25,75],axis=0)
    scale = q[1]-q[0]
    scale = np.where(scale>1e-12,scale,1.0)
    z = (x-median)/scale
    center = z.mean(axis=0)
    z -= center
    eig, vec = np.linalg.eigh(z.T @ z / (len(z)-1))
    whitening = (vec * (np.maximum(eig,0)+epsilon)**-0.5) @ vec.T
    return Transform(median,scale,center,whitening)

def sample_training(batches, limit=FIT_SAMPLE_SIZE, seed=RANDOM_SEED):
    """Uniform bounded priority sample across training rows only."""
    if limit<2:
        raise ValueError("Sample limit must be >=2")
    rng=np.random.default_rng(seed)
    keys=np.empty(0)
    rows=None
    for x in batches:
        k=rng.random(len(x))
        keys=np.concatenate((keys,k))
        rows=x.copy() if rows is None else np.concatenate((rows,x),axis=0)
        if len(keys)>limit:
            idx=np.argpartition(keys,limit-1)[:limit]
            keys,rows=keys[idx],rows[idx]
    if rows is None:
        raise ValueError("No training rows")
    return rows
