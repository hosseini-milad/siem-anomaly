"""Pilot hybrid label-free representative selection with bounded matrix blocks."""
import numpy as np

def embedding(x,b,c):
    x=x.astype(np.float64)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    valid=norm[:,0]>0
    z=np.divide(x,norm,out=np.zeros_like(x),where=norm>0)
    # Exact discrete matching via one-hot; continuous cosine weight 0.6.
    parts=[np.sqrt(.6)*z]
    d=np.column_stack((b,c))
    for j in range(d.shape[1]):
        _,inv=np.unique(d[:,j],return_inverse=True)
        parts.append(np.sqrt(.4/d.shape[1])*np.eye(inv.max()+1,dtype=np.float32)[inv])
    return np.column_stack(parts).astype(np.float32),valid

def assign(e, ids, block=256):
    nearest=np.empty(len(e),dtype=np.int64)
    scores=np.empty(len(e),dtype=np.float32)
    for start in range(0,len(e),block):
        s=e[start:start+block]@e[ids].T
        j=s.argmax(axis=1)
        nearest[start:start+len(j)]=j
        scores[start:start+len(j)]=s[np.arange(len(j)),j]
    return nearest,scores

def select_representatives(x,b,c,ratio=.1,seed=42,block=256,anchors=256):
    if not 0<ratio<=1 or len(x)==0 or min(block,anchors)<1:
        raise ValueError("invalid selection parameters")
    e,valid=embedding(x,b,c)
    n=len(e); budget=max(1,int(np.ceil(n*ratio)))
    rng=np.random.default_rng(seed)
    centers=rng.choice(n,min(n,anchors,budget),replace=False)
    group,score=assign(e,centers,block)
    sizes=np.bincount(group,minlength=len(centers))
    chosen=np.zeros(n,dtype=bool)
    chosen[centers]=True
    # Coverage via inverse square-root density weighted sampling.
    remaining=budget-len(centers)
    cover=int(.6*remaining); rare=int(.15*remaining)
    def take(order,k):
        ids=order[~chosen[order]][:k]
        chosen[ids]=True
    weights=1/np.sqrt(sizes[group])
    keys=-np.log(np.maximum(rng.random(n),1e-15))/weights
    take(np.argsort(keys),cover)
    take(np.argsort(np.where(valid,score,-np.inf)),rare)
    ids=np.flatnonzero(chosen)
    _,s=assign(e,ids,block)
    take(np.argsort(np.where(valid,s,-np.inf)),budget-len(ids))
    ids=np.flatnonzero(chosen)
    group,s=assign(e,ids,block)
    group[ids]=np.arange(len(ids))
    s[ids]=np.sum(e[ids]*e[ids],axis=1)
    covered=(s>.85)&valid&valid[ids[group]]
    return {"indices":ids,"assignment":ids[group],"similarity":s,"covered":covered,"multiplicity":np.bincount(group,minlength=len(ids))}
