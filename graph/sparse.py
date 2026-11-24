"""Exact bounded pilot: union top-k then symmetric greedy degree cap."""
import numpy as np
from scipy.sparse import csr_matrix
from graph.isolation_validation import allocate

def build_graph(x,b,c,k=12,cap=20,threshold=.85,block=128,policy="coverage_first"):
    if policy not in ("coverage_first","weight_first"): raise ValueError("unknown policy")
    n=len(x)
    if min(k,cap,block)<1 or n<1: raise ValueError("invalid graph settings")
    x=np.asarray(x,dtype=np.float64)
    norms=np.linalg.norm(x,axis=1)
    valid=np.isfinite(norms)&(norms>0)
    z=np.divide(x,norms[:,None],out=np.zeros_like(x),where=valid[:,None])
    d=np.column_stack((b,c))
    candidates={}
    for start in range(0,n,block):
        end=min(start+block,n)
        s=.6*(z[start:end]@z.T)
        for j in range(d.shape[1]):
            s+=.4/d.shape[1]*(d[start:end,j,None]==d[None,:,j])
        s[~valid[start:end],:]=-np.inf
        s[:,~valid]=-np.inf
        s[np.arange(end-start),np.arange(start,end)]=-np.inf
        # Stable tie ordering by node ID.
        ids=np.argsort(-s,axis=1,kind="stable")[:,:min(k,n)]
        weights=np.take_along_axis(s,ids,axis=1)
        rr,cc=np.where(weights>threshold)
        for i,j,w in zip(rr+start,ids[rr,cc],weights[rr,cc]):
            key=(min(int(i),int(j)),max(int(i),int(j)))
            candidates[key]=max(float(w),candidates.get(key,-1.))
    edges=sorted(candidates.items(),key=lambda t:(-t[1],t[0]))
    _,chosen=allocate(n,edges,cap=cap,coverage=policy=="coverage_first")
    rows=[];cols=[];weights=[]
    for i,j in sorted(chosen):
        w=candidates[i,j]
        rows.extend((i,j));cols.extend((j,i));weights.extend((w,w))
    return csr_matrix((np.asarray(weights,dtype=np.float64),(rows,cols)),shape=(n,n)),len(candidates)
