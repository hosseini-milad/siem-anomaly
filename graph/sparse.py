"""Exact bounded pilot: union top-k then symmetric greedy degree cap."""
import numpy as np
from scipy.sparse import csr_matrix

def build_graph(x,b,c,k=12,cap=20,threshold=.85,block=128):
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
    degree=np.zeros(n,dtype=np.int32)
    rows=[];cols=[];weights=[]
    for (i,j),w in sorted(candidates.items(),key=lambda t:(-t[1],t[0])):
        if degree[i]>=cap or degree[j]>=cap: continue
        degree[i]+=1;degree[j]+=1
        rows.extend((i,j));cols.extend((j,i));weights.extend((w,w))
    return csr_matrix((np.asarray(weights,dtype=np.float64),(rows,cols)),shape=(n,n)),len(candidates)
