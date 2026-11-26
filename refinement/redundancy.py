"""Conservative nearest-neighbour redundancy candidates; never deletes representatives."""
import numpy as np
from graph.sc_fagc import mixed_similarity

def audit(reference, mass, threshold=.999, block=128):
    mass=np.asarray(mass,dtype=np.float64)
    n=len(mass)
    if mass.ndim!=1 or n==0 or not np.isfinite(mass).all() or np.any(mass<0):
        raise ValueError("mass must be a non-empty finite non-negative vector")
    if not -1<=threshold<=1 or block<1:
        raise ValueError("invalid redundancy threshold or block")
    if any(len(a)!=n for a in reference):
        raise ValueError("unaligned representative descriptors")
    nearest=np.full(n,-1,dtype=np.int64); score=np.full(n,-1.,dtype=np.float64)
    for i in range(0,n,block):
        end=min(n,i+block)
        sim=mixed_similarity(*(a[i:end] for a in reference),*reference)
        sim[np.arange(end-i),np.arange(i,end)]=-np.inf
        if n>1:
            j=sim.argmax(axis=1)
            nearest[i:end]=j
            score[i:end]=sim[np.arange(end-i),j]
    # A candidate is a representative participating in a high-similarity pair.
    # Do not require zero mass: high mass does not make redundancy impossible.
    candidate=np.zeros(n,dtype=bool)
    if n>1:
        rows=np.flatnonzero((nearest>=0)&(score>=threshold))
        candidate[rows]=True
        candidate[nearest[rows]]=True
    return nearest,score,candidate

def candidate_pairs(nearest, score, threshold):
    nearest=np.asarray(nearest,dtype=np.int64); score=np.asarray(score,dtype=float)
    pairs={}
    for i,j in enumerate(nearest):
        if j<0 or score[i]<threshold or i==j: continue
        a,b=sorted((int(i),int(j)))
        pairs[(a,b)]=max(float(score[i]),pairs.get((a,b),-np.inf))
    return [(a,b,s) for (a,b),s in sorted(pairs.items())]
