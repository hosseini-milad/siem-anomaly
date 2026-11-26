import numpy as np
from graph.sc_fagc import mixed_similarity

def audit(reference, mass, threshold=.999, block=128):
    n=len(mass); nearest=np.full(n,-1,dtype=int); score=np.full(n,-1.)
    for i in range(0,n,block):
        s=mixed_similarity(*(a[i:i+block] for a in reference),*reference)
        s[np.arange(len(s)),np.arange(i,i+len(s))]=-np.inf
        if n>1:
            j=s.argmax(axis=1)
            nearest[i:i+len(s)]=j
            score[i:i+len(s)]=s[np.arange(len(s)),j]
    return nearest,score,(score>=threshold)&(mass==0)
