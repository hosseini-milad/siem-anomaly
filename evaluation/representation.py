"""Exact mixed-similarity evaluation in bounded blocks."""
import numpy as np
from graph.sc_fagc import mixed_similarity

def nearest(query, reference, assigned, block=128):
    n=len(query[0])
    best=np.empty(n); own=np.empty(n); ids=np.empty(n,dtype=int)
    for i in range(0,n,block):
        sl=slice(i,i+block)
        s=mixed_similarity(*(a[sl] for a in query),*reference)
        j=s.argmax(axis=1)
        ids[sl]=j
        best[sl]=s[np.arange(len(j)),j]
        own[sl]=s[np.arange(len(j)),assigned[sl]]
    return best,own,ids

def summarize(s):
    s=np.asarray(s)
    if not len(s): return None
    d=1-s
    return {"rows":len(s),"mean_distance":float(d.mean()),"max_distance":float(d.max()),
            "distance_quantiles":dict(zip(("p50","p90","p95","p99"),np.quantile(d,[.5,.9,.95,.99]).tolist())),
            "coverage":{str(t):float(np.mean(s>t)) for t in (.85,.90,.95)},
            "mean_similarity":float(s.mean()),"minimum_similarity":float(s.min())}
