"""Diagnose isolation on saved pilot nodes, without refitting or rereading raw data."""
import argparse
import json
import time
from pathlib import Path
import numpy as np

def scan(x,b,c,k=12,block=128):
    n=len(x)
    x=np.asarray(x,dtype=np.float64)
    norm=np.linalg.norm(x,axis=1)
    valid=np.isfinite(norm)&(norm>0)
    z=np.divide(x,norm[:,None],out=np.zeros_like(x),where=valid[:,None])
    d=np.column_stack((b,c))
    best=np.full(n,-np.inf)
    pairs={}
    for start in range(0,n,block):
        end=min(start+block,n)
        s=.6*(z[start:end]@z.T)
        for j in range(d.shape[1]):
            s+=.4/d.shape[1]*(d[start:end,j,None]==d[None,:,j])
        s[~valid[start:end],:]=-np.inf
        s[:,~valid]=-np.inf
        s[np.arange(end-start),np.arange(start,end)]=-np.inf
        best[start:end]=s.max(axis=1)
        ids=np.argsort(-s,axis=1,kind="stable")[:,:k]
        w=np.take_along_axis(s,ids,axis=1)
        rr,cc=np.where(w>.775)
        for i,j,v in zip(rr+start,ids[rr,cc],w[rr,cc]):
            key=(min(int(i),int(j)),max(int(i),int(j)))
            pairs[key]=max(float(v),pairs.get(key,-1.))
    return best,valid,pairs

def allocate(n,edges,cap=20,coverage=False):
    degree=np.zeros(n,dtype=np.int32)
    chosen=set()
    def add(i,j):
        if (i,j) not in chosen and degree[i]<cap and degree[j]<cap:
            chosen.add((i,j));degree[i]+=1;degree[j]+=1
    if coverage:
        # Match two uncovered endpoints first, then attach remaining isolates.
        for (i,j),w in edges:
            if degree[i]==degree[j]==0:add(i,j)
        for (i,j),w in edges:
            if degree[i]==0 or degree[j]==0:add(i,j)
    for (i,j),w in edges:add(i,j)
    return degree,chosen

def evaluate(best,valid,pairs,chunk):
    n=len(best);out={}
    for threshold in (.85,.825,.8,.775):
        edges=sorted(((p,w) for p,w in pairs.items() if w>threshold),key=lambda t:(-t[1],t[0]))
        eligible=valid&(best>threshold)
        candidate=np.zeros(n,dtype=bool)
        for (i,j),w in edges:candidate[[i,j]]=True
        row={"undefined_cosine":int((~valid).sum()),"no_threshold_neighbor":int((valid&~eligible).sum()),"candidate_isolates":int((~candidate).sum()),"eligible_without_candidate":int((eligible&~candidate).sum()),"union_candidates":len(edges)}
        for name,coverage in (("weight_first",False),("coverage_first",True)):
            t=time.perf_counter()
            degree,chosen=allocate(n,edges,coverage=coverage)
            weights=np.array([pairs[p] for p in sorted(chosen)])
            row[name]={"isolates":int((degree==0).sum()),"isolate_fraction":float((degree==0).mean()),"cap_induced_isolates":int(((degree==0)&candidate).sum()),"edges":len(chosen),"max_degree":int(degree.max()),"mean_similarity":float(weights.mean()) if len(weights) else None,"min_similarity":float(weights.min()) if len(weights) else None,"cross_chunk_edges":sum(int(chunk[i]!=chunk[j]) for i,j in chosen),"allocation_seconds":time.perf_counter()-t}
        out[str(threshold)]=row
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes",type=Path,default=Path("artifacts/graph/pilot/nodes.npz"))
    p.add_argument("--output",type=Path,default=Path("artifacts/graph/pilot/isolation_report.json"))
    a=p.parse_args()
    with np.load(a.nodes,allow_pickle=False) as f:
        t=time.perf_counter()
        best,valid,pairs=scan(f["x"],f["b"],f["c"])
        seconds=time.perf_counter()-t
        results=evaluate(best,valid,pairs,f["chunk_id"])
    report={"nodes":len(best),"scan_seconds":seconds,"thresholds":results,"limitations":"Same saved pilot nodes only; no attack labels, no downstream detection metrics; coverage-first heuristic not globally optimal."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
