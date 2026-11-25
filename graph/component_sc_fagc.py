"""Run SC-FAGC over existing connected components as graph representatives."""
import argparse,json
from pathlib import Path
import numpy as np
from graph.sc_fagc import run

def _mode_rows(values):
    out=np.empty(values.shape[1],dtype=values.dtype)
    for j in range(values.shape[1]):
        vals,counts=np.unique(values[:,j],return_counts=True); out[j]=vals[np.argmax(counts)]
    return out

def aggregate_components(nodes_path,components_path):
    with np.load(nodes_path,allow_pickle=False) as n,np.load(components_path,allow_pickle=False) as c:
        x,b,cat=n["x"],n["b"],n["c"]; component_id=c["component_id"].astype(np.int64); mass=c["raw_rows"].astype(np.int64)
    if len(component_id)!=len(x) or len(b)!=len(x) or len(cat)!=len(x): raise ValueError("node features and component_id are not aligned")
    count=len(mass)
    if count==0 or component_id.min(initial=0)<0 or component_id.max(initial=-1)>=count: raise ValueError("invalid component ids")
    cx=np.zeros((count,x.shape[1]),dtype=np.float64); cb=np.zeros((count,b.shape[1]),dtype=b.dtype); cc=np.zeros((count,cat.shape[1]),dtype=cat.dtype)
    for i in range(count):
        rows=np.flatnonzero(component_id==i)
        if not len(rows): raise ValueError(f"component {i} has no graph representatives")
        cx[i]=x[rows].mean(axis=0); cb[i]=_mode_rows(b[rows]); cc[i]=_mode_rows(cat[rows])
    return cx,cb,cc,mass

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--nodes",type=Path,required=True); p.add_argument("--components",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--anchors",type=int,default=None); p.add_argument("--clusters",type=int,default=None)
    p.add_argument("--capacity",type=int,default=100); p.add_argument("--anchor-k",type=int,default=5); p.add_argument("--epsilon",type=float,default=.01)
    p.add_argument("--max-iter",type=int,default=100); p.add_argument("--sinkhorn-iter",type=int,default=1000)
    a=p.parse_args(); x,b,c,mass=aggregate_components(a.nodes,a.components)
    result=run(x,b,c,mass,n_anchors=a.anchors,n_clusters=a.clusters,max_component_group_mass=a.capacity,anchor_k=a.anchor_k,
               epsilon=a.epsilon,max_iter=a.max_iter,sinkhorn_iter=a.sinkhorn_iter)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output.with_suffix(".npz"),assignment=np.asarray(result["assignment"],dtype=np.int64),
        probabilities=np.asarray(result["probabilities"],dtype=np.float32),component_mass=mass,anchor_indices=np.asarray(result["anchor_indices"],dtype=np.int64))
    a.output.write_text(json.dumps({k:v for k,v in result.items() if k not in ("probabilities","anchor_indices")},indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("assignment","probabilities","anchor_indices")},indent=2))

if __name__=="__main__": main()
