"""Sparse connected-component audit; labels are post-hoc only."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components

def partition_labels(source):
    s=np.char.replace(source.astype(str),chr(92),"/")
    a=np.char.find(s,"/type=attacks/")>=0
    b=np.char.find(s,"/type=normal/")>=0
    if not np.all(a^b):raise ValueError("unknown source partition")
    return a

def validate(g,threshold=.85,cap=20):
    if g.shape[0]!=g.shape[1] or not g.shape[0]:raise ValueError("invalid shape")
    g=g.tocsr()
    if not g.has_canonical_format:raise ValueError("noncanonical CSR")
    if g.diagonal().any() or not np.isfinite(g.data).all() or np.any(g.data<=threshold) or np.any(g.data>1+1e-6):raise ValueError("invalid weights or loops")
    diff=g-g.T
    if diff.nnz and np.max(np.abs(diff.data))>1e-10:raise ValueError("asymmetric graph")
    if np.diff(g.indptr).max()>cap:raise ValueError("degree cap exceeded")
    return g

def mapping_stats(folder,n,chunks):
    mass=np.zeros(n,dtype=np.int64);attack=mass.copy();covered=mass.copy()
    rep=np.full(n,-1,dtype=np.int8)
    for i in range(chunks):
        with np.load(folder/f"mapping_{i:03d}.npz",allow_pickle=False) as f:
            ids=f["node"];r=f["representative_rows"];labels=partition_labels(f["source"])
            if np.any(ids<0) or np.any(ids>=n):raise ValueError("mapping range")
            nodes=ids[r]
            if len(np.unique(nodes))!=len(nodes) or np.any(rep[nodes]!=-1):raise ValueError("duplicate representatives")
            unique,counts=np.unique(ids,return_counts=True)
            if not np.array_equal(unique,nodes) or not np.array_equal(counts,f["multiplicity"]):raise ValueError("multiplicity mismatch")
            np.add.at(mass,ids,1);np.add.at(attack,ids,labels);np.add.at(covered,ids,f["covered"])
            rep[nodes]=labels[r]
    if np.any(rep<0) or np.any(mass<1):raise ValueError("missing mapping")
    return mass,attack,covered,rep

def analyze(g,x,b,c,mass,attack,covered,rep):
    g=validate(g);n=g.shape[0]
    if any(len(v)!=n for v in (x,b,c,mass,attack,covered,rep)):raise ValueError("node alignment")
    if not all(np.isfinite(v).all() for v in (x,b,c)):raise ValueError("nonfinite features")
    k,lab=connected_components(g,directed=False)
    size=np.bincount(lab,minlength=k)
    def sums(v):return np.bincount(lab,weights=v,minlength=k)
    degree=np.diff(g.indptr);edges=sums(degree)/2
    weights=sums(np.asarray(g.sum(axis=1)).ravel())/2
    m=sums(mass);a=sums(attack);cov=sums(covered);ra=sums(rep)
    density=np.divide(2*edges,size*(size.astype(float)-1),out=np.zeros(k),where=size>1)
    norm=np.linalg.norm(x,axis=1);valid=norm>0
    z=np.divide(x,norm[:,None],out=np.zeros_like(x,dtype=float),where=valid[:,None])
    v=sums(valid);sq=np.zeros(k)
    for col in z.T:sq+=sums(col)**2
    # Exact mean pair cosine over valid continuous vectors, excluding self.
    cosine=np.divide(sq-v,v*(v-1),out=np.full(k,np.nan),where=v>1)
    # Discrete agreement on the same valid pairs, using grouped counts.
    agreement=np.zeros(k)
    for col in np.column_stack((b,c)).T:
        pairs=np.column_stack((lab[valid],col[valid]))
        keys,counts=np.unique(pairs,axis=0,return_counts=True)
        agreement+=np.bincount(keys[:,0].astype(int),weights=counts.astype(float)*(counts-1),minlength=k)
    agreement=np.divide(agreement,v*(v-1)*(b.shape[1]+c.shape[1]),out=np.full(k,np.nan),where=v>1)
    mixed=.6*cosine+.4*agreement
    edge_mean=np.divide(weights,edges,out=np.full(k,np.nan),where=edges>0)
    order=np.argsort(-size,kind="stable")
    def profile(i):
        return {"id":int(i),"nodes":int(size[i]),"edges":int(edges[i]),"raw_rows":int(m[i]),"density":float(density[i]),"mean_edge_similarity":float(edge_mean[i]) if np.isfinite(edge_mean[i]) else None,"mean_all_pair_similarity":float(mixed[i]) if np.isfinite(mixed[i]) else None,"attack_rows":int(a[i]),"attack_representatives":int(ra[i]),"coverage":float(cov[i]/m[i])}
    bins=[]
    for lo,hi in [(1,1),(2,5),(6,10),(11,50),(51,100),(101,1000),(1001,10000),(10001,n)]:
        mask=(size>=lo)&(size<=hi)
        bins.append({"min":lo,"max":hi,"components":int(mask.sum()),"nodes":int(size[mask].sum()),"raw_rows":int(m[mask].sum()),"attack_rows":int(a[mask].sum())})
    iso=size==1
    report={"nodes":n,"edges":g.nnz//2,"components":k,"singleton_components":int(iso.sum()),"non_singleton_components":int((~iso).sum()),"largest_component":profile(order[0]),"largest_node_fraction":float(size.max()/n),"largest_raw_fraction":float(m[order[0]]/m.sum()),"size_bins":bins,"top20":[profile(i) for i in order[:20]],"total_raw_rows":int(m.sum()),"total_attack_rows":int(a.sum()),"total_attack_representatives":int(ra.sum()),"isolated_raw_rows":int(m[iso].sum()),"isolated_attack_rows":int(a[iso].sum()),"mixed_label_components":int(((a>0)&(a<m)).sum()),"pure_attack_components":int((a==m).sum()),"pure_normal_components":int((a==0).sum()),"undefined_cosine_nodes":int((~valid).sum()),"coverage":float(cov.sum()/m.sum()),"limitations":"All-pair similarity uses valid representative pairs, not raw members; connectivity is not clustering quality. No spectral or path validation yet."}
    return report,{"component_id":lab,"size":size,"edges":edges,"raw_rows":m,"attack_rows":a,"attack_representatives":ra,"coverage":cov/m,"density":density,"mean_edge_similarity":edge_mean,"mean_all_pair_similarity":mixed}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();t=time.perf_counter()
    meta=json.loads((args.input/"report.json").read_text())
    g=load_npz(args.input/"adjacency.npz")
    with np.load(args.input/"nodes.npz",allow_pickle=False) as f:
        stats=mapping_stats(args.input,g.shape[0],meta["chunks"])
        report,profiles=analyze(g,f["x"],f["b"],f["c"],*stats)
    if report["total_raw_rows"]!=meta["rows"] or report["nodes"]!=meta["nodes"] or report["edges"]!=meta["edges"] or report["singleton_components"]!=meta["isolates"]:raise ValueError("artifact/report mismatch")
    report["day"]=meta["day"];report["analysis_seconds"]=time.perf_counter()-t
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/"components.npz",**profiles)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2,allow_nan=False))
if __name__=="__main__":main()
