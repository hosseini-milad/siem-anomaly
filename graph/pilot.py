"""Pilot over chronological logical-day chunks; local artifacts only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import save_npz
from config.settings import BATCH_SIZE,RANDOM_SEED
from data.reader import canonicalize
from datetime import date
import pyarrow.parquet as pq
import pyarrow.compute as pc
from preprocessing.features import extract_features
from preprocessing.transform import Transform
from preprocessing.reduction import select_representatives
from graph.sparse import build_graph,build_indexed_graph

def chunks(root,day_filter,batch_size=BATCH_SIZE,sample_rows=50000,temporal_bins=5,seed=RANDOM_SEED):
    """Sort one logical day by canonical timestamp; memory scales with one day."""
    day=date.fromisoformat(day_filter)
    paths=sorted((root/f"date={day_filter}").glob("type=*/*.parquet"))
    data=[];times=[];files=[];offsets=[];names=[]
    for path in paths:
        if path.parent.name not in ("type=normal","type=attacks"):continue
        fid=len(names);names.append(str(path.relative_to(root)))
        offset=0
        with pq.ParquetFile(path) as pf:
            for raw in pf.iter_batches(batch_size=batch_size):
                batch=canonicalize(raw,day,path.name)
                x,b,c=extract_features(batch)
                data.append(np.column_stack((x,b,c)))
                t=batch.column(batch.schema.get_field_index("flow_start"))
                times.append(pc.fill_null(pc.cast(t,"int64"),np.iinfo(np.int64).max).to_numpy(zero_copy_only=False))
                files.append(np.full(len(x),fid,dtype=np.int32))
                offsets.append(np.arange(offset,offset+len(x),dtype=np.int64))
                offset+=len(x)
    if not data:return
    data,times,files,offsets=map(np.concatenate,(data,times,files,offsets))
    order=np.argsort(times,kind="stable")
    if sample_rows is not None:
        if sample_rows < 1 or temporal_bins < 1:
            raise ValueError("sample_rows and temporal_bins must be positive")
        if sample_rows % temporal_bins:
            raise ValueError("sample_rows must be divisible by temporal_bins for equal temporal quotas")
        quota=sample_rows//temporal_bins
        ordered_times=times[order]
        lower=int(ordered_times[0]); upper=int(ordered_times[-1])
        if upper<=lower:
            raise ValueError("cannot temporally stratify records with a single timestamp")
        rng=np.random.default_rng(seed)
        selected=[]
        edges=np.linspace(lower,upper+1,temporal_bins+1,dtype=np.int64)
        for bin_id in range(temporal_bins):
            lo,hi=edges[bin_id],edges[bin_id+1]
            positions=np.flatnonzero((ordered_times>=lo)&(ordered_times<hi))
            if len(positions)<quota:
                raise ValueError(f"temporal bin {bin_id} has {len(positions)} rows; {quota} required")
            selected.append(rng.choice(positions,size=quota,replace=False))
        chosen=np.sort(np.concatenate(selected))
        order=order[chosen]
    names=np.asarray(names)
    for start in range(0,len(order),batch_size):
        ids=order[start:start+batch_size]
        yield data[ids],names[files[ids]],offsets[ids]

def label_summary(degree,representative_labels,member_attack,member_total):
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("dataset/parquet"))
    p.add_argument("--day",default="2026-06-21")
    p.add_argument("--sample-rows",type=int,default=50000)
    p.add_argument("--temporal-bins",type=int,default=5)
    p.add_argument("--seed",type=int,default=RANDOM_SEED)
    p.add_argument("--transform",type=Path,default=Path("artifacts/preprocessing/transform.npz"))
    p.add_argument("--output",type=Path,default=Path("artifacts/graph/pilot"))
    p.add_argument("--backend",choices=("exact","indexed"),default="exact")
    p.add_argument("--full-day",action="store_true")
    a=p.parse_args()
    if a.sample_rows<1 or a.temporal_bins<1 or a.sample_rows%a.temporal_bins:p.error("--sample-rows must be positive and divisible by --temporal-bins")
    if a.full_day and a.backend!="indexed":p.error("full-day requires indexed backend")
    a.output.mkdir(parents=True,exist_ok=True)
    with np.load(a.transform,allow_pickle=False) as f:model=Transform(*(f[k] for k in ("median","scale","center","whitening")))
    xs=[];bs=[];cs=[];chunk_ids=[];total=0;covered=0;start=time.perf_counter()
    for i,(data,source,offset) in enumerate(chunks(a.root,a.day,batch_size=BATCH_SIZE,sample_rows=None if a.full_day else a.sample_rows,temporal_bins=a.temporal_bins,seed=a.seed)):
        x=model.apply(data[:,:11]);b=data[:,11:22];c=data[:,22:25]
        r=select_representatives(x,b,c,seed=RANDOM_SEED+i)
        ids=r["indices"];base=sum(map(len,xs))
        mapping=base+np.searchsorted(ids,r["assignment"])
        np.savez_compressed(a.output/f"mapping_{i:03d}.npz",source=source,offset=offset,node=mapping,similarity=r["similarity"],covered=r["covered"],representative_rows=ids,multiplicity=r["multiplicity"])
        xs.append(x[ids]);bs.append(b[ids]);cs.append(c[ids]);chunk_ids.append(np.full(len(ids),i))
        total+=len(x);covered+=int(r["covered"].sum())
    if not total:raise ValueError("No records for selected day")
    x,b,c,chunk_id=map(np.concatenate,(xs,bs,cs,chunk_ids))
    prep=time.perf_counter()-start;t=time.perf_counter()
    g,candidates=(build_graph if a.backend=="exact" else build_indexed_graph)(x,b,c)
    graph_seconds=time.perf_counter()-t
    degree=np.diff(g.indptr);row,col=g.nonzero()
    save_npz(a.output/"adjacency.npz",g)
    np.savez_compressed(a.output/"nodes.npz",x=x,b=b,c=c,chunk_id=chunk_id)
    report={"day":a.day,"rows":total,"chunks":len(xs),"nodes":len(x),"edges":g.nnz//2,"union_candidates":candidates,"cross_chunk_edges":int(np.count_nonzero(chunk_id[row]!=chunk_id[col])//2),"isolates":int(np.count_nonzero(degree==0)),"max_degree":int(degree.max()),"mean_degree":float(degree.mean()),"min_edge_similarity":float(g.data.min()) if g.nnz else None,"coverage":covered/total,"prepare_seconds":prep,"graph_seconds":graph_seconds,"csr_bytes":g.data.nbytes+g.indices.nbytes+g.indptr.nbytes,"limitations":"Chronological logical day; one-day feature arrays in RAM; indexed candidate recall, peak RAM and downstream validation pending."}
    report["backend"]=a.backend
    report["full_day"]=a.full_day
    report["allocation_policy"]="coverage_first_then_weight_first"
    report["sampling"]={"method":"uniform random sampling within equal-duration temporal bins","sample_rows":None if a.full_day else a.sample_rows,"temporal_bins":None if a.full_day else a.temporal_bins,"seed":a.seed,"label_blind":True,"label_ratio_target":"natural population ratio; never enforced using labels"}
    report["isolate_fraction"]=float((degree==0).mean())
    (a.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
