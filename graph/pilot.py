"""Bounded pilot over real coalesced file-order chunks; local artifacts only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import save_npz
from config.settings import BATCH_SIZE,RANDOM_SEED
from data.reader import iter_batches
from preprocessing.features import extract_features
from preprocessing.transform import Transform
from preprocessing.reduction import select_representatives
from graph.sparse import build_graph

def chunks(root,day_filter,batch_size=BATCH_SIZE):
    parts=[];sources=[];offsets=[];count=0;previous=None;offset=0
    for day,kind,path,batch in iter_batches(root,batch_size):
        if str(day)<day_filter:continue
        if str(day)>day_filter:break
        if path!=previous:offset=0;previous=path
        x,b,c=extract_features(batch)
        data=np.column_stack((x,b,c))
        pos=0
        while pos<len(data):
            take=min(batch_size-count,len(data)-pos)
            parts.append(data[pos:pos+take])
            sources.append(np.full(take,str(path.relative_to(root))))
            offsets.append(np.arange(offset+pos,offset+pos+take,dtype=np.int64))
            count+=take;pos+=take
            if count==batch_size:
                yield np.concatenate(parts),np.concatenate(sources),np.concatenate(offsets)
                parts=[];sources=[];offsets=[];count=0
        offset+=len(data)
    if count:yield np.concatenate(parts),np.concatenate(sources),np.concatenate(offsets)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("dataset/parquet"))
    p.add_argument("--day",default="2026-06-21")
    p.add_argument("--max-chunks",type=int,default=5)
    p.add_argument("--transform",type=Path,default=Path("artifacts/preprocessing/transform.npz"))
    p.add_argument("--output",type=Path,default=Path("artifacts/graph/pilot"))
    a=p.parse_args()
    if not 1<=a.max_chunks<=20:p.error("pilot supports 1 to 20 chunks")
    a.output.mkdir(parents=True,exist_ok=True)
    with np.load(a.transform,allow_pickle=False) as f:model=Transform(*(f[k] for k in ("median","scale","center","whitening")))
    xs=[];bs=[];cs=[];chunk_ids=[];total=0;covered=0;start=time.perf_counter()
    from itertools import islice
    for i,(data,source,offset) in enumerate(islice(chunks(a.root,a.day),a.max_chunks)):
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
    g,candidates=build_graph(x,b,c)
    graph_seconds=time.perf_counter()-t
    degree=np.diff(g.indptr);row,col=g.nonzero()
    save_npz(a.output/"adjacency.npz",g)
    np.savez_compressed(a.output/"nodes.npz",x=x,b=b,c=c,chunk_id=chunk_id)
    report={"day":a.day,"rows":total,"chunks":len(xs),"nodes":len(x),"edges":g.nnz//2,"union_candidates":candidates,"cross_chunk_edges":int(np.count_nonzero(chunk_id[row]!=chunk_id[col])//2),"isolates":int(np.count_nonzero(degree==0)),"max_degree":int(degree.max()),"mean_degree":float(degree.mean()),"min_edge_similarity":float(g.data.min()) if g.nnz else None,"coverage":covered/total,"prepare_seconds":prep,"graph_seconds":graph_seconds,"csr_bytes":g.data.nbytes+g.indices.nbytes+g.indptr.nbytes,"limitations":"Exact pilot only, file-order chunks not chronological or class-balanced; no full-day ANN, peak RAM or downstream validation."}
    (a.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
