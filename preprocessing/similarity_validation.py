"""Compare fixed-transform similarity on uniform daily samples, without graph construction."""
import argparse
import json
from pathlib import Path
from itertools import groupby
import numpy as np
from config.settings import BATCH_SIZE, RANDOM_SEED
from data.reader import iter_batches
from preprocessing.features import extract_features, CONTINUOUS
from preprocessing.transform import Transform, sample_training

THRESHOLD = 0.85
ETA = 0.6

def daily_samples(root, batch_size, limit, seed):
    for day, group in groupby(iter_batches(root, batch_size), key=lambda r:r[0]):
        def arrays():
            for _, kind, _, batch in group:
                x,b,c = extract_features(batch)
                yield np.column_stack((x,b,c,np.full(len(x),kind=="attacks")))
        yield day, sample_training(arrays(),limit,seed+day.toordinal())

def normalize(x):
    norm=np.linalg.norm(x,axis=1)
    valid=np.isfinite(norm)&(norm>0)
    return np.divide(x,norm[:,None],out=np.zeros_like(x),where=valid[:,None]),valid

def evaluate_sample(sample, model, block=256):
    x=sample[:,:11]
    discrete=sample[:,11:25]
    labels=sample[:,25].astype(bool)
    x=np.where(np.isfinite(x),x,model.median)
    scaled=(x-model.median)/model.scale
    variants={"raw":x,"robust":scaled,"zca":model.apply(x).astype(np.float64)}
    vectors={k:normalize(v) for k,v in variants.items()}
    n=len(x)
    names=(*variants,"mixed")
    counts={k:np.zeros(n,dtype=np.int64) for k in names}
    hist={k:np.zeros(200,dtype=np.int64) for k in names}
    stats={k:np.zeros(7,dtype=np.float64) for k in names}
    for start in range(0,n,block):
        end=min(start+block,n)
        upper=np.arange(start,end)[:,None]<np.arange(n)[None,:]
        same=labels[start:end,None]==labels[None,:]
        scores={}
        for k,(v,valid) in vectors.items():
            scores[k]=(np.clip(v[start:end]@v.T,-1,1),valid[start:end,None]&valid[None,:])
        z,valid=scores["zca"]
        match=np.zeros_like(z)
        for j in range(discrete.shape[1]):
            match += discrete[start:end,j,None]==discrete[None,:,j]
        scores["mixed"]=(ETA*z+(1-ETA)*match/discrete.shape[1],valid)
        for k,(s,valid) in scores.items():
            mask=upper&valid
            a=s[mask]
            edge=mask&(s>THRESHOLD)
            counts[k][start:end]+=edge.sum(axis=1)
            counts[k]+=edge.sum(axis=0)
            hist[k]+=np.histogram(a,bins=200,range=(-1,1))[0]
            stats[k]+= [len(a),a.sum(),np.dot(a,a),np.count_nonzero(a>THRESHOLD),np.count_nonzero(a>0.995),np.count_nonzero(edge&same),np.count_nonzero(mask&same)]
    result={}
    for k,t in stats.items():
        total,s,ss,edges,high,same_edges,same_total=t
        result[k]={"valid_pairs":int(total),"undefined_pairs":int(n*(n-1)//2-total),"mean":float(s/total) if total else None,"std":float(np.sqrt(max(0,ss/total-(s/total)**2))) if total else None,"above_085_fraction":float(edges/total) if total else None,"above_0995_fraction":float(high/total) if total else None,"mean_threshold_neighbors":float(counts[k].mean()),"max_threshold_neighbors":int(counts[k].max()),"isolates":int(np.count_nonzero(counts[k]==0)),"same_partition_edge_fraction":float(same_edges/same_total) if same_total else None,"cross_partition_edge_fraction":float((edges-same_edges)/(total-same_total)) if total>same_total else None,"histogram":hist[k].tolist()}
    return {"rows":n,"attack_rows":int(labels.sum()),"zero_ratio":dict(zip(CONTINUOUS,np.mean(x==0,axis=0).tolist())),"variance":dict(zip(CONTINUOUS,np.var(x,axis=0).tolist())),"methods":result}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("dataset/parquet"))
    p.add_argument("--transform",type=Path,default=Path("artifacts/preprocessing/transform.npz"))
    p.add_argument("--output",type=Path,default=Path("artifacts/preprocessing/similarity_report.json"))
    p.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    p.add_argument("--sample-per-day",type=int,default=10000)
    p.add_argument("--block-size",type=int,default=256)
    a=p.parse_args()
    if min(a.batch_size,a.block_size)<1 or a.sample_per_day<2:
        p.error("positive batch/block and sample >=2 required")
    with np.load(a.transform,allow_pickle=False) as f:
        model=Transform(*(f[k] for k in ("median","scale","center","whitening")))
    report={"threshold":THRESHOLD,"mixed_weight":ETA,"mixed_policy":"candidate signed cosine plus mean exact discrete match; upstream parity pending","sample_per_day":a.sample_per_day,"batch_size":a.batch_size,"block_size":a.block_size,"histogram_range":[-1,1],"days":{},"limitations":"Daily uniform samples, threshold-only degree before kNN/degree caps; partition labels diagnostic only; no independent temporal validation."}
    for day,sample in daily_samples(a.root,a.batch_size,a.sample_per_day,RANDOM_SEED):
        report["days"][str(day)]=evaluate_sample(sample,model,a.block_size)
        print(f"{day}: {len(sample)} sampled rows",flush=True)
    if not report["days"]:
        raise ValueError("No data")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(a.output)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
