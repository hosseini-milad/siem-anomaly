"""Compare hybrid and random reduction on identical daily samples."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from config.settings import BATCH_SIZE,RANDOM_SEED
from preprocessing.transform import Transform
from preprocessing.similarity_validation import daily_samples
from preprocessing.reduction import embedding,assign,select_representatives

def metrics(s,covered,labels):
    return {"coverage":float(covered.mean()),"mean_similarity":float(s.mean()),"p05_similarity":float(np.quantile(s,.05)), "normal_coverage":float(covered[~labels].mean()) if (~labels).any() else None,"attack_coverage":float(covered[labels].mean()) if labels.any() else None}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("dataset/parquet"))
    p.add_argument("--transform",type=Path,default=Path("artifacts/preprocessing/transform.npz"))
    p.add_argument("--output",type=Path,default=Path("artifacts/preprocessing/reduction_report.json"))
    p.add_argument("--anchors",type=int,default=256)
    p.add_argument("--block-size",type=int,default=256)
    a=p.parse_args()
    if min(a.anchors,a.block_size)<1:p.error("positive anchors and block size required")
    with np.load(a.transform,allow_pickle=False) as f:
        model=Transform(*(f[k] for k in ("median","scale","center","whitening")))
    report={"ratio":.1,"threshold":.85,"anchors":a.anchors,"block_size":a.block_size,"batch_size":BATCH_SIZE,"days":{},"limitations":"Daily sampled chunks, not production stream; mixed formula provisional; no cross-chunk or downstream validation; coverage is assignment similarity, not detection recall."}
    for day,sample in daily_samples(a.root,BATCH_SIZE,BATCH_SIZE,RANDOM_SEED):
        x=model.apply(sample[:,:11]);b=sample[:,11:22];c=sample[:,22:25];labels=sample[:,25].astype(bool)
        seed=RANDOM_SEED+day.toordinal()
        t=time.perf_counter()
        r=select_representatives(x,b,c,seed=seed,block=a.block_size,anchors=a.anchors)
        elapsed=time.perf_counter()-t
        h=metrics(r["similarity"],r["covered"],labels)
        h.update(seconds=elapsed,representatives=len(r["indices"]),attack_representatives=int(labels[r["indices"]].sum()))
        t=time.perf_counter()
        e,valid=embedding(x,b,c)
        ids=np.random.default_rng(seed).choice(len(x),len(r["indices"]),replace=False)
        group,s=assign(e,ids,a.block_size)
        group[ids]=np.arange(len(ids));s[ids]=np.sum(e[ids]*e[ids],axis=1)
        q=metrics(s,(s>.85)&valid&valid[ids[group]],labels)
        q.update(seconds=time.perf_counter()-t,representatives=len(ids),attack_representatives=int(labels[ids].sum()))
        report["days"][str(day)]={"rows":len(x),"hybrid":h,"random":q}
        print(day,h,flush=True)
    if not report["days"]:raise ValueError("No data")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
if __name__=="__main__":main()
