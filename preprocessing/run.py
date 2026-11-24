"""Fit a provisional normal-reference transform and validate batch output."""
import argparse
import json
from pathlib import Path
from datetime import date
import numpy as np
from config.settings import BATCH_SIZE, FIT_SAMPLE_SIZE
from data.reader import iter_batches
from preprocessing.features import extract_features, CONTINUOUS, BINARY, CATEGORICAL
from preprocessing.transform import sample_training, fit_transform

def reference_batches(root,batch_size):
    for day,kind,path,batch in iter_batches(root,batch_size):
        if day>date(2026,6,20):
            break
        if kind=="normal" and day>=date(2026,6,15):
            yield extract_features(batch)[0]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("dataset/parquet"))
    p.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    p.add_argument("--fit-sample-size",type=int,default=FIT_SAMPLE_SIZE)
    p.add_argument("--output",type=Path,default=Path("artifacts/preprocessing"))
    a=p.parse_args()
    if a.batch_size<=0:
        p.error("batch size must be positive")
    sample=sample_training(reference_batches(a.root,a.batch_size),a.fit_sample_size)
    model=fit_transform(sample)
    rows=0
    for day,kind,path,batch in iter_batches(a.root,a.batch_size):
        x,b,c=extract_features(batch)
        z=model.apply(x)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite transformed feature")
        rows+=len(z)
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez(a.output/"transform.npz",median=model.median,scale=model.scale,center=model.center,whitening=model.whitening)
    report={"rows":rows,"fit_rows":len(sample),"batch_size":a.batch_size,"continuous":CONTINUOUS,"binary":BINARY,"categorical":CATEGORICAL,"fit_policy":"provisional reconstructed clean reference; not independent temporal evaluation"}
    (a.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
