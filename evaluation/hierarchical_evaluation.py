"""Read existing daily artifacts; never refit or regroup."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz,save_npz,csr_matrix
from graph.pilot import chunks
from preprocessing.transform import Transform
from hierarchy.temporal import _aggregate_descriptors
from graph.sc_fagc import mixed_similarity
from evaluation.representation import nearest,summarize
from evaluation.topology import audit,contract
from evaluation.compression import summarize_compression,distribution

def evaluate(root,day,folder,transform,output):
    root,folder,output=map(Path,(root,folder,output))
    output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    report=json.loads((folder/"report.json").read_text(encoding="utf-8"))
    if report["day"]!=day or report["status"]!="completed": raise ValueError("wrong or incomplete daily artifact")
    manifest=json.loads((folder/"chunks.json").read_text(encoding="utf-8"))["chunks"]
    groups=json.loads((folder/"hierarchy.json").read_text(encoding="utf-8"))["levels"]["day"]
    with np.load(folder/"atoms.npz",allow_pickle=False) as f:
        x,b,c,mass,chunk_ids=(f[k] for k in ("x","b","c","mass","chunk_id"))
    members=[np.asarray(g["child_component_indices"],dtype=int) for g in groups]
    if not np.array_equal(np.sort(np.concatenate(members)),np.arange(len(x))): raise ValueError("invalid hierarchy partition")
    parent=np.empty(len(x),dtype=int)
    for j,idx in enumerate(members): parent[idx]=j
    dx,db,dc,dm=_aggregate_descriptors(x,b,c,mass,members)
    centroid=(dx,db,dc)
    # Select actual retained nodes nearest each assigned daily descriptor.
    best=np.full(len(groups),-np.inf)
    med=[np.empty_like(a) for a in centroid]
    topology=[]; q=csr_matrix((len(groups),len(groups)))
    atom_graph=csr_matrix((len(x),len(x)))
    rep_count=0
    for item in manifest:
        p=folder/"chunks"/item["chunk_id"]
        ai=np.flatnonzero(chunk_ids==item["chunk_id"])
        with np.load(p/"nodes.npz") as f: nodes=tuple(f[k] for k in ("x","b","c"))
        with np.load(p/"scrac.npz") as f: assignment=f["assignment"]
        labels=parent[ai[assignment]]
        rep_count+=len(labels)
        for j in np.unique(labels):
            ix=np.flatnonzero(labels==j)
            s=mixed_similarity(*(a[ix] for a in nodes),*(a[j:j+1] for a in centroid)).ravel()
            k=int(s.argmax())
            if s[k]>best[j]:
                best[j]=s[k]
                for a,v in zip(med,nodes): a[j]=v[ix[k]]
        a=load_npz(p/"adjacency.npz")
        local,stats=audit(a,assignment,len(ai))
        topology.append(stats)
        q=q+contract(a,labels,len(groups))
        co=local.tocoo()
        atom_graph=atom_graph+csr_matrix((co.data,(ai[co.row],ai[co.col])),shape=atom_graph.shape)
    if not np.isfinite(best).all(): raise ValueError("missing actual representative")
    save_npz(output/"daily_quotient.npz",q)
    save_npz(output/"component_quotient.npz",atom_graph)
    np.savez_compressed(output/"daily_representatives.npz",x=dx,b=db,c=dc,mass=dm,actual_x=med[0],actual_b=med[1],actual_c=med[2])
    with np.load(transform) as f: model=Transform(*(f[k] for k in ("median","scale","center","whitening")))
    scores={k:[] for k in ("centroid_global","centroid_assigned","actual_global","actual_assigned","initial_assigned")}
    assignments=[]; nearest_ids=[]; attacks=[]; count=0
    for i,(data,source,offset) in enumerate(chunks(root,day,batch_size=10000,sample_rows=None)):
        item=manifest[i]; p=folder/"chunks"/item["chunk_id"]
        ai=np.flatnonzero(chunk_ids==item["chunk_id"])
        with np.load(p/"mapping.npz") as f:
            if not np.array_equal(f["source"],source) or not np.array_equal(f["offset"],offset): raise ValueError("source mapping mismatch")
            node=f["node"]; scores["initial_assigned"].append(f["similarity"])
        with np.load(p/"scrac.npz") as f: assigned=parent[ai[f["assignment"][node]]]
        query=(model.apply(data[:,:11]),data[:,11:22],data[:,22:25])
        for name,ref in (("centroid",centroid),("actual",med)):
            g,a,j=nearest(query,ref,assigned)
            scores[name+"_global"].append(g); scores[name+"_assigned"].append(a)
            if name=="actual": nearest_ids.append(j)
        assignments.append(assigned)
        attacks.append(np.asarray(["/type=attacks/" in ("/"+str(p).replace(chr(92),"/")) for p in source]))
        count+=len(data)
        print(f"evaluated {count} rows",flush=True)
    if count!=report["raw_rows"]: raise ValueError("row count mismatch")
    scores={k:np.concatenate(v) for k,v in scores.items()}
    assigned=np.concatenate(assignments); attack=np.concatenate(attacks); nearest_id=np.concatenate(nearest_ids)
    if not np.array_equal(np.bincount(assigned,minlength=len(groups)),dm): raise ValueError("mass mismatch")
    order=np.argsort(assigned,kind="stable")
    cuts=np.r_[0,np.cumsum(np.bincount(assigned,minlength=len(groups)))]
    with (output/"representatives.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["id","assigned_rows","nearest_actual_rows","attack_rows","mean_distance","p95_distance","coverage_085"])
        nc=np.bincount(nearest_id,minlength=len(groups))
        for j in range(len(groups)):
            ix=order[cuts[j]:cuts[j+1]]; s=summarize(scores["actual_assigned"][ix])
            w.writerow([j,len(ix),nc[j],int(attack[ix].sum()),s["mean_distance"],s["distance_quantiles"]["p95"],s["coverage"]["0.85"]])
    np.savez_compressed(output/"row_scores.npz",assigned=assigned,nearest_actual=nearest_id,**scores)
    source_bytes=sum(p.stat().st_size for p in (root/f"date={day}").glob("type=*/*.parquet"))
    artifact_bytes=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    result={"day":day,"status":"completed","runtime_seconds":time.perf_counter()-start,
            "compression":summarize_compression(count,rep_count,len(x),len(groups),source_bytes,artifact_bytes),
            "assigned_mass_distribution":distribution(dm),
            "representation":{k:{"all":summarize(v),"normal":summarize(v[~attack]),"attack":summarize(v[attack])} for k,v in scores.items()},
            "topology":{"scope":"observed independent chunk representative graphs","chunks":topology,
                        "original_weight":sum(t["weight"] for t in topology),"daily_weight":float(q.sum()),
                        "daily_internal_weight":float(q.diagonal().sum()),"daily_external_weight":float(q.sum()-q.diagonal().sum())},
            "limitations":["Actual representative is retained node closest to centroid, not exact raw medoid.","No raw or cross-chunk reference graph; path stretch and spectral distortion not yet measured.","Storage includes existing provenance and graph artifacts; source is compressed Parquet."]}
    (output/"evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="dataset/parquet"); p.add_argument("--day",required=True)
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--transform",default="artifacts/preprocessing/transform.npz")
    a=p.parse_args(); evaluate(a.root,a.day,a.input,a.transform,a.output)

if __name__=="__main__": main()
