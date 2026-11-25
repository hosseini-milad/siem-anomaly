"""Run the established representative-selection + SC-FAGC/SCRAC model independently
for every chronological 10k-row chunk of a complete day, then build daily/monthly/
yearly super-components from the resulting chunk SCRAC groups.

No source partition/attack labels are read. Raw-row mass comes only from the
representative mapping multiplicities.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
from graph.pilot import chunks
from graph.sparse import build_indexed_graph
from graph.sc_fagc import run
from preprocessing.reduction import select_representatives
from preprocessing.transform import Transform
from hierarchy.temporal import build_temporal_hierarchy


def run_day(root, day, transform_path, output, capacity=200, sample_chunk_size=10000,
            representative_ratio=.10, graph_k=12, degree_cap=20, edge_threshold=.85,
            anchor_threshold=.85, anchor_k=5, lambda_size=.20,
            acceptance_threshold=.75, temporal_threshold=.85, block_size=128):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with np.load(transform_path,allow_pickle=False) as f:
        model=Transform(*(f[k] for k in ("median","scale","center","whitening")))
    chunk_root=output/"chunks"; chunk_root.mkdir(parents=True,exist_ok=True)
    manifest_chunks=[]; summaries=[]; total_rows=0; total_representatives=0
    started=time.perf_counter()
    for chunk_index,(data,source,offset) in enumerate(chunks(
        Path(root),day,batch_size=sample_chunk_size,sample_rows=None)):
        chunk_id=f"{day}-c{chunk_index:04d}"
        x=model.apply(data[:,:11]); b=data[:,11:22]; c=data[:,22:25]
        selected=select_representatives(x,b,c,ratio=representative_ratio,seed=42+chunk_index)
        rep_ids=selected["indices"]
        rx,rb,rc=x[rep_ids],b[rep_ids],c[rep_ids]
        mass=np.asarray(selected["multiplicity"],dtype=np.int64)
        if int(mass.sum())!=len(data):
            raise ValueError(f"{chunk_id}: representative multiplicities do not conserve raw rows")
        adjacency,candidate_count=build_indexed_graph(rx,rb,rc,k=graph_k,cap=degree_cap,
                                                      threshold=edge_threshold)
        n_cc,topology=connected_components(adjacency,directed=False)
        result=run(rx,rb,rc,mass,anchor_threshold=anchor_threshold,anchor_k=anchor_k,
                   max_component_group_mass=capacity,scrac_capacity=capacity,
                   scrac_lambda=lambda_size,scrac_acceptance=acceptance_threshold,
                   topology_labels=topology)
        if result.get("status") == "optimization_not_validated" or "scrac_assignment" not in result:
            opt = result.get("optimization", {})
            diagnostic = {
                "status": "optimization_not_validated",
                "day": day,
                "chunk_id": chunk_id,
                "raw_rows": int(len(data)),
                "representative_nodes": int(len(rx)),
                "components": result.get("components"),
                "anchors": result.get("anchors"),
                "optimization": {
                    "converged": opt.get("converged"),
                    "iterations": opt.get("iterations"),
                    "sinkhorn_converged": opt.get("sinkhorn_converged"),
                    "sinkhorn_iterations": opt.get("sinkhorn_iterations"),
                    "row_mass_max_error": opt.get("row_mass_max_error"),
                    "row_relative_error": opt.get("row_relative_error"),
                    "column_relative_error": opt.get("column_relative_error"),
                    "column_mass_min": opt.get("column_mass_min"),
                    "column_mass_max": opt.get("column_mass_max"),
                    "objective_history": opt.get("objective_history"),
                },
                "anchor_expansion": result.get("anchor_expansion"),
                "anchor_counts": {
                    "requested": result.get("anchors_requested"),
                    "initial": result.get("initial_anchors"),
                    "after_expansion": result.get("anchors_after_expansion"),
                    "zero_mass_removed": result.get("zero_mass_anchors_removed"),
                },
            }
            diagnostic_path = output / f"{chunk_id}_optimization_diagnostic.json"
            diagnostic_path.write_text(
                json.dumps(diagnostic, indent=2, allow_nan=False),
                encoding="utf-8"
            )
            failure_report = {
                "status": "optimization_not_validated",
                "day": day,
                "failed_chunk": chunk_id,
                "completed_chunks": summaries,
                "diagnostic": str(diagnostic_path),
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "label_blind": True,
            }
            (output / "failure_report.json").write_text(
                json.dumps(failure_report, indent=2, allow_nan=False),
                encoding="utf-8"
            )
            print(json.dumps(diagnostic, indent=2))
            raise RuntimeError(
                f"{chunk_id}: SC-FAGC optimization not validated; "
                f"diagnostic saved to {diagnostic_path}"
            )
        assignment=np.asarray(result["assignment"],dtype=np.int64)
        if assignment.shape!=(len(rx),) or np.any(assignment<0):
            raise RuntimeError(f"{chunk_id}: incomplete SCRAC assignment")
        sizes=np.bincount(assignment)
        if sizes.max(initial=0)>capacity:
            raise RuntimeError(f"{chunk_id}: hard capacity violated")
        folder=chunk_root/chunk_id; folder.mkdir(parents=True,exist_ok=True)
        nodes_path=folder/"nodes.npz"; scrac_path=folder/"scrac.npz"
        np.savez_compressed(nodes_path,x=rx,b=rb,c=rc)
        np.savez_compressed(scrac_path,assignment=assignment,component_mass=mass)
        manifest_chunks.append({"chunk_id":chunk_id,"period":day,
                                "nodes":str(nodes_path.resolve()),"scrac":str(scrac_path.resolve())})
        summaries.append({"chunk_id":chunk_id,"raw_rows":int(len(data)),
                          "representative_nodes":int(len(rx)),"candidate_edges":int(candidate_count),
                          "connected_components":int(n_cc),"scrac_groups":int(len(sizes)),
                          "max_group_size":int(sizes.max()),"singleton_groups":int(np.count_nonzero(sizes==1)),
                          "runtime_seconds":result.get("runtime_seconds"),
                          "optimization":result.get("optimization")})
        total_rows+=len(data); total_representatives+=len(rx)
    if not manifest_chunks: raise ValueError(f"no records found for day {day}")
    manifest_path=output/"chunks.json"
    manifest_path.write_text(json.dumps({"chunks":manifest_chunks},indent=2),encoding="utf-8")
    # Build day-level atoms from each chunk's SCRAC output.
    xs=[];bs=[];cs=[];masses=[];periods=[];chunk_ids=[]
    from scripts.build_temporal_atoms import _chunk_atoms
    for item in manifest_chunks:
        ax,ab,ac,am,_=_chunk_atoms(item["nodes"],item["scrac"])
        xs.append(ax);bs.append(ab);cs.append(ac);masses.append(am)
        periods.extend([day]*len(am));chunk_ids.extend([item["chunk_id"]]*len(am))
    ax,ab,ac,am=map(np.concatenate,(xs,bs,cs,masses))
    atoms_path=output/"atoms.npz"
    np.savez_compressed(atoms_path,x=ax,b=ab,c=ac,mass=am,
                        period=np.asarray(periods,dtype="U10"),chunk_id=np.asarray(chunk_ids,dtype=str))
    hierarchy=build_temporal_hierarchy(ax,ab,ac,am,np.asarray(periods),np.asarray(chunk_ids),
                                       similarity_threshold=temporal_threshold,
                                       max_child_components=capacity,block_size=block_size)
    hierarchy_path=output/"hierarchy.json"
    hierarchy_path.write_text(json.dumps(hierarchy,indent=2,allow_nan=False),encoding="utf-8")
    report={"status":"completed","day":day,"label_blind":True,"labels_read":False,
            "raw_rows":int(total_rows),"chunk_count":len(summaries),
            "representative_nodes":int(total_representatives),
            "chunk_scrac_output_components":int(len(ax)),
            "daily_supercomponents":int(hierarchy["levels"]["day_summary"]["total_output"]),
            "capacity":int(capacity),"chunk_results":summaries,
            "runtime_seconds":round(time.perf_counter()-started,3),
            "manifest":str(manifest_path),"atoms":str(atoms_path),"hierarchy":str(hierarchy_path)}
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("dataset/parquet"))
    p.add_argument("--day",default="2026-06-21")
    p.add_argument("--transform",type=Path,default=Path("artifacts/preprocessing/transform.npz"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--capacity",type=int,default=200)
    p.add_argument("--representative-ratio",type=float,default=.10)
    p.add_argument("--graph-k",type=int,default=12)
    p.add_argument("--degree-cap",type=int,default=20)
    p.add_argument("--edge-threshold",type=float,default=.85)
    p.add_argument("--anchor-threshold",type=float,default=.85)
    p.add_argument("--anchor-k",type=int,default=5)
    p.add_argument("--lambda-size",type=float,default=.20)
    p.add_argument("--acceptance-threshold",type=float,default=.75)
    p.add_argument("--temporal-threshold",type=float,default=.85)
    p.add_argument("--block-size",type=int,default=128)
    a=p.parse_args()
    if a.capacity!=200 and a.capacity not in (300,500):
        p.error("capacity must be 200, 300, or 500")
    report=run_day(a.root,a.day,a.transform,a.output,a.capacity,
                   representative_ratio=a.representative_ratio,graph_k=a.graph_k,
                   degree_cap=a.degree_cap,edge_threshold=a.edge_threshold,
                   anchor_threshold=a.anchor_threshold,anchor_k=a.anchor_k,
                   lambda_size=a.lambda_size,acceptance_threshold=a.acceptance_threshold,
                   temporal_threshold=a.temporal_threshold,block_size=a.block_size)
    print(json.dumps({k:v for k,v in report.items() if k!="chunk_results"},indent=2))


if __name__=="__main__": main()
