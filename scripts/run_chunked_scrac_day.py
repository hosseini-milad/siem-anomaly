"""Run the established representative-selection + direct SCRAC model independently
for every chronological 10k-row chunk of a complete day, then build daily super-components from the resulting chunk SCRAC groups.

Partition labels are used only for post-hoc evaluation. Raw-row mass comes only from the
representative mapping multiplicities.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
from graph.pilot import chunks
from graph.sparse import build_indexed_graph
from graph.direct_scrac import direct_scrac
from scipy.sparse import save_npz
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
        assignment, stats = direct_scrac(rx, rb, rc, mass, threshold=anchor_threshold,
                                         capacity=capacity, lambda_size=lambda_size,
                                         acceptance_threshold=acceptance_threshold,
                                         topology_labels=topology, seed=42+chunk_index)
        if assignment.shape!=(len(rx),) or np.any(assignment<0):
            raise RuntimeError(f"{chunk_id}: incomplete SCRAC assignment")
        sizes=np.bincount(assignment)
        if sizes.max(initial=0)>capacity:
            raise RuntimeError(f"{chunk_id}: hard capacity violated")
        folder=chunk_root/chunk_id; folder.mkdir(parents=True,exist_ok=True)
        nodes_path=folder/"nodes.npz"; scrac_path=folder/"scrac.npz"
        np.savez_compressed(nodes_path,x=rx,b=rb,c=rc)
        save_npz(folder/"adjacency.npz", adjacency)
        mapping = np.searchsorted(rep_ids, selected["assignment"])
        np.savez_compressed(folder/"mapping.npz", source=source, offset=offset,
                            node=mapping, similarity=selected["similarity"],
                            covered=selected["covered"], representative_rows=rep_ids,
                            multiplicity=mass)
        # Partition metadata is consulted only after assignments are final.
        attacks = np.asarray(["/type=attacks/" in ("/"+str(p).replace(chr(92),"/")) for p in source])
        raw_groups = assignment[mapping]
        attack_counts = np.bincount(raw_groups, weights=attacks.astype(int), minlength=len(sizes)).astype(int)
        raw_counts = np.bincount(raw_groups, minlength=len(sizes))
        stats["posthoc_attack_rows"] = attack_counts.tolist()
        stats["posthoc_normal_rows"] = (raw_counts-attack_counts).tolist()
        stats["posthoc_mixed_groups"] = int(np.count_nonzero((attack_counts>0)&(attack_counts<raw_counts)))
        stats["raw_coverage"] = float(np.mean(selected["covered"]))
        np.savez_compressed(scrac_path,assignment=assignment,component_mass=mass)
        manifest_chunks.append({"chunk_id":chunk_id,"period":day,
                                "nodes":str(nodes_path.resolve()),"scrac":str(scrac_path.resolve())})
        summaries.append({"chunk_id":chunk_id,"raw_rows":int(len(data)),
                          "representative_nodes":int(len(rx)),"candidate_edges":int(candidate_count),
                          "connected_components":int(n_cc),"scrac_groups":int(len(sizes)),
                          "max_group_size":int(sizes.max()),"singleton_groups":int(np.count_nonzero(sizes==1)),
                          "scrac":stats})
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
                                       max_child_components=capacity,block_size=block_size, lambda_size=lambda_size,
                                       acceptance_threshold=acceptance_threshold, levels=("day",))
    hierarchy_path=output/"hierarchy.json"
    hierarchy_path.write_text(json.dumps(hierarchy,indent=2,allow_nan=False),encoding="utf-8")
    daily = hierarchy["levels"]["day"]
    atom_attacks = np.concatenate([r["scrac"]["posthoc_attack_rows"] for r in summaries])
    for group in daily:
        ids = group["child_component_indices"]
        group["posthoc_attack_rows"] = int(atom_attacks[ids].sum())
        group["posthoc_normal_rows"] = int(group["represented_raw_rows"]) - group["posthoc_attack_rows"]
    hierarchy_path.write_text(json.dumps(hierarchy,indent=2,allow_nan=False),encoding="utf-8")
    report={"status":"completed","day":day,"label_blind":True,"labels_read":True,"labels_used_for_grouping":False,
            "raw_rows":int(total_rows),"chunk_count":len(summaries),
            "representative_nodes":int(total_representatives),
            "chunk_scrac_output_components":int(len(ax)),
            "daily_supercomponents":int(hierarchy["levels"]["day_summary"]["total_output"]),
            "capacity":int(capacity),"method":"direct_scrac","chunk_results":summaries,
            "daily_mixed_groups":sum(g["posthoc_attack_rows"]>0 and g["posthoc_normal_rows"]>0 for g in daily),
            "daily_attack_rows":sum(g["posthoc_attack_rows"] for g in daily),
            "daily_normal_rows":sum(g["posthoc_normal_rows"] for g in daily),
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
