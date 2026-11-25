"""Post-run audit for adaptive-anchor SC-FAGC/SCRAC assignments."""
import argparse,json,time
from pathlib import Path
import numpy as np
from graph.component_sc_fagc import aggregate_components
from graph.sc_fagc import mixed_similarity

def evaluate(nodes,components,report_path,threshold=.85):
    t0=time.perf_counter()
    x,b,c,mass=aggregate_components(nodes,components)
    report=json.loads(Path(report_path).read_text(encoding="utf-8"))
    labels=np.asarray(report.get("assignment",[]),dtype=np.int64)
    if labels.shape!=(len(x),) or np.any(labels<0):
        raise ValueError("report assignment is missing, invalid, or not aligned with component data")
    groups=np.unique(labels); sizes=[]; masses=[]; min_sims=[]; violations=[]
    for gid in groups:
        idx=np.flatnonzero(labels==gid); sizes.append(int(len(idx))); masses.append(int(mass[idx].sum()))
        if len(idx)<2:
            min_sims.append(1.0); continue
        min_sim=1.0
        for i in range(len(idx)):
            sims=mixed_similarity(x[idx[i:i+1]],b[idx[i:i+1]],c[idx[i:i+1]],x[idx[i+1:]],b[idx[i+1:]],c[idx[i+1:]]).ravel()
            if sims.size: min_sim=min(min_sim,float(sims.min()))
        min_sims.append(float(min_sim))
        if min_sim<=threshold: violations.append({"group":int(gid),"size":int(len(idx)),"minimum_similarity":float(min_sim)})
    out={
      "components":int(len(x)),"raw_rows":int(mass.sum()),"groups":int(len(groups)),
      "component_group_size":{"min":int(min(sizes)),"median":float(np.median(sizes)),"mean":float(np.mean(sizes)),"p95":float(np.percentile(sizes,95)),"max":int(max(sizes))},
      "raw_mass_group_size":{"min":int(min(masses)),"median":float(np.median(masses)),"mean":float(np.mean(masses)),"p95":float(np.percentile(masses,95)),"max":int(max(masses))},
      "groups_over_100_components":int(sum(v>100 for v in sizes)),
      "groups_over_100_raw_rows":int(sum(v>100 for v in masses)),
      "singleton_groups":int(sum(v==1 for v in sizes)),
      "complete_link_threshold":float(threshold),
      "complete_link_violations":violations,
      "complete_link_valid":len(violations)==0,
      "anchor_expansion":report.get("anchor_expansion",{}),
      "anchors_initial":report.get("initial_anchors"),
      "anchors_after_expansion":report.get("anchors_after_expansion",report.get("anchors")),
      "zero_mass_anchors_removed":report.get("zero_mass_anchors_removed"),
      "runtime_audit_seconds":round(time.perf_counter()-t0,3)
    }
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes",type=Path,required=True); p.add_argument("--components",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threshold",type=float,default=.85); a=p.parse_args()
    result=evaluate(a.nodes,a.components,a.report,a.threshold)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
