"""Run BRR for all available days >= cutoff, then build monthly super-components.

Daily inputs are existing evaluation artifacts. No raw labels are read. Days before
the cutoff are excluded (the user's pre-2026-06-21 data are normal-only).
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from refinement.run import run as run_brr
from graph.direct_scrac import direct_scrac
from hierarchy.temporal import _aggregate_descriptors

CUTOFF="2026-06-21"

def discover(root):
    found={}
    for report_path in Path(root).rglob("evaluation.json"):
        try:
            report=json.loads(report_path.read_text(encoding="utf-8"))
            day=str(report["day"])
        except Exception:
            continue
        if len(day)!=10 or day<CUTOFF or report.get("status")!="completed":
            continue
        folder=report_path.parent
        if not (folder/"daily_representatives.npz").is_file() or not (folder/"row_scores.npz").is_file():
            continue
        if day in found: raise ValueError(f"multiple complete evaluations found for {day}: {found[day]} and {folder}")
        found[day]=folder
    return dict(sorted(found.items()))

def build_monthly(day_outputs, output, threshold=.85, capacity=200, lambda_size=.20, acceptance_threshold=.75):
    buckets=defaultdict(list)
    for day,folder in day_outputs.items():
        with np.load(folder/"daily_representatives.npz",allow_pickle=False) as f:
            x,b,c,mass=(np.asarray(f[k]) for k in ("actual_x","actual_b","actual_c","mass"))
        if not (len(x)==len(b)==len(c)==len(mass)) or np.any(mass<=0):
            raise ValueError(f"{day}: invalid daily representative arrays")
        for rid in range(len(mass)):
            buckets[day[:7]].append((day,rid,x[rid],b[rid],c[rid],float(mass[rid])))
    monthly_root=Path(output)/"monthly"; monthly_root.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for month,items in sorted(buckets.items()):
        x=np.asarray([v[2] for v in items]); b=np.asarray([v[3] for v in items]); c=np.asarray([v[4] for v in items])
        mass=np.asarray([v[5] for v in items],dtype=np.float64)
        assignment,stats=direct_scrac(x,b,c,mass,threshold=threshold,capacity=capacity,
            lambda_size=lambda_size,acceptance_threshold=acceptance_threshold,anchor_ratio=.10,seed=42)
        groups=[np.flatnonzero(assignment==gid) for gid in range(stats["groups"])]
        gx,gb,gc,gm=_aggregate_descriptors(x,b,c,mass,groups)
        day_ids=np.asarray([v[0] for v in items],dtype="U10")
        daily_ids=np.asarray([v[1] for v in items],dtype=np.int64)
        folder=monthly_root/month; folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/"monthly_components.npz",x=gx,b=gb,c=gc,mass=gm,
            child_assignment=assignment,child_day=day_ids,child_daily_representative=daily_ids)
        records=[]
        for gid,members in enumerate(groups):
            records.append({"monthly_component_id":gid,"daily_representative_count":len(members),
                "represented_raw_rows":float(gm[gid]),"children":[{"day":str(day_ids[j]),
                "daily_representative_id":int(daily_ids[j])} for j in members]})
        result={"status":"completed","month":month,"days_included":sorted(set(day_ids.tolist())),
            "input_daily_representatives":len(items),"monthly_supercomponents":len(groups),
            "capacity":capacity,"capacity_unit":"immediate daily representative children",
            "capacity_violations":stats["capacity_violations"],
            "represented_raw_rows":float(mass.sum()),"conserved_raw_rows":bool(np.isclose(gm.sum(),mass.sum())),
            "group_size_distribution":{"min":int(min(map(len,groups))),"median":float(np.median(list(map(len,groups)))),
                "p95":float(np.quantile(list(map(len,groups)),.95)),"max":int(max(map(len,groups)))},
            "scrac":{k:v for k,v in stats.items() if k not in ("anchor_indices","group_sizes","group_mass_raw_rows","group_minimum_pair_similarity")},
            "components":records}
        (folder/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
        summaries.append({"month":month,"days":len(result["days_included"]),
            "daily_representatives":len(items),"monthly_supercomponents":len(groups),
            "represented_raw_rows":float(mass.sum()),"capacity_violations":stats["capacity_violations"]})
    (monthly_root/"summary.json").write_text(json.dumps({"cutoff":CUTOFF,"months":summaries},indent=2),encoding="utf-8")
    return summaries

def run_batch(input_root,output,days=None,threshold=.85,rare_max=10,redundancy_threshold=.999,
              capacity=200,lambda_size=.20,acceptance_threshold=.75):
    available=discover(input_root)
    selected=sorted(available) if days is None else sorted(set(days))
    if not selected: raise ValueError(f"no complete daily evaluations on/after {CUTOFF} found")
    invalid=[d for d in selected if d<CUTOFF or d not in available]
    if invalid: raise ValueError(f"days are before cutoff or unavailable/incomplete: {invalid}; available={list(available)}")
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    day_outputs={}
    for day in selected:
        dest=output/"daily"/day
        print(f"BRR {day}: {available[day]}",flush=True)
        run_brr(available[day],dest,threshold,rare_max,redundancy_threshold)
        day_outputs[day]=available[day]
    monthly=build_monthly(day_outputs,output,threshold,capacity,lambda_size,acceptance_threshold)
    result={"status":"completed","cutoff_exclusive_before":CUTOFF,"normal_only_days_excluded":True,
        "days_processed":selected,"daily_count":len(selected),"monthly":monthly,
        "monthly_capacity":capacity,"output":str(output.resolve())}
    (output/"batch_report.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",default="artifacts/evaluation",
        help="Root containing per-day evaluation folders; recursively discovers completed days.")
    p.add_argument("--output",default="artifacts/period_batch")
    p.add_argument("--days",default=None,
        help="Optional comma-separated ISO dates. Omit to process ALL available complete days >= 2026-06-21.")
    p.add_argument("--threshold",type=float,default=.85)
    p.add_argument("--rare-max",type=int,default=10)
    p.add_argument("--redundancy-threshold",type=float,default=.999)
    p.add_argument("--capacity",type=int,default=200)
    p.add_argument("--lambda-size",type=float,default=.20)
    p.add_argument("--acceptance-threshold",type=float,default=.75)
    a=p.parse_args(); days=None if not a.days else [v.strip() for v in a.days.split(",") if v.strip()]
    run_batch(a.input_root,a.output,days,a.threshold,a.rare_max,a.redundancy_threshold,
              a.capacity,a.lambda_size,a.acceptance_threshold)
if __name__=="__main__": main()
