"""Reuse exact row scores from hierarchical evaluation; no regrouping or refitting."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from evaluation.representation import summarize
from refinement.behavioral_mapping import refine
from refinement.redundancy import audit as redundancy_audit, candidate_pairs
from refinement.rare_behavior import audit as rare_audit

def run(input_dir,output,threshold=.85,rare_max=10,redundancy_threshold=.999):
    start=time.perf_counter()
    input_dir,output=Path(input_dir),Path(output)
    if output.resolve()==input_dir.resolve() or input_dir.resolve() in output.resolve().parents:
        raise ValueError("use independent output directory")
    if not -1<=threshold<1 or rare_max<1 or not -1<=redundancy_threshold<=1:
        raise ValueError("invalid parameters")
    report=json.loads((input_dir/"evaluation.json").read_text(encoding="utf-8"))
    with np.load(input_dir/"daily_representatives.npz",allow_pickle=False) as f:
        reference=tuple(f[k] for k in ("actual_x","actual_b","actual_c")); mass=f["mass"]
    with np.load(input_dir/"row_scores.npz",allow_pickle=False) as f:
        assigned,nearest,own,best=(f[k] for k in ("assigned","nearest_actual","actual_assigned","actual_global"))
    n=len(mass)
    target,score,valid=refine(assigned,nearest,own,best,n,threshold)
    rare,original,mapped,covered,rare_mask=rare_audit(assigned,target,score,valid,n,rare_max)
    if not np.array_equal(original,mass) or len(assigned)!=report["compression"]["counts"][0]:
        raise ValueError("structural mass mismatch")
    neighbor,similarity,redundant=redundancy_audit(reference,covered,redundancy_threshold)
    pairs=candidate_pairs(neighbor,similarity,redundancy_threshold)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"behavioral_mapping.npz",structural=assigned,behavioral=np.where(valid,target,-1),
                        nearest_candidate=target,similarity=score,covered=valid)
    np.savez_compressed(output/"uncovered_rows.npz",row_index=np.flatnonzero(~valid),candidate=target[~valid],similarity=score[~valid])
    sums=np.bincount(target[valid],weights=score[valid],minlength=n)
    with (output/"representatives.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["id","structural_rows","candidate_rows","valid_behavioral_rows","mean_valid_similarity","rare","nearest_other_id","nearest_other_similarity","redundancy_candidate"])
        for j in range(n):
            w.writerow([j,original[j],mapped[j],covered[j],sums[j]/covered[j] if covered[j] else "",int(rare_mask[j]),neighbor[j],similarity[j],int(redundant[j])])
    with (output/"redundancy_pairs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["representative_a","representative_b","similarity","mass_a","mass_b"])
        for a,b,s in pairs: w.writerow([a,b,s,float(mass[a]),float(mass[b])])
    result={"status":"completed","day":report["day"],"rows":len(assigned),"representatives":n,
            "threshold":threshold,"structural_mapping_unchanged":True,"representatives_removed":0,
            "before":summarize(own),"after":summarize(score),
            "valid_coverage":float(valid.mean()),"coverage_target_met":bool(valid.mean()>.999),
            "changed_valid_rows":int(((target!=assigned)&valid).sum()),
            "rescued_rows":int(((own<=threshold)&valid).sum()),"uncovered_rows":int((~valid).sum()),
            "representatives_with_at_least_1000_valid_rows":int((covered>=1000).sum()),
            "zero_valid_mass_representatives":int((covered==0).sum()),
            "redundancy_candidates":int(redundant.sum()),"redundancy_candidate_pairs":len(pairs),
            "redundancy_threshold":redundancy_threshold,
            "rare_behavior":rare,"runtime_seconds":time.perf_counter()-start,
            "limitations":["Uses existing exact scores; does not optimize or add representatives.",
             "Rare means low structural mass, not independently established behavioral rarity.",
             "Redundancy is a nearest-neighbour candidate flag only; no deletion or automatic merge.",
             "Pair audit is nearest-neighbour based, not exhaustive all-pairs enumeration.",
             "Row indices follow original chronological evaluation order."]}
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--threshold",type=float,default=.85); p.add_argument("--rare-max",type=int,default=10)
    p.add_argument("--redundancy-threshold",type=float,default=.999)
    a=p.parse_args(); run(a.input,a.output,a.threshold,a.rare_max,a.redundancy_threshold)
if __name__=="__main__": main()
