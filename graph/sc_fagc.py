"""Component-level SC-FAGC adaptation for sparse behavioral graph summarization.
Input rows are existing graph components (structural representatives), not raw
flows. The paper's IRW objective and DB-OT solver are retained. Component mass
weights transport marginals; final assignment reports indivisible oversized components.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

def _validate_features(x,b,c,mass):
    x=np.asarray(x,dtype=np.float64); b=np.asarray(b); c=np.asarray(c); mass=np.asarray(mass,dtype=np.float64); n=len(x)
    if x.ndim!=2 or b.ndim!=2 or c.ndim!=2 or len(b)!=n or len(c)!=n or mass.shape!=(n,): raise ValueError("feature and mass arrays must have aligned rows")
    if n==0 or not np.isfinite(x).all() or not np.isfinite(mass).all() or np.any(mass<=0): raise ValueError("empty, non-finite, or non-positive component input")
    if b.shape[1]+c.shape[1]==0: raise ValueError("at least one discrete feature is required")
    return x,b,c,mass

def mixed_similarity(xa,ba,ca,xb,bb,cb):
    xa=np.asarray(xa,dtype=np.float64); xb=np.asarray(xb,dtype=np.float64)
    na=np.linalg.norm(xa,axis=1); nb=np.linalg.norm(xb,axis=1); den=na[:,None]*nb[None,:]
    cosine=np.divide(xa@xb.T,den,out=np.zeros((len(xa),len(xb))),where=den>0)
    da=np.column_stack((ba,ca)); db=np.column_stack((bb,cb))
    if da.shape[1]!=db.shape[1]: raise ValueError("discrete feature dimensions differ")
    categorical=np.mean(da[:,None,:]==db[None,:,:],axis=2)
    return np.clip(.6*cosine+.4*categorical,-1.,1.)

def select_anchor_indices(x,b,c,n_anchors,seed=42):
    """Deterministic farthest-first medoids; BKHK is a replaceable paper component."""
    n=len(x); m=min(int(n_anchors),n)
    if m<1: raise ValueError("n_anchors must be positive")
    center=np.mean(x,axis=0); first=int(np.argmin(np.linalg.norm(x-center,axis=1)))
    chosen=[first]; best=np.full(n,-np.inf,dtype=np.float64); best[first]=np.inf
    while len(chosen)<m:
        last=chosen[-1]
        sim=mixed_similarity(x,b,c,x[last:last+1],b[last:last+1],c[last:last+1]).ravel()
        best=np.maximum(best,sim); best[np.asarray(chosen,dtype=int)]=np.inf
        candidate=int(np.argmin(best))
        if best[candidate]>=1.-1e-12: break
        chosen.append(candidate)
    return np.asarray(chosen,dtype=np.int64)

def build_bipartite(x,b,c,anchor_indices,k=5,block=512):
    """Parameter-free k-nearest-anchor assignment from SC-FAGC Eq. (3)."""
    n=len(x); anchors=np.asarray(anchor_indices,dtype=np.int64); m=len(anchors)
    if anchors.ndim!=1 or m==0 or np.any(anchors<0) or np.any(anchors>=n): raise ValueError("invalid anchor indices")
    if len(np.unique(anchors))!=m or k<1 or block<1: raise ValueError("anchors must be unique and k/block positive")
    kk=min(int(k),m-1) if m>1 else 1; rows=[]; cols=[]; vals=[]; own={int(a):j for j,a in enumerate(anchors)}
    for start in range(0,n,block):
        end=min(n,start+block)
        sim=mixed_similarity(x[start:end],b[start:end],c[start:end],x[anchors],b[anchors],c[anchors]); dist=2.*(1.-sim)
        for local in range(end-start):
            i=start+local
            if i in own:
                rows.append(i); cols.append(own[i]); vals.append(1.); continue
            order=np.lexsort((anchors,dist[local])); active=order[:kk]
            cutoff=dist[local,order[kk]] if m>kk else dist[local,order[-1]]+1.
            weights=np.maximum(cutoff-dist[local,active],0.); denom=float(weights.sum())
            if denom<=1e-15: weights[:]=1.; denom=float(len(weights))
            rows.extend([i]*len(active)); cols.extend(active.tolist()); vals.extend((weights/denom).tolist())
    B=csr_matrix((vals,(rows,cols)),shape=(n,m)); B.eliminate_zeros(); return B

def _sinkhorn_double_bounded(D,row_mass,lower,upper,epsilon,max_iter,tol):
    """Weighted DB-OT subproblem with exact row marginals and bounded columns."""
    m,c=D.shape
    if np.any(row_mass<=0) or lower<0 or upper<=0 or lower>upper: raise ValueError("invalid DB-OT marginals")
    total=float(row_mass.sum())
    if c*lower>total+tol or c*upper<total-tol: raise ValueError("infeasible bilateral cluster-mass bounds")
    from scipy.special import logsumexp
    logK=(D-D.max(axis=1,keepdims=True))/epsilon
    logw=np.log(row_mass); q=np.zeros(c); v=np.zeros(c)
    loglo=np.log(lower) if lower>0 else -np.inf; loghi=np.log(upper)
    for it in range(max_iter):
        u=logw-logsumexp(logK+(q+v)[None,:],axis=1)
        col=logsumexp(logK+u[:,None],axis=0)
        q=np.maximum(loglo-col-v,0.)
        v=np.minimum(loghi-col-q,0.)
        F=np.exp(logK+u[:,None]+(q+v)[None,:])
        cs=F.sum(axis=0)
        re=float(np.max(np.abs(F.sum(axis=1)/row_mass-1.)))
        ce=float(max(np.maximum(lower-cs,0).max(),np.maximum(cs-upper,0).max())/max(1.,total))
        if max(re,ce)<=tol: break
    return F/row_mass[:,None],{"sinkhorn_iterations":it+1,"row_mass_max_error":float(np.max(np.abs(F.sum(axis=1)-row_mass))),"row_relative_error":re,"column_relative_error":ce,"column_mass_min":float(cs.min()),"column_mass_max":float(cs.max()),"sinkhorn_converged":bool(max(re,ce)<=tol)}

def sc_fagc(B,anchor_mass,n_clusters,lower_mass,upper_mass,epsilon=.01,max_iter=100,sinkhorn_iter=1000,tol=1e-8):
    """Maximize Tr(U.T (B.T B) U)+epsilon H(U) using IRW/MM and DB-OT."""
    B=csr_matrix(B,dtype=np.float64); w=np.asarray(anchor_mass,dtype=np.float64); m=B.shape[1]; c=int(n_clusters)
    if B.shape[0]==0 or m==0 or w.shape!=(m,) or c<1: raise ValueError("invalid SC-FAGC dimensions")
    if not np.isfinite(B.data).all() or np.any(B.data<0): raise ValueError("B must be finite and nonnegative")
    if not np.allclose(np.asarray(B.sum(axis=1)).ravel(),1.,atol=1e-8): raise ValueError("B rows must sum to one")
    if epsilon<=0 or max_iter<1 or sinkhorn_iter<1: raise ValueError("invalid optimization parameters")
    A=(B.T@B).tocsr(); V=np.random.default_rng(42).normal(0,.01,(m,c))
    if V.shape[1]<c: V=np.pad(V,((0,0),(0,c-V.shape[1])))
    U=np.exp(V-V.max(axis=1,keepdims=True)); U/=U.sum(axis=1,keepdims=True); history=[]; last_sink={}; delta=float('inf')
    U,last_sink=_sinkhorn_double_bounded(np.log(U)*epsilon,w,lower_mass,upper_mass,epsilon,sinkhorn_iter,tol)
    if not last_sink['sinkhorn_converged']: return U,{'objective_history':[], 'iterations':0,'converged':False,**last_sink}
    def objective(T): return float(np.sum(T*(A@T))-epsilon*np.sum(w[:,None]*T*np.log(np.maximum(T,1e-300))))
    history.append(objective(U))
    for _ in range(max_iter):
        D=2.*(A@U)/w[:,None]; U_new,last_sink=_sinkhorn_double_bounded(D,w,float(lower_mass),float(upper_mass),epsilon,sinkhorn_iter,tol)
        if not last_sink['sinkhorn_converged']: break
        value=objective(U_new)
        if value<history[-1]-1e-8*max(1.,abs(history[-1])): break
        history.append(value)
        delta=float(np.max(np.abs(U_new-U))); U=U_new
        if delta<=tol: break
    return U,{"objective_history":history,"iterations":len(history),"converged":bool(delta<=tol and last_sink.get("sinkhorn_converged",False)),"anchor_mass":w.tolist(),**last_sink}

def capacity_assign(probabilities,component_mass,max_size=100):
    """Assign whole components; never split an oversized component."""
    P=np.asarray(probabilities,dtype=np.float64); mass=np.asarray(component_mass,dtype=np.int64)
    if P.ndim!=2 or mass.shape!=(P.shape[0],) or np.any(mass<=0): raise ValueError("invalid probabilities or component masses")
    if max_size<1 or not np.isfinite(P).all(): raise ValueError("invalid capacity or probabilities")
    n,c=P.shape; target=min(n,max(1,int(np.ceil(mass.sum()/max_size))))
    labels=np.full(n,-1,dtype=np.int64); loads=np.zeros(target,dtype=np.int64); oversized=[]
    sorted_p=np.sort(P,axis=1); confidence=sorted_p[:,-1]-(sorted_p[:,-2] if c>1 else 0.)
    order=np.lexsort((np.arange(n),confidence,-mass))
    for i in order:
        if mass[i]>max_size: oversized.append(int(i)); continue
        pref=np.argsort(-P[i],kind="stable"); candidates=[j for j in pref if j<target and loads[j]+mass[i]<=max_size]
        if not candidates: loads=np.append(loads,0); candidates=[len(loads)-1]
        j=int(candidates[0]); labels[i]=j; loads[j]+=mass[i]
    assignable_loads=loads.copy()
    for i in oversized: labels[i]=len(loads); loads=np.append(loads,mass[i])
    return labels,{"cluster_count":int(len(loads)),"cluster_mass":loads.tolist(),"max_cluster_mass":int(loads.max(initial=0)),
                   "oversized_component_indices":oversized,"capacity_respected_for_assignable_components":bool(np.all(assignable_loads<=max_size))}


def size_penalty(new_raw_mass,lambda1=.02,lambda2=.10):
    """Approved SCRAC penalty on proposed represented raw-row mass."""
    n=float(new_raw_mass)
    return float(lambda1*np.log1p(n/50.)+lambda2*(max(0.,n-100.)/100.)**2)

def scrac_assign(x,b,c,component_mass,probabilities,anchor_indices,anchor_assignments,threshold=.85):
    """Complete-Link plus size-regularized assignment; every component is retained."""
    P=np.asarray(probabilities,dtype=np.float64); mass=np.asarray(component_mass,dtype=np.float64)
    U=np.asarray(anchor_assignments,dtype=np.float64); anchors=np.asarray(anchor_indices,dtype=np.int64); n=len(x)
    if P.shape[0]!=n or U.shape[0]!=len(anchors) or P.shape[1]!=U.shape[1] or mass.shape!=(n,):
        raise ValueError("SCRAC assignment inputs are not aligned")
    groups=[]; seeds=[]; used=set()
    for j in range(P.shape[1]):
        for ai in np.argsort(-U[:,j],kind="stable"):
            candidate=int(anchors[ai])
            if candidate not in used:
                used.add(candidate); seeds.append(candidate); groups.append([candidate]); break
    if not groups: raise ValueError("SC-FAGC did not provide any group anchors")
    remaining=np.asarray([i for i in range(n) if i not in used],dtype=np.int64)
    confidence=np.max(P,axis=1); remaining=remaining[np.argsort(-confidence[remaining],kind="stable")]
    unassigned=[]; records=[]
    def score(i,gid):
        members=np.asarray(groups[gid],dtype=np.int64)
        sim=mixed_similarity(x[i:i+1],b[i:i+1],c[i:i+1],x[members],b[members],c[members]).ravel()
        smin=float(sim.min())
        if not smin>threshold: return None
        smean=float(sim.mean())
        sa=float(mixed_similarity(x[i:i+1],b[i:i+1],c[i:i+1],x[seeds[gid]:seeds[gid]+1],
                                  b[seeds[gid]:seeds[gid]+1],c[seeds[gid]:seeds[gid]+1])[0,0])
        new_mass=float(mass[members].sum()+mass[i]); penalty=size_penalty(len(members)+1)
        return .4*sa+.3*smean+.3*smin-penalty,{"anchor":sa,"mean":smean,"minimum":smin,"new_raw_mass":new_mass,"penalty":penalty}
    for i in remaining:
        options=[]
        for gid in range(len(groups)):
            item=score(int(i),int(gid))
            if item is not None: options.append((item[0],int(gid),item[1]))
        if options and max(z[0] for z in options)>=.75:
            options.sort(key=lambda z:(-z[0],z[1])); value,gid,parts=options[0]
            groups[gid].append(int(i)); records.append({"component":int(i),"group":gid,"score":float(value),**parts})
        else:
            groups.append([int(i)]); seeds.append(int(i)); unassigned.append(int(i))
    recovered=[]
    recovered=[i for i in unassigned if len(groups[seeds.index(i)])==1]
    labels=np.empty(n,dtype=np.int64)
    for gid,members in enumerate(groups): labels[np.asarray(members,dtype=np.int64)]=gid
    group_mass=[int(mass[np.asarray(g,dtype=np.int64)].sum()) for g in groups]
    return labels,{"groups":len(groups),"group_mass":group_mass,"group_sizes":[len(g) for g in groups],"opened_groups":len(unassigned),"max_group_mass":max(group_mass,default=0),
      "oversized_groups":int(sum(v>100 for v in group_mass)),"recovered_singleton_components":recovered,
      "recovered_singleton_count":len(recovered),"complete_link_threshold":float(threshold),"comparison":"strict_greater_than",
      "size_penalty":"0.02*ln(1+n/50)+0.10*(max(0,n-100)/100)^2",
      "score":"0.4*S_anchor+0.3*S_mean+0.3*S_min-P(new_group_size)","score_records":records}


def run(x,b,c,component_mass,n_anchors=None,n_clusters=None,max_component_group_mass=100,anchor_k=5,epsilon=.01,max_iter=100,sinkhorn_iter=1000,seed=42):
    x,b,c,mass=_validate_features(x,b,c,component_mass); n=len(x)
    # Cluster count is a controllable SC-FAGC hyperparameter; sqrt(n) is only
    # a bounded default heuristic. Final raw-mass capacity is enforced below.
    n_clusters=n_clusters or min(n,32,max(1,int(np.ceil(np.sqrt(n)))))
    if n_clusters>n: raise ValueError("n_clusters cannot exceed number of components")
    n_anchors=min(n,max(int(n_anchors or np.ceil(.05*n)),n_clusters))
    anchors=select_anchor_indices(x,b,c,n_anchors,seed); B=build_bipartite(x,b,c,anchors,k=anchor_k)
    anchor_mass=np.asarray(B.T@mass).ravel()
    requested_anchors=n_anchors; selected_anchors=len(anchors); requested_clusters=n_clusters
    active=anchor_mass>0
    if not np.any(active): raise ValueError("no positive-mass anchors")
    B=B[:,active].tocsr(); anchors=anchors[active]; anchor_mass=anchor_mass[active]
    if not np.allclose(np.asarray(B.sum(axis=1)).ravel(),1.) or not np.isclose(anchor_mass.sum(),mass.sum()):
        raise ValueError("anchor filtering violated mass conservation")
    n_clusters=min(n_clusters,len(anchors))
    target=float(mass.sum()/n_clusters); lower=.4*target; upper=max(target,float(max_component_group_mass))
    U,opt=sc_fagc(B,anchor_mass,n_clusters,lower,upper,epsilon,max_iter,sinkhorn_iter)
    if not opt['converged']:
        return {'components':n,'anchors':len(anchors),'anchors_requested':requested_anchors,'zero_mass_anchors_removed':selected_anchors-len(anchors),'redundant_anchors_not_selected':requested_anchors-selected_anchors,'optimization':opt,'status':'optimization_not_validated','assignment':list(range(n)),'probabilities':np.asarray(B@U).tolist(),'anchor_indices':anchors.tolist()}
    probabilities=np.asarray(B@U); labels,scrac=scrac_assign(x,b,c,mass,probabilities,anchors,U)
    return {"components":n,"anchors":len(anchors),"clusters_requested":requested_clusters,"clusters_effective":n_clusters,"anchors_requested":requested_anchors,"zero_mass_anchors_removed":selected_anchors-len(anchors),"redundant_anchors_not_selected":requested_anchors-selected_anchors,"anchor_external_support":(np.asarray(B.getnnz(axis=0)).ravel()-1).tolist(),"represented_raw_rows":int(mass.sum()),
            "component_mass_max":int(mass.max()),"anchor_mass_lower_bound":lower,"anchor_mass_upper_bound":upper,
            "optimization":opt,"scrac_assignment":scrac,"unassigned_components":int(np.count_nonzero(labels<0)),
            "assignment":labels.tolist(),"probabilities":probabilities.tolist(),"anchor_indices":anchors.tolist(),
            "method_notes":["SC-FAGC IRW and entropy-regularized DB-OT operate on existing graph components.",
             "Deterministic farthest-first medoids replace the paper's BKHK anchor generator; this is an explicit adaptation.",
             "SCRAC Complete-Link requires minimum mixed similarity strictly above 0.85.",
             "Size penalty is 0.02*ln(1+n/50)+0.10*(max(0,n-100)/100)^2 on proposed group member count.",
             "Unmatched components are retained as standalone groups; no source component is discarded."]}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--anchors",type=int,default=None); p.add_argument("--clusters",type=int,default=None); p.add_argument("--capacity",type=int,default=100)
    p.add_argument("--anchor-k",type=int,default=5); p.add_argument("--epsilon",type=float,default=.01); p.add_argument("--max-iter",type=int,default=100); p.add_argument("--sinkhorn-iter",type=int,default=1000)
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as f:
        result=run(f["x"],f["b"],f["c"],f["component_mass"],n_anchors=a.anchors,n_clusters=a.clusters,
                   max_component_group_mass=a.capacity,anchor_k=a.anchor_k,epsilon=a.epsilon,max_iter=a.max_iter,sinkhorn_iter=a.sinkhorn_iter)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("assignment","probabilities","anchor_indices")},indent=2))

if __name__=="__main__": main()
