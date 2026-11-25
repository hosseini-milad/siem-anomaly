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
    chosen=[first]; best=np.ones(n,dtype=np.float64); best[first]=np.inf
    while len(chosen)<m:
        last=chosen[-1]
        sim=mixed_similarity(x,b,c,x[last:last+1],b[last:last+1],c[last:last+1]).ravel()
        best=np.minimum(best,sim); best[np.asarray(chosen,dtype=int)]=np.inf
        chosen.append(int(np.argmin(best)))
    return np.asarray(chosen,dtype=np.int64)

def build_bipartite(x,b,c,anchor_indices,k=5,block=512):
    """Parameter-free k-nearest-anchor assignment from SC-FAGC Eq. (3)."""
    n=len(x); anchors=np.asarray(anchor_indices,dtype=np.int64); m=len(anchors)
    if anchors.ndim!=1 or m==0 or np.any(anchors<0) or np.any(anchors>=n): raise ValueError("invalid anchor indices")
    if len(np.unique(anchors))!=m or k<1 or block<1: raise ValueError("anchors must be unique and k/block positive")
    kk=min(int(k),m-1) if m>1 else 1; B=np.zeros((n,m),dtype=np.float64)
    for start in range(0,n,block):
        end=min(n,start+block)
        sim=mixed_similarity(x[start:end],b[start:end],c[start:end],x[anchors],b[anchors],c[anchors]); dist=2.*(1.-sim)
        for local in range(end-start):
            order=np.lexsort((anchors,dist[local])); active=order[:kk]
            cutoff=dist[local,order[kk]] if m>kk else dist[local,order[-1]]+1.
            weights=np.maximum(cutoff-dist[local,active],0.); denom=float(weights.sum())
            if denom<=1e-15: weights[:]=0.; weights[0]=1.; denom=1.
            B[start+local,active]=weights/denom
    return csr_matrix(B)

def _sinkhorn_double_bounded(D,row_mass,lower,upper,epsilon,max_iter,tol):
    """Weighted DB-OT subproblem with exact row marginals and bounded columns."""
    m,c=D.shape
    if np.any(row_mass<=0) or lower<0 or upper<=0 or lower>upper: raise ValueError("invalid DB-OT marginals")
    total=float(row_mass.sum())
    if c*lower>total+tol or c*upper<total-tol: raise ValueError("infeasible bilateral cluster-mass bounds")
    logits=(D-np.max(D,axis=1,keepdims=True))/epsilon; K=np.exp(np.clip(logits,-700.,0.))
    u=np.ones(m); q=np.ones(c); v=np.ones(c); tiny=np.finfo(float).tiny
    for it in range(max_iter):
        u=row_mass/np.maximum(K@(q*v),tiny); col=K.T@u
        q=np.maximum(lower/np.maximum(col*v,tiny),1.); col=K.T@u
        v=np.minimum(upper/np.maximum(col*q,tiny),1.)
        if it%5==0 or it==max_iter-1:
            F=(u[:,None]*K)*(q*v)[None,:]; re=float(np.max(np.abs(F.sum(axis=1)-row_mass))); cs=F.sum(axis=0)
            ce=max(float(np.maximum(lower-cs,0).max()),float(np.maximum(cs-upper,0).max()))
            if max(re,ce)<=tol*max(1.,total): break
    F=(u[:,None]*K)*(q*v)[None,:]; U=F/row_mass[:,None]
    return U,{"sinkhorn_iterations":it+1,"row_mass_max_error":float(np.max(np.abs(F.sum(axis=1)-row_mass))),
              "column_mass_min":float(F.sum(axis=0).min()),"column_mass_max":float(F.sum(axis=0).max())}

def sc_fagc(B,anchor_mass,n_clusters,lower_mass,upper_mass,epsilon=.01,max_iter=100,sinkhorn_iter=1000,tol=1e-8):
    """Maximize Tr(U.T (B.T B) U)+epsilon H(U) using IRW/MM and DB-OT."""
    B=csr_matrix(B,dtype=np.float64); w=np.asarray(anchor_mass,dtype=np.float64); m=B.shape[1]; c=int(n_clusters)
    if B.shape[0]==0 or m==0 or w.shape!=(m,) or c<1: raise ValueError("invalid SC-FAGC dimensions")
    if not np.isfinite(B.data).all() or np.any(B.data<0): raise ValueError("B must be finite and nonnegative")
    if not np.allclose(np.asarray(B.sum(axis=1)).ravel(),1.,atol=1e-8): raise ValueError("B rows must sum to one")
    if epsilon<=0 or max_iter<1 or sinkhorn_iter<1: raise ValueError("invalid optimization parameters")
    A=(B.T@B).toarray(); vals,vecs=np.linalg.eigh(A); order=np.argsort(vals)[::-1][:min(c,m)]; V=vecs[:,order]
    if V.shape[1]<c: V=np.pad(V,((0,0),(0,c-V.shape[1])))
    U=np.exp(V-V.max(axis=1,keepdims=True)); U/=U.sum(axis=1,keepdims=True); history=[]; last_sink={}
    for _ in range(max_iter):
        D=2.*(A@U); U_new,last_sink=_sinkhorn_double_bounded(D,w,float(lower_mass),float(upper_mass),epsilon,sinkhorn_iter,tol)
        entropy=-float(np.sum(np.where(U_new>0,U_new*np.log(np.maximum(U_new,1e-300)),0.)))
        history.append(float(np.trace(U_new.T@A@U_new)+epsilon*entropy))
        delta=float(np.max(np.abs(U_new-U))); U=U_new
        if delta<=tol: break
    return U,{"objective_history":history,"iterations":len(history),"converged":bool(delta<=tol),"anchor_mass":w.tolist(),**last_sink}

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

def run(x,b,c,component_mass,n_anchors=None,n_clusters=None,max_component_group_mass=100,anchor_k=5,epsilon=.01,max_iter=100,sinkhorn_iter=1000,seed=42):
    x,b,c,mass=_validate_features(x,b,c,component_mass); n=len(x)
    n_clusters=n_clusters or max(1,int(np.ceil(mass.sum()/max_component_group_mass)))
    if n_clusters>n: raise ValueError("n_clusters cannot exceed number of components")
    n_anchors=min(n,max(int(n_anchors or np.ceil(np.sqrt(n))),n_clusters))
    anchors=select_anchor_indices(x,b,c,n_anchors,seed); B=build_bipartite(x,b,c,anchors,k=anchor_k)
    anchor_mass=np.asarray(B.T@mass).ravel(); target=float(mass.sum()/n_clusters); lower=.4*target; upper=max(target,float(max_component_group_mass))
    U,opt=sc_fagc(B,anchor_mass,n_clusters,lower,upper,epsilon,max_iter,sinkhorn_iter)
    probabilities=np.asarray(B@U); labels,cap=capacity_assign(probabilities,mass,max_component_group_mass)
    return {"components":n,"anchors":len(anchors),"clusters_requested":n_clusters,"represented_raw_rows":int(mass.sum()),
            "component_mass_max":int(mass.max()),"anchor_mass_lower_bound":lower,"anchor_mass_upper_bound":upper,
            "optimization":opt,"capacity":cap,"unassigned_components":int(np.count_nonzero(labels<0)),
            "assignment":labels.tolist(),"probabilities":probabilities.tolist(),"anchor_indices":anchors.tolist(),
            "method_notes":["SC-FAGC IRW and entropy-regularized DB-OT operate on existing graph components.",
             "Deterministic farthest-first medoids replace the paper's BKHK anchor generator; this is an explicit adaptation.",
             "Anchor-level mass bounds are exact for the OT subproblem; final raw-mass capacity is enforced separately.",
             "Components individually larger than capacity remain intact and are reported as oversized."]}

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
