import numpy as np
import pytest
from scipy.sparse import csr_matrix
from graph.sc_fagc import build_bipartite,sc_fagc,capacity_assign,run,expand_anchors

def toy():
    x=np.array([[1.,0.],[.99,.01],[0.,1.],[.01,.99],[-1.,0.],[-.99,.01]])
    b=np.array([[1],[1],[0],[0],[1],[1]]); c=np.array([[0],[0],[1],[1],[2],[2]])
    return x,b,c,np.array([20,15,20,15,20,10])

def test_bipartite_rows_are_probabilities():
    x,b,c,_=toy(); B=build_bipartite(x,b,c,np.array([0,2,4]),k=2)
    assert np.allclose(np.asarray(B.sum(axis=1)).ravel(),1.) and np.all(B.data>=0)

def test_sc_fagc_mass_bounds_and_row_simplex():
    B=csr_matrix([[1.,0.],[.8,.2],[0.,1.]]); w=np.array([50.,50.])
    U,info=sc_fagc(B,w,2,40.,60.,max_iter=20); col=w@U
    assert np.allclose(U.sum(axis=1),1.,atol=1e-7)
    assert np.all(col>=40.-1e-5) and np.all(col<=60.+1e-5) and info["iterations"]>=1

def test_capacity_assignment_respects_cap_when_feasible():
    p=np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8]]); labels,info=capacity_assign(p,np.array([30,20,25,25]),50)
    assert np.all(labels>=0) and max(info["cluster_mass"])<=50 and info["capacity_respected_for_assignable_components"]

def test_oversized_component_is_not_split():
    p=np.array([[.9,.1],[.1,.9]]); labels,info=capacity_assign(p,np.array([120,20]),100)
    assert labels[0]!=labels[1] and info["oversized_component_indices"]==[0]

def test_run_end_to_end():
    x,b,c,mass=toy(); result=run(x,b,c,mass,n_anchors=4,n_clusters=2,max_component_group_mass=100,max_iter=10,sinkhorn_iter=500)
    assert result["components"]==6 and len(result["assignment"])==6
    assert len(result["probabilities"])==6 and result["represented_raw_rows"]==100

def test_rejects_infeasible_bounds():
    with pytest.raises(ValueError,match="infeasible"): sc_fagc(csr_matrix(np.eye(2)),np.array([10.,10.]),2,11.,12.)

def test_component_aggregation(tmp_path):
    from graph.component_sc_fagc import aggregate_components
    nodes=tmp_path/"nodes.npz"; comps=tmp_path/"components.npz"
    np.savez(nodes,x=np.array([[1.,0.],[.9,.1],[0.,1.]]),b=np.array([[1],[1],[0]]),c=np.array([[0],[0],[1]]))
    np.savez(comps,component_id=np.array([0,0,1]),raw_rows=np.array([25,10]))
    x,b,c,mass=aggregate_components(nodes,comps)
    assert x.shape==(2,2) and b.ravel().tolist()==[1,0] and c.ravel().tolist()==[0,1] and mass.tolist()==[25,10]


def test_size_penalty_piecewise_formula():
    from graph.sc_fagc import size_penalty
    assert size_penalty(50)==pytest.approx(.02*np.log1p(1.))
    assert size_penalty(200)==pytest.approx(.02*np.log1p(4.)+.10)


def test_scrac_complete_link_and_total_mapping():
    from graph.sc_fagc import scrac_assign
    x,b,c,mass=toy()
    probabilities=np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8],[.7,.3],[.6,.4]])
    U=np.array([[.9,.1],[.1,.9]])
    labels,report=scrac_assign(x,b,c,mass,probabilities,np.array([0,2]),U,threshold=.85)
    assert np.all(labels>=0) and len(labels)==len(x)
    assert report["comparison"]=="strict_greater_than"
    assert report["recovered_singleton_count"]>=0


def test_irw_objective_is_monotone_on_toy_data():
    x,b,c,mass=toy()
    result=run(x,b,c,mass,n_anchors=4,n_clusters=2,max_iter=30,sinkhorn_iter=1000)
    history=np.asarray(result["optimization"]["objective_history"])
    assert len(history)>=2
    assert np.all(np.diff(history)>=-1e-7)


def test_zero_mass_anchors():
    x=np.tile([1.,0.],(6,1)); b=np.ones((6,1)); c=np.zeros((6,1))
    result=run(x,b,c,np.ones(6,dtype=int),n_anchors=4,n_clusters=3,max_iter=5)
    assert result["zero_mass_anchors_removed"]==0
    assert result["anchors"]==1
    assert result["represented_raw_rows"]==6
    assert result["unassigned_components"]==0


def test_anchor_self_support_and_ties():
    x,b,c,_=toy()
    initial=np.array([0,1,2])
    anchors,expansion=expand_anchors(x,b,c,initial,threshold=.85)
    B=build_bipartite(x,b,c,anchors,k=2,threshold=.85)
    assert np.allclose(B[anchors].toarray(),np.eye(len(anchors)))
    assert np.allclose(np.asarray(B.sum(axis=1)).ravel(),1.)
    assert expansion["final_anchor_count"]==len(anchors)


def test_adaptive_anchor_expansion_creates_anchor_for_uncovered_component():
    x,b,c,_=toy()
    anchors,report=expand_anchors(x,b,c,np.array([0]),threshold=.85)
    assert 2 in anchors and 4 in anchors
    assert report["components_requiring_new_anchor"]>=1
    assert report["expanded_anchor_count"]==len(anchors)-1
    B=build_bipartite(x,b,c,anchors,k=2,threshold=.85)
    assert np.allclose(np.asarray(B.sum(axis=1)).ravel(),1.)


def test_adaptive_anchor_expansion_keeps_initial_anchor_when_covered():
    x,b,c,_=toy()
    anchors,report=expand_anchors(x,b,c,np.array([0,2,4]),threshold=.85)
    assert len(anchors)==3
    assert report["expanded_anchor_count"]==0
