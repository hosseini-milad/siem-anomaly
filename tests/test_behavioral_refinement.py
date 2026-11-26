import numpy as np
from refinement.behavioral_mapping import refine
from refinement.rare_behavior import audit
from refinement.redundancy import audit as redundancy, candidate_pairs

def test_refinement_ties_and_uncovered():
    a=np.array([0,1,0]); n=np.array([1,0,1])
    t,s,v=refine(a,n,[.7,.95,.6],[.9,.95,.8],2)
    assert t.tolist()==[1,1,1]
    assert v.tolist()==[True,True,False]
    assert a.tolist()==[0,1,0]
    r,original,mapped,covered,rare=audit(a,t,s,v,2)
    assert original.sum()==mapped.sum()==3 and covered.sum()==2
    assert r["uncovered_rows"]==1

def test_redundancy_flags_both_members_even_with_mass():
    x=np.array([[1.,0.],[1.,0.],[0.,1.]])
    b=c=np.zeros((3,1))
    nearest,score,flag=redundancy((x,b,c),np.array([3,7,2]),threshold=.999)
    assert flag.tolist()==[True,True,False]
    assert candidate_pairs(nearest,score,.999)==[(0,1,1.0)]

def test_redundancy_empty_and_singleton():
    x=np.array([[1.,0.]])
    b=c=np.zeros((1,1))
    nearest,score,flag=redundancy((x,b,c),np.array([5]))
    assert nearest.tolist()==[-1] and score.tolist()==[-1.] and not flag.any()
