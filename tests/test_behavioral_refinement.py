import numpy as np
from refinement.behavioral_mapping import refine
from refinement.rare_behavior import audit
from refinement.redundancy import audit as redundancy

def test_refinement_ties_and_uncovered():
    a=np.array([0,1,0]); n=np.array([1,0,1])
    t,s,v=refine(a,n,[.7,.95,.6],[.9,.95,.8],2)
    assert t.tolist()==[1,1,1]
    assert v.tolist()==[True,True,False]
    assert a.tolist()==[0,1,0]
    r,original,mapped,covered,rare=audit(a,t,s,v,2)
    assert original.sum()==mapped.sum()==3 and covered.sum()==2
    assert r["uncovered_rows"]==1

def test_redundancy_is_only_flag():
    x=np.array([[1.,0.],[1.,0.]])
    b=c=np.zeros((2,1))
    _,_,flag=redundancy((x,b,c),np.array([3,0]))
    assert flag.tolist()==[False,True]
