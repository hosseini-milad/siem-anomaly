import numpy as np
from scipy.sparse import csr_matrix
from evaluation.topology import audit
from evaluation.representation import nearest,summarize

def test_quotient_weight():
    a=csr_matrix([[0.,1.,0.],[1.,0.,2.],[0.,2.,0.]])
    q,s=audit(a,np.array([0,0,1]),2)
    assert q.sum()==a.sum()
    assert q[0,0]==2 and q[0,1]==2
    assert s["cross_original_component_groups"]==0

def test_nearest_and_assigned():
    x=np.eye(2); b=c=np.zeros((2,1))
    best,own,ids=nearest((x,b,c),(x,b,c),np.array([1,0]),block=1)
    assert np.allclose(best,1) and np.allclose(own,.4)
    assert np.array_equal(ids,[0,1])
    assert summarize(best)["coverage"]["0.85"]==1
