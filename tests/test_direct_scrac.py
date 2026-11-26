import numpy as np
from graph.direct_scrac import direct_scrac
from graph.sc_fagc import mixed_similarity

def test_direct_scrac_conserves_and_respects_complete_link():
    x = np.array([[1.,0.],[.8,.6],[.28,.96],[-1.,0.]])
    b = c = np.zeros((4,1), dtype=int)
    mass = np.array([2,3,5,7])
    labels, stats = direct_scrac(x,b,c,mass,capacity=2,lambda_size=0.)
    assert stats["represented_raw_rows"] == 17
    assert stats["capacity_violations"] == 0
    assert stats["groups"] == 3
    for gid in np.unique(labels):
        ids = np.flatnonzero(labels == gid)
        if len(ids)>1:
            assert mixed_similarity(x[ids],b[ids],c[ids],x[ids],b[ids],c[ids]).min() > .85

def test_direct_scrac_topology_and_capacity():
    x = np.ones((6,2))
    b = c = np.zeros((6,1), dtype=int)
    topology = np.array([0,0,0,1,1,1])
    labels, stats = direct_scrac(x,b,c,np.ones(6),capacity=2,topology_labels=topology)
    assert stats["max_group_size"] <= 2
    assert stats["groups"] == 4
    for gid in np.unique(labels):
        assert len(np.unique(topology[labels == gid])) == 1


def test_cubic_penalty_rejects_otherwise_eligible_pair():
    x = np.array([[1.,0.],[.8,.6]])
    b = c = np.zeros((2,1), dtype=int)
    mass = np.ones(2)
    _, unpenalized = direct_scrac(x,b,c,mass,capacity=2,lambda_size=0.)
    _, penalized = direct_scrac(x,b,c,mass,capacity=2,lambda_size=.20)
    assert unpenalized["groups"] == 1
    assert penalized["groups"] == 2
