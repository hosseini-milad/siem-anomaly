import unittest
import numpy as np
from graph.sparse import build_graph
class GraphTest(unittest.TestCase):
    def test_symmetry_cap_threshold_isolate(self):
        x=np.array([[1.,0.]]*5+[[0.,0.]])
        b=np.zeros((6,11));c=np.zeros((6,3))
        g,_=build_graph(x,b,c,k=3,cap=2,block=2)
        self.assertEqual((g-g.T).nnz,0)
        self.assertFalse(g.diagonal().any())
        self.assertLessEqual(np.diff(g.indptr).max(),2)
        self.assertEqual(g[5].nnz,0)
        self.assertTrue((g.data>.85).all())
        self.assertGreater(g.nnz,0)

    def test_policy_and_indexed_invariants(self):
        from graph.sparse import build_indexed_graph
        x=np.random.default_rng(42).normal(size=(80,11))
        b=np.zeros((80,11));c=np.zeros((80,3))
        for builder in (build_graph,build_indexed_graph):
            g,_=builder(x,b,c,k=4,cap=3,threshold=.5)
            self.assertEqual((g-g.T).nnz,0)
            self.assertFalse(g.diagonal().any())
            self.assertLessEqual(np.diff(g.indptr).max(),3)
            self.assertTrue((g.data>.5).all())
    def test_label_summary(self):
        from graph.pilot import label_summary
        r=label_summary(np.array([0,1,0]),np.array([True,False,False]),np.array([2,1,0]),np.array([3,2,1]))
        self.assertEqual(r["attack_fraction_of_isolates"],.5)
        self.assertEqual(r["attack_fraction_of_isolated_members"],.5)
