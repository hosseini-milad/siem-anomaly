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
