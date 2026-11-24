import unittest
import numpy as np
from scipy.sparse import csr_matrix
from graph.components import analyze,partition_labels,validate

class ComponentsTest(unittest.TestCase):
    def test_components(self):
        g=csr_matrix([[0,.9,0],[.9,0,0],[0,0,0]])
        x=np.ones((3,2));b=np.zeros((3,1));c=b.copy()
        r,p=analyze(g,x,b,c,np.array([2,3,4]),np.array([1,0,4]),np.array([2,3,4]),np.array([1,0,1]))
        self.assertEqual(r["components"],2)
        self.assertEqual(r["singleton_components"],1)
        self.assertEqual(r["total_raw_rows"],9)
        self.assertEqual(r["isolated_attack_rows"],4)
        self.assertAlmostEqual(r["largest_component"]["mean_all_pair_similarity"],1)
    def test_paths(self):
        self.assertEqual(partition_labels(np.array(["date=x/type=attacks/a",r"date=x\type=normal\b"])).tolist(),[True,False])
    def test_invalid(self):
        with self.assertRaises(ValueError):validate(csr_matrix([[0,.9],[0,0]]))
