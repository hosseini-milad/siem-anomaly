import unittest
import numpy as np
from preprocessing.reduction import select_representatives
class ReductionTest(unittest.TestCase):
    def test_budget_and_mapping(self):
        x=np.random.default_rng(42).normal(size=(100,11))
        b=np.zeros((100,11));c=np.zeros((100,3))
        r=select_representatives(x,b,c,anchors=4)
        self.assertEqual(len(r["indices"]),10)
        self.assertEqual(r["multiplicity"].sum(),100)
        self.assertTrue(np.isin(r["assignment"],r["indices"]).all())
        self.assertTrue(np.array_equal(r["assignment"][r["indices"]],r["indices"]))
