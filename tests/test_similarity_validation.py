import unittest
import numpy as np
from preprocessing.similarity_validation import evaluate_sample
from preprocessing.transform import Transform

class SimilarityTest(unittest.TestCase):
    def test_pairs_and_degree(self):
        x=np.zeros((3,26))
        x[0,0]=x[1,0]=1
        x[2,1]=1
        m=Transform(np.zeros(11),np.ones(11),np.zeros(11),np.eye(11))
        r=evaluate_sample(x,m,2)["methods"]["zca"]
        self.assertEqual(r["valid_pairs"],3)
        self.assertAlmostEqual(r["above_085_fraction"],1/3)
        self.assertAlmostEqual(r["mean_threshold_neighbors"],2/3)
        self.assertEqual(r["isolates"],1)
