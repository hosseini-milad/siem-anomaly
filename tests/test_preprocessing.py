import unittest
import numpy as np
from preprocessing.features import port_category
from preprocessing.transform import fit_transform, sample_training

class PreprocessingTest(unittest.TestCase):
    def test_ports(self):
        np.testing.assert_array_equal(port_category(np.array([np.nan,3.1,0,1023,1024,49151,49152,65535,65536])),[0,0,1,1,2,2,3,3,0])
    def test_batch_invariance(self):
        x=np.random.default_rng(42).normal(size=(100,11))
        m=fit_transform(x)
        np.testing.assert_allclose(m.apply(x),np.concatenate([m.apply(x[:30]),m.apply(x[30:])]),atol=1e-6)
    def test_sample(self):
        x=np.arange(300).reshape(100,3)
        self.assertEqual(sample_training([x[:50],x[50:]],20).shape,(20,3))
