"""Temporal sensitivity, fixed controls and feature layout."""
import unittest
import numpy as np
from features import extract,TF_DIM,DYNAMIC_DIM


class FeatureTest(unittest.TestCase):
    def test_shared_baseline_and_temporal_order(self):
        rng=np.random.default_rng(15);x=rng.normal(size=(2,41,480)).astype(np.float32)
        rms=np.sqrt(np.mean(x*x,axis=(0,2))).astype(np.float32)
        tf,dyn=extract(x,rms,0); again,dyn2=extract(x,rms,0)
        self.assertEqual(tf.shape,(2,TF_DIM));self.assertEqual(dyn.shape,(2,DYNAMIC_DIM))
        np.testing.assert_array_equal(tf,again);np.testing.assert_array_equal(dyn,dyn2)
        self.assertTrue(np.isfinite(np.concatenate((tf,dyn),axis=1)).all())
        tf_rev,dyn_rev=extract(x[:,:,::-1].copy(),rms,0)
        means=tf[:,:164].reshape(2,41,4); means_rev=tf_rev[:,:164].reshape(2,41,4)[:,:,::-1]
        np.testing.assert_allclose(means,means_rev,atol=2e-6)
        power=tf[:,164:].reshape(2,17,41,4);power_rev=tf_rev[:,164:].reshape(2,17,41,4)[:,:,:,::-1]
        np.testing.assert_allclose(power,power_rev,atol=2e-5,rtol=1e-5)
        reverse=dyn_rev.reshape(2,16,12,4,4,2)[:,:,:,:,::-1,:]
        self.assertGreater(float(np.max(np.abs(dyn.reshape(2,16,12,4,4,2)-reverse))),.05)


if __name__=='__main__': unittest.main()
