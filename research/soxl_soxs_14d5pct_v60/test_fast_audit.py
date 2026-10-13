import unittest
from path_learning import np
from audit_dense_calibration import solve as original
from audit_dense_calibration_fast import solve as vectorized

class FastAuditTests(unittest.TestCase):
    def compare(self,k,constant=False,repeated=False):
        n=64;clock=np.tile(np.arange(k),n);x=np.column_stack([np.random.default_rng(19).normal(size=(n*k,2)),clock]);y=np.random.default_rng(29).normal(size=(n*k,2));w=np.exp2(np.repeat(np.arange(n)-n,k)/63.)/k
        if constant:x[:,:2]=1.
        if repeated:
            ix=np.random.default_rng(20).integers(0,n,size=n);rows=np.concatenate([np.arange(i*k,(i+1)*k) for i in ix]);x=x[rows];y=y[rows];w=w[rows]
        a=original(x,y,w,n,k);b=vectorized(x,y,w,n,k)
        for f in ['input_center','input_scale','intercepts','coefficients']:np.testing.assert_array_equal(a[f],b[f])
    def test_eight_clocks(self):self.compare(8)
    def test_four_clocks(self):self.compare(4)
    def test_constant_scores(self):self.compare(8,constant=True)
    def test_repeated_bootstrap_dates(self):self.compare(8,repeated=True)

if __name__=='__main__':unittest.main()
