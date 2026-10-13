"""Block normalization, exact nesting, frozen transforms, and full equations."""
import unittest
from block_learning import *
from audit_blocks import reference,compare
from quadratic_learning import fit_family as scalar_fit
from partial_pool import fit as linear_fit,predict as linear_predict


class BlockTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(74);self.x=rng.normal(size=(90,18));self.y=rng.normal(size=(90,2))
        self.names=np.repeat(['a','b','c'],30);self.w=np.tile(2.**(np.arange(30)/126),3)/3

    def test_registered_penalties(self):
        f=block_factors([18,171]);np.testing.assert_array_equal(f[:18],1.);np.testing.assert_array_equal(f[18:],9.5)
        m=fit_family(self.x,self.y,self.names,self.w,30)
        np.testing.assert_array_equal(m['common'],100*f);np.testing.assert_array_equal(m['deviation'],300*f)

    def test_original_coordinate_full_equations(self):
        m=fit_family(self.x,self.y,self.names,self.w,30)
        compare(m,reference(self.x,self.y,self.names,self.w,30))

    def test_unequal_pair_weights(self):
        w=self.w.copy();w[self.names=='b']*=4
        compare(fit_family(self.x,self.y,self.names,w,30),reference(self.x,self.y,self.names,w,30))

    def test_degree_one_exact_linear(self):
        m=fit_family(self.x,self.y,self.names,self.w,30,degree=1)
        for mode in VARIANTS:
            old=linear_fit(self.x,self.y,self.names,self.w,30,mode)
            np.testing.assert_allclose(forecast(m,self.x,self.names,mode),linear_predict(old,self.x,self.names),atol=1e-10,rtol=1e-9)

    def test_uniform_factors_exact_old_families(self):
        for factor in [1.,10.5]:
            m=fit_family(self.x,self.y,self.names,self.w,30,factors=np.full(189,factor))
            compare(m,scalar_fit(self.x,self.y,self.names,self.w,30,common=100*factor,deviation=300*factor))

    def test_copy_each_constructed_block_independently(self):
        # Already constructed features: no polynomial expansion of replicated blocks.
        x=self.x[:,:6];w=self.w.copy();w[self.names=='b']*=4
        baseline=fit_family(x,self.y,self.names,w,30,degree=1,factors=block_factors([2,4]))
        for a,b in [(2,1),(1,3),(2,3)]:
            repeated=np.column_stack([np.tile(x[:,:2],(1,a)),np.tile(x[:,2:],(1,b))])
            model=fit_family(repeated,self.y,self.names,w,30,degree=1,factors=block_factors([2*a,4*b]))
            for mode in VARIANTS:
                np.testing.assert_allclose(forecast(model,repeated,self.names,mode),forecast(baseline,x,self.names,mode),atol=1e-11,rtol=1e-10)

    def test_copy_without_block_scaling_negative_control(self):
        x=self.x[:,:6];f=block_factors([2,4]);a=fit_family(x,self.y,self.names,self.w,30,degree=1,factors=f)
        xx=np.column_stack([np.tile(x[:,:2],(1,3)),x[:,2:]])
        b=fit_family(xx,self.y,self.names,self.w,30,degree=1,factors=np.r_[np.tile(f[:2],3),f[2:]])
        self.assertGreater(np.max(np.abs(forecast(a,x,self.names,'partial')-forecast(b,xx,self.names,'partial'))),1e-3)

    def test_constant_column_and_frozen_query(self):
        x=self.x.copy();x[:,2]=1.;m=fit_family(x,self.y,self.names,self.w,30);compare(m,reference(x,self.y,self.names,self.w,30))
        before=[m[k].copy() for k in ['original_center','original_scale','centers','scales']]
        forecast(m,np.full((1,18),100.),['a'],'partial')
        for k,v in zip(['original_center','original_scale','centers','scales'],before):np.testing.assert_array_equal(m[k],v)

    def test_repeated_rows_preserve_real_date_count(self):
        a=fit_family(self.x,self.y,self.names,self.w,30)
        b=fit_family(np.repeat(self.x,2,axis=0),np.repeat(self.y,2,axis=0),np.repeat(self.names,2),np.repeat(self.w,2),30)
        for mode in VARIANTS:np.testing.assert_allclose(forecast(a,self.x,self.names,mode),forecast(b,self.x,self.names,mode),atol=1e-10,rtol=1e-9)

    def test_invalid_blocks_and_factors(self):
        for widths,ref in [([],18),([0,2],18),([18,171],0),([1.5],18)]:
            with self.assertRaises(ValueError):block_factors(widths,ref)
        for factors in [np.zeros(189),np.full(189,np.nan),np.ones(18)]:
            with self.assertRaises(ValueError):fit_family(self.x,self.y,self.names,self.w,30,factors=factors)


if __name__=='__main__':unittest.main()
