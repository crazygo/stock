"""Normalization applies after constructing the common basis, with fixed reference."""
import unittest
from normalized_learning import *
from partial_pool import fit,predict
from audit_quadratic import reference,compare


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(73);self.x=rng.normal(size=(90,6));self.y=rng.normal(size=(90,2))
        self.groups=np.repeat(['a','b','c'],30);self.w=np.tile(2.**(np.arange(30)/126),3)/3

    def test_registered_width_formula(self):
        self.assertEqual(penalties(189),{'multiplier':10.5,'common':1050.,'deviation':3150.,'columns':189,'reference_columns':18})

    def test_whole_basis_replication_invariance(self):
        for copies in [2,3]:
            repeated=np.tile(self.x,(1,copies));a=penalties(self.x.shape[1]);b=penalties(repeated.shape[1])
            for mode in VARIANTS:
                first=fit(self.x,self.y,self.groups,self.w,30,mode,a['common'],a['deviation'])
                second=fit(repeated,self.y,self.groups,self.w,30,mode,b['common'],b['deviation'])
                np.testing.assert_allclose(predict(first,self.x,self.groups),predict(second,repeated,self.groups),atol=1e-11,rtol=1e-10)

    def test_unnormalized_replication_changes_prediction(self):
        repeated=np.tile(self.x,(1,3));a=fit(self.x,self.y,self.groups,self.w,30);b=fit(repeated,self.y,self.groups,self.w,30)
        self.assertGreater(np.max(np.abs(predict(a,self.x,self.groups)-predict(b,repeated,self.groups))),1e-3)

    def test_unequal_group_weights_replication(self):
        w=self.w.copy();w[self.groups=='b']*=4;repeated=np.tile(self.x,(1,2));a=penalties(6);b=penalties(12)
        for mode in VARIANTS:
            first=fit(self.x,self.y,self.groups,w,30,mode,a['common'],a['deviation'])
            second=fit(repeated,self.y,self.groups,w,30,mode,b['common'],b['deviation'])
            np.testing.assert_allclose(predict(first,self.x,self.groups),predict(second,repeated,self.groups),atol=1e-11,rtol=1e-10)

    def test_normalized_polynomial_matches_full_equations(self):
        x=np.random.default_rng(74).normal(size=(90,18));model=normalized_fit(x,self.y,self.groups,self.w,30)
        compare(model,reference(x,self.y,self.groups,self.w,30,common=1050.,deviation=3150.))

    def test_reference_width_recovers_original_penalty(self):
        setting=penalties(18);self.assertEqual(setting['common'],100.);self.assertEqual(setting['deviation'],300.)

    def test_future_query_does_not_mutate_frozen_transform(self):
        x=np.random.default_rng(75).normal(size=(90,18));m=normalized_fit(x,self.y,self.groups,self.w,30)
        old=[m[k].copy() for k in ['original_center','original_scale','centers','scales']]
        forecast(m,np.full((1,18),100.),['a'],'partial')
        for k,value in zip(['original_center','original_scale','centers','scales'],old):np.testing.assert_array_equal(m[k],value)

    def test_invalid_reference_rejected(self):
        for width,reference_width in [(0,18),(189,0),(1.5,18)]:
            with self.assertRaises(ValueError):penalties(width,reference_width)


if __name__=='__main__':unittest.main()
