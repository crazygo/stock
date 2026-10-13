import unittest
from quadratic_learning import *
from audit_quadratic import reference,compare
from partial_pool import fit as original_fit,predict as original_predict
from sklearn.preprocessing import PolynomialFeatures


class QuadraticTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(72);self.X=rng.normal(size=(90,4));self.Y=rng.normal(size=(90,2))
        self.pair=np.repeat(['a','b','c'],30);self.w=np.tile(2.**(np.arange(30)/126),3)/3

    def test_enumeration_matches_polynomial_features(self):
        np.testing.assert_array_equal(polynomial(self.X),PolynomialFeatures(2,include_bias=False).fit_transform(self.X))
        self.assertEqual(polynomial(np.zeros((1,18))).shape,(1,189))

    def test_block_solution_matches_full_equations(self):
        for common,deviation in [(100.,300.),(300.,100.)]:
            saved=fit_family(self.X,self.Y,self.pair,self.w,30,common=common,deviation=deviation)
            compare(saved,reference(self.X,self.Y,self.pair,self.w,30,common=common,deviation=deviation))

    def test_linear_map_matches_original_augmented_solver(self):
        family=fit_family(self.X,self.Y,self.pair,self.w,30,degree=1)
        for mode in VARIANTS:
            old=original_fit(self.X,self.Y,self.pair,self.w,30,mode=mode)
            np.testing.assert_allclose(forecast(family,self.X,self.pair,mode),original_predict(old,self.X,self.pair),atol=1e-11,rtol=1e-10)

    def test_unequal_pair_mass_and_constant_inputs(self):
        x=self.X.copy();x[self.pair=='b',2]=0;w=self.w.copy();w[self.pair=='a']*=3
        compare(fit_family(x,self.Y,self.pair,w,30),reference(x,self.Y,self.pair,w,30))

    def test_duplicate_rows_do_not_create_dates(self):
        a=fit_family(self.X,self.Y,self.pair,self.w,30)
        b=fit_family(np.tile(self.X,(2,1)),np.tile(self.Y,(2,1)),np.tile(self.pair,2),np.tile(self.w,2),30)
        for k in VARIANTS:np.testing.assert_allclose(forecast(a,self.X,self.pair,k),forecast(b,self.X,self.pair,k),atol=1e-11,rtol=1e-10)

    def test_translation_and_positive_units_are_invariant(self):
        a=fit_family(self.X,self.Y,self.pair,self.w,30);changed=self.X*np.array([3,1,7,2])+np.array([2,9,-3,5])
        b=fit_family(changed,self.Y,self.pair,self.w,30)
        for k in VARIANTS:np.testing.assert_allclose(forecast(a,self.X,self.pair,k),forecast(b,changed,self.pair,k),atol=1e-11,rtol=1e-10)

    def test_independent_matches_single_pair(self):
        allmodel=fit_family(self.X,self.Y,self.pair,self.w,30);m=self.pair=='b'
        solo=fit_family(self.X[m],self.Y[m],self.pair[m],self.w[m],30)
        np.testing.assert_allclose(forecast(allmodel,self.X[m],self.pair[m],'independent'),forecast(solo,self.X[m],self.pair[m],'independent'),atol=1e-11,rtol=1e-10)

    def test_missing_data_fails(self):
        x=self.X.copy();x[2,1]=np.nan
        with self.assertRaises(ValueError):fit_family(x,self.Y,self.pair,self.w,30)


if __name__=='__main__':unittest.main()
