import unittest
from geometry import *
from conditional_calibration import fit_condition,forecast_condition

class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(13);self.X=np.c_[rng.normal(size=(160,10)),np.arange(160)%4];self.Y=rng.normal(size=(160,2));self.w=rng.uniform(.1,2,160);self.m=fit_condition(self.X,self.Y,self.w,40)
    def test_trace_against_full_smoother(self):
        g=geometry(self.X,self.w,self.m,40);A=g['train_design'];S=A@np.linalg.solve(A.T@(g['weights'][:,None]*A)+penalty(10),A.T*g['weights'])
        self.assertAlmostEqual(g['df'],np.trace(S),places=11)
        eig=np.linalg.eigvalsh((A*np.sqrt(g['weights'])[:,None])@g['inverse']@(A*np.sqrt(g['weights'])[:,None]).T)
        self.assertGreaterEqual(eig.min(),-1e-12);self.assertLessEqual(eig.max(),1+1e-12)
    def test_sensitivity_against_response_perturbation(self):
        g=geometry(self.X,self.w,self.m,40);A=g['train_design'];a,_=design(self.X[[3]],self.m)
        influence=a@g['inverse']@(A.T*np.sqrt(g['weights']));terms=query_terms(self.X[3],self.m,g)
        self.assertAlmostEqual(float(np.sum(influence**2)),terms['response_sensitivity'],places=11)
    def test_decomposition_matches_prediction(self):
        for p in [2,10]:
            X=self.X if p==10 else self.X[:,[0,1,10]];m=fit_condition(X,self.Y,self.w,40);g=geometry(X,self.w,m,40)
            for i in [0,1,2,3,159]:np.testing.assert_allclose(query_terms(X[i],m,g)['correction'],forecast_condition(m,X[i])[0],atol=1e-12)
    def test_label_change_does_not_change_geometry(self):
        m=fit_condition(self.X,self.Y*3+4,self.w,40);g1=geometry(self.X,self.w,self.m,40);g2=geometry(self.X,self.w,m,40)
        np.testing.assert_array_equal(g1['inverse'],g2['inverse']);self.assertEqual(g1['df'],g2['df'])
    def test_date_weight_concentration_not_row_count(self):
        dates=np.array([1.,2.,3.]);rows=np.repeat(dates/4,4);self.assertAlmostEqual(dates.sum()**2/(dates@dates),4*0+36/14);self.assertAlmostEqual(rows.sum()**2/(rows@rows),4*36/14)
    def test_zero_weight_rows_no_effect(self):
        w=self.w.copy();w[-4:]=0;m=fit_condition(self.X,self.Y,w,39);g1=geometry(self.X,w,m,39);g2=geometry(self.X[:-4],w[:-4],m,39);np.testing.assert_array_equal(g1['inverse'],g2['inverse'])

if __name__=='__main__':unittest.main()
