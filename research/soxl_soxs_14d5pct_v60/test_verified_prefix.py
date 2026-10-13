import unittest
from audit_verified_prefix import exact_model,np

class PrefixTests(unittest.TestCase):
    def test_exact_nested_arrays(self):
        a={'selected_blocks':np.array([[1,2]]),'result':{'net':np.array([.01,-.02])}};exact_model(a,{'selected_blocks':a['selected_blocks'].copy(),'result':{'net':a['result']['net'].copy()}})
    def test_one_ulp_change_rejected(self):
        a=np.array([.01]);b=np.nextafter(a,np.inf)
        with self.assertRaises(AssertionError):exact_model({'net':a},{'net':b})
    def test_missing_result_rejected(self):
        with self.assertRaises(AssertionError):exact_model({'cash':np.zeros(1)},{})
    def test_dtype_change_rejected(self):
        with self.assertRaises(AssertionError):exact_model(np.array([1],dtype='int64'),np.array([1.],dtype='float64'))

if __name__=='__main__':unittest.main()
