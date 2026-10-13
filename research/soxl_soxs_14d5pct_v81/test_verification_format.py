import json,unittest
import numpy as np
from verification_format import serialized_field


class FormatTests(unittest.TestCase):
    def test_empty_pair_coefficient_json_roundtrip(self):
        m={k:[] for k in ['ids','center','scale','design_center','coefficients','penalty']};m['mode']='mean'
        np.testing.assert_array_equal(serialized_field(json.loads(json.dumps(m)),'coefficients'),np.zeros((0,2)))

    def test_nonempty_mean_slopes_rejected(self):
        for field in ['ids','center','scale','design_center','coefficients','penalty']:
            m={k:[] for k in ['ids','center','scale','design_center','coefficients','penalty']};m.update(mode='mean');m[field]=[1]
            with self.assertRaises(ValueError):serialized_field(m,'coefficients')

    def test_other_models_not_reshaped_or_repaired(self):
        m={'mode':'state','coefficients':[]};self.assertEqual(serialized_field(m,'coefficients'),[])
        with self.assertRaises(AssertionError):np.testing.assert_allclose(serialized_field(m,'coefficients'),np.zeros((0,2)))


if __name__=='__main__':unittest.main()
