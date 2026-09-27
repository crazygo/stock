import math
import unittest

import torch

from .v7_c_group import shape_features,group_contrast


def observed_path(prices):
    """Contiguous no-gap bars from a sequence of boundary prices."""
    x=torch.zeros(1,len(prices)-1,14)
    for i,(a,b) in enumerate(zip(prices[:-1],prices[1:])):
        x[0,i,0]=math.log(b/a)
        x[0,i,12]=math.log(a)/5
        x[0,i,10]=1
    return x


class ShapeSemantics(unittest.TestCase):
    def test_no_gap_sustained_rise_is_positive_path(self):
        z=shape_features(observed_path([1,1.01,1.02,1.03]))[0]
        self.assertAlmostEqual(float(z[:3].sum()),100*math.log(1.03),places=4)

    def test_same_endpoint_different_drawdown(self):
        smooth=shape_features(observed_path([1,1.01,1.03,1.05]))[0]
        reversal=shape_features(observed_path([1,1.10,1.00,1.05]))[0]
        self.assertAlmostEqual(float(smooth[:3].sum()),float(reversal[:3].sum()),places=4)
        self.assertGreater(float(reversal[3]),float(smooth[3])+.05)

    def test_unobserved_tail_does_not_change_shape(self):
        x=torch.cat((observed_path([1,1.01,1.02]),torch.zeros(1,2,14)),1)
        expected=shape_features(x)
        x[0,-1,0]=.3; x[0,-1,12]=3; x[0,-2,0]=-.4
        self.assertTrue(torch.allclose(expected,shape_features(x)))

    def test_unknown_group_is_masked_not_numeric_zero(self):
        group=torch.zeros(1,6,6,10)
        group[0,0,0,3]=.4;group[0,0,0,9]=1
        group[0,-1,0,3]=.6;group[0,-1,0,9]=1
        delta,valid=group_contrast(group)
        self.assertAlmostEqual(float(delta[0,0,3]),.2,places=5)
        self.assertEqual(float(valid[0,0,0]),1)
        self.assertEqual(float(valid[0,1,0]),0)
        group[0,-1,0,9]=0
        delta,valid=group_contrast(group)
        self.assertEqual(float(delta[0,0,3]),0)
        self.assertEqual(float(valid[0,0,0]),0)


if __name__=="__main__": unittest.main()
