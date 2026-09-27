import math
import unittest

import torch

from .runner import shape_features


def candles(opens, closes, valid=None):
    x=torch.zeros((1,len(opens),14),dtype=torch.float32)
    valid=valid or [True]*len(opens)
    for i,(o,c,m) in enumerate(zip(opens,closes,valid)):
        if m:
            x[0,i,0]=math.log(c/o)
            x[0,i,12]=math.log(o)/5
            x[0,i,10]=1
    return x


class ShapeSemantics(unittest.TestCase):
    def test_contiguous_path_uses_close_not_only_gap(self):
        x=candles([100,101,102],[101,102,103])
        x[...,3]=0  # Gap-only computation would incorrectly infer flat path.
        z=shape_features(x)[0]
        self.assertAlmostEqual(float(z[:3].sum()),100*math.log(103/100),places=4)
        self.assertGreater(float(z[4]),.99)

    def test_open_gap_is_included_after_prior_close(self):
        x=candles([100,105],[101,106])
        z=shape_features(x)[0]
        self.assertAlmostEqual(float(z[:3].sum()),100*math.log(106/100),places=4)

    def test_same_endpoint_different_drawdown(self):
        smooth=shape_features(candles([100,101,102],[101,102,105]))[0]
        reversal=shape_features(candles([100,110,99],[110,99,105]))[0]
        self.assertAlmostEqual(float(smooth[:3].sum()),float(reversal[:3].sum()),places=4)
        self.assertGreater(float(reversal[3]),float(smooth[3])+.05)
        self.assertLess(float(reversal[4]),float(smooth[4]))

    def test_invalid_tail_is_not_future_shape(self):
        x=candles([100,101,0,0],[101,102,0,0],[True,True,False,False])
        expected=shape_features(x)
        x[0,-1,0]=.9;x[0,-1,12]=2;x[0,-2,0]=-.7
        self.assertTrue(torch.allclose(expected,shape_features(x)))

    def test_internal_unknown_gap_is_marked(self):
        full=shape_features(candles([100,101,102],[101,102,103]))[0]
        gap=shape_features(candles([100,0,102],[101,0,103],[True,False,True]))[0]
        self.assertEqual(float(full[6]),0)
        self.assertGreater(float(gap[6]),0)
        self.assertLess(float(gap[7]),float(full[7]))


if __name__=="__main__":unittest.main()
