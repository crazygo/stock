import unittest
import numpy as np
import pandas as pd
import torch

from .evaluate import select_signals, summarize, choose_thresholds, GROUPS
from .train import pairs, rank_gradient, rank_objective_value, strip_group_categories, model_c


class PrecisionTests(unittest.TestCase):
    def frame(self):
        dates = ['2026-03-02', '2026-03-03', '2026-03-06']
        return pd.DataFrame({'sample_id': ['a','b','c'], 'symbol': ['ALAB']*3,
            'session_date': dates, 'decision_at': [d+'T16:30:30+00:00' for d in dates],
            'signal_end_at': ['2026-03-05T16:35:00+00:00', '2026-03-06T16:35:00+00:00', '2026-03-11T15:35:00+00:00'],
            'raw_3d_5pct': [.8,.99,.8], 'y_3d_5pct': [1,0,1], 'terminal_3d': [.1,-.1,.1]})

    def test_cooldown_ignores_outcome_and_group_overlap(self):
        f = self.frame()
        a = select_signals(f, {'chips': .7, 'optics': .7})
        self.assertEqual(a.sample_id.tolist(), ['a','c'])
        self.assertEqual(a.trigger_groups.tolist(), [['chips','optics']]*2)
        f.y_3d_5pct = 1-f.y_3d_5pct
        b = select_signals(f, {'chips': .7, 'optics': .7})
        self.assertEqual(a.sample_id.tolist(), b.sample_id.tolist())

    def test_no_signal_is_null_and_all_sessions_count(self):
        f = self.frame()
        signals = select_signals(f, {'chips': .7})
        dates = pd.bdate_range('2026-03-02', periods=10).strftime('%Y-%m-%d').tolist()
        m = summarize(f, signals, 'chips', dates)
        self.assertEqual(m['signals_per_5_sessions'], 1)
        empty = select_signals(f, {'chips': None})
        m = summarize(f, empty, 'chips', dates)
        self.assertIsNone(m['precision']); self.assertFalse(m['point_and_supply_pass'])

    def test_small_perfect_cal_rejected(self):
        f = self.frame()
        chosen, curves = choose_thresholds(f, f.session_date.tolist())
        self.assertTrue(all(v is None for v in chosen.values()))
        self.assertEqual(len(curves['chips']), 14)

    def test_membership_does_not_bypass_group_threshold(self):
        s = select_signals(self.frame(), {'chips': .7, 'optics': .95})
        self.assertEqual(s.trigger_groups.tolist(), [['chips'], ['chips']])

    def test_future_rows_cannot_change_past_selection(self):
        f = self.frame()
        a = select_signals(f.iloc[:2], {'chips': .7})
        b = select_signals(f, {'chips': .7})
        self.assertEqual(a.sample_id.tolist(), b[b.session_date<'2026-03-06'].sample_id.tolist())

    def test_pairs_are_same_stock_and_gradient_is_numeric(self):
        syms = np.array(['A','A','B','B','C'])
        y = np.array([1.,0,1,0,0]); z = np.array([.1,.2,-.4,.1,.7]); w=np.ones(5)
        ij = pairs(syms, y)
        self.assertTrue(np.all(syms[ij[0]] == syms[ij[1]]))
        self.assertTrue(np.all(y[ij[0]]==1)); self.assertTrue(np.all(y[ij[1]]==0))
        g,h=rank_gradient(y,z,w,ij)
        numeric=[]
        for i in range(len(z)):
            dz=np.zeros(len(z)); dz[i]=1e-5
            numeric.append((rank_objective_value(y,z+dz,w,ij)-rank_objective_value(y,z-dz,w,ij))/2e-5)
        np.testing.assert_allclose(g,numeric,atol=1e-7)
        self.assertTrue(np.all(h>0))

    def test_category_transform_only_changes_categories(self):
        a=np.random.default_rng(1).normal(size=(2,6,6,10)).astype(np.float32)
        b=strip_group_categories(a)
        np.testing.assert_array_equal(a[...,3:],b[...,3:])
        self.assertTrue(np.all(b[...,:3]==0)); self.assertFalse(np.all(a[...,:3]==0))
        np.testing.assert_array_equal(strip_group_categories(a[:1]),b[:1])

    def test_neural_route_isolation_and_dropout_inference(self):
        torch.set_num_threads(1)
        a=torch.zeros(2,8,192,14); b=torch.zeros(2,31,17,14); d=torch.zeros(2,126,14)
        g=torch.randn(2,6,6,10); p=torch.full((2,9),.5)
        for x in (a,b,d): x[...,10]=1
        m=model_c('C_no_group',{'within_stock_rank':True}).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(a,b,d,g,p),m(a,b,d,torch.full_like(g,float('nan')),p))
        m=model_c('C_no_daily',{'capacity':True,'support':True}).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(a,b,d,g,p),m(a,b,torch.full_like(d,float('nan')),g,p))
        m=model_c('C_group',{'support':True,'group_dropout':True}).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(a,b,d,g,p),m(a,b,d,g,p),rtol=0,atol=0)


if __name__=='__main__': unittest.main()
