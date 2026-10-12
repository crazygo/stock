import unittest
import numpy as np
import pandas as pd
from linear_entry import core, factors, date_weights, selected_calibration, decision, fit, predict, FACTOR_COLUMNS


class LinearCausality(unittest.TestCase):
    def test_direction_volatility_and_clipping(self):
        row = {'SOXX_vol20': .02, 'QQQ_vol20': .01, 'SOXL_vol20': .06, 'SOXS_vol20': .05,
            'SOXX_ret1': .02, 'SOXX_ret5': .04, 'SOXX_ret20': .08, 'QQQ_ret5': .03,
            'SOXX_today_return': .01, 'breadth_today_mean': .005}
        a = factors(row,'SOXL'); b = factors(row,'SOXS')
        self.assertAlmostEqual(a['sector_1'],1.)
        for name in FACTOR_COLUMNS[:-1]: self.assertAlmostEqual(a[name],-b[name])
        self.assertAlmostEqual(a['own_log_volatility'],np.log(.06))
        row['SOXX_ret1'] = 100.; self.assertEqual(factors(row,'SOXL')['sector_1'],5.)

    def test_parent_weight_half_life_and_total(self):
        w = date_weights(253,True)
        self.assertAlmostEqual(w.sum(),253.); self.assertAlmostEqual(w[0]/w[-1],.25)
        self.assertTrue(np.array_equal(date_weights(253,False),np.ones(253)))

    def make_history(self, n=70, inverted=False):
        rows = []; labels = []
        for i,date in enumerate(pd.date_range('2025-01-01',periods=n)):
            at = date.tz_localize(core.ET)+pd.Timedelta(hours=9)
            x = (i-n/2)/1000
            rows.append({'date':date,'at':at,'top_symbol':'SOXL','raw':{'SOXL':{'score':x}}})
            labels.append({'id':at.isoformat()+'|SOXL','log_return':(-x if inverted else 2*x)+.001})
        return rows,pd.DataFrame(labels).set_index('id')

    def test_future_labels_and_scores_cannot_enter_calibration(self):
        history,panel = self.make_history(); valid = [e['date'] for e in history[:60]]
        a = selected_calibration(history,panel,valid)
        panel.loc[panel.index[-1],'log_return'] = 999.
        history[-1]['raw']['SOXL']['score'] = -999.
        self.assertEqual(a,selected_calibration(history,panel,valid))
        self.assertEqual(len(a['dates']),60); self.assertTrue(a['ready'])

    def test_negative_relation_is_recorded_but_not_rank_reversed(self):
        history,panel = self.make_history(inverted=True)
        cal = selected_calibration(history,panel,[e['date'] for e in history])
        self.assertLess(cal['unconstrained_coefficient'],0.)
        self.assertEqual(cal['coefficient'],0.)
        raw = {s:{'score':.01 if s=='SOXL' else -.01} for s in core.ETFS}
        self.assertEqual(decision(raw,cal)['top_symbol'],'SOXL')
        self.assertAlmostEqual(decision(raw,cal)['adjusted'],panel.log_return.mean())

    def test_constant_score_and_shared_warmup(self):
        history,panel = self.make_history()
        for e in history: e['raw']['SOXL']['score'] = .01
        cal = selected_calibration(history,panel,[e['date'] for e in history])
        self.assertEqual(cal['coefficient'],0.); self.assertEqual(cal['scale'],1.)
        raw = {s:{'score':.01} for s in core.ETFS}
        cal['ready'] = False
        for method in ['conditional','offset','raw']:
            self.assertEqual(decision(raw,cal,method)['action'],'CASH')

    def test_scaler_and_one_class_use_only_training_rows(self):
        rows = []
        dates = list(pd.date_range('2025-01-01',periods=3))
        for i,d in enumerate(dates):
            for s in core.ETFS:
                rows.append({'id':str(d)+s,'date':d,'symbol':s,'hit':1,'log_return':.04,
                             **{'factor_'+c:float(i+1) for c in FACTOR_COLUMNS}})
        panel = pd.DataFrame(rows).set_index('id')
        model,info = fit(panel,dates[:2],'logit_factor_uniform',minimum=2)
        np.testing.assert_allclose(model['scaler'].mean_,1.5)
        panel.loc[panel.date.eq(dates[2]),'factor_sector_1'] = 1e9
        repeated,_ = fit(panel,dates[:2],'logit_factor_uniform',minimum=2)
        np.testing.assert_array_equal(model['scaler'].mean_,repeated['scaler'].mean_)
        self.assertIsNone(model['estimator'])
        out = predict(model,panel[panel.date.eq(dates[2])])
        self.assertEqual(out['SOXL']['p'],1.); self.assertAlmostEqual(out['SOXL']['score'],.04)


if __name__=='__main__': unittest.main()
