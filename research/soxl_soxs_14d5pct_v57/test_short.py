"""Counterexamples for proxy feedback timing and unchanged actual target."""
import unittest
from copy import deepcopy
from short_learning import *
from audit_short_inputs import reference_label
from test_contract import market_fixture


class ShortTests(unittest.TestCase):
    def setUp(self):
        self.market = market_fixture()
        self.at = pd.Timestamp('2026-05-04 09:25:30', tz=core.ET).tz_convert('UTC')

    def label(self, market=None):
        return short_label(market or self.market, 'SOXL', self.at)

    def change(self, when, **values):
        at = pd.Timestamp(when, tz=core.ET).tz_convert('UTC')
        f = self.market.bars['SOXL']
        for key, value in values.items():
            f.loc[at, key] = value
            if key in ['open', 'high', 'low', 'close']:
                f.loc[at, 'econ_' + key] = value * f.loc[at, 'share_factor']

    def test_flat_costed_next_close(self):
        r = self.label(); self.assertTrue(r['known']); self.assertEqual(r['reason'], 'short_close')
        self.assertEqual(r['exit_at'], pd.Timestamp('2026-05-05 16:00', tz=core.ET))
        self.assertAlmostEqual(r['net_return'], 100*.9995*.9995/(100*1.0005*1.0005)-1)

    def test_early_hit_not_early_training(self):
        self.change('2026-05-04 09:35', high=110)
        r = self.label(); self.assertEqual(r['reason'], 'target')
        self.assertEqual(r['release'], pd.Timestamp('2026-05-05 16:00:01', tz=core.ET))

    def test_missing_after_early_win_is_unknown(self):
        self.change('2026-05-04 09:35', high=110)
        f = self.market.bars['SOXL']; self.market.bars['SOXL'] = f.drop(pd.Timestamp('2026-05-05 15:00', tz=core.ET))
        self.assertFalse(self.label()['known'])

    def test_stop_wins_ambiguous_bar(self):
        self.change('2026-05-04 09:35', low=90, high=110)
        r = self.label(); self.assertEqual(r['reason'], 'risk_stop'); self.assertTrue(r['same_bar_both_barriers'])

    def test_adverse_overnight_gap(self):
        self.change('2026-05-05 09:35', open=85, low=84)
        r = self.label(); self.assertTrue(r['gap_below_stop']); self.assertAlmostEqual(r['exit_fill_economic'], 85*.9995)

    def test_split_units_and_rounding(self):
        f = self.market.bars['SOXL']; mask = f.date >= pd.Timestamp('2026-05-05')
        f.loc[mask, 'share_factor'] = 2
        for c in ['open', 'high', 'low', 'close']: f.loc[mask, c] /= 2
        a = self.label(); b = reference_label(self.market, 'SOXL', self.at)
        self.assertAlmostEqual(a['net_return'], b['net_return']); self.assertEqual(a['reason'], 'short_close')

    def test_dividend_receivable_included(self):
        self.market.actions['SOXL'] = [{'type':'CashDividends','date':'2026-05-05','value':1.,'payment_date':'2026-06-01'}]
        self.assertEqual(self.label()['dividend_per_economic_unit'], 1.)

    def test_dividend_after_exit_excluded(self):
        self.change('2026-05-04 09:35', high=110)
        self.market.actions['SOXL'] = [{'type':'CashDividends','date':'2026-05-05','value':1.,'payment_date':'2026-06-01'}]
        self.assertEqual(self.label()['dividend_per_economic_unit'], 0.)

    def test_future_after_horizon_immune(self):
        a = self.label(); self.change('2026-05-06 09:35', low=1, high=1000)
        self.assertEqual(a, self.label())

    def test_late_available_delays_feedback(self):
        self.change('2026-05-05 16:00', available_at=pd.Timestamp('2026-05-06 02:00',tz=core.ET))
        self.assertEqual(self.label()['release'], pd.Timestamp('2026-05-06 02:00',tz=core.ET))

    def test_available_before_end_invalid(self):
        self.change('2026-05-05 16:00', available_at=pd.Timestamp('2026-05-05 15:59',tz=core.ET))
        self.assertFalse(self.label()['known'])

    def test_future_window_missing_stays_unknown(self):
        m = restrict_market(self.market, pd.Timestamp('2026-05-05 12:00',tz=core.ET))
        self.assertFalse(self.label(m)['known'])

    def test_official_holiday_skipped(self):
        self.at = pd.Timestamp('2026-05-22 09:25:30',tz=core.ET).tz_convert('UTC')
        self.assertEqual(self.label()['next_date'], pd.Timestamp('2026-05-26'))

    def test_halfday_short_close(self):
        day = pd.Timestamp('2026-05-05'); close = pd.Timestamp('2026-05-05 13:00',tz=core.ET).tz_convert('UTC')
        self.market.calendar[day]['close_at'] = str(close)
        f = self.market.bars['SOXL']; self.market.bars['SOXL'] = f[~((f.date == day) & (f.end_at > close))]
        self.assertEqual(self.label()['exit_at'], close)

    def test_one_unknown_side_keeps_pair_unknown(self):
        original = pd.DataFrame({'C':[9.], 'D':[8.], 'M':[7.], 'release':[pd.NaT], 'terminal_release':[pd.NaT], 'feature_ready':[True]},index=[self.at])
        rows = [self.label(), {**self.label(), 'symbol':'SOXS','known':False,'net_return':None}]
        short = build_short_panel(original, rows)
        self.assertTrue(short.loc[self.at,['C','D','M']].isna().all()); self.assertEqual(original.C.iloc[0], 9.)

    def test_lagged_waits_both_feedbacks(self):
        short = pd.DataFrame({'terminal_release':[self.at, self.at], 'release':[self.at,self.at]},index=[0,1])
        original = pd.DataFrame({'terminal_release':[self.at+pd.Timedelta(days=2),pd.NaT]},index=[0,1])
        panels = training_panels(original, short)
        self.assertEqual(panels['lagged'].terminal_release.iloc[0], self.at+pd.Timedelta(days=2))
        self.assertTrue(pd.isna(panels['lagged'].terminal_release.iloc[1])); self.assertEqual(panels['fast'].terminal_release.iloc[0],self.at)

    def test_feedback_gap_abstains(self):
        d = {'feedback':{'expired_unresolved_prefix':False}}
        self.assertTrue(feedback_for_action(d,{'short_unresolved_expired':True})['expired_unresolved_prefix'])

    def test_original_acceptance_unchanged(self):
        p = read(SOURCE/'protocol.json'); old = read(prior.SOURCE/'protocol.json')
        for key in ['acceptance','development','cases','capital','contract']: self.assertEqual(p[key],old[key])
        for c in CLOCKS:self.assertEqual(columns(c),prior.columns(c,'full'))

    def synthetic_panels(self):
        rng = np.random.default_rng(510)
        days = pd.bdate_range('2025-01-02', periods=145)
        ids = pd.DatetimeIndex([pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC') for d in days])
        p = pd.DataFrame(rng.normal(size=(len(days),len(columns(CLOCKS[0])))),index=ids,columns=columns(CLOCKS[0]))
        p['date']=days;p['clock']=CLOCKS[0];p['feature_ready']=True
        p['C']=rng.normal(0,.01,len(p));p['D']=rng.normal(0,.03,len(p));p['M']=abs(p.D)
        p['terminal_release']=ids+pd.Timedelta(days=2);p['release']=p.terminal_release;p['short_nominal_release']=p.terminal_release
        old=p.copy();old['terminal_release']=ids+pd.Timedelta(days=14)
        return old,p,ids[-1]

    def test_fast_feedback_admits_more_recent_members(self):
        old,p,at=self.synthetic_panels();_,fits,_=fit_bases(training_panels(old,p),at)
        self.assertGreater(len(fits['fast']['fit']['members']),len(fits['lagged']['fit']['members']))
        self.assertTrue(all(pd.Timestamp(t)<day_cutoff(at) for t in fits['fast']['fit']['members']))

    def test_future_target_cannot_change_issued_fit(self):
        from run_planning import json_ready
        old,p,at=self.synthetic_panels();_,a,_=fit_bases(training_panels(old,p),at)
        changed=p.copy();changed.loc[at,['C','D','M']]=999
        _,b,_=fit_bases(training_panels(old,changed),at)
        self.assertEqual(json_ready(a),json_ready(b))


if __name__ == '__main__': unittest.main()
