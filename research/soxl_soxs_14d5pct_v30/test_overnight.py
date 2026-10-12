import unittest
from types import SimpleNamespace
from overnight import *


class OvernightTests(unittest.TestCase):
    def fixture(self,half=False):
        days=pd.DatetimeIndex(['2026-01-05','2026-01-06']);calendar={};frames=[];forecasts={}
        for day in days:
            opening=pd.Timestamp(str(day.date())+' 09:30',tz=core.ET).tz_convert('UTC')
            close=pd.Timestamp(str(day.date())+(' 13:00' if half else ' 16:00'),tz=core.ET).tz_convert('UTC')
            calendar[day]={'open_at':str(opening),'close_at':str(close)};ends=pd.date_range(opening+pd.Timedelta(minutes=5),close,freq='5min')
            frames.append(pd.DataFrame({'start_at':ends-pd.Timedelta(minutes=5),'end_at':ends,'available_at':ends+pd.Timedelta(seconds=1),
                'date':day,'session_type':'regular','open':100.,'high':101.,'low':99.,'close':100.,'share_factor':1.,'econ_open':100.,
                'econ_high':101.,'econ_low':99.,'econ_close':100.},index=ends))
            forecasts[(day,'SOXL')]={'ready':True,'adverse_tail_mean':.02,'date':day,'at':close-pd.Timedelta(minutes=15)+pd.Timedelta(seconds=30)}
        market=SimpleNamespace(bars={'SOXL':pd.concat(frames)},calendar=calendar,sessions=days)
        entry=pd.Timestamp(calendar[days[0]]['open_at']);base={'decision_at':entry-pd.Timedelta(minutes=4,seconds=30),'symbol':'SOXL',
            'entry_at':entry,'entry_date':days[0],'entry_fill_economic':100.,'limit_economic':105.,'expiry_date':days[-1],
            'expiry_at':pd.Timestamp(calendar[days[-1]]['close_at']),'dividends':[],'dividend_per_economic_unit':0.,'complete_window':True,
            'label_available_at':pd.Timestamp(calendar[days[-1]]['close_at'])+pd.Timedelta(seconds=1)}
        return market,base,forecasts

    def test_exit_uses_next_open_not_same_bar_price(self):
        m,b,f=self.fixture();close=pd.Timestamp(m.calendar[b['entry_date']]['close_at']);end=close-pd.Timedelta(minutes=5)
        m.bars['SOXL'].loc[end,['open','high','low','close']]=[102.,110.,90.,100.]
        label,reviews=managed_label(m,b,f,'buffer');self.assertEqual(label['reason'],'preclose_exit')
        self.assertEqual(label['exit_at'],close-pd.Timedelta(minutes=10));self.assertAlmostEqual(label['exit_fill_economic'],102*.9995)
        self.assertEqual(label['terminal_release'],end+pd.Timedelta(seconds=1));self.assertEqual(len(reviews),1)

    def test_original_target_can_fill_during_exit_latency(self):
        m,b,f=self.fixture();close=pd.Timestamp(m.calendar[b['entry_date']]['close_at']);end=close-pd.Timedelta(minutes=10)
        m.bars['SOXL'].loc[end,'high']=106.
        label,rs=managed_label(m,b,f,'buffer');self.assertEqual(label['reason'],'target');self.assertEqual(label['exit_at'],end)
        self.assertEqual(label['exit_fill_economic'],105.);self.assertEqual(rs[0]['action'],'EXIT')

    def test_gap_stop_precedes_discretionary_open_exit(self):
        m,b,f=self.fixture();end=pd.Timestamp(m.calendar[b['entry_date']]['close_at'])-pd.Timedelta(minutes=5)
        m.bars['SOXL'].loc[end,['open','high','low','close']]=[90.,95.,85.,91.]
        lab,_=managed_label(m,b,f,'buffer');self.assertEqual(lab['reason'],'risk_stop');self.assertAlmostEqual(lab['exit_fill_economic'],90*.9995)

    def test_half_day_review_tracks_official_close(self):
        m,b,f=self.fixture(True);lab,rs=managed_label(m,b,f,'buffer')
        self.assertEqual(rs[0]['at'].tz_convert(core.ET).strftime('%H:%M:%S'),'12:45:30')
        self.assertEqual(lab['exit_at'].tz_convert(core.ET).strftime('%H:%M:%S'),'12:50:00')

    def test_missing_prefix_does_not_jump_to_future_execution(self):
        m,b,f=self.fixture();missing=pd.Timestamp(m.calendar[b['entry_date']]['close_at'])-pd.Timedelta(minutes=10)
        m.bars['SOXL']=m.bars['SOXL'].drop(missing);lab,_=managed_label(m,b,f,'buffer')
        self.assertIsNone(lab['exit_at']);self.assertTrue(np.isnan(lab['net_return']))

    def test_unavailable_risk_or_mark_requests_exit(self):
        self.assertEqual(review(100,104,{'ready':False},1,'buffer')['action'],'EXIT')
        self.assertEqual(review(100,None,{'ready':True},1,'buffer',False)['action'],'EXIT')

    def test_future_gap_pair_cannot_change_current_tail(self):
        days=pd.bdate_range('2025-01-01',periods=90);records=[{'date':d,'ready':True,'release':d.tz_localize(core.ET),'loss':i/1000} for i,d in enumerate(days)]
        at=days[70].tz_localize(core.ET)+pd.Timedelta(hours=15);a=gap_forecast(records,days[70],at)
        for r in records[71:]:r['loss']=.99
        self.assertEqual(a,gap_forecast(records,days[70],at));self.assertEqual(a['tail_count'],8)

    def test_late_or_missing_pair_is_not_deleted(self):
        days=pd.bdate_range('2025-01-01',periods=70);records=[{'date':d,'ready':True,'release':d.tz_localize(core.ET),'loss':.01} for d in days]
        records[-2]['ready']=False;at=days[-1].tz_localize(core.ET)+pd.Timedelta(hours=15)
        r=gap_forecast(records,days[-1],at);self.assertFalse(r['ready']);self.assertEqual(len(r['members']),70)

    def test_tail_mean_uses_all_losses_in_worst_decile(self):
        days=pd.bdate_range('2025-01-01',periods=60);records=[{'date':d,'ready':True,'release':d.tz_localize(core.ET),'loss':i/1000} for i,d in enumerate(days)]
        r=gap_forecast(records,days[-1],days[-1].tz_localize(core.ET)+pd.Timedelta(hours=15))
        self.assertAlmostEqual(r['adverse_tail_mean'],np.mean(np.arange(54,60)/1000))

    def test_future_bar_cannot_change_prior_review(self):
        m,b,f=self.fixture();a,ra=managed_label(m,b,f,'buffer');other=m.bars['SOXL'].date==m.sessions[-1]
        m.bars['SOXL'].loc[other,['open','high','low','close']]=[10.,11.,9.,10.]
        z,rz=managed_label(m,b,f,'buffer');self.assertEqual(ra,rz);self.assertEqual(a['net_return'],z['net_return'])

    def test_carry_rule_does_not_guarantee_future_breakeven(self):
        self.assertEqual(review(100,104.9,{'ready':True,'adverse_tail_mean':.02},1,'buffer')['action'],'HOLD')
        self.assertEqual(review(100,101.,{'ready':True,'adverse_tail_mean':.02},1,'buffer')['action'],'EXIT')
        # A later open may exceed the historical tail; no fictitious protected fill.
        self.assertLess(90*.9995*.9995/(100*1.0005)-1,-.06)

    def test_old_future_exit_flags_and_dividends_do_not_survive_early_exit(self):
        m,b,f=self.fixture();b.update(gap_below_stop=True,same_bar_both_barriers=True)
        b['dividends']=[{'ex':m.sessions[-1],'pay':m.sessions[-1],'economic_amount':1.}];b['dividend_per_economic_unit']=1.
        lab,_=managed_label(m,b,f,'buffer');self.assertFalse(lab['gap_below_stop']);self.assertFalse(lab['same_bar_both_barriers'])
        self.assertEqual(lab['dividends'],[]);self.assertEqual(lab['dividend_per_economic_unit'],0.)

    def test_dividend_terminal_retains_conservative_full_window_release(self):
        m,b,f=self.fixture();mark=pd.Timestamp(m.calendar[b['entry_date']]['close_at'])-pd.Timedelta(minutes=15)
        m.bars['SOXL'].loc[mark,['high','close','econ_high','econ_close']]=[104.95,104.9,104.95,104.9]
        b['dividends']=[{'ex':m.sessions[-1],'pay':m.sessions[-1],'economic_amount':.1}];b['dividend_per_economic_unit']=.1
        lab,rs=managed_label(m,b,f,'buffer');self.assertEqual(rs[0]['action'],'HOLD');self.assertEqual(lab['dividend_per_economic_unit'],.1)
        self.assertEqual(lab['terminal_release'],b['label_available_at'])

    def test_independent_event_competition_matches_path_loop(self):
        from audit_paths import reference_label,compare,compare_records
        for half in [False,True]:
            for multiple in [1,2]:
                for opening in [90.,102.,108.]:
                    m,b,f=self.fixture(half);end=pd.Timestamp(m.calendar[b['entry_date']]['close_at'])-pd.Timedelta(minutes=5)
                    m.bars['SOXL'].loc[end,['open','high','low','close']]=[opening,max(opening,109),min(opening,90),opening]
                    lab,records=managed_label(m,b,f,'buffer',multiple);e,expected=reference_label(m,b,f,'buffer',multiple)
                    compare(lab,e);compare_records(records,expected)

    def test_unknown_exit_cannot_be_encoded_as_zero_return(self):
        from audit_paths import compare
        with self.assertRaises(AssertionError):compare({'exit_at':None,'terminal_release':None,'net_return':0.},None)


if __name__=='__main__':unittest.main()
