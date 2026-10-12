"""Scenario maturity, payoff semantics and independent daily fit checks."""
import unittest,json
from unittest.mock import patch
from cost_robust import *
from audit_cost import daily_check
from test_shared import fixture as old_fixture
from run_planning import json_ready


def fixture(n=63):
    p,h,m=old_fixture(n);p['base_L']=p.C+p.D;p['base_S']=p.C-p.D
    p['stress_L']=p.base_L-.002;p['stress_S']=p.base_S-.002
    p.loc[p.index[::13],'stress_L']-=.1;p.loc[p.index[::17],'stress_S']+=.04
    return p,h,m


def wire(x):return json.loads(json.dumps(json_ready(x),default=str))


class CostTests(unittest.TestCase):
    def test_worst_uses_worse_scenario_even_if_stress_improves(self):
        p,_,_=fixture();a=p[['base_L','base_S']].to_numpy();b=p[['stress_L','stress_S']].to_numpy()
        np.testing.assert_allclose(target_values(p,'worst'),np.minimum(a,b))
        self.assertTrue((b>a).any());self.assertTrue((b<a).any())

    def test_average_is_mean_log_not_log_mean(self):
        p,_,_=fixture();a=p[['base_L','base_S']].to_numpy();b=p[['stress_L','stress_S']].to_numpy()
        np.testing.assert_allclose(target_values(p,'average'),(a+b)/2)
        self.assertGreater(np.max(abs(target_values(p,'average')-np.log((np.exp(a)+np.exp(b))/2))),1e-5)

    def test_baseline_control_ignores_stress_target(self):
        p,_,_=fixture();want=target_values(p,'baseline');p[['stress_L','stress_S']]=np.nan
        np.testing.assert_array_equal(target_values(p,'baseline'),want)

    def test_all_bootstrap_models_independent_equations(self):
        p,h,m=fixture();at=p.index[-4];saved=fit_day(p,h,registry(p,m),at)
        _,checks=daily_check(p,m,h,at,wire(saved));self.assertEqual(checks,1200)

    def test_unknown_one_cost_keeps_pair_unreleased_and_source_unchanged(self):
        p,_,m=fixture(2);p=p.iloc[:1].copy();before=p.copy();at=p.index[0];base={};stress={}
        for s in core.ETFS:
            base[key(at,s)]={'net_return':.04,'r':{'known':True,'release':at+pd.Timedelta(hours=1)}}
            stress[key(at,s)]={'net_return':-.06,'r':{'known':s=='SOXL','release':at+pd.Timedelta(hours=2) if s=='SOXL' else None}}
        with patch('cost_robust.prefix_receipt',side_effect=lambda _,label:label['r']):b,r=joint_feedback(m,p,base,stress)
        self.assertTrue(pd.isna(b.loc[at,'terminal_release']));self.assertTrue(pd.isna(b.loc[at,'stress_S']))
        self.assertIsNone(r[0]['joint_terminal_release']);pd.testing.assert_frame_equal(before,p)

    def test_joint_release_is_latest_of_four_paths(self):
        p,_,m=fixture(2);p=p.iloc[:1];at=p.index[0];base={};stress={}
        for j,(name,lab) in enumerate([('base',base),('stress',stress)]):
            for k,s in enumerate(core.ETFS):lab[key(at,s)]={'net_return':.04,'r':{'known':True,'release':at+pd.Timedelta(hours=1+j*2+k)}}
        with patch('cost_robust.prefix_receipt',side_effect=lambda _,label:label['r']):b,_=joint_feedback(m,p,base,stress)
        self.assertEqual(b.loc[at,'terminal_release'],at+pd.Timedelta(hours=4))

    def test_late_cost_blocks_whole_day_and_later_dates(self):
        p,h,m=fixture();at=p.index[-4];target=p.index[20];p.loc[target,'terminal_release']=pd.NaT
        parents,fb=pool(registry(p,m),h,at)
        self.assertEqual(len(parents),5);self.assertTrue(fb['expired_unresolved_prefix'])
        self.assertTrue(all(x['date']<p.loc[target,'date'] for x in parents))

    def test_exact_cutoff_not_admitted(self):
        p,h,m=fixture();at=p.index[-4];cut=day_cutoff(at);target=p.index[20];p.loc[target,'terminal_release']=cut
        self.assertEqual(len(pool(registry(p,m),h,at)[0]),5)
        p.loc[target,'terminal_release']=cut-pd.Timedelta(nanoseconds=1)
        self.assertGreater(len(pool(registry(p,m),h,at)[0]),5)

    def test_future_labels_do_not_change_fit(self):
        p,h,m=fixture();at=p.index[-4];a=fit_day(p,h,registry(p,m),at)
        p.loc[p.date>=p.loc[at,'date'],['base_L','base_S','stress_L','stress_S']]=999
        b=fit_day(p,h,registry(p,m),at);self.assertEqual(wire(a),wire(b))

    def test_constant_cost_offset_preserved_in_band(self):
        p,h,m=fixture();p['stress_L']=p.base_L-.001;p['stress_S']=p.base_S-.001;at=p.index[-4]
        d=fit_day(p,h,registry(p,m),at);e=issue_cost(p,at,h[at],d)
        for field in ['point','lower']:np.testing.assert_allclose(np.array(e['worst'][field]),np.array(e['baseline'][field])-.001,atol=1e-12)
        np.testing.assert_allclose(e['worst']['radius'],e['baseline']['radius'],atol=1e-12)

    def test_halfday_weights_and_minimum_count_dates(self):
        p,h,m=fixture(59);at=p.index[-4];d=fit_day(p,h,registry(p,m),at);self.assertFalse(d['ready'])
        p,h,m=fixture();at=p.index[-4];d=fit_day(p,h,registry(p,m),at)
        self.assertAlmostEqual(sum(d['row_weights']),len(d['parents']))
        self.assertIn(3,[len(x['members']) for x in d['parents']])
        for j,p in enumerate(d['parents']):self.assertAlmostEqual(sum(np.array(d['row_weights'])[d['row_groups']==j]),d['day_weights'][j])

    def test_no_current_scenario_needed_for_query(self):
        p,h,m=fixture();at=p.index[-4];d=fit_day(p,h,registry(p,m),at);a=issue_cost(p,at,h[at],d)
        p.loc[at,['base_L','base_S','stress_L','stress_S']]=np.nan;b=issue_cost(p,at,h[at],d)
        self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
