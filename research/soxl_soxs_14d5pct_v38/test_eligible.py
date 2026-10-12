import unittest
from copy import deepcopy
from eligible_risk import *
from verify_eligible import daily_audit,canonical,check_event
from test_selected import Tests as SourceTests


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SourceTests.setUpClass();cls.panel=SourceTests.panel;cls.records=SourceTests.records;cls.market=SourceTests.market;cls.at=SourceTests.at
        cls.reg=registry(cls.panel,cls.market);cls.daily=fit_day(cls.panel,cls.records,cls.reg,cls.at)
        assert cls.daily['ready']

    def test_independent_all_bootstrap_fits(self):
        self.assertEqual(daily_audit(self.panel,self.market,self.records,self.at,canonical(self.daily)),600)

    def test_members_must_be_as_issued_eligible(self):
        ids=self.daily['inputs']['eligible_context']['members'];self.assertTrue(ids)
        self.assertTrue(all(self.records[t]['events']['full_band']['action'] in core.ETFS for t in ids))
        self.assertLess(len(ids),len(self.daily['inputs']['all_context']['members']))

    def test_actual_outcomes_do_not_determine_members(self):
        p=self.panel.copy();p['selected_y']*=-1
        d=fit_day(p,self.records,self.reg,self.at)
        for k in MODELS:self.assertEqual(self.daily['inputs'][k]['members'],d['inputs'][k]['members'])

    def test_future_records_do_not_change_fits(self):
        p=self.panel.copy();p.loc[p.index>=day_cutoff(self.at),'selected_y']=999;records=deepcopy(self.records)
        for at,r in records.items():
            if at>=day_cutoff(self.at):r['events']['full_band']['action']='CASH';r['estimates']['full']['point']=[999.,-999.]
        self.assertEqual(canonical(self.daily),canonical(fit_day(p,records,self.reg,self.at)))

    def test_unknown_date_stops_prefix_before_qualification(self):
        r=deepcopy(self.reg);r[12]['release']=None;d=fit_day(self.panel,self.records,r,self.at)
        self.assertEqual(len(d['parents']),12);self.assertFalse(d['ready']);self.assertTrue(d['feedback']['expired_unresolved_prefix'])

    def test_minimum_is_dates_not_qualified_rows(self):
        records=deepcopy(self.records)
        for i,p in enumerate(self.reg):
            if i>=29:
                for at in p['members']:records[at]['events']['full_band'].update(action='CASH',allocation=0.)
        d=fit_day(self.panel,records,self.reg,self.at);self.assertFalse(d['ready']);self.assertLessEqual(len(d['subsets']['eligible']),29)

    def test_eligible_date_weight_not_clock_count(self):
        pp=[{'parent_index':0,'members':[1]},{'parent_index':3,'members':[2,3,4]}]
        w,g=subset_weights(pp,4);self.assertAlmostEqual(w.sum(),2)
        self.assertAlmostEqual(w[g==0].sum()/w[g==3].sum(),2**(-3/63))

    def test_eligibility_does_not_compress_age(self):
        pp=[{'parent_index':0,'members':[1]},{'parent_index':63,'members':[2]}]
        w,g=subset_weights(pp,64);self.assertAlmostEqual(w[0]/w[1],.5)

    def test_support_exact_boundary_and_constant(self):
        s={'columns':['x','constant'],'minimum':[0,2],'maximum':[1,2]}
        self.assertEqual(outside_columns([0,2],s),[]);self.assertEqual(outside_columns([1,2],s),[])
        self.assertEqual(outside_columns([1.01,2.1],s),['x','constant'])

    def test_supported_override_and_unsupported_fallback(self):
        d=deepcopy(self.daily);r=deepcopy(self.records[self.at]);r['events']['full_band'].update(action='SOXL',allocation=1.)
        estimates=issue(self.panel,self.at,r,d)
        for v in estimates.values():v.update(quality=.1,lower_quality=.05,outside_columns=[])
        es=events(self.at,self.panel.loc[self.at,'date'],estimates,d,r,r['events']['full_band'])
        self.assertEqual(es['eligible_supported']['action'],'CASH')
        estimates['eligible_context']['outside_columns']=['volatility']
        es=events(self.at,self.panel.loc[self.at,'date'],estimates,d,r,r['events']['full_band'])
        self.assertEqual(es['eligible_supported']['action'],'SOXL');self.assertEqual(es['eligible_gate']['action'],'CASH')
        self.assertEqual(es['eligible_supported']['reason'],'source_outside_support')

    def test_source_rejection_is_never_overridden(self):
        r=deepcopy(self.records[self.at]);r['events']['full_band'].update(action='CASH',allocation=0.)
        estimates=issue(self.panel,self.at,r,self.daily)
        for v in estimates.values():v.update(quality=.99,lower_quality=.99,outside_columns=['volatility'])
        es=events(self.at,self.panel.loc[self.at,'date'],estimates,self.daily,r,r['events']['full_band'])
        for e in es.values():self.assertEqual(e['action'],'CASH')

    def test_warmup_preserves_source(self):
        d=deepcopy(self.daily);d['ready']=False;r=self.records[self.at];estimates=issue(self.panel,self.at,r,d)
        es=events(self.at,self.panel.loc[self.at,'date'],estimates,d,r,r['events']['full_band'])
        for name,e in es.items():
            if name!='cash':self.assertEqual(e['action'],r['events']['full_band']['action'])

    def test_empty_resample_stays_unknown(self):
        d=deepcopy(self.daily);d['bootstrap'][0]['eligible_context']=None;r=deepcopy(self.records[self.at]);r['events']['full_band'].update(action='SOXL',allocation=1.)
        estimates=issue(self.panel,self.at,r,d);v=estimates['eligible_context'];self.assertFalse(v['bootstrap_complete']);self.assertIsNone(v['lower_quality'])
        self.assertIsNone(v['bootstrap_qualities'][0]);v.update(outside_columns=[],quality=.9)
        es=events(self.at,self.panel.loc[self.at,'date'],estimates,d,r,r['events']['full_band'])
        self.assertEqual(es['eligible_supported']['action'],'SOXL');self.assertEqual(es['eligible_band']['action'],'CASH')

    def test_full_event_audit(self):
        r=self.records[self.at];info=issue(self.panel,self.at,r,self.daily)
        es=events(self.at,self.panel.loc[self.at,'date'],info,self.daily,r,r['events']['full_band'])
        for k,v in es.items():check_event(k,canonical(v),canonical(r),canonical(self.daily),canonical(info),canonical(r['events']['full_band']))


if __name__=='__main__':unittest.main()
