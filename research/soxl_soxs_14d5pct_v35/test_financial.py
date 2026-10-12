"""Financial period normalization, causal source availability and daily learning."""
import unittest,json,copy
from financial_learning import *
from audit_features import audit_features
from audit_calibration import daily_check
from audit_models import model_equations
from test_shared import fixture as old_fixture
from run_planning import json_ready


def wire(x):return json.loads(json.dumps(json_ready(x),default=str))


def issuer_fixture():
    tags={'revenue':TAGS['revenue'][0],'net_income':TAGS['net_income'][0]};rows={k:[] for k in tags};fil=[]
    periods=[('2023-07-01','2023-09-30',100,10,'2023-11-01','a'),('2023-10-01','2023-12-31',100,10,'2024-02-01','b'),
        ('2024-07-01','2024-09-30',150,30,'2024-11-01','c'),('2024-01-01','2024-09-30',450,90,'2024-11-01','c'),
        ('2024-01-01','2024-12-31',630,126,'2025-02-10','d')]
    for start,end,rev,net,day,a in periods:
        if not any(x['accessionNumber']==a for x in fil):fil.append({'accessionNumber':a,'filingDate':day,'acceptanceDateTime':day+'T21:00:00.000Z','form':'10-K' if a in ['b','d'] else '10-Q','items':''})
        for field,value in [('revenue',rev),('net_income',net)]:rows[field].append({'start':start,'end':end,'val':value,'filed':day,'accn':a,'form':fil[-1]['form']})
    fil.append({'accessionNumber':'earn','filingDate':'2025-02-10','acceptanceDateTime':'2025-02-10T21:00:00.000Z','form':'8-K','items':'9.01, 2.02'})
    recent={k:[x[k] for x in fil] for k in fil[0]};facts={'cik':1,'facts':{'us-gaap':{tags[k]:{'units':{'USD':v}} for k,v in rows.items()}}}
    return {'companyfacts':facts,'submissions':{'cik':'0000000001','filings':{'recent':recent}}}


def data_fixture(day='2025-02-11'):
    sources={s:issuer_fixture() for s in PANEL};at=pd.Timestamp(day+' 09:25:30',tz=core.ET).tz_convert('UTC')
    features=pd.DataFrame({'date':[pd.Timestamp(day)],'clock':[CLOCKS[0]]},index=pd.DatetimeIndex([at]))
    sessions=pd.bdate_range('2023-01-02','2025-04-30');return features,sources,sessions


def learning_fixture(n=126):
    p,h,m=old_fixture(n);rng=np.random.default_rng(346);p['release']=p.index+pd.Timedelta(days=14)
    for c in sorted(set(sum((columns(clock,'full') for clock in CLOCKS),[]))):
        if c not in p:p[c]=rng.normal(0,.5,len(p))
    return p,h,m


class FinancialTests(unittest.TestCase):
    def test_revenue_growth_acceleration_and_net_margin(self):
        f,s,d=data_fixture();x,r=build(f,s,d)
        self.assertTrue(x.financial_ready.iloc[0]);self.assertAlmostEqual(x.revenue_growth.iloc[0],np.log(1.8))
        self.assertAlmostEqual(x.growth_acceleration.iloc[0],np.log(1.2));self.assertAlmostEqual(x.margin_improvement.iloc[0],.1)
        self.assertEqual(x.recent_earnings_share.iloc[0],1.)

    def test_cumulative_q4_sources(self):
        a=issuer_fixture();p,_=extract(a['companyfacts'],filings(a['submissions']));q=quarters(p['revenue'],'2025-02-11')[-1]
        self.assertEqual(q['value'],180);self.assertTrue(q['derived']);self.assertEqual(q['start'],'2024-10-01');self.assertEqual(len(q['sources']),2)

    def test_same_priority_conflicting_facts_raise(self):
        a=issuer_fixture();p,_=extract(a['companyfacts'],filings(a['submissions']));p['revenue'].append({**p['revenue'][0],'value':999})
        with self.assertRaises(ValueError):quarters(p['revenue'],'2025-02-11')

    def test_same_day_disclosure_excluded(self):
        f,s,d=data_fixture('2025-02-10');x,r=build(f,s,d);self.assertFalse(x.financial_ready.iloc[0]);self.assertTrue(x[FINANCIAL].isna().all().all())

    def test_future_filing_values_do_not_change_past_inputs(self):
        f,s,d=data_fixture('2025-02-10');a,r=build(f,s,d)
        for v in s.values():
            for item in v['companyfacts']['facts']['us-gaap'].values():
                for row in item['units']['USD']:
                    if row['filed']>='2025-02-10':row['val']=999999
        b,q=build(f,s,d);pd.testing.assert_frame_equal(a,b);self.assertEqual(wire(r),wire(q))

    def test_later_revision_never_rewrites_earlier_state(self):
        f,s,d=data_fixture();a,_=build(f,s,d)
        for v in s.values():
            rec=v['submissions']['filings']['recent'];row={'accessionNumber':'revision','filingDate':'2025-03-03','acceptanceDateTime':'2025-03-03T21:00:00Z','form':'10-K/A','items':''}
            for k in rec:rec[k].append(row[k])
            v['companyfacts']['facts']['us-gaap'][TAGS['revenue'][0]]['units']['USD'].append({'start':'2024-01-01','end':'2024-12-31','val':900,'filed':'2025-03-03','accn':'revision','form':'10-K/A'})
        b,_=build(f,s,d);pd.testing.assert_frame_equal(a,b)
        g,_,_=data_fixture('2025-03-04');c,_=build(g,s,d);self.assertGreater(c.revenue_growth.iloc[0],a.revenue_growth.iloc[0])

    def test_missing_one_issuer_abstains(self):
        f,s,d=data_fixture();s[PANEL[0]]['companyfacts']['facts']['us-gaap']={};x,r=build(f,s,d);self.assertFalse(x.financial_ready.iloc[0])

    def test_wrong_currency_not_used_as_dollars(self):
        a=issuer_fixture();g=a['companyfacts']['facts']['us-gaap'][TAGS['revenue'][0]];g['units']['EUR']=g['units'].pop('USD')
        p,_=extract(a['companyfacts'],filings(a['submissions']));self.assertEqual(p['revenue'],[])

    def test_unknown_accession_is_not_assumed_known(self):
        a=issuer_fixture();a['companyfacts']['facts']['us-gaap'][TAGS['revenue'][0]]['units']['USD'][-1]['accn']='missing'
        p,e=extract(a['companyfacts'],filings(a['submissions']));self.assertEqual(e['accession_not_in_cached_catalog'],1);self.assertEqual(len(p['revenue']),4)

    def test_filed_date_mismatch_is_retained_as_exclusion(self):
        a=issuer_fixture();a['companyfacts']['facts']['us-gaap'][TAGS['revenue'][0]]['units']['USD'][-1]['filed']='2025-02-09'
        p,e=extract(a['companyfacts'],filings(a['submissions']));self.assertEqual(e['filing_metadata_mismatch'],1)

    def test_company_amount_units_do_not_change_ratios(self):
        f,s,d=data_fixture();a,_=build(f,s,d)
        for j,v in enumerate(s.values()):
            for item in v['companyfacts']['facts']['us-gaap'].values():
                for row in item['units']['USD']:row['val']*=j+2
        b,_=build(f,s,d);np.testing.assert_allclose(a[FINANCIAL],b[FINANCIAL],atol=1e-12)

    def test_weekend_disclosure_first_observed_next_session(self):
        f,s,d=data_fixture('2025-02-17')
        for v in s.values():
            r=v['submissions']['filings']['recent'];r['filingDate'][-1]='2025-02-14';r['acceptanceDateTime'][-1]='2025-02-14T21:00:00Z'
        x,r=build(f,s,d);self.assertEqual(x.recent_earnings_share.iloc[0],1.)

    def test_late_utc_date_never_uses_earlier_et_midnight(self):
        a=issuer_fixture();r=a['submissions']['filings']['recent'];r['acceptanceDateTime'][-1]='2025-02-11T02:00:00Z'
        self.assertEqual(filings(a['submissions'])['earn']['available_date'],'2025-02-12')

    def test_independent_feature_reconstruction(self):
        f,s,d=data_fixture();x,r=build(f,s,d);a=audit_features(f,x,wire(r),s,d);self.assertTrue(a['passed']);self.assertEqual(a['ready_ticks'],1)

    def test_raw_model_equations_and_future_labels(self):
        p,h,m=learning_fixture();at=p.index[-8];models,records,raw=fit_bases(p,at);same=p[p.clock==p.loc[at,'clock']]
        for k in VARIANTS:
            ids=records[k]['fit']['members'];want,n=model_equations(same,ids,at,models[k],records[k]['fit'],columns(p.loc[at,'clock'],k));self.assertEqual(n,3);np.testing.assert_allclose(raw[k],[want[s] for s in core.ETFS])
        p.loc[p.date>p.loc[at,'date'],FINANCIAL+['C','D']]=99999;p.loc[p.date>=p.loc[at,'date'],['C','D']]=99999
        _,new,newraw=fit_bases(p,at);self.assertEqual(wire(records),wire(new))
        for k in VARIANTS:np.testing.assert_array_equal(raw[k],newraw[k])

    def test_calibration_uses_issued_history(self):
        p,h,m=learning_fixture(63);at=p.index[-4];hist={k:{t:v+(j-1)*.005 for t,v in h.items()} for j,k in enumerate(VARIANTS)}
        d=fit_calibration(p,hist,registry(p,m),at);_,n=daily_check(p,m,hist,at,wire(d));self.assertEqual(n,1200)
        for k in VARIANTS:hist[k][at]=np.array([999.,999.])
        self.assertEqual(wire(d),wire(fit_calibration(p,hist,registry(p,m),at)))

    def test_unknown_terminal_blocks_later_labels(self):
        p,h,m=learning_fixture(63);at=p.index[-4];p.loc[p.index[20],'terminal_release']=pd.NaT
        d=fit_calibration(p,{k:h.copy() for k in VARIANTS},registry(p,m),at);self.assertFalse(d['ready']);self.assertEqual(len(d['parents']),5)


if __name__=='__main__':unittest.main()
