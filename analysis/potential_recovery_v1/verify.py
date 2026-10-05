#!/usr/bin/env python3
"""Audit source amounts, candle fidelity, temporal bounds and honest exclusions."""
from pathlib import Path
import hashlib,json,re,sys
from datetime import datetime,timezone
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    report=json.loads((HERE/'screen_results.json').read_text());stocks=report['stocks'];meta=report['metadata'];source=json.loads((HERE/'source_registration.json').read_text())
    html=(HERE/'index.html').read_text();data=json.loads(re.search(r'<script type="application/json" id="embedded-data">(.*?)</script>',html,re.S).group(1))
    registry=source['backtest_registration'];panel=pd.read_parquet(registry['panel_path']);groups={t:g for t,g in panel.groupby('ticker')}
    official={s['session_date'] for s in calendar('2023-01-01',meta['daily_asof'])['sessions']};checks=0;annual_facts=0
    for r in data['stocks']:
        t=r['ticker'];g=groups[t].copy();g['day']=g.date.dt.strftime('%Y-%m-%d');g=g[(g.day<=meta['daily_asof'])&g.day.isin(official)].sort_values('date')
        cols=['open','high','low','close'];g=g[g[cols].gt(0).all(axis=1)&g.high.ge(g[cols].max(axis=1))&g.low.le(g[cols].min(axis=1))&g.volume.ge(0)]
        year=g[g.day>='2026-01-01'];peak=year.loc[year.high.idxmax()];low=year[year.date>=peak.date].low.min()
        assert abs(r['year_peak']-peak.high)<1e-8 and r['year_peak_date']==peak.day,t
        assert abs(r['peak_after_low']-low)<1e-8 and r['peak_after_low_date']>=r['year_peak_date'],t
        assert abs(r['year_ratio']-peak.high/r['reference_price'])<1e-10,t
        assert abs(r['recovery_fraction']-(r['reference_price']-low)/(peak.high-low))<1e-10,t
        expected=g[g.day>=meta['navigation_start']]
        bars=r['bars'];assert len(bars)==len(expected) and len({b[0] for b in bars})==len(bars),t
        assert [b[0] for b in bars]==expected.day.tolist(),t
        assert np.allclose(np.array([b[1:] for b in bars]),expected[cols+['volume']].to_numpy(),atol=1e-7,rtol=0),t
        checks+=len(bars)
        raw=json.loads((ROOT/f".cache/doubling_opportunity_v1/coverage_v4/companyfacts/CIK{r['cik']:010d}.json").read_text())
        for year_rec in r['financial']['annual']:
            for key,fact in year_rec['fact_sources'].items():
                if fact is None:assert year_rec.get(key) is None;continue
                assert fact['start']==year_rec['start'] and fact['end']==year_rec['end'] and fact['filed']<source['financial_cutoff_exclusive'],(t,key)
                namespace=next(n for n in ['us-gaap','dei'] if fact['tag'] in raw['facts'].get(n,{}));unit='shares' if key=='diluted_shares' else 'USD'
                direct=raw['facts'][namespace][fact['tag']]['units'][unit]
                assert any(x.get('start')==fact['start'] and x['end']==fact['end'] and x.get('accn')==fact['accn'] and x['val']==fact['val'] for x in direct),(t,key,'raw amount mismatch')
                annual_facts+=1
            if year_rec['fcf'] is not None:assert year_rec['fcf']==year_rec['operating_cashflow']-year_rec['capex']
        if r['quality_tier'].startswith('A'):
            assert all(r['financial']['checks'].values()) and r['quality_score']==100 and r['price_basis_clean'] and r['market_cap']>=2e9,t
    ledger=pd.read_csv(HERE/'all_pool_outcomes.csv',keep_default_na=False).set_index('ticker')
    lineage=json.loads((Path(registry['path'])/'lineage.json').read_text());universe=lineage['registration']['universe']['stocks']
    assert len(ledger)==meta['registered_pool'] and ledger.index.is_unique and set(ledger.index)==set(universe)
    coverage=json.loads((HERE/'coverage_audit.json').read_text());audit={r['ticker']:r for r in coverage['stocks']}
    assert set(audit)==set(universe) and len(data['audit'])==len(universe)
    assert {r['ticker']:r for r in data['audit']}==audit
    valid=panel[cols].gt(0).all(axis=1)&np.isfinite(panel[cols+['volume']]).all(axis=1)&panel.high.ge(panel[cols].max(axis=1))&panel.low.le(panel[cols].min(axis=1))&panel.volume.ge(0)
    source_dates=panel.loc[valid&panel.date.le(meta['daily_asof'])].copy();source_dates['day']=source_dates.date.dt.strftime('%Y-%m-%d')
    source_dates=source_dates[source_dates.day.isin(official)]
    days_by_ticker=source_dates.groupby('ticker').day.agg(set).to_dict()
    coverage_checks=0
    for w,start,end in [('YTD','2026-01-01',meta['daily_asof']),('AprJun','2026-04-01','2026-06-30'),('M04','2026-04-01','2026-04-30'),('M05','2026-05-01','2026-05-31'),('M06','2026-06-01','2026-06-30')]:
        wanted={s['session_date'] for s in calendar(start,end)['sessions']};counts={'complete':0,'partial':0,'zero':0}
        for t in universe:
            days=days_by_ticker.get(t,set());missing=sorted(wanted-days);r=audit[t];seen=len(wanted)-len(missing)
            assert r[w+'_missing_dates']==missing and r[w+'_expected']==len(wanted) and r[w+'_observed']==seen,(t,w)
            assert r[w+'_before_first']==sum(d<min(days) for d in missing) if days else r[w+'_before_first']==0
            assert int(ledger.loc[t,w+'_missing'])==len(missing),(t,w)
            counts['complete' if not missing else 'zero' if not seen else 'partial']+=1;coverage_checks+=1
        assert all(counts[k]==meta['coverage'][w][k] for k in counts),w
    original=pd.read_csv(Path(source['source_report_path'])/'all_stocks.csv',keep_default_na=False).set_index('ticker')
    nums=original[['operating_margin','net_margin','ocf_margin_ttm','revenue_yoy','report_age_days','dollar_volume_actual']].apply(pd.to_numeric,errors='coerce')
    last_close=source_dates.sort_values('date').groupby('ticker').close.last()
    core=nums.operating_margin.ge(.08)&nums.net_margin.ge(.04)&nums.ocf_margin_ttm.ge(.06)&nums.revenue_yoy.ge(-.1)&nums.report_age_days.le(150)&nums.dollar_volume_actual.ge(1e7)&last_close.reindex(nums.index).ge(5)&pd.Series({t:m['security_description_confirmed'] for t,m in universe.items()}).reindex(nums.index)
    expected_core=set(nums.index[core]);actual_core={r['ticker'] for r in stocks if not r['comparison_only']}
    assert actual_core==expected_core and len(expected_core)==meta['preliminary_core_candidates'],'whole-pool candidate omissions'
    quotes=json.loads((ROOT/'.cache/potential_recovery_v1/quotes/latest_market_snapshots.json').read_text())
    for r in stocks:
        q=quotes['stocks'].get(r['ticker'],{})
        if r['ratio_source']=='current_snapshot':
            assert str(q['update_time']).startswith(source['quote_session_date']) and r['reference_price']==q['last_price'],r['ticker']
        else:assert r['reference_price']==r['daily_close'],r['ticker']
    cat=pd.read_csv(HERE/'catalog_outcomes.csv',keep_default_na=False)
    assert len(cat)==meta['catalog']['directory_count'] and cat.ticker.is_unique
    assert set(cat[cat.catalog_status=='registered'].ticker)==set(universe)
    assert set(cat[cat.catalog_status=='outside_registered'].ticker)=={r['ticker'] for r in lineage['registration']['universe']['exclusions']}
    candidates=[r for r in stocks if not r['comparison_only']]
    for v in [1.2,1.5,1.8,1.9,2.,2.5]:
        assert sum(r['year_ratio']>=v for r in candidates)==meta['ratio_counts'][str(v)]
        assert sum(r['daily_year_ratio']>=v for r in candidates)==meta['daily_core_ratio_counts'][str(v)]
        assert sum(r['year_ratio']>=v and r['quality_tier'].startswith('A') for r in candidates)==meta['A_ratio_counts'][str(v)]
    assert meta['ratio_prescreen_removed'] and meta['new_1_2_to_1_5']['outside_prior_prescreen']>0
    assert next(r for r in stocks if r['ticker']=='MXL')['comparison_only']
    assert meta['probabilities_or_signals_created'] is False
    assert 'src="http' not in html and 'href="http' not in html.split('<script')[0] # source links built from embedded notes
    receipt=json.loads((HERE/'SOURCE_EVIDENCE.json').read_text())
    for p,s in receipt['source_paths_and_sha256'].items():assert sha(p)==s,p
    out={'verified_at':datetime.now(timezone.utc).isoformat(),'revision':meta['revision'],'stocks':len(stocks),'daily_bars_compared_to_source':checks,'annual_facts_compared_to_SEC_payload':annual_facts,'all_pool_outcomes':meta['registered_pool'],'stock_window_date_sets_independently_checked':coverage_checks,'catalog_securities_accounted':len(cat),'whole_pool_core_membership_exact_match':True,'ratio_prescreen_removed':True,'source_hashes_checked':len(receipt['source_paths_and_sha256']),'checks_passed':True,'financial_strength_is_not_predictive_validation':True,'business_risks_retained':True,'html_sha256':sha(HERE/'index.html'),'limits':['Daily session decomposition not verified','Whole-pool current quotes remain incomplete; all basic candidates have same-session snapshots','Registered current catalog is not all world equities or historical PIT','Incomplete dates cannot establish complete annual peak','No predictive probability or independent future outcome validation']}
    (HERE/'DATA_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
