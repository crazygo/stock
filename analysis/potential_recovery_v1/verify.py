#!/usr/bin/env python3
"""Audit source amounts, candle fidelity, temporal bounds and honest exclusions."""
from pathlib import Path
import hashlib,json,re,sys
from datetime import datetime,timezone
import pandas as pd
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
        for b,(_,native) in zip(bars,expected.iterrows()):
            assert b[0]==native.day and all(abs(v-native[k])<1e-7 for v,k in zip(b[1:],cols+['volume'])),(t,b[0]);checks+=1
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
    assert len(pd.read_csv(HERE/'all_pool_outcomes.csv'))==meta['registered_pool']
    candidates=[r for r in stocks if not r['comparison_only']]
    for v in [1.5,1.8,1.9,2.,2.5]:
        assert sum(r['year_ratio']>=v for r in candidates)==meta['ratio_counts'][str(v)]
    assert next(r for r in stocks if r['ticker']=='ISRG')['year_ratio']<1.5
    assert next(r for r in stocks if r['ticker']=='MXL')['comparison_only']
    assert meta['probabilities_or_signals_created'] is False
    assert 'src="http' not in html and 'href="http' not in html.split('<script')[0] # source links built from embedded notes
    receipt=json.loads((HERE/'SOURCE_EVIDENCE.json').read_text())
    for p,s in receipt['source_paths_and_sha256'].items():assert sha(p)==s,p
    out={'verified_at':datetime.now(timezone.utc).isoformat(),'stocks':len(stocks),'daily_bars_compared_to_source':checks,'annual_facts_compared_to_SEC_payload':annual_facts,'all_pool_outcomes':meta['registered_pool'],'source_hashes_checked':len(receipt['source_paths_and_sha256']),'checks_passed':True,'financial_strength_is_not_predictive_validation':True,'business_risks_retained':True,'html_sha256':sha(HERE/'index.html'),'limits':['Daily session decomposition not verified','Current quotes only prefiltered universe','No predictive probability or independent future outcome validation']}
    (HERE/'DATA_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
