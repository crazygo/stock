"""Read-only feasibility for a later price-consistent state experiment, not v68 logic."""
import argparse
from datetime import datetime,timezone
from path_learning import *


def audit(inputs):
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');assert m['phase']=='development'
    market=core.load_market();assert market.manifest==m['full_market'];cutoff=pd.Timestamp(m['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    leaders=load_panel(market.calendar,PANEL_ROOT);origin=core.ROOT/'.cache/soxl_soxs_14d5pct_v55/runs/development_20261013/input_manifest.json';assert leaders[2]==read(origin)['panel']
    symbols=['SOXX','QQQ',*read(SOURCE/'protocol.json')['panel']];bars={s:(market.bars[s] if s in ['SOXX','QQQ'] else leaders[0][s]) for s in symbols};daily={};quote_ready={};sessions=list(market.sessions);pos={d:i for i,d in enumerate(sessions)}
    for symbol,f in bars.items():
        f=f[(f.end_at<cutoff)&(f.available_at<cutoff)];daily[symbol]={};quote_ready[symbol]={}
        for day,g in f.groupby('date'):
            cal=market.calendar.get(day)
            if cal is None:continue
            expected=pd.date_range(cal['open_at'],cal['close_at'],freq='5min',inclusive='right');reg=g[g.index.isin(expected)]
            good=reg.index.equals(expected) and reg.available_at.notna().all() and reg.available_at.ge(reg.end_at).all() and reg.econ_close.gt(0).all() and np.isfinite(reg.econ_close).all()
            if good:daily[symbol][day]=(float(reg.econ_close.iloc[-1]),reg.available_at.max())
            for clock in CLOCKS:
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
                if end not in g.index:continue
                row=g.loc[end]
                if pd.notna(row.available_at) and end<=row.available_at<=at and row.start_at==end-pd.Timedelta(minutes=5) and np.isfinite(row.econ_close) and row.econ_close>0:quote_ready[symbol][at]=row.available_at
    panel=pd.read_parquet(inputs/'activity_panel.parquet');bank=dict(np.load(inputs/'path_bank.npz'));datechecks=0;quote_failures=0;history_failures=0;statechecks=0;preserved_missing=0;source_missing=set();query_rows=[]
    for day,g in panel.groupby('date'):
        prior=sessions[max(0,pos[day]-20-1):pos[day]];known=day.tz_localize(core.ET).tz_convert('UTC');ready=len(prior)==21 and all(d in daily[s] and daily[s][d][1]<known for s in symbols for d in prior)
        if not ready:history_failures+=1;continue
        prices={s:np.array([daily[s][d][0] for d in prior]) for s in symbols};rets={s:prices[s][-1]/prices[s][-6]-1 for s in symbols};vol=max(float(np.std(prices['SOXX'][1:]/prices['SOXX'][:-1]-1,ddof=1)),1e-4)
        wanted=[float(np.clip((rets['SOXX']-rets['QQQ'])/(vol*np.sqrt(5)),-5,5)),float(2*np.mean([rets[s]>0 for s in symbols[2:]])-1),float(np.log(vol))]
        saved=g[PRIOR].to_numpy(float);finite=np.isfinite(saved);expected=np.tile(wanted,(len(g),1));assert np.isfinite(g.loc[g.feature_ready,PRIOR].to_numpy(float)).all()
        np.testing.assert_allclose(saved[finite],expected[finite],rtol=1e-10,atol=1e-12);datechecks+=1;statechecks+=int(finite.sum());preserved_missing+=int((~finite).sum())
        for at,row in g.iterrows():
            query_ok=all(at in quote_ready[s] for s in symbols);quote_failures+=int(not query_ok)
            members=price_known_pool(bank,at,row.clock);added_fail=[]
            for i in members:
                missing=[]
                for dayid in bank['blocks'][i]:
                    source_day=pd.Timestamp(bank['date_ns'][dayid]);j=pos[source_day];previous=sessions[j-1] if j else None
                    for s in symbols:
                        if source_day not in daily[s] or previous not in daily[s] or max(daily[s][source_day][1],daily[s][previous][1])>=known:missing.append(s+':daily')
                source_day=pd.Timestamp(bank['date_ns'][bank['blocks'][i,0]]);source_at=pd.Timestamp(str(source_day.date())+' '+row.clock,tz=core.ET).tz_convert('UTC')
                for s in symbols:
                    if source_at not in quote_ready[s] or quote_ready[s][source_at]>=known:missing.append(s+':quote')
                if missing:added_fail.append(int(i));source_missing.update(missing)
            if '2026-01-01'<=str(day.date())<='2026-04-30':query_rows.append({'at':at,'date':day,'current_quotes_ready':query_ok,'source_members':len(members),'missing_reference_members':added_fail})
    out={'passed':True,'scope':'Coverage and initial-state formula feasibility only; no new dynamics or capital strategy implemented, no guarantee that future path weights remain supported. Not part of frozen v68 strategy.',
         'checked_at':datetime.now(timezone.utc).isoformat(),'capital_outcomes_read_by_this_audit':False,'new_data_downloaded':False,'symbols':symbols,'reference_manifest_sha256':sha(origin),'reference_bars_and_action_hashes':leaders[2],'initial_history_dates_ready':datechecks,'initial_history_dates_missing':history_failures,'initial_state_formula_checks':statechecks,'original_missing_state_fields_preserved':preserved_missing,'current_quote_missing_queries':quote_failures,'evaluation_queries':len(query_rows),'evaluation_queries_all_references_available':sum(x['current_quotes_ready'] and not x['missing_reference_members'] for x in query_rows),'source_missing_kinds':sorted(source_missing),'rows':query_rows,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'source_sha256':sha(Path(__file__))}
    dump(SOURCE/'evidence/next_inputs_audit.json',out);print({k:v for k,v in out.items() if k not in ['rows','reference_bars_and_action_hashes']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
