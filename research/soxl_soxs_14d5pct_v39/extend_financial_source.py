"""Extend v35 daily forecasts under verified v39 permission; no upstream qualification."""
import argparse,gzip,pickle,sys
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[1]/"soxl_soxs_14d5pct_v35"))
from datetime import datetime,timezone
from pathlib import Path
from threadpoolctl import threadpool_limits
from financial_learning import *
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay
SOURCE=Path(__file__).resolve().parents[1]/"soxl_soxs_14d5pct_v35"
from supported import authorize as consumer_authorize, load_frozen as consumer_frozen


def run(output,case_after=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json')
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(core.ROOT/f)==h
    assert case_after is not None
    consumer=consumer_authorize(case_after);cp,cr=consumer_frozen()
    auth={'run':str((core.ROOT/cp['forecast_source']).resolve()),'consumer_authorization':consumer,'upstream_policy_qualified':False};phase='cases'
    prereg={**prereg,'code_sha256':cr['code_sha256']}
    output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve every attempt')
    for sub in ['','training','models','executed_sources']:(output/sub).mkdir(parents=True,exist_ok=True)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,
        'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':auth,'consumer_authorization':consumer,'source_sha256':prereg['code_sha256']})
    for f in prereg['code_sha256']:
        src=core.ROOT/f;dest=output/'executed_sources'/src.parent.name/src.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(src.read_bytes())
    market,features,panel,base,stress,manifest=prepare_inputs(p,phase,output);dump(output/'input_manifest.json',manifest)
    original_panel=panel.copy();financial_sources,financial_manifest=load_sources()
    new_features,feature_receipts=build(features,financial_sources,market.sessions)
    new_features.to_parquet(output/'financial_features.parquet');dump(output/'feature_receipts.json',feature_receipts)
    panel=panel.join(new_features);panel['feature_ready']=panel.feature_ready & panel.financial_ready
    panel.to_parquet(output/'financial_panel.parquet');manifest['financial_panel_source']=financial_manifest;dump(output/'input_manifest.json',manifest)
    histories={kind:{} for kind in VARIANTS};reg=registry(panel,market);source_reg=registry(original_panel,market)
    source_history={};source_daily=None;predictions={n:[] for n in ROUTES};previous_day=None;daily=None
    oldroot=core.ROOT/p['control_source'];old_delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v32/evidence/delivery_manifest.json')
    for file in ['predictions.json','partial_band_equity.csv','partial_band_trades.csv']:
        assert sha(oldroot/file)==old_delivery['cache_outputs_sha256'][file]
    old={pd.Timestamp(e['at']):e for e in read(oldroot/'predictions.json')['partial_band']}
    (output/'daily').mkdir();(output/'source_daily').mkdir();dump(output/'parent_registry.json',reg)
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            c=row.clock;source_raw,source_origin,source_model=base_forecast(original_panel,at,manifest)
            if row.date!=previous_day:
                daily=fit_calibration(panel,histories,reg,at);dump(output/'daily'/(str(row.date.date())+'.json'),daily);previous_day=row.date
                if at not in old:
                    source_daily=source_fit_day(original_panel,source_history,source_reg,at);dump(output/'source_daily'/(str(row.date.date())+'.json'),source_daily)
            models,fits,raws=fit_bases(panel,at);estimates=issue(panel,at,raws,daily)
            old_decision=None if at in old else source_issue(original_panel,at,source_raw,source_daily)
            prior=old[at] if at in old else decide(old_decision['partial'],source_daily['feedback'],True)
            record={'at':at,'date':row.date,'clock':c,'cutoff':day_cutoff(at),'raws':raws,'fits':fits,
                'source_raw':source_raw,'source_origin':source_origin,'daily_record':str(row.date.date())+'.json',
                'estimates':estimates,'old':prior,'old_decision':old_decision}
            issued=events(at,row.date,c,estimates,daily['feedback'],prior);record['events']=issued
            for n,e in issued.items():predictions[n].append(e)
            dump(output/'training'/(tag(at)+'.json'),record)
            for kind in VARIANTS:
                histories[kind][at]=raws[kind]
                if models[kind] is not None:
                    with gzip.open(output/'models'/(kind+'_'+tag(at)+'.pkl.gz'),'wb') as stream:pickle.dump(models[kind],stream)
            source_history[at]=source_raw
            if source_model is not None:
                with gzip.open(output/'models'/('source_'+tag(at)+'.pkl.gz'),'wb') as stream:pickle.dump(source_model,stream)
            if (i+1)%200==0:print('trained through',at,flush=True)
    dump(output/'predictions.json',predictions);settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end']
    frozen={n:[e for e in events if start<=str(e['date'].date())<=end] for n,events in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_v35_capital_outcomes_seen':False,
        'prior_versions_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':frozen})
    test=features[features.date.between(start,end)];months=settings.get('months',settings.get('regression_months'));books={}
    result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for n,h in frozen.items():
        multiple=2 if n=='double_cost' else 1
        book=replay(market,test,list((stress if multiple==2 else base).values()),{e['at']:e['action'] for e in h},start,end,
            cost_multiple=multiple,allocations={e['at']:e['allocation'] for e in h},require_full_cash=False)
        books[n]=book;save_replay(output,n,book);result['routes'][n]=score(book,months)
    expected=[]
    for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)]:
        for c in CLOCKS:
            t=pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC')
            if t<pd.Timestamp(market.calendar[d]['close_at']):expected.append(t)
    covered=list(test.index)==expected
    result['coverage']={'complete':covered,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique(),
        'same_day_ticks_are_not_independent_dates':True};result['primary']=result['routes'][p['primary']]
    result['paired_increments']={n:paired_uncertainty(books[p['primary']][0],books[n][0]) for n in ROUTES if n not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],covered,p);result['development_gate']=gate
        result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['upstream_diagnostic_numbers']={'passed':bool(covered and result['primary']['primary_pass']),'policy_qualification_granted':False,'consumer_only':True}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    dump(output/'results.json',result);print({n:{'mean':s['primary_average'],'total':s['continuous_return']} for n,s in result['routes'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case-after',type=Path)
    a=p.parse_args();run(a.output,a.case_after)
