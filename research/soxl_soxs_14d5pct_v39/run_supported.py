"""Freeze top-choice risk-filtered actions, then evaluate a continuous cash ledger."""
import argparse
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from supported import *
import shutil
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay


def run(output,case_after=None,source_extension=None):
    p,prereg=load_frozen();auth=authorize(case_after) if case_after else None
    phase='cases' if auth else 'development';output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve every attempt')
    prior,prior_delivery=history_source()
    if auth:
        if source_extension is None:raise ValueError('Verified financial extension required')
        root,records,source_hashes=case_source(source_extension,auth)
    else:
        assert source_extension is None
        root,records,source_hashes=load_source()
    manifest=read(root/'input_manifest.json')
    market=core.load_market();assert market.manifest==manifest['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    panel=selected_panel(pd.read_parquet(root/'financial_panel.parquet'),records);features=pd.read_parquet(root/'features.parquet')
    for sub in ['','daily','training','executed_sources']:(output/sub).mkdir(parents=True,exist_ok=True)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,
        'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':auth,'source_sha256':prereg['code_sha256']})
    panel.to_parquet(output/'selected_panel.parquet')
    dump(output/'input_manifest.json',{'verified_source':str(root),'source_files_sha256':source_hashes,'market':manifest,'selected_history_source':str(prior),'selected_history_delivery_sha256':sha(core.ROOT/'research/soxl_soxs_14d5pct_v38/evidence/delivery_manifest.json')})
    for f in prereg['code_sha256']:
        src=core.ROOT/f;dest=output/'executed_sources'/src.parent.name/src.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(src.read_bytes())
    predictions={n:[] for n in ROUTES};reg=registry(panel,market);previous=None;daily=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            if row.date!=previous:
                source_day=read(root/'daily'/(str(row.date.date())+'.json'))
                daily=read(prior/'daily'/(str(row.date.date())+'.json')) if row.date<pd.Timestamp('2026-05-01') else fit_day(panel,records,reg,at,source_day)
                dump(output/'daily'/(str(row.date.date())+'.json'),daily);previous=row.date
            estimates=issue(panel,at,records[at],daily)
            issued=selected_events(parent.events(at,row.date,estimates,daily,records[at],records[at]['events']['full_band']))
            dump(output/'training'/(tag(at)+'.json'),{'at':at,'date':row.date,'clock':row.clock,'cutoff':day_cutoff(at),
                'estimates':estimates,'events':issued,'source_record_sha256':sha(root/'training'/(tag(at)+'.json'))})
            for n,e in issued.items():predictions[n].append(e)
            if (i+1)%200==0:print('eligible-risk through',at,flush=True)
    dump(output/'predictions.json',predictions);settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end']
    frozen={n:[e for e in es if start<=str(e['date'].date())<=end] for n,es in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_v39_capital_outcomes_seen':False,
        'prior_versions_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':frozen})
    test=features[features.date.between(start,end)];months=settings.get('months',settings.get('regression_months'));base=labels(root/'baseline_labels.json');stress=labels(root/'stress_labels.json');books={}
    result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for n,es in frozen.items():
        multiple=2 if n=='double_cost' else 1
        book=replay(market,test,list((stress if multiple==2 else base).values()),{e['at']:e['action'] for e in es},start,end,
            cost_multiple=multiple,allocations={e['at']:e['allocation'] for e in es},require_full_cash=False)
        books[n]=book;save_replay(output,n,book);result['routes'][n]=score(book,months)
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)]
              for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    complete=list(test.index)==expected
    result['coverage']={'complete':complete,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique()}
    result['primary']=result['routes'][p['primary']]
    result['paired_increments']={n:paired_uncertainty(books[p['primary']][0],books[n][0]) for n in ROUTES if n not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],complete,p);result['development_gate']=gate
        result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['case_acceptance']={'passed':bool(complete and result['primary']['primary_pass'])}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    dump(output/'results.json',result);print({n:{'mean':s['primary_average'],'total':s['continuous_return']} for n,s in result['routes'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-after',type=Path)
    parser.add_argument('--source-extension',type=Path)
    a=parser.parse_args();run(a.output,a.case_after,a.source_extension)
